// COPYRIGHT (C) 2025 M-Patek. ALL RIGHTS RESERVED.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use log::{info, debug, warn};
use rand::seq::SliceRandom;

use crate::net::node::NodeRole;

/// ⏱️ Peer Configuration
const PEER_TTL_SECS: u64 = 60;   // 超过 60秒 没心跳视为下线
const GOSSIP_INTERVAL_MS: u64 = 2000; // 每 2秒 八卦一次
const FANOUT: usize = 3;         // 每次随机告诉 3 个邻居

/// 🏷️ PeerInfo: 邻居节点的身份卡片
#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub id: String,
    pub address: String, // IP:Port
    pub role: NodeRole,
    pub last_seen: SystemTime,
    // 💡 Future: 加入 latency 或 load 指标用于更优的路由选择
}

/// 🌳 Topology: 我在网络中的位置
///
/// 这是一个逻辑上的“树”结构，用于 HyperFolder 的折叠路径。
/// 数据流向：Leaves -> Children -> Self -> Parent -> Root (PS)
#[derive(Debug, Clone)]
pub struct Topology {
    pub parent: Option<PeerInfo>,    // 我把结果汇报给谁
    pub children: Vec<PeerInfo>,     // 我负责汇总谁的结果
    pub is_root: bool,               // 我是否是最终的 Parameter Server
}

/// 📡 DiscoveryService: 负责节点发现与拓扑维护
pub struct DiscoveryService {
    local_id: String,
    local_role: NodeRole,
    local_addr: String,
    
    /// 📖 Routing Table: 这是一个线程安全的动态邻居表
    peers: Arc<RwLock<HashMap<String, PeerInfo>>>,
}

impl DiscoveryService {
    pub fn new(id: String, role: NodeRole, addr: String) -> Self {
        DiscoveryService {
            local_id: id,
            local_role: role,
            local_addr: addr,
            peers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 🌱 Seeding: 注入初始种子节点 (Bootstrapping)
    pub async fn add_seed_peer(&self, id: String, addr: String, role: NodeRole) {
        let mut peers = self.peers.write().await;
        peers.insert(id.clone(), PeerInfo {
            id,
            address: addr,
            role,
            last_seen: SystemTime::now(),
        });
    }

    /// 💓 Heartbeat: 更新某个节点的状态 (“我听到它的心跳了”)
    pub async fn register_heartbeat(&self, id: String, addr: String, role: NodeRole) {
        let mut peers = self.peers.write().await;
        peers.insert(id.clone(), PeerInfo {
            id,
            address: addr,
            role,
            last_seen: SystemTime::now(),
        });
    }

    /// 🗑️ GC: 清理掉线的节点
    pub async fn purge_dead_peers(&self) {
        let mut peers = self.peers.write().await;
        let now = SystemTime::now();
        peers.retain(|id, info| {
            if let Ok(duration) = now.duration_since(info.last_seen) {
                if duration.as_secs() < PEER_TTL_SECS {
                    return true;
                }
            }
            info!("💀 Peer [{}] timed out. Removing from topology.", id);
            false
        });
    }

    /// 🗣️ Gossip Protocol: 生成要发送给邻居的“八卦”信息
    /// 返回：(目标地址列表, 这里的全网视图)
    pub async fn generate_gossip(&self) -> (Vec<String>, Vec<PeerInfo>) {
        let peers = self.peers.read().await;
        
        // 1. 获取当前所有活着的节点列表
        let all_peers: Vec<PeerInfo> = peers.values().cloned().collect();
        
        // 2. 随机选择 k 个目标进行传播 (Fan-out)
        let mut rng = rand::thread_rng();
        let targets: Vec<String> = all_peers
            .choose_multiple(&mut rng, FANOUT)
            .map(|p| p.address.clone())
            .collect();
            
        // 3. 构建只有 ID/Addr/Role 的轻量级列表用于交换
        // (实际中可能只交换增量，这里为了演示交换全量)
        (targets, all_peers)
    }

    /// 🗣️ Gossip Handler: 处理收到的“八卦”
    pub async fn handle_gossip(&self, incoming_peers: Vec<PeerInfo>) {
        let mut local_peers = self.peers.write().await;
        for p in incoming_peers {
            // 不记录自己
            if p.id == self.local_id { continue; }

            // 简单的 LWW (Last-Write-Wins) 策略
            // 如果对方发来的节点我们没见过，或者比我们要新，就更新
            // 注意：这里用 SystemTime 其实有分布式时钟问题，
            // 严谨做法应使用 Logical Clock (Lamport Clock) 或 Vector Clock。
            // 但对于 Peer Discovery 的“存活”判定，本地时间收到消息的时间点即可。
            
            // 这里我们简化为：只要收到八卦，就认为该节点还活着
            local_peers.entry(p.id.clone())
                .and_modify(|local| local.last_seen = SystemTime::now())
                .or_insert_with(|| {
                    info!("✨ Discovered new peer via Gossip: [{}]", p.id);
                    PeerInfo {
                        last_seen: SystemTime::now(),
                        ..p
                    }
                });
        }
    }

    /// 📐 Topology Builder: 构建确定性聚合树
    ///
    /// 这是一个无中心算法。只要所有节点的 PeerTable 最终一致，
    /// 它们计算出的 Topology 就是一致的，无需额外协商。
    ///
    /// 规则：
    /// 1. 所有节点按 ID 排序。
    /// 2. 只有 PS 节点有资格成为 Tree 的 Root。
    /// 3. Worker 节点根据 Hash/ID 顺序挂载到 PS 或其他 Worker 下面。
    ///
    /// 简化实现：所有 Worker 组成一个平铺列表，分片挂载到可用的 PS 上。
    /// 如果只有一个 PS，那就是典型的 Master-Slave。
    /// 如果有多个 PS，Worker 会通过取模 (Hash % PS_Count) 自动负载均衡。
    pub async fn build_topology(&self) -> Topology {
        let peers_guard = self.peers.read().await;
        
        // 1. 区分角色
        let mut ps_nodes: Vec<&PeerInfo> = peers_guard.values()
            .filter(|p| p.role == NodeRole::ParameterServer)
            .collect();
        // 确保 PS 列表顺序确定
        ps_nodes.sort_by_key(|p| &p.id);

        // 如果我是 PS
        if self.local_role == NodeRole::ParameterServer {
            // 简单的逻辑：PS 负责所有连接到它的 Workers
            // 在更复杂的树中，PS 也可以有层级
            return Topology {
                parent: None, // Root 没爸爸
                children: Vec::new(), // 实际上 Worker 会主动连我，这里无需预设，或者作为白名单
                is_root: true,
            };
        }

        // 如果我是 Worker
        // 2. 寻找我的 Parent (Uplink)
        // 策略：Rendezvous Hashing (最高效的无状态负载均衡)
        // Parent = Max(Hash(SelfID + PotentialParentID))
        // 这里简化为：取模
        
        if ps_nodes.is_empty() {
            // 孤儿模式：没有发现 PS
            warn!("⚠️ No Parameter Server found! Topology is broken.");
            return Topology { parent: None, children: vec![], is_root: false };
        }

        // 简单的 Sharding: 根据我的 ID 决定我归哪个 PS 管
        // 假设 ID 是字符串，简单的 Hash 算法
        let my_hash: u64 = self.local_id.bytes().fold(0, |acc, b| acc.wrapping_add(b as u64));
        let ps_index = (my_hash as usize) % ps_nodes.len();
        let selected_parent = ps_nodes[ps_index].clone();

        // 3. 构建结果
        // 目前 Worker 是叶子节点 (Leaf)，没有 Children
        // 未来如果做多级聚合 (Tree-AllReduce)，Worker 也可以有 Children
        Topology {
            parent: Some(selected_parent),
            children: Vec::new(),
            is_root: false,
        }
    }
}
