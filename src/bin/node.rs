// COPYRIGHT (C) 2025 M-Patek. ALL RIGHTS RESERVED.

use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use log::{info, error, warn, debug};
use tokio::sync::mpsc;

// 引入我们之前构建的模块
use htp_core::net::node::{HTPNode, NodeRole};
use htp_core::net::discovery::{DiscoveryService, PeerBrief}; // 假设 PeerBrief 已在 wire 或 discovery 中定义
use htp_core::net::wire::{PacketType, PROTOCOL_VERSION};
use htp_core::core::param::HyperParams;

/// 🚀 Evolver Node CLI
/// 启动一个 Hyper-Tensor 神经节点
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// 节点唯一标识 (如: "node-01")
    #[arg(short, long)]
    id: String,

    /// 监听地址 (如: "127.0.0.1:5000")
    #[arg(short, long)]
    listen: SocketAddr,

    /// 节点角色 (worker 或 ps)
    #[arg(short, long, default_value = "worker")]
    role: String,

    /// 种子节点地址 (可选，用于加入集群)
    #[arg(short, long)]
    seed: Option<String>, // 格式: "id@ip:port"
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // 1. 初始化日志
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));
    let args = Args::parse();

    info!("🚀 Starting Evolver Node [{}]...", args.id);

    // 2. 确定角色
    let role = match args.role.as_str() {
        "ps" => NodeRole::ParameterServer,
        "worker" => NodeRole::Worker,
        _ => panic!("Invalid role. Use 'worker' or 'ps'."),
    };
    info!("🎭 Identity: {:?} | Listening on: {}", role, args.listen);

    // 3. 初始化核心组件
    // (a) 大脑: HTPNode (负责推理与梯度)
    let node = Arc::new(HTPNode::new(
        args.id.clone(),
        role.clone(),
        12, // 默认深度，实际应从 Config 读取
    ));

    // (b) 感官: DiscoveryService (负责发现邻居)
    let discovery = Arc::new(DiscoveryService::new(
        args.id.clone(),
        role.clone(),
        args.listen.to_string(),
    ));

    // (c) 神经: Quinn Networking (QUIC Transport)
    let (endpoint, mut incoming) = make_server_endpoint(args.listen)?;

    // 4. 处理种子节点 (Bootstrapping)
    if let Some(seed_str) = args.seed {
        // 简单解析 "node-00@127.0.0.1:5000"
        if let Some((seed_id, seed_addr)) = seed_str.split_once('@') {
            info!("🌱 Bootstrapping via Seed: {} @ {}", seed_id, seed_addr);
            // 假设 Seed 默认为 PS，实际应查询
            discovery.add_seed_peer(seed_id.to_string(), seed_addr.to_string(), NodeRole::ParameterServer).await;
        }
    }

    // ==================================================================
    // 🔁 Background Tasks (后台生命维持系统)
    // ==================================================================
    
    let disc_clone = discovery.clone();
    let endpoint_clone = endpoint.clone();
    
    // Task A: Gossip & Heartbeat Loop
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(2000));
        loop {
            interval.tick().await;
            
            // 1. 清理死节点
            disc_clone.purge_dead_peers().await;

            // 2. 生成八卦信息
            let (targets, peer_list) = disc_clone.generate_gossip().await;
            
            // 3. 发送八卦
            if !targets.is_empty() {
                // 转换 PeerInfo -> PeerBrief (Wire Protocol)
                let briefs: Vec<PeerBrief> = peer_list.iter().map(|p| PeerBrief {
                    id: p.id.clone(),
                    address: p.address.clone(),
                    role_code: if p.role == NodeRole::ParameterServer { 1 } else { 0 },
                }).collect();

                let gossip_packet = PacketType::PeerDiscovery {
                    sender_id: disc_clone.local_id(), // 需在 DiscoveryService 暴露此 getter
                    peers: briefs,
                };

                // 尝试发送给随机选中的邻居
                for target_addr in targets {
                    let _ = send_packet(&endpoint_clone, &target_addr, &gossip_packet).await;
                }
            }
        }
    });

    // ==================================================================
    // 🔁 Main Loop (主事件循环)
    // ==================================================================
    info!("👂 Node is active. Waiting for signals...");

    while let Some(conn) = incoming.next().await {
        let node_ref = node.clone();
        let disc_ref = discovery.clone();
        let endpoint_ref = endpoint.clone();

        // 为每个连接启动一个处理协程
        tokio::spawn(async move {
            let connection = match conn.await {
                Ok(c) => c,
                Err(e) => { warn!("🔥 Connection failed: {}", e); return; },
            };

            // 每一个流代表一个请求/消息包
            loop {
                // 读取流
                let mut recv_stream = match connection.accept_uni().await {
                    Ok(s) => s,
                    Err(_) => break, // 连接关闭
                };

                // 读取二进制数据
                let payload = match recv_stream.read_to_end(1024 * 1024).await {
                    Ok(data) => data,
                    Err(_) => break,
                };

                // 反序列化
                if let Ok(packet) = PacketType::from_bytes(&payload) {
                    // 1. 拦截 Discovery 包 (Gossip)
                    if let PacketType::PeerDiscovery { sender_id, peers } = &packet {
                        // 更新路由表
                        // 这里需要把 PeerBrief 转回 PeerInfo，并记录来源 IP
                        // 简化处理: 直接交给 DiscoveryService
                        debug!("🗣️ Received Gossip from {}", sender_id);
                        // disc_ref.handle_gossip(...).await; 
                        continue;
                    }

                    // 2. 交给大脑处理 (Inference / Gradient)
                    if let Some(response) = node_ref.process_packet(packet).await {
                        // 3. 如果有回执，发回去 (例如 InferenceResponse)
                        // 注意：这里我们收的是 Uni stream，如果要回复，需要建立反向流或双向流
                        // 为了简化，这里假设对方监听地址在 Packet payload 里或通过 discovery 查找
                        // 真实实现中 QUIC 通常用 Bi-stream (双向流)
                        // 这里仅演示逻辑: 查路由表 -> 发送
                    }
                }
            }
        });
    }

    Ok(())
}

// ==================================================================
// 🛠️ Network Utilities (QUIC Boilerplate)
// ==================================================================

/// 创建 QUIC 服务端 Endpoint
fn make_server_endpoint(bind_addr: SocketAddr) -> Result<(quinn::Endpoint, quinn::Incoming), Box<dyn Error>> {
    // 1. 生成自签名证书 (Ephemeral)
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()])?;
    let cert_der = cert.serialize_der()?;
    let priv_key = cert.serialize_private_key_der();
    let priv_key = rustls::PrivateKey(priv_key);
    let cert_chain = vec![rustls::Certificate(cert_der)];

    // 2. 配置 Server TLS
    let mut server_config = rustls::ServerConfig::builder()
        .with_safe_defaults()
        .with_no_client_auth()
        .with_single_cert(cert_chain, priv_key)?;
    server_config.alpn_protocols = vec![b"htp-v1".to_vec()]; // Application Layer Protocol Negotiation

    // 3. 构建 Quinn Server Config
    let server_config = quinn::ServerConfig::with_crypto(Arc::new(server_config));
    
    // 4. 绑定端口
    let endpoint = quinn::Endpoint::server(server_config, bind_addr)?;
    
    Ok((endpoint, incoming)) // 注意: quinn 0.10 API 略有不同，这里是概念代码
}

/// 发送 UDP/QUIC 包的辅助函数
async fn send_packet(endpoint: &quinn::Endpoint, target_addr: &str, packet: &PacketType) -> Result<(), Box<dyn Error>> {
    // 解析地址
    let remote: SocketAddr = target_addr.parse()?;
    
    // 建立连接 (如果已连接 Quinn 会复用)
    let connection = endpoint.connect(remote, "localhost")?.await?;
    
    // 打开单向流
    let mut send_stream = connection.open_uni().await?;
    
    // 序列化并发送
    let bytes = packet.to_bytes().map_err(|s| s.to_string())?; // Convert String error to Box<dyn Error>
    send_stream.write_all(&bytes).await?;
    send_stream.finish().await?;

    Ok(())
}
