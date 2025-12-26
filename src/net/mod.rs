// COPYRIGHT (C) 2025 M-Patek. ALL RIGHTS RESERVED.

/// 📡 Wire Protocol: 分布式传输协议
///
/// 定义了 White-Box 架构中节点间的数据包格式。
/// 与旧版 Evolver 传输 "ZK-Proofs" 不同，新版传输的是：
/// 1. Forward Pass: 逻辑状态向量 (Inference State)
/// 2. Backward Pass: 梯度更新 (Gradient Updates)
/// 3. Synchronization: 模型参数快照 (Model Snapshots)
pub mod wire;

// 🔮 Future Roadmap (待实现模块):
//
// pub mod node;      // P2P 节点逻辑 (Worker / Parameter Server)
// pub mod discovery; // 节点发现与拓扑构建
// pub mod sync;      // 梯度聚合算法 (Ring-AllReduce / Gossip)
