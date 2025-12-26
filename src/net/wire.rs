// COPYRIGHT (C) 2025 M-Patek. ALL RIGHTS RESERVED.

use serde::{Serialize, Deserialize};
use crate::core::algebra::{Vector, Matrix, Float};

/// 📦 WireProtocol: 网络传输协议版本
pub const PROTOCOL_VERSION: u32 = 2; // White-Box Era

/// 📡 PacketType: 定义消息的意图
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PacketType {
    /// 🤝 Handshake: 节点加入网络
    Handshake { node_id: String, protocol_ver: u32 },
    
    /// 🧠 ForwardPass: 推理请求 (传输输入状态)
    /// "这是前提 A，请推导结论。"
    InferenceRequest { 
        request_id: u64,
        input_state: Vector 
    },
    
    /// 💡 InferenceResult: 推理响应 (传输输出状态)
    /// "根据逻辑 A，导出的结论坐标是 B。"
    InferenceResponse { 
        request_id: u64, 
        output_state: Vector 
    },

    /// 📉 GradientUpdate: 分布式训练 (传输梯度)
    /// "我算出了这个 Batch 的误差，这是我对权重的修正建议。"
    GradientPush(GradientUpdate),

    /// 🧬 ModelSync: 权重同步 (传输模型参数)
    /// "这是最新的全局共识逻辑参数。"
    ParameterBroadcast(ModelSnapshot),
}

/// 📉 GradientUpdate: 梯度传输包
/// 包含了一个 Layer 的权重梯度和偏差梯度
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientUpdate {
    /// 目标层级 ID
    pub layer_index: usize,
    
    /// ∇W (Weight Gradient): 扁平化的矩阵梯度
    pub weight_grad: Vec<Float>,
    
    /// ∇b (Bias Gradient): 向量梯度
    pub bias_grad: Vec<Float>,
    
    /// Batch Size (用于聚合平均)
    pub batch_size: usize,
}

/// 📸 ModelSnapshot: 模型快照
/// 用于新节点同步或 Parameter Server 广播
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSnapshot {
    pub epoch: u64,
    pub layers: Vec<LayerState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerState {
    pub layer_index: usize,
    pub weights: Matrix,
    pub bias: Vector,
}

/// 🛠️ Serialization Utilities
impl PacketType {
    /// 序列化为二进制流 (Bincode / Protobuf)
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self).map_err(|e| e.to_string())
    }

    /// 从二进制流反序列化
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        bincode::deserialize(data).map_err(|e| e.to_string())
    }
}
