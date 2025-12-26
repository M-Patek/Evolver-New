// COPYRIGHT (C) 2025 M-Patek. ALL RIGHTS RESERVED.

use super::algebra::{Float, MANIFOLD_DIM};
use serde::{Serialize, Deserialize};

/// ⚙️ HyperParams: 逻辑流形的物理法则配置
///
/// 在白盒架构中，我们不再需要 "Discriminant" (判别式) 或 "Class Group" 参数。
/// 取而代之的是定义流形几何形状和动力学特性的超参数。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperParams {
    /// 📏 Manifold Dimension (流形维度)
    /// 必须与编译时常量 MANIFOLD_DIM 保持一致 (用于运行时校验)。
    /// 维度越高，能表达的逻辑概念越复杂。
    pub dimension: usize,

    /// 🧱 Network Depth (逻辑深度)
    /// 定义了因果链的最大长度。这决定了模型能进行多长步骤的连续推理。
    pub depth: usize,

    /// ⚡ Learning Rate (学习率 / Eta)
    /// 用于梯度下降 (Gradient Descent) 的步长。
    /// 在 White-Box 模式下，这是显式可调的。
    pub learning_rate: Float,

    /// 🛡️ Lipschitz Bound (稳定性约束 K)
    /// 权重矩阵的谱范数上限 (Spectral Norm Bound)。
    /// 约束 ||W|| <= K。如果 K > 1 太多，系统会陷入混沌 (蝴蝶效应)；
    /// 如果 K < 1，梯度会消失。理想值略大于 1.0。
    pub lipschitz_bound: Float,

    /// 🎯 Zero-Hallucination Tolerance (Epsilon)
    /// 判定逻辑是否“闭合”的几何误差阈值。
    /// 如果 ||Prediction - Target|| > Epsilon，则判定为幻觉。
    pub tolerance_epsilon: Float,
}

impl Default for HyperParams {
    /// 标准配置 (Standard Mode)
    /// 平衡了推理深度和训练稳定性。
    fn default() -> Self {
        HyperParams {
            dimension: MANIFOLD_DIM,
            depth: 12,           // 12层逻辑深度，足够处理一般因果推断
            learning_rate: 1e-3, // 典型的 AdamW 学习率
            lipschitz_bound: 1.05, // 允许轻微的扩张，保持信号流动
            tolerance_epsilon: 1e-4, // 标准几何检查精度
        }
    }
}

impl HyperParams {
    /// 🔬 高保真模式 (High-Fidelity Mode)
    /// 用于需要极高逻辑精度的场景 (如数学证明生成)。
    /// 特点：更深的网络，更严格的约束，更慢的学习率。
    pub fn high_fidelity() -> Self {
        HyperParams {
            dimension: MANIFOLD_DIM, 
            depth: 24,             // 深度加倍
            learning_rate: 5e-4,   // 慢速精细调整
            lipschitz_bound: 1.01, // 极严格的稳定性，接近等距映射 (Isometry)
            tolerance_epsilon: 1e-6, // 显微镜级别的误差容忍
        }
    }

    /// 🚀 快速直觉模式 (Fast-Inference Mode)
    /// 用于实时响应，允许一定的模糊性，换取速度和泛化能力。
    pub fn fast_inference() -> Self {
        HyperParams {
            dimension: MANIFOLD_DIM,
            depth: 6,              // 浅层推理
            learning_rate: 1e-2,   // 快速收敛
            lipschitz_bound: 1.10, // 允许更大幅度的状态跳跃
            tolerance_epsilon: 1e-3, // 较低的容忍度
        }
    }

    /// 验证参数的物理合理性
    pub fn validate(&self) -> Result<(), String> {
        if self.dimension != MANIFOLD_DIM {
            return Err(format!("Dimension Mismatch: Config expects {}, but binary compiled with {}", self.dimension, MANIFOLD_DIM));
        }
        if self.lipschitz_bound < 0.9 {
            return Err("Lipschitz constant too low: Will cause Vanishing Gradient.".to_string());
        }
        if self.lipschitz_bound > 2.0 {
            return Err("Lipschitz constant too high: Will cause Exploding Gradient / Chaos.".to_string());
        }
        Ok(())
    }
}
