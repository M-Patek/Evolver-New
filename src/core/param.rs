// COPYRIGHT (C) 2025 M-Patek. ALL RIGHTS RESERVED.

use super::algebra::{Float, MANIFOLD_DIM};
use serde::{Serialize, Deserialize};

/// ⚙️ HyperParams: 逻辑流形的物理法则配置
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperParams {
    /// 📏 Manifold Dimension
    pub dimension: usize,

    /// 🧱 Network Depth
    pub depth: usize,

    /// ⚡ Learning Rate
    pub learning_rate: Float,

    /// 🛡️ Lipschitz Bound (稳定性约束 K)
    /// 权重矩阵的谱范数上限 (Spectral Norm Bound)。
    /// 
    /// 修正: 使用 estimate_spectral_norm() 进行真实估算。
    /// 理想值应略大于 1.0 (如 1.05) 以允许信息在层间有效传递，
    /// 但必须小于混沌阈值。
    pub lipschitz_bound: Float,

    /// 🎯 Zero-Hallucination Tolerance (Epsilon)
    pub tolerance_epsilon: Float,
}

impl Default for HyperParams {
    fn default() -> Self {
        HyperParams {
            dimension: MANIFOLD_DIM,
            depth: 12,
            learning_rate: 1e-3,
            lipschitz_bound: 1.05, // 修正后的安全阈值
            tolerance_epsilon: 1e-4,
        }
    }
}

impl HyperParams {
    pub fn high_fidelity() -> Self {
        HyperParams {
            dimension: MANIFOLD_DIM, 
            depth: 24,
            learning_rate: 5e-4,
            lipschitz_bound: 1.01, // 接近等距映射
            tolerance_epsilon: 1e-6,
        }
    }

    pub fn fast_inference() -> Self {
        HyperParams {
            dimension: MANIFOLD_DIM,
            depth: 6,
            learning_rate: 1e-2,
            lipschitz_bound: 1.10, 
            tolerance_epsilon: 1e-3,
        }
    }

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
