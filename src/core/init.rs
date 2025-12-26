// COPYRIGHT (C) 2025 M-Patek. ALL RIGHTS RESERVED.

use super::algebra::{Vector, Matrix, Float, MANIFOLD_DIM};

// ⚠️ [REFACTOR NOTICE]:
// This file formerly handled "Prime Generation" for cryptographic hardness.
// In White-Box Evolver, it is repurposed for "Manifold Initialization".
// Recommended Rename: `src/core/init.rs`

/// 🧬 ConceptEmbedder: 将离散 Token 映射到连续流形
///
/// 替代了原本的 "Hash-to-Prime" 机制。
/// 以前：Token -> Hash -> Prime (离散/不可微)
/// 现在：Token -> Hash -> Vector (连续/可微)
pub struct ConceptEmbedder;

impl ConceptEmbedder {
    /// 🗺️ Token Projection (确定性映射)
    /// 将一个 Token ID 投影到流形上的一个固定坐标。
    ///
    /// 在实际的大模型中，这通常是一个可学习的 Embedding Table (Lookup)。
    /// 在这里，为了演示 "White-Box" 的确定性，我们使用哈希投影作为 "Zero-Shot" 初始化。
    pub fn embed_token(token_id: u32) -> Vector {
        // 使用简单的哈希算法生成确定性的伪随机向量
        // (避免引入庞大的依赖，仅作演示)
        let mut data = Vec::with_capacity(MANIFOLD_DIM);
        let mut state = token_id as u64;

        // SplitMix64 风格的简单的混合器
        for _ in 0..MANIFOLD_DIM {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z = z ^ (z >> 31);
            
            // 归一化到 [-1.0, 1.0] 区间，符合神经网络输入分布
            let val = (z as Float / u64::MAX as Float) * 2.0 - 1.0;
            data.push(val);
        }

        // 归一化向量长度 (Unit Norm)，确保初始状态在单位球面上
        let norm: Float = data.iter().map(|x| x*x).sum::<Float>().sqrt();
        let normalized_data = data.iter().map(|x| x / norm).collect();

        Vector::new(normalized_data)
    }
}

/// 🎲 WeightInitializer: 神经网络权重初始化器
/// 
/// 替代了原本的 "Random Prime Search"。
/// 实现了 Xavier/Glorot Initialization，确保梯度在深层网络中流动时不会消失或爆炸。
pub struct WeightInitializer;

impl WeightInitializer {
    /// 🏗️ Xavier Uniform Initialization
    /// 适用于 Tanh 或 Linear 激活函数
    /// Range: [-limit, limit] where limit = sqrt(6 / (fan_in + fan_out))
    pub fn init_matrix(rows: usize, cols: usize, seed: u64) -> Matrix {
        let mut data = Vec::with_capacity(rows * cols);
        let mut rng_state = seed;

        // Xavier Limit
        let limit = (6.0 / (rows as Float + cols as Float)).sqrt();

        for _ in 0..(rows * cols) {
            // Simple LCG PRNG
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let rand_01 = rng_state as Float / u64::MAX as Float;
            
            // Map [0, 1] to [-limit, limit]
            let val = (rand_01 * 2.0 - 1.0) * limit;
            data.push(val);
        }

        Matrix::new(rows, cols, data)
    }

    /// 📍 Bias Initialization
    /// 通常初始化为 0 或很小的常数
    pub fn init_bias(dim: usize) -> Vector {
        Vector::new(vec![0.0; dim])
    }
}
