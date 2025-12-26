// COPYRIGHT (C) 2025 M-Patek. ALL RIGHTS RESERVED.

use super::algebra::{Vector, Matrix, Float, MANIFOLD_DIM};
use super::affine::AffineTuple;

/// 🔮 LogicOracle: 逻辑导师与真理裁决者
///
/// 在白盒架构中，Oracle 扮演 "Ground Truth" 的角色。
/// 它负责生成训练任务，并计算逻辑推演的误差。
pub struct LogicOracle;

impl LogicOracle {
    /// ⚖️ [Loss Function]: Geodesic Error Calculation
    /// 计算预测状态与目标真值之间的几何距离。
    /// 在欧几里得近似下，使用 MSE (Mean Squared Error)。
    /// 
    /// L = || S_pred - S_target ||^2
    pub fn calculate_loss(predicted: &Vector, target: &Vector) -> Float {
        let diff = predicted.sub(target);
        // L2 Norm Squared
        diff.data.iter().map(|x| x * x).sum()
    }

    /// 🛡️ [Verification]: Geometric Consistency Check
    /// 验证推理结果是否在允许的误差范围内 (Epsilon Ball)。
    /// 这是 "Zero Hallucination" 的判定标准。
    pub fn verify_logic(predicted: &Vector, target: &Vector, epsilon: Float) -> bool {
        let loss = Self::calculate_loss(predicted, target);
        loss < epsilon
    }

    /// 🎓 [The Solver]: One-Shot Weight Solver (Delta Rule)
    /// 
    /// 这是 White-Box 架构的杀手锏。
    /// 给定输入 S_in 和目标 S_target，以及当前的权重 W_old，
    /// 计算出让 S_in 精确映射到 S_target 所需的最小权重修正量 ΔW。
    /// 
    /// Math:
    /// Error E = S_target - (W_old * S_in)
    /// We want ΔW such that ΔW * S_in = E
    /// Solution (Minimal Norm): ΔW = (E * S_in^T) / ||S_in||^2
    pub fn compute_ideal_update(
        input: &Vector, 
        target: &Vector, 
        current_gate: &AffineTuple
    ) -> Matrix {
        // 1. Calculate Prediction Error: E = Target - (W * Input + b)
        let current_pred = current_gate.linear.matmul_vec(input);
        let current_pos = current_pred.add(&current_gate.translation);
        let error = target.sub(&current_pos); // Error Vector

        // 2. Calculate Input Norm Squared: ||x||^2
        let input_norm_sq: Float = input.data.iter().map(|x| x*x).sum();
        
        // Prevent division by zero
        if input_norm_sq < 1e-9 {
            return Matrix { rows: MANIFOLD_DIM, cols: MANIFOLD_DIM, data: vec![0.0; MANIFOLD_DIM * MANIFOLD_DIM] };
        }

        // 3. Compute Outer Product: E * x^T
        // Result is a matrix where M_ij = E_i * x_j
        let mut delta_data = vec![0.0; MANIFOLD_DIM * MANIFOLD_DIM];
        for i in 0..MANIFOLD_DIM {
            for j in 0..MANIFOLD_DIM {
                delta_data[i * MANIFOLD_DIM + j] = (error.data[i] * input.data[j]) / input_norm_sq;
            }
        }

        Matrix {
            rows: MANIFOLD_DIM,
            cols: MANIFOLD_DIM,
            data: delta_data,
        }
    }

    /// 🎲 [Synthetic Data]: Generate Random Premise
    /// 生成一个随机的单位向量作为逻辑前提。
    /// (Simple placeholder implementation)
    pub fn genesis_premise(seed: u64) -> Vector {
        // Simple LCG based generation to avoid external 'rand' crate dependency for now
        let mut data = Vec::with_capacity(MANIFOLD_DIM);
        let mut state = seed;
        for _ in 0..MANIFOLD_DIM {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = (state as f64 / u64::MAX as f64) as Float; // 0.0 to 1.0
            data.push(val * 2.0 - 1.0); // -1.0 to 1.0
        }
        Vector::new(data)
    }
}
