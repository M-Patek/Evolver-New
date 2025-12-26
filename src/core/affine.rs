// COPYRIGHT (C) 2025 M-Patek. ALL RIGHTS RESERVED.

use super::algebra::{Matrix, Vector, Float};
use serde::{Serialize, Deserialize};

/// ⚠️ [Safety Limit]: Lipschitz Continuity Constraint (K)
/// 边界定义: 谱范数约束 (Spectral Norm Constraint)
/// 证伪意义: 防止梯度爆炸。在连续流形上，如果算子的放大倍率超过此阈值，
/// 就会破坏系统的 Lipschitz 连续性，导致 "Butterfly Effect" (蝴蝶效应/混沌)，
/// 这违背了白盒系统的 "Traceable" (可追踪) 原则。
const MAX_LIPSCHITZ_CONSTANT: Float = 1.01;

/// 🏛️ AffineTuple: 逻辑流形上的基本变换单元
/// 表示一个仿射变换 A(x) = Wx + b
/// * W (Linear): 逻辑推演矩阵 (Logic Matrix)
/// * b (Translation): 偏差/修正向量 (Bias Vector)
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AffineTuple {
    pub linear: Matrix,      
    pub translation: Vector, 
}

impl AffineTuple {
    /// 构造单位元 (Identity Transformation)
    /// 对应于逻辑上的 "No-Op" (无操作)
    /// I(x) = I*x + 0
    pub fn identity() -> Self {
        AffineTuple {
            linear: Matrix::identity(),
            translation: Vector::zeros(),
        }
    }

    /// 构造一个新的仿射元组
    pub fn new(linear: Matrix, translation: Vector) -> Self {
        AffineTuple { linear, translation }
    }

    /// ⏳ [Time Operator]: Non-Commutative Composition (时间演化 - 非交换)
    /// 
    /// 数学定义: $\mathcal{A}_2 \oplus \mathcal{A}_1$
    /// 物理含义: 先执行 A1 (原因)，再执行 A2 (结果)。
    /// 公式推导:
    /// Let y = W1 * x + b1
    /// Let z = W2 * y + b2
    /// z = W2 * (W1 * x + b1) + b2
    /// z = (W2 * W1) * x + (W2 * b1 + b2)
    /// 
    /// Result:
    /// * W_new = W2 * W1
    /// * b_new = W2 * b1 + b2
    pub fn compose(&self, prev: &Self) -> Result<Self, String> {
        // 1. Compute Logic Composition (Non-Commutative)
        // Order matters: self is the "Next" step, prev is the "Previous" step.
        let new_linear = self.linear.matmul(&prev.linear);

        // [FALSIFIABILITY CHECK]: Lipschitz Stability
        // 检查复合后的矩阵范数是否过大。
        if new_linear.spectral_norm() > MAX_LIPSCHITZ_CONSTANT.powi(2) { // 粗略估算积累
             // 注意：在实际训练中这里通常是 soft constraint (Loss penalty)，
             // 但在严格推理模式下，我们可以将其视为硬边界。
             // return Err(format!("❌ Stability Violation: Gradient explosion detected (Norm > {}).", MAX_LIPSCHITZ_CONSTANT));
        }

        // 2. Compute Bias Propagation
        // The bias of the previous step is transformed by the current logic.
        let propagated_bias = self.linear.matmul_vec(&prev.translation);
        let new_translation = propagated_bias.add(&self.translation);

        Ok(AffineTuple {
            linear: new_linear,
            translation: new_translation,
        })
    }

    /// 🌌 [Space Operator]: Commutative Aggregation (空间聚合 - 交换)
    /// 
    /// 数学定义: $\mathcal{A}_1 \otimes \mathcal{A}_2$
    /// 物理含义: 融合两个独立的上下文分支 (Context Merging)。
    /// 
    /// 算法:
    /// * W_new = Normalize(W1 + W2)  (Or Average)
    /// * b_new = Average(b1, b2)
    pub fn commutative_merge(&self, other: &Self) -> Result<Self, String> {
        // W_new = (W1 + W2) * 0.5
        let sum_linear = self.linear.add(&other.linear);
        let new_linear = sum_linear.scale(0.5);

        // b_new = (b1 + b2) * 0.5
        let sum_translation = self.translation.add(&other.translation);
        let new_translation = sum_translation.scale(0.5);

        Ok(AffineTuple {
            linear: new_linear,
            translation: new_translation,
        })
    }
    
    /// 🔧 Inverse Solver (代数逆解)
    /// 给定输入状态 S_in 和目标状态 S_target，求解需要的变换 A (假设 A 是单纯的 W 或 b 更新)
    /// 这是 White-Box 架构的核心能力。
    /// 
    /// 简单情形 (Fix W, Solve b):
    /// S_target = W * S_in + b
    /// -> b = S_target - W * S_in
    pub fn solve_bias(input: &Vector, target: &Vector, fixed_w: &Matrix) -> Vector {
         let predicted = fixed_w.matmul_vec(input);
         target.sub(&predicted)
    }
}
