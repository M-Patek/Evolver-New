// COPYRIGHT (C) 2025 M-Patek. ALL RIGHTS RESERVED.

use super::affine::AffineTuple;
use super::algebra::{Vector, Matrix};
use serde::{Serialize, Deserialize};

/// 🧠 HTPNeuron: 逻辑流形上的基本神经单元
///
/// 与输出标量激活值的传统神经元不同，HTP 神经元维护着一个高维坐标 (Vector)。
/// 它不仅仅是“激活”，它是“思考”的一个快照。
///
/// 状态方程: S_t = W * S_{t-1} + b
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HTPNeuron {
    /// 📍 Current Logic Coordinate (当前思维坐标)
    /// 代表该神经元当前持有的逻辑状态 S
    pub state: Vector,

    /// ⚙️ Intrinsic Logic Gate (内在逻辑门 / 权重)
    /// 定义了该神经元如何处理输入信息：(W, b)
    pub logic_gate: AffineTuple,
}

impl HTPNeuron {
    /// Genesis: 在原点创建一个空白神经元
    /// 初始状态为 0，逻辑门为恒等变换 (Identity)
    pub fn new() -> Self {
        HTPNeuron {
            state: Vector::zeros(),
            logic_gate: AffineTuple::identity(),
        }
    }

    /// 使用特定的权重初始化神经元
    pub fn with_weights(linear: Matrix, bias: Vector) -> Self {
        HTPNeuron {
            state: Vector::zeros(),
            logic_gate: AffineTuple::new(linear, bias),
        }
    }

    /// 🔄 Time Evolution / Forward Pass (时间演化)
    ///
    /// 物理含义: 神经元 "吸收" 输入状态，应用自己的逻辑规则，推导出新的状态。
    /// 公式: S_new = W * S_input + b
    pub fn absorb(&mut self, input: &Vector) -> Vector {
        // 1. Apply Linear Logic (W * x)
        // 这一步代表 "推理" (Deduction)
        let linear_part = self.logic_gate.linear.matmul_vec(input);

        // 2. Apply Bias/Correction (+ b)
        // 这一步代表 "修正" (Adjustment)
        let new_state = linear_part.add(&self.logic_gate.translation);

        // 3. Update Internal Memory
        self.state = new_state.clone();

        new_state
    }

    /// 🧬 Algebraic One-Shot Learning (代数逆解 / 瞬间学习)
    ///
    /// 这是一个 "Solver" 的微观实现。
    /// 场景: 如果我们知道对于输入 Input，正确的输出必须是 Target。
    /// 假设 W 固定，我们可以在一步之内求解出需要的偏差 b。
    ///
    /// 公式: b = Target - W * Input
    pub fn force_learn_bias(&mut self, input: &Vector, target: &Vector) {
        // 计算 W * Input
        let predicted_linear = self.logic_gate.linear.matmul_vec(input);
        
        // 求解 b = Target - Prediction
        let new_bias = target.sub(&predicted_linear);
        
        // 瞬间更新权重，无需迭代
        self.logic_gate.translation = new_bias;
    }
    
    /// 🔍 Manifold Integrity Check (流形完整性检查)
    /// 防止 NaN (Not a Number) 或 Inf (无穷大) 污染网络。
    /// 這是 "Zero Hallucination" 的物理基础之一。
    pub fn verify_integrity(&self) -> Result<(), String> {
        for val in self.state.as_slice() {
            if !val.is_finite() {
                return Err("🔥 Neuron Meltdown: State contains NaN or Infinity. Logic manifold collapsed.".to_string());
            }
        }
        Ok(())
    }
}
