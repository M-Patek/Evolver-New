// COPYRIGHT (C) 2025 M-Patek. ALL RIGHTS RESERVED.

use crate::core::algebra::{Vector, Matrix, Float, MANIFOLD_DIM};
use crate::core::affine::AffineTuple;
use crate::core::neuron::HTPNeuron;
use crate::core::oracle::LogicOracle;
use crate::core::param::HyperParams;
use crate::topology::tensor::HyperTensor;

/// 🏋️ TrainingLoop: 逻辑进化训练器
///
/// White-Box 架构支持两种训练模式：
/// 1. Gradient Descent (通识学习): 通过大量样本慢慢调整权重，学习通用逻辑模式。
/// 2. Algebraic Solver (顿悟/One-Shot): 通过代数逆运算，瞬间学会特定事实。
pub struct TrainingLoop {
    params: HyperParams,
    optimizer: SimpleOptimizer,
}

impl TrainingLoop {
    pub fn new(params: HyperParams) -> Self {
        TrainingLoop {
            params: params.clone(),
            optimizer: SimpleOptimizer::new(params.learning_rate),
        }
    }

    /// 📉 Mode 1: Gradient Descent Step (反向传播)
    /// 适用于学习通用规律 (Generalization)
    pub fn train_step_sgd(
        &mut self, 
        inputs: &[AffineTuple], 
        target_root: &AffineTuple
    ) -> Float {
        // 1. Forward Pass (with Trace)
        // 开启 training_mode=true 以记录梯度磁带
        let hyper_tensor = HyperTensor::forward(inputs, true);
        
        // 2. Compute Loss
        // L = || Prediction - Target ||^2
        // 这里简化为只计算 Translation (Bias) 的误差，实际应包含 Linear 部分
        let loss = LogicOracle::calculate_loss(
            &hyper_tensor.root.translation, 
            &target_root.translation
        );

        // 3. Backward Pass (Auto-Diff)
        // 从 Trace 中反向推导梯度
        if let Some(trace) = &hyper_tensor.trace {
            // 计算输出层的梯度 dL/dOut
            // dL/dOut = 2 * (Pred - Target)
            let diff = hyper_tensor.root.translation.sub(&target_root.translation);
            let grad_output = AffineTuple::new(
                Matrix::new(MANIFOLD_DIM, MANIFOLD_DIM, vec![0.0; MANIFOLD_DIM*MANIFOLD_DIM]), // 简化: 忽略矩阵梯度
                diff.scale(2.0)
            );

            // 反向传播到叶子节点
            let _leaf_grads = trace.backward(&grad_output);

            // 4. Update Weights (Optimizer Step)
            // 在真实实现中，这里会根据 leaf_grads 更新对应的 Embedding 或 Neuron 权重
            // self.optimizer.step(&mut model_params, &leaf_grads);
        }

        loss
    }

    /// ⚡ Mode 2: Algebraic One-Shot Solver (瞬间学习)
    /// 适用于记忆特定事实 (Memorization)
    /// "Input A + Input B -> Must imply Target C"
    pub fn train_step_solver(
        &mut self,
        neuron: &mut HTPNeuron, // 目标神经元
        input_state: &Vector,
        target_state: &Vector
    ) -> Float {
        // 1. Check current error
        let current_output = neuron.absorb(input_state);
        let initial_loss = LogicOracle::calculate_loss(&current_output, target_state);

        // 如果误差已经很小，跳过
        if initial_loss < self.params.tolerance_epsilon {
            return initial_loss;
        }

        // 2. Solve for Delta W (The Magic)
        // 询问 Oracle：我需要怎么改权重，才能让 input 完美映射到 target？
        let delta_w = LogicOracle::compute_ideal_update(
            input_state, 
            target_state, 
            &neuron.logic_gate
        );

        // 3. Apply Update Immediately
        // W_new = W_old + Delta_W * Learning_Rate
        // (Solver 模式下 LR 通常为 1.0，即完全接受建议)
        let w_update = delta_w.scale(1.0); 
        neuron.logic_gate.linear = neuron.logic_gate.linear.add(&w_update);
        
        // 同时修正 Bias (Fix fixed-point drift)
        neuron.force_learn_bias(input_state, target_state);

        // 4. Verify
        let new_output = neuron.absorb(input_state);
        let final_loss = LogicOracle::calculate_loss(&new_output, target_state);

        final_loss
    }
}

/// 🔧 SimpleOptimizer: 基础梯度下降优化器
pub struct SimpleOptimizer {
    learning_rate: Float,
}

impl SimpleOptimizer {
    pub fn new(lr: Float) -> Self {
        SimpleOptimizer { learning_rate: lr }
    }

    /// W = W - lr * Grad
    pub fn apply_gradient(&self, weights: &mut Matrix, grad: &Matrix) {
        let step = grad.scale(-self.learning_rate);
        *weights = weights.add(&step);
    }
}
