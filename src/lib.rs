// COPYRIGHT (C) 2025 M-Patek. ALL RIGHTS RESERVED.

//! # White-Box Evolver (Hyper-Tensor Protocol)
//!
//! A pure-logic neural architecture based on:
//! 1. **Differentiable Manifolds** (Replacing Class Groups)
//! 2. **Non-Commutative Affine Algebra** (Preserving Causality)
//! 3. **Algebraic Inversion** (One-Shot Learning)
//!
//! This library provides the core mathematical kernels and topological structures
//! to build and train transparent logic machines.

// ==================================================================
// 1. Core Mathematical Kernels (The Heart)
// ==================================================================
// 包含：线性代数(algebra)、仿射算子(affine)、神经元(neuron)、
// 物理参数(param)、逻辑导师(oracle) 和 初始化器(primes/init)。
pub mod core;

// ==================================================================
// 2. Topological Structures (The Brain)
// ==================================================================
// 包含：全息张量(tensor)、并行折叠(folding) 和 梯度磁带(merkle)。
pub mod topology;

// ==================================================================
// 3. Training & Evolution (The Gym)
// ==================================================================
// 包含：梯度下降循环 和 代数逆解器。
pub mod train_loop;

// ==================================================================
// 4. Distributed Networking (The Nervous System)
// ==================================================================
// 包含：梯度传输协议 和 状态同步。
pub mod net;

// ==================================================================
// 5. Tests Module
// ==================================================================
// 包含流式折叠验证和代数求解验证。
#[cfg(test)]
mod tests {
    pub mod streaming_test;
}

// ==================================================================
// 🌟 Prelude: The All-in-One Import
// ==================================================================
/// 方便用户快速开始开发。
/// Usage: use evolver::prelude::*;
pub mod prelude {
    // 1. Math Basics
    pub use crate::core::algebra::{Vector, Matrix, Float};
    pub use crate::core::affine::AffineTuple;
    
    // 2. Core Units
    pub use crate::core::neuron::HTPNeuron;
    pub use crate::core::param::HyperParams;
    pub use crate::core::oracle::LogicOracle;
    
    // 3. Initialization (Mapping "Primes" to "Embeddings")
    pub use crate::core::primes::{ConceptEmbedder, WeightInitializer};

    // 4. Topology
    pub use crate::topology::tensor::HyperTensor;

    // 5. Training
    pub use crate::train_loop::{TrainingLoop, SimpleOptimizer};
}
