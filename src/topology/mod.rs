// COPYRIGHT (C) 2025 M-Patek. ALL RIGHTS RESERVED.

/// 🧠 Topology: 逻辑流形的拓扑结构
///
/// 本模块定义了 White-Box Evolver 的"大脑皮层"。
/// 它负责将微观的神经元 (Neurons) 组织成宏观的全息张量 (HyperTensor)。
///
/// 包含三大支柱：
/// 1. HyperTensor (tensor.rs): 动态计算图容器，支持推理模式和训练模式。
/// 2. HyperFolder (folding.rs): 基于 Rayon 的高速并行折叠算法 (Inference Engine)。
/// 3. CausalTrace (merkle.rs): 梯度磁带与因果追踪器 (Training Engine / 原 Merkle 树)。

pub mod tensor;
pub mod folding;
pub mod merkle;
