// COPYRIGHT (C) 2025 M-Patek. ALL RIGHTS RESERVED.

// 1. Algebra: 线性代数基础 (Matrix, Vector, Float)
// 这是整个 White-Box 架构的数学基石。
pub mod algebra;

// 2. Affine: 仿射变换算子 (AffineTuple)
// 定义了非交换的时间算子和交换的空间算子。
pub mod affine;

// 3. Param: 物理超参数 (HyperParams)
// 定义流形维度、Lipschitz 约束和学习率。
pub mod param;

// 4. Primes (Refactored to Init): 初始化与嵌入 (ConceptEmbedder)
// 虽然文件名叫 primes (历史遗留)，但现在负责 Xavier 初始化和 Token 嵌入。
pub mod primes;

// 5. Neuron: 神经单元 (HTPNeuron)
// 具体的流形坐标处理器。
pub mod neuron;

// 6. Oracle: 逻辑导师 (LogicOracle)
// 负责计算 Loss、验证几何一致性和提供代数逆解。
pub mod oracle;
