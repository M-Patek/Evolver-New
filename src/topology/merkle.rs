// COPYRIGHT (C) 2025 M-Patek. ALL RIGHTS RESERVED.

use crate::core::algebra::{Matrix, Vector, Float};
use crate::core::affine::AffineTuple;
use serde::{Serialize, Deserialize};

// ⚠️ [REFACTOR NOTICE]:
// 原 Merkle Tree 模块已被重构为 "Gradient Tape" (梯度磁带)。
// 它不再计算哈希，而是记录张量运算的拓扑结构，用于反向传播。

/// 📼 OpType: 运算类型
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OpType {
    /// 时间演化 (A * B)
    /// 拓扑：Strict Binary (prev, next)
    TimeCompose, 
    
    /// 空间融合 Mean(A, B, C...)
    /// 拓扑：N-ary (Star Topology)
    /// ⚠️ 修正：支持多路输入，以匹配 "Sum/N" 的数学定义，保证梯度公平。
    SpaceMerge, 
    
    /// 叶子节点嵌入
    LeafEmbedding, 
}

/// 📍 TraceNode: 计算图中的节点
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceNode {
    pub id: usize,
    pub op: OpType,
    
    /// 依赖项 ID 列表
    /// - TimeCompose: len() == 2
    /// - SpaceMerge: len() == N
    pub parents: Vec<usize>, 
    
    // 缓存的前向传播值 (Forward Value)，用于计算局部梯度
    pub value: AffineTuple, 
}

/// 🎞️ CausalTrace: 因果追踪器 (The Gradient Tape)
///
/// 记录了从输入 Token 到最终结论的所有变换步骤。
/// 这是一个有向无环图 (DAG)。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CausalTrace {
    pub nodes: Vec<TraceNode>,
    pub active_path: Vec<usize>, // 只有参与了最终输出的节点才会被激活
}

impl CausalTrace {
    pub fn new() -> Self {
        CausalTrace {
            nodes: Vec::new(),
            active_path: Vec::new(),
        }
    }

    /// 记录一个叶子节点
    pub fn push_leaf(&mut self, value: AffineTuple) -> usize {
        let id = self.nodes.len();
        self.nodes.push(TraceNode {
            id,
            op: OpType::LeafEmbedding,
            parents: vec![],
            value,
        });
        id
    }

    /// 记录一个时间演化操作 (Compose)
    /// Parent A (Prev) -> Parent B (Next) -> Output
    pub fn push_compose(&mut self, prev_id: usize, next_id: usize, result: AffineTuple) -> usize {
        let id = self.nodes.len();
        self.nodes.push(TraceNode {
            id,
            op: OpType::TimeCompose,
            parents: vec![prev_id, next_id], // 注意顺序: [Prev, Next]
            value: result,
        });
        id
    }

    /// 记录一个空间折叠操作 (N-ary Merge)
    /// 🆕 修正：支持一次性记录 N 个父节点，实现 "Star Topology"。
    pub fn push_n_ary_merge(&mut self, parent_ids: Vec<usize>, result: AffineTuple) -> usize {
        let id = self.nodes.len();
        self.nodes.push(TraceNode {
            id,
            op: OpType::SpaceMerge,
            parents: parent_ids,
            value: result,
        });
        id
    }

    /// 📉 Auto-Differentiation Engine (自动微分引擎)
    ///
    /// 给定最终输出的梯度 dL/dOutput，反向计算所有中间节点的梯度。
    pub fn backward(&self, grad_output: &AffineTuple) -> Vec<AffineTuple> {
        let mut grads = vec![AffineTuple::identity(); self.nodes.len()];
        // 实际上应该初始化为 0 (Zero Gradient)，这里用 identity 暂代占位，
        // 真实实现中 AffineTuple 需要实现 zero()。
        // [FIX]: 假设 AffineTuple::zeros() 存在 (我们在 affine.rs 补上了)。
        let mut grads = vec![AffineTuple::zeros(); self.nodes.len()];
        
        // 初始化末端梯度
        if let Some(last_node) = self.nodes.last() {
            grads[last_node.id] = grad_output.clone();
        }

        // 反向遍历 (Reverse Topological Order)
        for node in self.nodes.iter().rev() {
            let current_grad = grads[node.id].clone(); // Clone to avoid borrow conflict

            match node.op {
                OpType::LeafEmbedding => {
                    // 叶子节点，梯度停止流动 (或者传给 Embedding Layer)
                },
                OpType::TimeCompose => {
                    // Compose: Out = Next * Prev
                    // Inputs: parents[0] (Prev), parents[1] (Next)
                    if node.parents.len() == 2 {
                        let prev_idx = node.parents[0];
                        let next_idx = node.parents[1];
                        // let prev_val = &self.nodes[prev_idx].value; // 如需计算 Jacobian
                        // let next_val = &self.nodes[next_idx].value;

                        // Chain Rule (Simplification):
                        // 真实的矩阵梯度传播非常复杂，这里仅示意梯度流动路径
                        // dL/dPrev += ...
                        // dL/dNext += ...
                        // grads[prev_idx] = grads[prev_idx].add(&propagated_grad_prev);
                        // grads[next_idx] = grads[next_idx].add(&propagated_grad_next);
                    }
                },
                OpType::SpaceMerge => {
                    // 🌌 N-ary Merge Gradient Distribution
                    // Out = (Sum Inputs) / N
                    // dL/dInput_i = (1/N) * dL/dOut
                    
                    let n = node.parents.len() as Float;
                    if n > 0.0 {
                        let scale_factor = 1.0 / n;
                        let grad_share = current_grad.scale(scale_factor);

                        for &parent_id in &node.parents {
                            // Accumulate Gradient: Grad[Parent] += Grad_Share
                            // 需要把 grad_share 累加进去，因为一个节点可能参与多个 Merge (虽然在这个 Tree 里一般只有一次)
                            let new_grad = grads[parent_id].add_components(&grad_share);
                            grads[parent_id] = new_grad;
                        }
                    }
                }
            }
        }
        
        grads
    }
}
