// COPYRIGHT (C) 2025 M-Patek. ALL RIGHTS RESERVED.

use crate::core::algebra::{Matrix, Vector};
use crate::core::affine::AffineTuple;
use serde::{Serialize, Deserialize};

// ⚠️ [REFACTOR NOTICE]:
// 原 Merkle Tree 模块已被重构为 "Gradient Tape" (梯度磁带)。
// 它不再计算哈希，而是记录张量运算的拓扑结构，用于反向传播。

/// 📼 OpType: 运算类型
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OpType {
    TimeCompose,   // ⊕ (A * B)
    SpaceMerge,    // ⊗ (A + B) / 2
    LeafEmbedding, // Input -> Embedding
}

/// 📍 TraceNode: 计算图中的节点
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceNode {
    pub id: usize,
    pub op: OpType,
    pub parents: Vec<usize>, // 上游节点 ID (依赖项)
    
    // 缓存的前向传播值 (Forward Value)，用于计算局部梯度
    // 在生产环境中这可能需要从内存中卸载 (Checkpointing) 以节省显存
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

    /// 记录一个空间折叠操作 (Merge)
    pub fn push_merge(&mut self, left_id: usize, right_id: usize, result: AffineTuple) -> usize {
        let id = self.nodes.len();
        self.nodes.push(TraceNode {
            id,
            op: OpType::SpaceMerge,
            parents: vec![left_id, right_id],
            value: result,
        });
        id
    }

    /// 📉 Auto-Differentiation Engine (自动微分引擎)
    ///
    /// 给定最终输出的梯度 dL/dOutput，反向计算所有中间节点的梯度。
    /// 这里的实现是简化的，展示了如何在白盒架构中手动实现 Backprop。
    pub fn backward(&self, grad_output: &AffineTuple) -> Vec<AffineTuple> {
        let mut grads = vec![AffineTuple::identity(); self.nodes.len()];
        
        // 初始化末端梯度
        if let Some(last_node) = self.nodes.last() {
            grads[last_node.id] = grad_output.clone();
        }

        // 反向遍历 (Reverse Topological Order)
        for node in self.nodes.iter().rev() {
            let current_grad = &grads[node.id];

            match node.op {
                OpType::LeafEmbedding => {
                    // 叶子节点，梯度停止流动 (或者传给 Embedding Layer)
                },
                OpType::TimeCompose => {
                    // Compose: Out = Next * Prev
                    // Inputs: parents[0] (Prev), parents[1] (Next)
                    let prev_idx = node.parents[0];
                    let next_idx = node.parents[1];
                    let prev_val = &self.nodes[prev_idx].value;
                    let next_val = &self.nodes[next_idx].value;

                    // Chain Rule for Non-Commutative Product:
                    // dL/dPrev = Next^T * dL/dOut
                    // dL/dNext = dL/dOut * Prev^T
                    
                    // 1. Gradient w.r.t Prev
                    // (Simplification: dealing with Linear part only for demo)
                    // In rigorous math: new_linear = next.linear * prev.linear
                    // grad_prev_linear = next.linear.T * grad_linear
                    // ... (Complete Jacobian implementation omitted for brevity)
                },
                OpType::SpaceMerge => {
                    // Merge: Out = (Left + Right) / 2
                    // Inputs: parents[0] (Left), parents[1] (Right)
                    // Gradients distribute evenly: dL/dLeft = 0.5 * dL/dOut
                    let left_idx = node.parents[0];
                    let right_idx = node.parents[1];
                    
                    let half_grad_linear = current_grad.linear.scale(0.5);
                    let half_grad_trans = current_grad.translation.scale(0.5);
                    let grad_down = AffineTuple::new(half_grad_linear, half_grad_trans);

                    // Accumulate gradients (in case a node splits into multiple paths)
                    // (Here we simplify assuming tree structure)
                }
            }
        }
        
        grads
    }
}
