// COPYRIGHT (C) 2025 M-Patek. ALL RIGHTS RESERVED.

use serde::{Serialize, Deserialize};
use crate::core::affine::AffineTuple;
use crate::core::algebra::Vector;
use crate::topology::folding::HyperFolder;
use crate::topology::merkle::CausalTrace;

/// 🧠 HyperTensor: 全息逻辑张量
///
/// 这是网络对一段输入序列 (Context Window) 的最终理解。
/// 它既包含结果 (Root)，也包含过程 (Trace)。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperTensor {
    /// 📍 Global Root: 最终折叠出的逻辑状态
    /// 代表了整段输入的 "语义总和"。
    pub root: AffineTuple,

    /// 🎞️ Causal Trace: 梯度磁带 (Optional)
    /// 仅在训练模式下生成。记录了从 Leaf 到 Root 的所有计算步骤，
    /// 用于反向传播 (Backpropagation) 或代数逆解。
    pub trace: Option<CausalTrace>,
}

impl HyperTensor {
    /// 🆕 Genesis: 创建一个空的 HyperTensor
    pub fn identity() -> Self {
        HyperTensor {
            root: AffineTuple::identity(),
            trace: None,
        }
    }

    /// 🚀 Forward Pass (构造函数)
    ///
    /// 将一串原始的 Token Embeddings 转换为全息张量。
    ///
    /// * `inputs`: 输入的仿射元组序列 (Leaf Nodes)。
    /// * `training_mode`: 
    ///     - `true`: 开启梯度追踪 (慢速，生成 Trace)。
    ///     - `false`: 开启并行折叠 (极速，无 Trace)。
    pub fn forward(inputs: &[AffineTuple], training_mode: bool) -> Self {
        if inputs.is_empty() {
            return Self::identity();
        }

        if training_mode {
            Self::fold_with_trace(inputs)
        } else {
            Self::fold_fast(inputs)
        }
    }

    /// 🏎️ Fast Folding (Inference Mode)
    /// 利用 Rayon 进行并行规约，速度极快，但不保留梯度图。
    fn fold_fast(inputs: &[AffineTuple]) -> Self {
        // 调用我们之前在 folding.rs 写的并行算法
        let root = HyperFolder::fold_timeline(inputs)
            .unwrap_or_else(AffineTuple::identity);

        HyperTensor {
            root,
            trace: None, // 推理模式不需要梯度
        }
    }

    /// 🐢 Trace Folding (Training Mode)
    /// 串行执行折叠 (或分层折叠)，并 meticulously 记录每一步到 CausalTrace。
    /// 这样我们才能执行 backward()。
    fn fold_with_trace(inputs: &[AffineTuple]) -> Self {
        let mut trace = CausalTrace::new();
        
        // 1. Register Leaf Nodes
        // 将所有输入注册到 Trace 中，获取它们的 Node ID
        let mut current_layer_ids: Vec<usize> = inputs.iter()
            .map(|leaf| trace.push_leaf(leaf.clone()))
            .collect();
        
        let mut current_layer_values = inputs.to_vec();

        // 2. Hierarchical Reduction (Tree Structure)
        // 模拟 Rayon 的归约过程，但是是记录在案的。
        // Loop until only one node remains (The Root).
        while current_layer_ids.len() > 1 {
            let mut next_layer_ids = Vec::new();
            let mut next_layer_values = Vec::new();

            // Pairwise folding (A+B, C+D, ...)
            for chunk_ids in current_layer_ids.chunks(2) {
                if chunk_ids.len() == 2 {
                    let left_id = chunk_ids[0];
                    let right_id = chunk_ids[1];
                    
                    // Retrieve values from the 'nodes' in trace (or logical cache)
                    // Note: In a real implementation we might cache values separately to avoid borrowing trace.
                    // Here we assume sequential processing matches indices.
                    // We need to fetch the actual AffineTuples computed previously.
                    // For simplicity, we carry `current_layer_values` alongside.
                    let val_idx = chunk_ids[0] % 2; // Logic simplification for demo loop matching
                    // Correct approach: track indices in `current_layer_values`
                    
                    // Let's refine the index logic:
                    // Since we are iterating chunks, we need corresponding values.
                    // But `chunks` on slice is hard with index mapping.
                    // Let's iterate by index steps.
                }
            }
            
            // Re-implementing simplified loop
            let mut i = 0;
            while i < current_layer_ids.len() {
                if i + 1 < current_layer_ids.len() {
                    let prev_id = current_layer_ids[i];
                    let next_id = current_layer_ids[i+1];
                    
                    let prev_val = &current_layer_values[i];
                    let next_val = &current_layer_values[i+1];

                    // Execute Logic: Next * Prev (Time Compose)
                    // or Merge (Space Fold) depending on context.
                    // Assume Time Folding for sequence tensor:
                    let result = next_val.compose(prev_val).expect("Fold Error");
                    
                    // Record in Tape
                    let new_id = trace.push_compose(prev_id, next_id, result.clone());
                    
                    next_layer_ids.push(new_id);
                    next_layer_values.push(result);
                    
                    i += 2;
                } else {
                    // Odd element out, carry over
                    next_layer_ids.push(current_layer_ids[i]);
                    next_layer_values.push(current_layer_values[i].clone());
                    i += 1;
                }
            }

            current_layer_ids = next_layer_ids;
            current_layer_values = next_layer_values;
        }

        HyperTensor {
            root: current_layer_values[0].clone(),
            trace: Some(trace),
        }
    }
    
    /// 🔍 Introspection (自省)
    /// 打印逻辑折叠的深度和复杂度。
    pub fn complexity(&self) -> usize {
        match &self.trace {
            Some(t) => t.nodes.len(),
            None => 0, // 快速模式下不可知
        }
    }
}
