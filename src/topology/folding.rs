// COPYRIGHT (C) 2025 M-Patek. ALL RIGHTS RESERVED.

use rayon::prelude::*;
use crate::core::affine::AffineTuple;

/// 📂 HyperFolder: 拓扑折叠器 (Topological Folder)
///
/// 负责将大量的逻辑单元 (AffineTuple) 通过时间或空间算子压缩成单一的“全息摘要”。
/// 由于我们的代数算子满足结合律，我们可以利用 Rayon 实现 Log(N) 复杂度的自动并行折叠。
pub struct HyperFolder;

impl HyperFolder {
    /// ⏳ Time Folding (Sequential -> Instant)
    /// 
    /// 物理含义: 将时间线上的一系列连续步骤 A -> B -> C -> ... -> Z 
    /// 压缩为一个单一的等效变换矩阵 T_total。
    /// 
    /// T_total = A_z * ... * A_c * A_b * A_a
    /// 
    /// 并行化原理: 
    /// 虽然矩阵乘法不满足交换律 (A*B != B*A)，但满足结合律 ((A*B)*C = A*(B*C))。
    /// 因此我们可以将长链切分为 Chunk 并行计算，最后再合并。
    pub fn fold_timeline(timeline: &[AffineTuple]) -> Option<AffineTuple> {
        if timeline.is_empty() { return None; }

        // Rayon's reduce_with uses a tree-based reduction algorithm,
        // which naturally fits the associativity requirement.
        let result = timeline.par_iter()
            .cloned()
            .reduce_with(|prev_step, next_step| {
                // ⚠️ Crucial: Maintain Causal Order
                // compose(prev) means: new_matrix = self * prev
                // So we want: next_step.compose(&prev_step)
                next_step.compose(&prev_step).expect("Time Folding Error: Lipschitz bound violated?")
            });

        result
    }

    /// 🌌 Space Folding (Parallel -> Unified)
    /// 
    /// 物理含义: 将多个独立的上下文分支 (Branches) 融合为一个统一的上下文。
    /// 类似于 Transformer 中的 Multi-Head Attention 的结果聚合，但这里是几何融合。
    /// 
    /// 算法: Tree Reduction using Commutative Merge (Average/Normalize).
    pub fn fold_context(branches: &[AffineTuple]) -> Option<AffineTuple> {
        if branches.is_empty() { return None; }

        // 由于 commutative_merge 实现为 (A+B)/2，
        // 树状归约 (Tree Reduction) 能够保证所有分支的权重相对均衡。
        // Rayon 默认使用树状归约。
        let result = branches.par_iter()
            .cloned()
            .reduce_with(|branch_a, branch_b| {
                branch_a.commutative_merge(&branch_b).expect("Space Folding Error")
            });

        result
    }
    
    /// 🧱 Layer Folding (Deep Stacking)
    /// 
    /// 用于将上一层的输出折叠为下一层的输入。
    /// (简单的 wrapper，但在深度网络拓扑中有语义价值)
    pub fn fold_layers(layer_outputs: &[AffineTuple]) -> Option<AffineTuple> {
        // Layers imply sequence (Bottom -> Up), so we use Time Folding logic
        // strictly speaking, layer composition is functional composition.
        Self::fold_timeline(layer_outputs)
    }
}
