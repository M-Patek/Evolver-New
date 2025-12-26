// COPYRIGHT (C) 2025 M-Patek. ALL RIGHTS RESERVED.

use rayon::prelude::*;
use crate::core::affine::AffineTuple;
use crate::core::algebra::Float;

/// 📦 Accumulator (Monoid Structure)
/// 
/// 引入 Monoid 结构以修复空间折叠的结合律问题。
/// 原理：(Sum, Count) 是满足结合律的，而 Average 不是。
/// 
/// (S1, N1) + (S2, N2) = (S1+S2, N1+N2)
/// Associativity: ((A+B)+C) == (A+(B+C))
struct Accumulator {
    sum: AffineTuple,
    count: usize,
}

impl Accumulator {
    fn new(leaf: AffineTuple) -> Self {
        Accumulator {
            sum: leaf,
            count: 1,
        }
    }

    // Identity element for the Monoid
    fn zero() -> Self {
        Accumulator {
            sum: AffineTuple::zeros(),
            count: 0,
        }
    }

    fn merge(self, other: Self) -> Self {
        // 使用纯加法合并，避免中间平均导致的精度损失和结合律破坏
        Accumulator {
            sum: self.sum.add_components(&other.sum),
            count: self.count + other.count,
        }
    }
    
    fn finalize(self) -> Option<AffineTuple> {
        if self.count == 0 {
            None
        } else {
            // 最后一步统一归一化：Mean = Sum / Count
            let scale = 1.0 / (self.count as Float);
            Some(self.sum.scale(scale))
        }
    }
}

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
    /// 🛠️ 修正 (Fix): 
    /// 原先直接使用 Average 不满足结合律，导致并行结果不确定。
    /// 现改为 "Map-Reduce-Finalize" 模式，使用 Accumulator (Monoid) 保证数学确定性。
    pub fn fold_context(branches: &[AffineTuple]) -> Option<AffineTuple> {
        if branches.is_empty() { return None; }

        // Phase 1: Map (Lift to Monoid) & Reduce (Parallel Sum)
        let final_acc = branches.par_iter()
            .map(|branch| Accumulator::new(branch.clone()))
            .reduce(
                || Accumulator::zero(), 
                |a, b| a.merge(b)
            );

        // Phase 2: Finalize (Normalize)
        final_acc.finalize()
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
