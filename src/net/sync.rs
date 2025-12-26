// COPYRIGHT (C) 2025 M-Patek. ALL RIGHTS RESERVED.

use std::collections::{HashMap, HashSet};
use crate::core::algebra::{Matrix, Vector, Float};
use crate::net::wire::GradientUpdate;

/// 📊 AggregationResult: 聚合器的输出
pub enum AggregationResult {
    /// ⏳ 尚未收齐，继续等待
    Pending,
    /// ✅ 已收齐，输出聚合后的梯度（准备发给父节点或应用到模型）
    Complete(GradientUpdate),
    /// ⚠️ 这是一个过期的梯度（Epoch 落后），已丢弃
    Stale,
}

/// 🧠 LayerAccumulator: 单层的累加器
/// 负责处理 (g1*n1 + g2*n2) / (n1+n2) 的加权逻辑
struct LayerAccumulator {
    /// 累积的权重梯度和 (Σ g_w * n)
    weighted_sum_w: Vec<Float>,
    /// 累积的偏置梯度和 (Σ g_b * n)
    weighted_sum_b: Vec<Float>,
    /// 总样本数 (Σ n)
    total_batch: usize,
    /// 已贡献的节点 ID 集合 (防重复提交)
    contributors: HashSet<String>,
}

impl LayerAccumulator {
    fn new() -> Self {
        LayerAccumulator {
            weighted_sum_w: Vec::new(),
            weighted_sum_b: Vec::new(),
            total_batch: 0,
            contributors: HashSet::new(),
        }
    }

    /// ➕ 吸收一个新的梯度包
    fn absorb(&mut self, grad: &GradientUpdate, from_node: &str) {
        if self.contributors.contains(from_node) {
            return; // 幂等性保护：忽略重复提交
        }

        let n = grad.batch_size as Float;

        // 1. 初始化或累加 Weight 梯度
        if self.weighted_sum_w.is_empty() {
            // Init: g * n
            self.weighted_sum_w = grad.weight_grad.iter().map(|&g| g * n).collect();
        } else {
            // Accumulate: += g * n
            for (i, &g) in grad.weight_grad.iter().enumerate() {
                if i < self.weighted_sum_w.len() {
                    self.weighted_sum_w[i] += g * n;
                }
            }
        }

        // 2. 初始化或累加 Bias 梯度
        if self.weighted_sum_b.is_empty() {
            self.weighted_sum_b = grad.bias_grad.iter().map(|&g| g * n).collect();
        } else {
            for (i, &g) in grad.bias_grad.iter().enumerate() {
                if i < self.weighted_sum_b.len() {
                    self.weighted_sum_b[i] += g * n;
                }
            }
        }

        self.total_batch += grad.batch_size;
        self.contributors.insert(from_node.to_string());
    }

    /// ➗ 归一化并输出最终梯度
    /// New_Avg = Sum(Weighted_Grads) / Total_Batch
    fn finalize(&self, layer_idx: usize) -> GradientUpdate {
        let scale = if self.total_batch > 0 {
            1.0 / (self.total_batch as Float)
        } else {
            1.0
        };

        GradientUpdate {
            layer_index: layer_idx,
            weight_grad: self.weighted_sum_w.iter().map(|&x| x * scale).collect(),
            bias_grad: self.weighted_sum_b.iter().map(|&x| x * scale).collect(),
            batch_size: self.total_batch,
        }
    }
}

/// 🌊 GradientAggregator: 梯度同步聚合器
/// 管理所有层级的聚合状态
pub struct GradientAggregator {
    /// 全局 Epoch 计数器 (防止接收上一轮的延迟包)
    current_epoch: u64,
    
    /// 缓冲区: LayerIndex -> Accumulator
    buffers: HashMap<usize, LayerAccumulator>,
}

impl GradientAggregator {
    pub fn new() -> Self {
        GradientAggregator {
            current_epoch: 0,
            buffers: HashMap::new(),
        }
    }

    /// 🔄 设置新纪元 (清空旧缓冲)
    pub fn advance_epoch(&mut self, new_epoch: u64) {
        if new_epoch > self.current_epoch {
            self.current_epoch = new_epoch;
            self.buffers.clear();
        }
    }

    /// 📥 处理梯度更新
    ///
    /// * `grad`: 收到的梯度包
    /// * `from_node`: 来源节点 ID (如果是自己产生的，可传 "SELF")
    /// * `expected_children`: 根据拓扑，我应该等待哪些子节点 (ID List)
    pub fn aggregate(
        &mut self, 
        grad: GradientUpdate, 
        from_node: String, 
        expected_children: &[String]
    ) -> AggregationResult {
        // 简单起见，这里假设 GradientUpdate 结构里未来应该带 epoch 字段。
        // 目前假设网络是同步的，只处理当前逻辑。
        
        let layer_idx = grad.layer_index;
        
        // 1. 获取或创建累加器
        let acc = self.buffers
            .entry(layer_idx)
            .or_insert_with(LayerAccumulator::new);

        // 2. 吸收梯度
        acc.absorb(&grad, &from_node);

        // 3. 检查完整性 (Completeness Check)
        // 我们需要等待：所有子节点 + 我自己 ("SELF")
        // expected_count = children.len() + 1
        let mut all_needed: HashSet<String> = expected_children.iter().cloned().collect();
        all_needed.insert("SELF".to_string()); // 必须包含本地计算的梯度

        if acc.contributors.is_superset(&all_needed) {
            // ✅ 召唤神龙：所有碎片已集齐
            let final_grad = acc.finalize(layer_idx);
            
            // 清理缓冲区 (该层本轮已完成)
            self.buffers.remove(&layer_idx);
            
            return AggregationResult::Complete(final_grad);
        }

        AggregationResult::Pending
    }
}
