// COPYRIGHT (C) 2025 M-Patek. ALL RIGHTS RESERVED.

use std::sync::Arc;
use tokio::sync::RwLock;
use log::{info, warn, error};

use crate::core::algebra::{Vector, Matrix};
use crate::core::affine::AffineTuple;
use crate::core::neuron::HTPNeuron;
use crate::core::oracle::LogicOracle;
use crate::topology::tensor::HyperTensor;
use crate::net::wire::{PacketType, GradientUpdate, ModelSnapshot, LayerState};
use crate::train_loop::SimpleOptimizer;

/// 🎭 NodeRole: 节点身份
#[derive(Debug, Clone, PartialEq)]
pub enum NodeRole {
    /// 👷 Worker: 负责执行前向推理和反向传播计算
    Worker,
    /// 🧠 ParameterServer: 负责维护全局真理（权重）并执行更新
    ParameterServer,
}

/// 🤖 HTPNode: 神经节点实体
pub struct HTPNode {
    pub id: String,
    pub role: NodeRole,
    
    /// 🧠 Local Memory: 本地存储的神经网络模型
    /// Worker 存的是副本 (Cache)，PS 存的是真理 (Master)
    /// 使用 Arc<RwLock> 实现线程安全的并发访问
    pub model: Arc<RwLock<Vec<HTPNeuron>>>,

    /// ⚡ Optimizer: 仅 PS 节点持有，用于更新权重
    pub optimizer: Option<SimpleOptimizer>,
}

impl HTPNode {
    /// 初始化一个新节点
    pub fn new(id: String, role: NodeRole, model_depth: usize) -> Self {
        // 初始化空白模型 (实际应用中应从磁盘加载或通过网络同步)
        let mut neurons = Vec::with_capacity(model_depth);
        for _ in 0..model_depth {
            neurons.push(HTPNeuron::new());
        }

        let optimizer = match role {
            NodeRole::ParameterServer => Some(SimpleOptimizer::new(1e-3)), // 默认学习率
            NodeRole::Worker => None,
        };

        HTPNode {
            id,
            role,
            model: Arc::new(RwLock::new(neurons)),
            optimizer,
        }
    }

    /// 📨 Packet Processor: 核心消息处理循环
    /// 模拟接收到一个网络包并处理 (实际应配合 Quinn/Tokio Stream 使用)
    pub async fn process_packet(&self, packet: PacketType) -> Option<PacketType> {
        match packet {
            PacketType::Handshake { node_id, protocol_ver } => {
                info!("🤝 Handshake received from [{}] (v{})", node_id, protocol_ver);
                // 这里可以返回一个 HandshakeAck，暂时略过
                None
            }

            PacketType::InferenceRequest { request_id, input_state } => {
                if self.role != NodeRole::Worker {
                    warn!("⚠️ PS received InferenceRequest. Ignoring.");
                    return None;
                }
                self.handle_inference(request_id, input_state).await
            }

            PacketType::GradientPush(grad) => {
                if self.role != NodeRole::ParameterServer {
                    warn!("⚠️ Worker received GradientPush. Ignoring.");
                    return None;
                }
                self.handle_gradient_update(grad).await
            }

            PacketType::ParameterBroadcast(snapshot) => {
                if self.role != NodeRole::Worker {
                    return None; // PS 通常不接收广播，除非是多级 PS 架构
                }
                self.handle_parameter_sync(snapshot).await
            }

            _ => None,
        }
    }

    /// 🧠 [Worker Logic]: 执行推理
    async fn handle_inference(&self, request_id: u64, input: Vector) -> Option<PacketType> {
        info!("🧠 Worker [{}] processing Request #{}", self.id, request_id);

        let model_guard = self.model.read().await;
        
        // 1. 构建计算图输入
        // 这里简化处理：假设模型是单层或简单的串行结构，将输入包装为 AffineTuple
        // 实际的 Evolver 会构建复杂的 HyperTensor
        let input_tuple = AffineTuple::new(Matrix::identity(), input);
        
        // 2. 模拟网络前向传播 (Forward Pass)
        // 这里的逻辑是将输入通过所有神经元折叠。
        // 为了演示，我们取第一个神经元进行处理。
        let mut result_vector = Vector::zeros();
        if let Some(first_neuron) = model_guard.first() {
             // Clone 神经元状态以避免由于借用检查器导致的冲突，
             // 在实际高性能场景下应使用 Zero-copy。
             let mut neuron_clone = first_neuron.clone(); 
             result_vector = neuron_clone.absorb(&input_tuple.translation);
        }

        // 3. 返回结果
        Some(PacketType::InferenceResponse {
            request_id,
            output_state: result_vector,
        })
    }

    /// 📉 [PS Logic]: 梯度下降更新
    async fn handle_gradient_update(&self, grad: GradientUpdate) -> Option<PacketType> {
        info!("📉 PS [{}] applying gradients to Layer {}", self.id, grad.layer_index);

        if let Some(opt) = &self.optimizer {
            let mut model_guard = self.model.write().await;
            
            if let Some(target_neuron) = model_guard.get_mut(grad.layer_index) {
                // 1. 重构梯度矩阵
                // GradientUpdate 传输的是扁平化的 Vec<Float>，需要还原为 Matrix
                let weight_grad_mat = Matrix::new(
                    target_neuron.logic_gate.linear.rows,
                    target_neuron.logic_gate.linear.cols,
                    grad.weight_grad
                );

                // 2. 执行优化器步骤 (W = W - lr * grad)
                opt.apply_gradient(&mut target_neuron.logic_gate.linear, &weight_grad_mat);
                
                // 3. 更新 Bias (简单相减)
                // 实际 SimpleOptimizer 也应该支持 Bias，这里手动演示
                let bias_grad_vec = Vector::new(grad.bias_grad);
                let lr = 1e-3; // 暂时硬编码，应从 params 读取
                target_neuron.logic_gate.translation = target_neuron.logic_gate.translation
                    .sub(&bias_grad_vec.scale(lr));

                info!("✅ Weights updated via Gradient Descent.");
                
                // 4. (可选) 触发广播：如果更新累计到一定程度，广播新参数
                // 这里为了演示，每次更新都广播（效率极低，仅作逻辑展示）
                return Some(self.create_snapshot(&model_guard));
            }
        }
        None
    }

    /// 🧬 [Worker Logic]: 同步全局参数
    async fn handle_parameter_sync(&self, snapshot: ModelSnapshot) -> Option<PacketType> {
        info!("🧬 Worker [{}] syncing with Global Truth (Epoch {})", self.id, snapshot.epoch);
        
        let mut model_guard = self.model.write().await;
        
        for layer_state in snapshot.layers {
            if layer_state.layer_index < model_guard.len() {
                // 覆盖本地权重
                model_guard[layer_state.layer_index].logic_gate.linear = layer_state.weights;
                model_guard[layer_state.layer_index].logic_gate.bias = layer_state.bias; // 修正: LayerState 定义里是 bias
            }
        }
        None
    }

    /// 📸 Helper: 创建模型快照
    fn create_snapshot(&self, neurons: &[HTPNeuron]) -> PacketType {
        let layers = neurons.iter().enumerate().map(|(idx, n)| {
            LayerState {
                layer_index: idx,
                weights: n.logic_gate.linear.clone(),
                bias: n.logic_gate.translation.clone(),
            }
        }).collect();

        PacketType::ParameterBroadcast(ModelSnapshot {
            epoch: 0, // 实际应维护全局 Epoch 计数器
            layers,
        })
    }
}
