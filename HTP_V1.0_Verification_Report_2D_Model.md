# Hyper-Tensor Protocol: 抗压测试与验证报告 (2D Toy Model)

**版本:** 1.0  
**日期:** 2025-12-27  
**测试对象:** 空间算子 (Space Operator) 与 代数求解器 (Algebraic Solver)  
**测试环境:** 纸面推演 (Theoretical Derivation) vs 代码逻辑核查 (Codebase Verification)

---

## 1. 测试背景与目标
本测试旨在验证 **White-Box Evolver** 架构在极端简化条件下的数学刚性。我们构建了一个 $D=2$ 的二维线性系统，包含多分支上下文输入和多层神经元推理链，以验证以下核心断言：

* **空间算子的结合律 (Associativity):** 证明新版算子消除了并行归约树（Reduction Tree）的结构敏感性，实现了全局上下文的一致性。
* **时间演化的确定性 (Causality):** 验证神经元的前向推理是否严格遵循仿射变换逻辑。
* **代数逆解的有效性 (Invertibility):** 验证 "One-Shot Solver" 是否能精确计算出权重修正量，实现零误差学习。

---

## 2. 实验设置 (Setup)

### 2.1 拓扑结构
* **流形维度:** $D=2$
* **输入上下文:** 3 个并行分支 $\mathcal{B}_1, \mathcal{B}_2, \mathcal{B}_3$。
* **推理链路:** 2 层神经元串行结构 ($Neuron_1 \to Neuron_2$)。

### 2.2 初始参数
* **上下文向量:**
    * $v_1 = [1, 0]^T$
    * $v_2 = [0, 1]^T$
    * $v_3 = [1, 1]^T$
* **神经元权重 (初始化):**
    * $W_1 = \begin{bmatrix} 1 & 0.2 \\ 0 & 0.9 \end{bmatrix}, \quad b_1 = [0.1, -0.1]^T$
    * $W_2 = \begin{bmatrix} 0.95 & 0 \\ 0.1 & 1 \end{bmatrix}, \quad b_2 = [0, 0.05]^T$

---

## 3. 验证 I：空间算子 (Space Operator)

### 3.1 问题陈述
旧版“二元平均”算子 $\text{Mean}(A, B) = (A+B)/2$ 不满足结合律，导致并行计算结果依赖于折叠顺序（括号位置）。
* **分支 A:** $((v_1 \otimes v_2) \otimes v_3) \to [0.75, 0.75]^T$
* **分支 B:** $(v_1 \otimes (v_2 \otimes v_3)) \to [0.75, 0.50]^T$
* **结论:** 结果不收敛，系统存在“幻觉”风险。

### 3.2 解决方案与验证
新版架构采用了 **Monoid Accumulator (Sum-then-Normalize)** 模式。

**理论推导:**
$$\bar{v} = \frac{v_1 + v_2 + v_3}{3} = \left[\frac{2}{3}, \frac{2}{3}\right]^T \approx [0.6667, 0.6667]^T$$
无论归约树如何构建，总和与计数是不变的，因此最终均值唯一。

**代码核查:**
在 `src/topology/folding.rs` 中，`HyperFolder` 使用了 `Accumulator` 结构体：
1.  `merge` 函数仅执行累加：`sum = s1 + s2`, `count = c1 + c2`。
2.  `finalize` 函数在最后执行除法：`sum.scale(1.0 / count)`。

这确保了代码层面的实现严格遵循了结合律，消除了计算不确定性。

**验证结论:** **PASSED.** 空间折叠结果具有数学唯一性。

---

## 4. 验证 II：时间演化 (Time Evolution)

### 4.1 推理过程
基于统一上下文 $S_{ctx} = [2/3, 2/3]^T$ 进行前向推理。

**Layer 1:**
$$S_1 = W_1 S_{ctx} + b_1 = [0.9, 0.5]^T$$

**Layer 2:**
$$S_2 = W_2 S_1 + b_2 = [0.855, 0.64]^T$$

**代码核查:**
`src/core/neuron.rs` 中的 `absorb` 方法实现了标准的仿射变换：
`linear_part.add(&self.logic_gate.translation)`。系统在推理阶段未使用非线性激活函数，保证了推演过程与线性代数预测完全一致。

**验证结论:** **PASSED.** 逻辑推演路径清晰可追踪。

---

## 5. 验证 III：代数求解器 (The Solver)

### 5.1 学习目标 (One-Shot Learning)
* 设定目标输出 $S_{target} = [0.855, 0.80]^T$。
* 计算误差 $E = [0, 0.16]^T$。

### 5.2 逆解计算
利用正则化最小二乘法 (Damped Least Squares) 求解 $\Delta W$。

**理论公式:**
$$\Delta W = \frac{E \cdot S_1^T}{\|S_1\|^2 + \lambda}$$
其中 $\|S_1\|^2 = 1.06$。

**计算结果:**
$$\Delta W \approx \begin{bmatrix} 0 & 0 \\ 0.1358 & 0.0755 \end{bmatrix}$$

**代码核查:**
`src/core/oracle.rs` 中的 `compute_ideal_update` 方法完整复现了该逻辑：
1.  计算 `error`。
2.  计算 `denominator = input_norm_sq + lambda` (代码中 $\lambda=1e-6$)。
3.  计算外积 `factor * input` 更新矩阵。

**结果验证:**
应用 $\Delta W$ 后，新权重 $W_2'$ 再次作用于 $S_1$，输出精确为 $S_{target}$。

**验证结论:** **PASSED.** 系统具备单次迭代即收敛到精确解的能力。

---

## 6. 稳定性与微扰分析 (Stability Analysis)

### 6.1 扰动传播
输入微扰 $\delta v_1 = [0.003, -0.002]^T$。
由于系统全局由仿射变换组成（Lipschitz 连续且光滑），扰动传播呈现严格的线性关系：
$$\delta S_{final} = (W_2 W_1) \cdot \frac{1}{3} \delta v_1$$

### 6.2 结论
系统没有引入类似于 ReLU 的非光滑截断，因此不存在梯度消失的“死区”或梯度爆炸的突变点。这符合 White-Box 设计中对“可解释性”和“可控性”的要求。

---

## 7. 总体结论 (Final Verdict)

本次抗压测试证明：**Hyper-Tensor Protocol 的核心数学逻辑在理论和代码实现上是自洽的。**

1.  **空间算子**修复了旧版的结合律缺陷，使得分布式并行折叠（Hyper-Folding）成为可能。
2.  **求解器**展示了超越传统梯度下降的“顿悟”学习能力，能够精确地捕捉逻辑因果。
3.  **代码实现** (`folding.rs`, `neuron.rs`, `oracle.rs`) 忠实地还原了数学设计。

**建议:** 后续在大规模测试中，需持续关注深层网络中连乘导致的数值稳定性问题（尽管有 Lipschitz 约束），以及纯线性结构在复杂非线性任务上的表达能力限制。

---
**M-Patek Research | Pure Logic. Zero Magic.**
