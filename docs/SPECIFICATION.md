# Hyper-Tensor Protocol: Technical Specification v1.1

## 1. System Constants & Configuration
The **White-Box architecture** is defined by geometric hyper-parameters rather than cryptographic security parameters.

### 1.1 Logical Manifold ($\mathcal{M}$)
* **Definition**: The continuous space where all logical concepts reside.
* **Dimension ($D$):**
    * **Standard**: $D = 768$ (Balanced)
    * **High-Fidelity**: $D = 2048$ (Deep Logic)
* **Manifold Type**: General Linear Group $GL(D, \mathbb{R})$ extended with Affine Translations.

### 1.2 Numerical Precision ($\mathbb{F}$)
* **Training**: IEEE 754 **Float32** (Required for precise gradient traces).
* **Inference**: **BFloat16** (Brain Floating Point) for hardware acceleration.
* **Reasoning**: We strictly avoid quantization below 16-bit to preserve causal rigidity.

---

## 2. Core Data Structures

### 2.1 The Logical State ($S$)
Represents a snapshot of "Thought" at a specific timestamp.

```rust
/// A point on the Differentiable Manifold
struct LogicState {
    /// Dense coordinate vector of dimension D
    coords: Tensor<f32, D>, 
}
```

### 2.2 The Causal Operator ($\mathcal{A}$)
Represents a "Logic Step" or "Transition Rule". This replaces the "Weights" in traditional NNs.

```rust
/// An Affine Transformation Tuple (W, b)
struct AffineTuple {
    /// The linear transformation matrix (The "Logic")
    linear: Matrix<f32, D, D>, 
    /// The translation vector (The "Bias/Correction")
    translation: Vector<f32, D>, 
}
```

---

## 3. Algebraic Kernels (The Dual-Operator System)
This section defines the rigid mathematical rules for Logic Evolution.

### 3.1 Time Evolution Kernel ($\oplus_{time}$)
* **Purpose**: Sequential Logic Composition (Cause $\rightarrow$ Effect).
* **Math**: Non-Commutative Affine Composition.
* **Evolution**: Given current state $S_{t}$ and operator $\mathcal{A} = (W, \vec{b})$:
  $$S_{t+1} = \mathcal{A} \cdot S_{t} = W \times S_{t} + \vec{b}$$

**Composition Rule:** Combining two consecutive steps $\mathcal{A}_1$ then $\mathcal{A}_2$:

```rust
fn compose(first: &AffineTuple, second: &AffineTuple) -> AffineTuple {
    // Note: Matrix multiplication is non-commutative (A*B != B*A)
    AffineTuple {
        linear: second.linear.matmul(&first.linear),
        translation: second.linear.matmul(&first.translation) + second.translation,
    }
}
```

### 3.2 Space Folding Kernel ($\otimes_{space}$)
* **Purpose**: Parallel Context Aggregation.
* **Math**: N-ary Commutative Mean (implemented via Monoid).

```rust
fn merge_all(branches: &[AffineTuple]) -> AffineTuple {
    // 1. Accumulate Sum (Commutative & Associative)
    let sum = branches.iter().sum();
    
    // 2. Normalize by Count
    let result = sum / branches.len();
    
    result
}
```

---

## 4. Hyper-Tensor Topology

### 4.1 The Embedding Layer
Unlike Transformers which use static lookups, we use a **Differentiable Manifold Projector**.
* **Input**: Token Index $i \in \mathbb{N}$
* **Output**: Initial Affine Tuple $\mathcal{A}_{init} = (I, \text{Embed}(i))$
* **Note**: The linear part starts as Identity ($I$), the bias carries the semantic meaning.

### 4.2 The Recursive Cortex
The network is structured as a fractal tree (**Hyper-Tensor Folding**):
1. **Leaf Layer**: Processes raw tokens in parallel.
2. **Folding Layers**: Recursively applies $\otimes_{space}$ to merge context windows (N-ary topology).
3. **Causal Layers**: Applies deep $\oplus_{time}$ chains to derive conclusions (Binary topology).

---

## 5. Navigation & Decoding (The Solver)

### 5.1 Inverse Projection
To generate text, we must map the abstract coordinate $S_{final}$ back to the vocabulary.
$$Logits = S_{final} \cdot W_{vocab}^T$$
* **Standard Mode**: Select $\text{argmax}(Logits)$.
* **Solver Mode**: If $Target$ is known, we calculate the error vector $\vec{e} = S_{target} - S_{final}$ and estimate the weight update $\Delta W$ using Tikhonov Regularization ($\lambda \approx 10^{-6}$).

### 5.2 Zero-Hallucination Check
Before outputting a token, the system verifies **Geometric Consistency**:

```rust
fn verify_consistency(state: &LogicState) -> Result<(), Error> {
    if !manifold.contains(state, epsilon=1e-5) {
        return Err("Logic Derivation Failed: Coordinate out of bounds");
    }
    Ok(())
}
```

---

## 6. Training Protocol

### 6.1 Gradient Descent
* **Optimizer**: AdamW or SOAP (Second-Order).
* **Loss Function**: Geodesic Distance on the Manifold (MSE in Euclidean approximation).

### 6.2 Algebraic One-Shot Learning (Adaptive)
For rapid fact acquisition:
1. **Input**: Premise $S_{in}$, Target Fact $S_{out}$.
2. **Estimate**: Compute $\Delta W$ with damping factor $\lambda$.
3. **Update**: Apply $\Delta W$ directly to the specific memory neuron.

---

## 7. Storage Format
Weights are serialized as raw tensors.
* **File Format**: `.safetensors` (Zero-copy memory mapping).
* **Layout**:
    * `layers.0.linear.weight`: $[D, D]$
    * `layers.0.linear.bias`: $[D]$
    * `embeddings`: $[Vocab, D]$
