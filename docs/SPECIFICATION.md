# White-Box Hyper-Tensor Protocol: Technical Specification v1.0

## 1. System Parameters & Constants

Unlike the previous version which relied on cryptographic parameters (Discriminants), the White-Box architecture relies on **Geometric Hyper-Parameters**.

### 1.1 Manifold Configuration
* **Dimension ($D$)**: The dimensionality of the logic state vector.
    * *Default*: $D = 512$ (standard) or $1024$ (high-fidelity).
* **Precision ($\mathbb{F}$)**: The floating-point standard used for algebraic operations.
    * *Type*: IEEE 754 `Float32` (training) or `BFloat16` (inference).
    * *Reasoning*: High precision is required for deep causal chains to prevent vanishing gradients, but exact integer arithmetic is no longer needed.

### 1.2 Tensor Bounds
To ensure stability during continuous differentiation:
* **Lipschitz Constant ($K$)**: Maximum allowed norm for any weight matrix $W$.
    * Constraint: $\|W\|_2 \le 1.0 + \epsilon$
* **Bias Clamp**: Maximum absolute value for bias vector components.

---

## 2. Core Data Structures

### 2.1 The Logical State ($S$)
A neuron's activation state is no longer a Class Group element, but a dense vector.

```rust
struct LogicState {
    coords: Tensor<f32, D>, // The position in the logical manifold
}
```

### 2.2 The Affine Transformation Tuple ($\mathcal{A}$)
The fundamental unit of logic, replacing the old `(Prime, Element)` pair. It represents a "Causal Transition".

```rust
struct AffineTuple {
    linear: Matrix<f32, D, D>, // Represents the logical implication (W)
    translation: Vector<f32, D>, // Represents the bias/correction (b)
}
```

---

## 3. Algebraic Protocols (The Kernel)

This section defines the rigorous implementation of the **Dual-Operator System**.

### 3.1 Time Evolution Operator ($\oplus_{time}$)
**Purpose**: To combine two sequential logical steps into a single equivalent step.
**Input**: $\mathcal{A}_1$ (Step 1), $\mathcal{A}_2$ (Step 2, occurring after Step 1).
**Output**: $\mathcal{A}_{result}$

**Algorithm**:
1.  **Linear Composition**:
    $$W_{new} = W_2 \times W_1$$
    *(Standard Matrix Multiplication)*
2.  **Bias Propagation**:
    $$b_{new} = (W_2 \times b_1) + b_2$$
    *(Affine Shift)*

**Code Representation**:
```rust
fn compose_time(prev: &AffineTuple, next: &AffineTuple) -> AffineTuple {
    AffineTuple {
        linear: next.linear.matmul(&prev.linear),
        translation: next.linear.matmul(&prev.translation) + next.translation,
    }
}
```

### 3.2 Space Merging Operator ($\otimes_{space}$)
**Purpose**: To fold context from parallel branches (e.g., Multi-Head Attention equivalent).
**Input**: $\mathcal{A}_{left}$, $\mathcal{A}_{right}$.
**Output**: $\mathcal{A}_{folded}$

**Algorithm**:
1.  **Symmetric Aggregation**:
    $$W_{folded} = \text{Normalize}(W_{left} + W_{right})$$
    $$b_{folded} = \text{Average}(b_{left}, b_{right})$$
    *(Note: The specific reduction function can be summation, averaging, or a Hadamard product depending on the specific layer type)*.

---

## 4. Hyper-Tensor Topology (The Architecture)

### 4.1 Recursive Folding (The "Pyramid")
The network structure remains a hierarchical tree.
* **Leaf Nodes**: Raw input tokens embedded into initial Affine Tuples $\mathcal{A}_{token} = (I, \text{Embedding}(token))$.
* **Branch Nodes**: Results of $\oplus_{time}$ operations representing logic chains.
* **Root Node**: The final folded state representing the conclusion of the entire context window.

### 4.2 Coordinate Mapping
Input tokens are mapped to the manifold via a trainable **Embedding Matrix**.
* $TokenID \rightarrow \vec{v} \in \mathbb{R}^D$
* This mapping is invertible, allowing the `InverseDecoder` to function.

---

## 5. Verification & Navigation

### 5.1 The Logical Trace
Instead of Zero-Knowledge Proofs, the system generates a **Differential Trace**.
* **Definition**: A record of all partial derivatives $\frac{\partial S_{final}}{\partial W_i}$ along the active path.
* **Usage**: Used by the optimizer to adjust weights.

### 5.2 Inverse Solving (Navigation)
To generate the next token, the system solves the navigation equation:

**Problem**: Find token $T$ such that $S_{next} \approx \text{Embedding}(T)$.
**Method**:
1.  Project the current logical state $S_{final}$ into the vocabulary space:
    $$Logits = S_{final} \times W_{vocab}^T$$
2.  Select the token with the highest alignment score (dot product) or minimal Euclidean distance.
3.  *(Optional One-Shot Correction)*: If the selected token is "almost" correct but implies a slight logic error, calculate the specific $\vec{b}_{correction}$ needed to make it perfect and back-propagate immediately.

---

## 6. Serialization (Wire Format)

For distributed training, we transmit raw tensors rather than cryptographic proofs.

### 6.1 Packet Structure
```protobuf
message GradientUpdate {
    uint32 layer_id = 1;
    bytes tensor_data = 2; // Serialized f32 array
    repeated uint32 active_path_indices = 3; // Sparse update indices
}
```

### 6.2 Storage
* **Format**: `.safetensors` or raw binary.
* **Endianness**: Little-endian.
