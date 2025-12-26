// COPYRIGHT (C) 2025 M-Patek. ALL RIGHTS RESERVED.

use serde::{Serialize, Deserialize};

// ==================================================================
// 1. 基础类型定义 (The Manifold Substrate)
// ==================================================================

/// 🎯 Precision Selection
/// - Training: f32 (Standard Backpropagation)
/// - High-Fidelity Logic: f64
/// 在 White-Box Evolver 中，默认使用 f32 以适配 GPU 张量核心。
pub type Float = f32;

/// 📏 Manifold Dimension (D)
/// 逻辑流形的维度。必须与 SPECIFICATION.md 中的定义一致。
/// 为了简化演示，这里硬编码为 512，实际工程中可能是泛型或配置项。
pub const MANIFOLD_DIM: usize = 512;

/// 🏛️ Vector: 逻辑流形上的点或位移向量
/// Represents a point $v \in \mathbb{R}^D$
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Vector {
    pub data: Vec<Float>,
}

/// 🏛️ Matrix: 线性变换算子
/// Represents a linear map $W: \mathbb{R}^D \to \mathbb{R}^D$
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Float>,
}

// ==================================================================
// 2. 线性代数核心实现 (Linear Algebra Kernel)
// ==================================================================

impl Vector {
    /// 创建新向量 (需要检查维度)
    pub fn new(data: Vec<Float>) -> Self {
        if data.len() != MANIFOLD_DIM {
            // 在严格模式下应该 panic 或返回 Result，这里为了简洁保持 lenient
            eprintln!("⚠️ Warning: Vector dimension mismatch. Expected {}, got {}", MANIFOLD_DIM, data.len());
        }
        Vector { data }
    }

    /// 零向量 (Origin)
    pub fn zeros() -> Self {
        Vector { data: vec![0.0; MANIFOLD_DIM] }
    }

    /// 向量加法: $v + u$
    pub fn add(&self, other: &Self) -> Self {
        let new_data = self.data.iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        Vector { data: new_data }
    }

    /// 向量减法: $v - u$
    pub fn sub(&self, other: &Self) -> Self {
        let new_data = self.data.iter()
            .zip(&other.data)
            .map(|(a, b)| a - b)
            .collect();
        Vector { data: new_data }
    }

    /// 标量乘法: $k \cdot v$
    pub fn scale(&self, scalar: Float) -> Self {
        let new_data = self.data.iter()
            .map(|a| a * scalar)
            .collect();
        Vector { data: new_data }
    }

    /// 原始数据访问
    pub fn as_slice(&self) -> &[Float] {
        &self.data
    }
}

impl Matrix {
    /// 创建新矩阵
    pub fn new(rows: usize, cols: usize, data: Vec<Float>) -> Self {
        assert_eq!(data.len(), rows * cols, "Matrix data size does not match dimensions");
        Matrix { rows, cols, data }
    }

    /// 单位矩阵 (Identity Matrix)
    /// $I \cdot v = v$
    pub fn identity() -> Self {
        let mut data = vec![0.0; MANIFOLD_DIM * MANIFOLD_DIM];
        for i in 0..MANIFOLD_DIM {
            data[i * MANIFOLD_DIM + i] = 1.0;
        }
        Matrix { 
            rows: MANIFOLD_DIM, 
            cols: MANIFOLD_DIM, 
            data 
        }
    }

    /// 矩阵乘法 (Matrix Multiplication): $C = A \cdot B$
    /// 注意：这是非交换操作的核心 (A*B != B*A)
    /// (Naive implementation O(N^3), production should use BLAS/Gemm)
    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows, "Matrix dimension mismatch for multiplication");
        let n = self.rows;
        let m = self.cols;
        let p = other.cols;
        
        let mut result = vec![0.0; n * p];
        
        // 简单的三重循环实现
        for i in 0..n {
            for k in 0..m {
                let r = self.data[i * m + k];
                // 优化：跳过 0 元素
                if r.abs() > 1e-9 {
                    for j in 0..p {
                        result[i * p + j] += r * other.data[k * p + j];
                    }
                }
            }
        }
        
        Matrix { rows: n, cols: p, data: result }
    }

    /// 矩阵-向量乘法 (Matrix-Vector Product): $y = A \cdot x$
    pub fn matmul_vec(&self, vec: &Vector) -> Vector {
        assert_eq!(self.cols, vec.data.len(), "Matrix-Vector dimension mismatch");
        let mut result = vec![0.0; self.rows];
        
        for i in 0..self.rows {
            let mut sum = 0.0;
            for j in 0..self.cols {
                sum += self.data[i * self.cols + j] * vec.data[j];
            }
            result[i] = sum;
        }
        
        Vector { data: result }
    }

    /// 矩阵加法 (Matrix Addition): $A + B$
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.data.len(), other.data.len(), "Matrix addition shape mismatch");
        let new_data = self.data.iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        Matrix { rows: self.rows, cols: self.cols, data: new_data }
    }

    /// 矩阵缩放 (Scalar Multiplication): $k \cdot A$
    pub fn scale(&self, scalar: Float) -> Self {
        let new_data = self.data.iter()
            .map(|a| a * scalar)
            .collect();
        Matrix { rows: self.rows, cols: self.cols, data: new_data }
    }

    /// 🛡️ [Safety Check]: Spectral Norm Calculation
    /// 用于验证 Lipschitz 连续性。为了性能，这里使用 Frobenius Norm 作为上界近似。
    /// $\|A\|_F = \sqrt{\sum a_{ij}^2} \ge \|A\|_2$
    pub fn spectral_norm(&self) -> Float {
        self.data.iter()
            .map(|x| x * x)
            .sum::<Float>()
            .sqrt()
    }
}
