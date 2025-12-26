// COPYRIGHT (C) 2025 M-Patek. ALL RIGHTS RESERVED.

#[cfg(test)]
mod tests {
    use crate::core::algebra::{Vector, Matrix, Float, MANIFOLD_DIM};
    use crate::core::affine::AffineTuple;
    use crate::core::neuron::HTPNeuron;
    use crate::core::oracle::LogicOracle;
    use crate::core::primes::{ConceptEmbedder, WeightInitializer};

    /// 🧪 Test 1: Causal Consistency (因果律验证)
    /// 验证结合律: (A2 * A1) * S == A2 * (A1 * S)
    /// 这是 "Time Folding" (时间并行折叠) 的数学基础。
    #[test]
    fn test_causal_associativity() {
        println!("🧪 [Test] Causal Consistency (Associativity)...");

        // 1. Init Random State
        let s0 = ConceptEmbedder::embed_token(42);

        // 2. Init Two Logic Steps (A1, A2)
        let w1 = WeightInitializer::init_matrix(MANIFOLD_DIM, MANIFOLD_DIM, 100);
        let b1 = WeightInitializer::init_bias(MANIFOLD_DIM);
        let a1 = AffineTuple::new(w1, b1);

        let w2 = WeightInitializer::init_matrix(MANIFOLD_DIM, MANIFOLD_DIM, 200);
        let b2 = WeightInitializer::init_bias(MANIFOLD_DIM);
        let a2 = AffineTuple::new(w2, b2);

        // 3. Path A: Sequential Execution (S -> S1 -> S2)
        let mut neuron_seq = HTPNeuron::new();
        neuron_seq.state = s0.clone();
        
        neuron_seq.logic_gate = a1.clone();
        let s1 = neuron_seq.absorb(&s0); // S1 = A1(S0)
        
        neuron_seq.logic_gate = a2.clone();
        let s2_seq = neuron_seq.absorb(&s1); // S2 = A2(S1)

        // 4. Path B: Folded Execution (A_total = A2 * A1, then S -> S2)
        let a_total = a2.compose(&a1).expect("Composition Failed");
        
        let mut neuron_fold = HTPNeuron::new();
        neuron_fold.state = s0.clone();
        neuron_fold.logic_gate = a_total;
        let s2_fold = neuron_fold.absorb(&s0); // S2 = (A2*A1)(S0)

        // 5. Verify Equivalence (Error should be floating-point negligible)
        let loss = LogicOracle::calculate_loss(&s2_seq, &s2_fold);
        println!("   > Sequential vs Folded Loss: {:.10e}", loss);
        
        assert!(loss < 1e-5, "❌ Associativity Broken! Time Folding is invalid.");
    }

    /// 🧪 Test 2: The Solver (代数逆解 / One-Shot Learning)
    /// 验证我们是否能通过 Oracle 瞬间算出所需的权重修正量。
    #[test]
    fn test_algebraic_solver() {
        println!("🧪 [Test] Algebraic One-Shot Solver...");

        // 1. Define Problem
        // Start: "Sky"
        // Target: "Blue"
        let s_in = ConceptEmbedder::embed_token(1); // "Sky"
        let s_target = ConceptEmbedder::embed_token(2); // "Blue"
        
        // Initial Logic: Random (Tabula Rasa)
        let w_init = WeightInitializer::init_matrix(MANIFOLD_DIM, MANIFOLD_DIM, 777);
        let b_init = WeightInitializer::init_bias(MANIFOLD_DIM);
        let current_gate = AffineTuple::new(w_init, b_init);

        // Check initial error
        let mut neuron = HTPNeuron::new();
        neuron.logic_gate = current_gate.clone();
        let s_pred_initial = neuron.absorb(&s_in);
        let initial_loss = LogicOracle::calculate_loss(&s_pred_initial, &s_target);
        println!("   > Initial Loss (Random): {:.4}", initial_loss);

        // 2. Invoke The Oracle (Solve for Delta W)
        // We want to correct W such that W_new * S_in ≈ S_target
        let delta_w = LogicOracle::compute_ideal_update(&s_in, &s_target, &current_gate);
        
        // 3. Apply Correction
        // W_new = W_old + Delta W
        let w_new = current_gate.linear.add(&delta_w);
        neuron.logic_gate.linear = w_new;

        // 4. Verify Learning
        let s_pred_solved = neuron.absorb(&s_in);
        let solved_loss = LogicOracle::calculate_loss(&s_pred_solved, &s_target);
        println!("   > Solved Loss (One-Shot): {:.10e}", solved_loss);

        assert!(solved_loss < 1e-4, "❌ Solver Failed! Could not derive logic analytically.");
        assert!(solved_loss < initial_loss, "❌ Solver made things worse!");
    }

    /// 🧪 Test 3: Deep Manifold Stability (深层稳定性)
    /// 模拟 100 层推理，检查数值是否保持稳定 (Lipschitz Check)。
    #[test]
    fn test_deep_stability() {
        println!("🧪 [Test] Deep Manifold Stability (100 Layers)...");

        let mut s = ConceptEmbedder::embed_token(100);
        
        // Use an identity-like matrix with slight noise to simulate stable logic
        // If we used random matrices, the value would explode or vanish quickly.
        let mut w = Matrix::identity();
        // Add tiny noise to identity
        w.data[0] += 0.01; 

        let b = WeightInitializer::init_bias(MANIFOLD_DIM);
        let gate = AffineTuple::new(w, b);
        let mut neuron = HTPNeuron::new();
        neuron.logic_gate = gate;

        for i in 0..100 {
            s = neuron.absorb(&s);
            
            // Check for NaN / Inf
            if let Err(e) = neuron.verify_integrity() {
                panic!("❌ Instability detected at layer {}: {}", i, e);
            }
        }
        
        // Check norm
        let norm: Float = s.data.iter().map(|x| x*x).sum::<Float>().sqrt();
        println!("   > Final State Norm after 100 steps: {:.4}", norm);
        
        assert!(norm.is_finite(), "Norm is not finite");
        // We expect some growth or shrinkage, but not explosion to Infinity
    }
}
