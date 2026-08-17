use nalgebra::{DMatrix, DVector};
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Kernel SHAP Weighting Function: calculates the Shapley kernel weight for a given coalition size.
///
/// Formula: k(M, |z'|) = (M - 1) / ( (M choose |z'|) * |z'| * (M - |z'|) )
pub fn shapley_kernel_weight(num_features: usize, coalition_size: usize) -> f64 {
    let m = num_features as f64;
    let z = coalition_size as f64;

    if coalition_size == 0 || coalition_size == num_features {
        // Enforce strong boundary conditions for empty and full coalitions
        return 1e6;
    }

    let n_choose_k = n_choose_k(num_features, coalition_size) as f64;
    let denominator = n_choose_k * z * (m - z);

    (m - 1.0) / denominator
}

/// Helper function to compute combinations (N choose K)
fn n_choose_k(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut c = 1;
    for i in 0..k {
        c = c * (n - i) / (i + 1);
    }
    c
}

/// Structural representation of generated coalition samples
pub struct CoalitionData {
    /// Binary coalition matrix Z_prime of shape [num_samples, num_features]
    pub z_matrix: DMatrix<f64>,
    /// Diagonal weight vector W of shape [num_samples]
    pub weights: DVector<f64>,
}

/// Generates binary coalition masks using Shapley Kernel sampling.
/// Ensures boundary conditions (empty and full coalitions) are explicitly included.
pub fn generate_coalitions(num_features: usize, num_samples: usize) -> CoalitionData {
    let mut z_rows = Vec::with_capacity(num_samples * num_features);
    let mut weights = Vec::with_capacity(num_samples);

    // 1. Boundary Coalition: Empty set (all zeros)
    z_rows.extend(vec![0.0; num_features]);
    weights.push(shapley_kernel_weight(num_features, 0));

    // 2. Boundary Coalition: Grand set (all ones)
    z_rows.extend(vec![1.0; num_features]);
    weights.push(shapley_kernel_weight(num_features, num_features));

    // 3. Sample random coalitions for the remaining budget
    let mut rng = thread_rng();
    let mut indices: Vec<usize> = (0..num_features).collect();

    for _ in 2..num_samples {
        indices.shuffle(&mut rng);
        // Randomly choose coalition size |z'| in range [1, M - 1]
        let size = (1..num_features).collect::<Vec<_>>().choose(&mut rng).copied().unwrap_or(1);

        let mut row = vec![0.0; num_features];
        for &idx in &indices[..size] {
            row[idx] = 1.0;
        }

        weights.push(shapley_kernel_weight(num_features, size));
        z_rows.extend(row);
    }

    let z_matrix = DMatrix::from_row_slice(num_samples, num_features, &z_rows);
    let weights = DVector::from_vec(weights);

    CoalitionData { z_matrix, weights }
}

/// Weighted Least Squares (WLS) Solver constrained to enforce Shapley efficiency.
///
/// Solves: min_phi || W^(1/2) * (Z * phi - (y - base_value)) ||^2
pub fn solve_wls(
    z_matrix: &DMatrix<f64>,
    weights: &DVector<f64>,
    predictions: &DVector<f64>,
    base_value: f64,
) -> Result<DVector<f64>, &'static str> {
    let (num_samples, num_features) = z_matrix.shape();

    // Adjust targets relative to the baseline prediction expectation
    let y_diff = predictions.map(|val| val - base_value);

    // Construct diagonal square-root weight matrix W^(1/2)
    let sqrt_weights = weights.map(|w| w.sqrt());
    let mut z_weighted = z_matrix.clone();
    let mut y_weighted = y_diff.clone();

    for i in 0..num_samples {
        let w_i = sqrt_weights[i];
        for j in 0..num_features {
            z_weighted[(i, j)] *= w_i;
        }
        y_weighted[i] *= w_i;
    }

    // Compute Normal Equations: (Z_w^T * Z_w) * phi = Z_w^T * Y_w
    let z_t_z = z_weighted.transpose() * &z_weighted;
    let z_t_y = z_weighted.transpose() * &y_weighted;

    // Solve the linear system using Cholesky (SVD / SVD-QR fallback for stability)
    z_t_z.svd(true, true)
        .solve(&z_t_y, 1e-7)
        .map_err(|_| "Failed to solve WLS system due to singular/ill-conditioned coalition matrix")
}

/// High-level Kernel SHAP explainer runner
pub fn explain_sample<F>(
    predict_fn: F,
    sample: &[f64],
    background: &[Vec<f64>],
    num_samples: usize,
) -> Result<Vec<f64>, &'static str>
where
    F: Fn(&[Vec<f64>]) -> Vec<f64>,
{
    let num_features = sample.len();

    // Calculate baseline expectation E[f(x)] over background dataset
    let bg_preds = predict_fn(background);
    let base_value: f64 = bg_preds.iter().sum::<f64>() / bg_preds.len() as f64;

    // 1. Generate coalition samples and Shapley kernel weights
    let CoalitionData { z_matrix, weights } = generate_coalitions(num_features, num_samples);

    // 2. Synthesize feature instances combining sample values with background baselines
    let mut synthetic_instances = Vec::with_capacity(num_samples * background.len());
    for row_idx in 0..num_samples {
        let mask = z_matrix.row(row_idx);
        for bg_row in background {
            let mut instance = vec![0.0; num_features];
            for j in 0..num_features {
                instance[j] = if mask[j] == 1.0 { sample[j] } else { bg_row[j] };
            }
            synthetic_instances.push(instance);
        }
    }

    // 3. Evaluate prediction function on synthetic coalition instances
    let raw_preds = predict_fn(&synthetic_instances);

    // Average predictions over background samples for each coalition mask
    let mut coalition_preds = vec![0.0; num_samples];
    let bg_len = background.len();
    for i in 0..num_samples {
        let chunk = &raw_preds[i * bg_len..(i + 1) * bg_len];
        coalition_preds[i] = chunk.iter().sum::<f64>() / bg_len as f64;
    }

    let y_vec = DVector::from_vec(coalition_preds);

    // 4. Solve Weighted Least Squares to recover Shapley values
    let phi = solve_wls(&z_matrix, &weights, &y_vec, base_value)?;

    Ok(phi.as_slice().to_vec())
}