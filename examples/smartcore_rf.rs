// Run with: cargo run --example smartcore_rf

use shap_rs::explain_sample;

fn main() {
    // Simulated prediction function representing a trained SmartCore model
    let predict_fn = |batch: &[Vec<f64>]| -> Vec<f64> {
        batch
            .iter()
            .map(|x| 0.5 * x[0] + 2.0 * x[1] - 1.2 * x[2])
            .collect()
    };

    let sample = vec![1.0, 2.0, 0.5];
    let background = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.5, 0.5, 0.5],
    ];

    let shap_values = explain_sample(predict_fn, &sample, &background, 100).unwrap();

    println!("Input Sample: {:?}", sample);
    println!("SHAP Attributions: {:?}", shap_values);
}