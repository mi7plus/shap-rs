# shap-rs

Fast, native Rust model explainability powered by Shapley values.

## Features
- **Kernel SHAP**: Model-agnostic black-box explanations via Weighted Least Squares (WLS).
- **Parallel Processing**: Multi-threaded sample batch calculations using Rayon.
- **JSON Serialization**: Direct export to web-ready Force and Waterfall plot formats.

## Quickstart
```rust
use shap_rs::explain_sample;

let sample = vec![1.0, 2.0];
let background = vec![vec![0.0, 0.0]];
let predict_fn = |batch: &[Vec<f64>]| batch.iter().map(|x| x[0] + x[1]).collect();

let attributions = explain_sample(predict_fn, &sample, &background, 64).unwrap();