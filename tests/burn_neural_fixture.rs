#![cfg(all(feature = "burn-adapter", feature = "json-adapters"))]

use burn_core::backend::Autodiff;
use burn_ndarray::NdArray;
use ndarray::{Array2, Array3};
use serde::Deserialize;
use shap_rs::{
    burn_adapter::BurnAffineModel,
    explainers::{DeepExplainer, GradientExplainer},
    Background, Explainer,
};

type Backend = Autodiff<NdArray<f32>>;

#[derive(Deserialize)]
struct Fixture {
    source: String,
    coefficients: Vec<Vec<f64>>,
    intercept: Vec<f64>,
    background: Vec<Vec<f64>>,
    samples: Vec<Vec<f64>>,
    base_values: Vec<Vec<f64>>,
    predictions: Vec<Vec<f64>>,
    attributions: Vec<Vec<Vec<f64>>>,
}

fn matrix(rows: &[Vec<f64>]) -> Array2<f64> {
    Array2::from_shape_vec(
        (rows.len(), rows.first().unwrap().len()),
        rows.iter().flatten().copied().collect(),
    )
    .unwrap()
}

fn tensor3(values: &[Vec<Vec<f64>>]) -> Array3<f64> {
    Array3::from_shape_vec(
        (values.len(), values[0].len(), values[0][0].len()),
        values.iter().flatten().flatten().copied().collect(),
    )
    .unwrap()
}

fn assert_close(actual: impl Iterator<Item = f64>, expected: impl Iterator<Item = f64>) {
    for (actual, expected) in actual.zip(expected) {
        assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
    }
}

#[test]
fn burn_gradient_and_deep_match_committed_affine_reference() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/burn_affine_reference.json")).unwrap();
    assert!(fixture.source.contains("Analytical affine reference"));
    let coefficients = matrix(&fixture.coefficients);
    let samples = matrix(&fixture.samples);
    let expected_values = tensor3(&fixture.attributions);
    let expected_bases = matrix(&fixture.base_values);
    let expected_predictions = matrix(&fixture.predictions);
    let background = Background::new(matrix(&fixture.background)).unwrap();

    let deep = DeepExplainer::new(
        BurnAffineModel::<Backend>::new(
            coefficients.clone(),
            fixture.intercept.clone(),
            Default::default(),
        )
        .unwrap(),
        background.clone(),
    )
    .explain(samples.view())
    .unwrap();
    let gradient = GradientExplainer::new(
        BurnAffineModel::<Backend>::new(coefficients, fixture.intercept, Default::default())
            .unwrap(),
        Background::new(ndarray::array![[1.0, 1.0]]).unwrap(),
    )
    .with_nsamples(32)
    .explain(samples.view())
    .unwrap();

    for explanation in [&deep, &gradient] {
        assert_close(
            explanation.values().iter().copied(),
            expected_values.iter().copied(),
        );
        assert_close(
            explanation.base_values().iter().copied(),
            expected_bases.iter().copied(),
        );
        assert_close(
            explanation.reconstructed().iter().copied(),
            expected_predictions.iter().copied(),
        );
    }
}
