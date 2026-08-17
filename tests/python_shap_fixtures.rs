#![cfg(feature = "json-adapters")]

use ndarray::{Array1, Array2, Array3, ArrayView2};
use serde::Deserialize;
use shap_rs::{
    explainers::{
        ExactExplainer, HierarchicalPartitionExplainer, KernelExplainer, LinearExplainer,
        PartitionNode, PartitionTree, PermutationExplainer,
    },
    Background, Explainer, Explanation, FnModel, Link,
};

#[derive(Deserialize)]
struct Expected {
    values: Vec<Vec<Vec<f64>>>,
    base_values: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct Fixture {
    background: Vec<Vec<f64>>,
    samples: Vec<Vec<f64>>,
    coefficients: Vec<Vec<f64>>,
    intercept: Vec<f64>,
    exact: Expected,
    permutation: Expected,
    kernel: Expected,
    linear: Expected,
    partition: Expected,
    exact_logit: Expected,
}

fn matrix(rows: &[Vec<f64>]) -> Array2<f64> {
    Array2::from_shape_vec(
        (rows.len(), rows.first().unwrap().len()),
        rows.iter().flatten().copied().collect(),
    )
    .unwrap()
}

fn assert_matches(actual: &Explanation, expected: &Expected, tolerance: f64) {
    let dimensions = (
        expected.values.len(),
        expected.values[0].len(),
        expected.values[0][0].len(),
    );
    let values = Array3::from_shape_vec(
        dimensions,
        expected
            .values
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect(),
    )
    .unwrap();
    let bases = matrix(&expected.base_values);
    assert_eq!(actual.values().dim(), values.dim());
    assert_eq!(actual.base_values().dim(), bases.dim());
    for (left, right) in actual.values().iter().zip(values.iter()) {
        assert!((left - right).abs() <= tolerance, "{left} != {right}");
    }
    for (left, right) in actual.base_values().iter().zip(bases.iter()) {
        assert!((left - right).abs() <= tolerance, "{left} != {right}");
    }
}

#[test]
fn core_explainers_match_python_shap_046() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/python_shap_linear.json")).unwrap();
    let background_data = matrix(&fixture.background);
    let samples = matrix(&fixture.samples);
    let coefficients = matrix(&fixture.coefficients);
    let intercept = Array1::from(fixture.intercept.clone());
    let make_model = || {
        let coefficients = coefficients.clone();
        let intercept = intercept.clone();
        FnModel::new(move |x: ArrayView2<'_, f64>| Ok(x.dot(&coefficients) + &intercept))
    };
    let make_background = || Background::new(background_data.clone()).unwrap();

    assert_matches(
        &ExactExplainer::new(make_model(), make_background())
            .explain(samples.view())
            .unwrap(),
        &fixture.exact,
        1e-10,
    );
    assert_matches(
        &PermutationExplainer::new(make_model(), make_background())
            .with_n_permutations(300)
            .with_seed(7)
            .explain(samples.view())
            .unwrap(),
        &fixture.permutation,
        1e-10,
    );
    assert_matches(
        &KernelExplainer::new(make_model(), make_background())
            .with_nsamples(256)
            .with_seed(7)
            .explain(samples.view())
            .unwrap(),
        &fixture.kernel,
        1e-8,
    );
    assert_matches(
        &LinearExplainer::new(coefficients.clone(), intercept.clone(), make_background())
            .unwrap()
            .explain(samples.view())
            .unwrap(),
        &fixture.linear,
        1e-10,
    );
    let hierarchy = PartitionTree::new(
        PartitionNode::Group(
            Box::new(PartitionNode::Group(
                Box::new(PartitionNode::Feature(0)),
                Box::new(PartitionNode::Feature(1)),
            )),
            Box::new(PartitionNode::Feature(2)),
        ),
        3,
    )
    .unwrap();
    assert_matches(
        &HierarchicalPartitionExplainer::new(make_model(), make_background(), hierarchy)
            .explain(samples.view())
            .unwrap(),
        &fixture.partition,
        1e-10,
    );
    let probability_model = FnModel::new(|x: ArrayView2<'_, f64>| {
        Ok(Array2::from_shape_fn((x.nrows(), 1), |(row, _)| {
            let margin = 0.7 * x[[row, 0]] - 0.4 * x[[row, 1]] + 0.2 * x[[row, 2]] + 0.1;
            1.0 / (1.0 + (-margin).exp())
        }))
    });
    assert_matches(
        &ExactExplainer::new(probability_model, make_background())
            .with_link(Link::Logit)
            .explain(samples.view())
            .unwrap(),
        &fixture.exact_logit,
        1e-10,
    );
}
