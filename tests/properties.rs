use ndarray::{array, Array2, ArrayView2};
use proptest::prelude::*;
use shap_rs::{
    explainers::ExactExplainer, interactions::ExactInteractionExplainer, Background, Explainer,
    FixedMasker, FnModel,
};

fn linear_model(
    weights: [f64; 3],
) -> FnModel<impl Fn(ArrayView2<'_, f64>) -> shap_rs::Result<Array2<f64>>> {
    FnModel::new(move |x: ArrayView2<'_, f64>| {
        Ok(Array2::from_shape_fn((x.nrows(), 1), |(i, _)| {
            (0..3).map(|j| weights[j] * x[[i, j]]).sum()
        }))
    })
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs()))
}

proptest! {
    #[test]
    fn exact_additivity(x in prop::array::uniform3(-10.0f64..10.0), w in prop::array::uniform3(-10.0f64..10.0)) {
        let input = Array2::from_shape_vec((1, 3), x.to_vec()).unwrap();
        let e = ExactExplainer::new(linear_model(w), Background::new(Array2::zeros((1, 3))).unwrap())
            .explain(input.view()).unwrap();
        let expected: f64 = (0..3).map(|j| w[j] * x[j]).sum();
        prop_assert!(close(e.reconstructed()[[0, 0]], expected));
    }

    #[test]
    fn interaction_symmetry(x in prop::array::uniform2(-10.0f64..10.0)) {
        let model = FnModel::new(|v: ArrayView2<'_, f64>| Ok(Array2::from_shape_fn((v.nrows(), 1), |(i, _)| v[[i, 0]] * v[[i, 1]])));
        let e = ExactInteractionExplainer::new(model, FixedMasker::new(array![0.0, 0.0]).unwrap())
            .explain(Array2::from_shape_vec((1, 2), x.to_vec()).unwrap().view()).unwrap();
        prop_assert!(close(e.values()[[0, 0, 1, 0]], e.values()[[0, 1, 0, 0]]));
    }

    #[test]
    fn feature_permutation_invariance(x in prop::array::uniform3(-10.0f64..10.0), w in prop::array::uniform3(-10.0f64..10.0)) {
        let original = ExactExplainer::new(linear_model(w), Background::new(Array2::zeros((1, 3))).unwrap())
            .explain(Array2::from_shape_vec((1, 3), x.to_vec()).unwrap().view()).unwrap();
        let px = [x[2], x[0], x[1]];
        let pw = [w[2], w[0], w[1]];
        let permuted = ExactExplainer::new(linear_model(pw), Background::new(Array2::zeros((1, 3))).unwrap())
            .explain(Array2::from_shape_vec((1, 3), px.to_vec()).unwrap().view()).unwrap();
        for (new, old) in [2, 0, 1].into_iter().enumerate() {
            prop_assert!(close(permuted.values()[[0, new, 0]], original.values()[[0, old, 0]]));
        }
    }

    #[test]
    fn background_row_replication_invariance(x in prop::array::uniform3(-10.0f64..10.0), w in prop::array::uniform3(-10.0f64..10.0)) {
        let input = Array2::from_shape_vec((1, 3), x.to_vec()).unwrap();
        let one = ExactExplainer::new(linear_model(w), Background::new(Array2::zeros((1, 3))).unwrap()).explain(input.view()).unwrap();
        let many = ExactExplainer::new(linear_model(w), Background::new(Array2::zeros((5, 3))).unwrap()).explain(input.view()).unwrap();
        prop_assert!(one.values().iter().zip(many.values()).all(|(&a, &b)| close(a, b)));
    }

    #[test]
    fn batching_invariance(a in prop::array::uniform3(-10.0f64..10.0), b in prop::array::uniform3(-10.0f64..10.0)) {
        let w = [1.0, -2.0, 3.0];
        let both = ExactExplainer::new(linear_model(w), Background::new(Array2::zeros((1, 3))).unwrap())
            .explain(Array2::from_shape_vec((2, 3), [a, b].concat()).unwrap().view()).unwrap();
        for (i, row) in [a, b].into_iter().enumerate() {
            let one = ExactExplainer::new(linear_model(w), Background::new(Array2::zeros((1, 3))).unwrap())
                .explain(Array2::from_shape_vec((1, 3), row.to_vec()).unwrap().view()).unwrap();
            prop_assert!((0..3).all(|j| close(both.values()[[i, j, 0]], one.values()[[0, j, 0]])));
        }
    }

    #[test]
    fn explanation_linearity(x in prop::array::uniform3(-10.0f64..10.0), a in prop::array::uniform3(-10.0f64..10.0), b in prop::array::uniform3(-10.0f64..10.0)) {
        let input = Array2::from_shape_vec((1, 3), x.to_vec()).unwrap();
        let ea = ExactExplainer::new(linear_model(a), Background::new(Array2::zeros((1, 3))).unwrap()).explain(input.view()).unwrap();
        let eb = ExactExplainer::new(linear_model(b), Background::new(Array2::zeros((1, 3))).unwrap()).explain(input.view()).unwrap();
        let sum = [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
        let es = ExactExplainer::new(linear_model(sum), Background::new(Array2::zeros((1, 3))).unwrap()).explain(input.view()).unwrap();
        prop_assert!((0..3).all(|j| close(es.values()[[0, j, 0]], ea.values()[[0, j, 0]] + eb.values()[[0, j, 0]])));
    }
}
