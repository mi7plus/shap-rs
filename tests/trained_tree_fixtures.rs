#![cfg(feature = "json-adapters")]

use ndarray::Array2;
use serde::Deserialize;
use serde_json::Value;
use shap_rs::{
    explainers::TreeExplainer,
    interactions::TreeInteractionExplainer,
    tree::adapters::{
        from_lightgbm_json, from_xgboost_json, from_xgboost_json_with_tree_weights,
        from_xgboost_model_json,
    },
    Explainer, Explanation, Node, Predict, Tree, TreeEnsemble,
};

#[derive(Deserialize)]
struct TreeCase {
    model: Value,
    #[serde(default)]
    full_model: Option<Value>,
    model_base: Vec<f64>,
    samples: Vec<Vec<Option<f64>>>,
    prediction: Vec<Vec<f64>>,
    values: Vec<Vec<Vec<f64>>>,
    base_values: Vec<Vec<f64>>,
    #[serde(default)]
    interactions: Option<Vec<Vec<Vec<Vec<f64>>>>>,
    #[serde(default)]
    base_margin: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    base_margin_prediction: Option<Vec<Vec<f64>>>,
}

#[derive(Deserialize)]
struct Fixture {
    xgboost_regression: TreeCase,
    lightgbm_regression: TreeCase,
    xgboost_binary: TreeCase,
    lightgbm_binary: TreeCase,
    xgboost_multiclass: TreeCase,
    lightgbm_multiclass: TreeCase,
    lightgbm_categorical: TreeCase,
    xgboost_dart_weights: DartCase,
}

#[derive(Deserialize)]
struct DartCase {
    full_model: Value,
    model: Value,
    tree_weights: Vec<f64>,
    model_base: Vec<f64>,
    samples: Vec<Vec<Option<f64>>>,
    prediction: Vec<Vec<f64>>,
}

fn samples(case: &TreeCase) -> Array2<f64> {
    Array2::from_shape_vec(
        (case.samples.len(), case.samples[0].len()),
        case.samples
            .iter()
            .flatten()
            .map(|value| value.unwrap_or(f64::NAN))
            .collect(),
    )
    .unwrap()
}

fn assert_case(name: &str, model: &TreeEnsemble, case: &TreeCase) {
    let x = samples(case);
    let prediction = model.predict(x.view()).unwrap();
    for (actual, expected) in prediction.iter().zip(case.prediction.iter().flatten()) {
        assert!(
            (actual - expected).abs() < 2e-5,
            "{name}: {actual} != {expected}"
        );
    }
    let explanation = TreeExplainer::new(model).explain(x.view()).unwrap();
    assert_explanation(name, &explanation, case);
}

fn assert_explanation(name: &str, explanation: &Explanation, case: &TreeCase) {
    for (actual, expected) in explanation
        .values()
        .iter()
        .zip(case.values.iter().flatten().flatten())
    {
        assert!(
            (actual - expected).abs() < 2e-5,
            "{name}: {actual} != {expected}"
        );
    }
    for (actual, expected) in explanation
        .base_values()
        .iter()
        .zip(case.base_values.iter().flatten())
    {
        assert!(
            (actual - expected).abs() < 2e-5,
            "{name}: {actual} != {expected}"
        );
    }
}

fn has_repeated_feature_on_path(tree: &Tree) -> bool {
    fn visit(tree: &Tree, node: usize, path: &mut Vec<usize>) -> bool {
        match &tree.nodes()[node] {
            Node::Leaf { .. } => false,
            split => {
                let feature = split.split_feature().unwrap();
                let (left, right) = split.children().unwrap();
                if path.contains(&feature) {
                    return true;
                }
                path.push(feature);
                let repeated = visit(tree, left, path) || visit(tree, right, path);
                path.pop();
                repeated
            }
        }
    }
    visit(tree, tree.root(), &mut Vec::new())
}

#[test]
fn trained_xgboost_and_lightgbm_models_match_python_shap() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/trained_trees.json")).unwrap();
    let xgboost = from_xgboost_json(
        &fixture.xgboost_regression.model.to_string(),
        3,
        fixture.xgboost_regression.model_base.len(),
        fixture.xgboost_regression.model_base.clone(),
    )
    .unwrap();
    assert!(xgboost
        .trees()
        .iter()
        .any(|(tree, _)| has_repeated_feature_on_path(tree)));
    assert!(xgboost.trees().iter().any(|(tree, _)| {
        let mut covers = tree.nodes().iter().map(Node::cover);
        covers
            .next()
            .is_some_and(|first| covers.any(|cover| cover != first))
    }));
    assert_case("xgboost regression", &xgboost, &fixture.xgboost_regression);
    let interactions = TreeInteractionExplainer::new(&xgboost)
        .explain(samples(&fixture.xgboost_regression).view())
        .unwrap();
    for (actual, expected) in interactions.values().iter().zip(
        fixture
            .xgboost_regression
            .interactions
            .as_ref()
            .unwrap()
            .iter()
            .flatten()
            .flatten()
            .flatten(),
    ) {
        assert!(
            (actual - expected).abs() < 2e-5,
            "interaction: {actual} != {expected}"
        );
    }
    let margins = Array2::from_shape_vec(
        (3, 1),
        fixture
            .xgboost_regression
            .base_margin
            .as_ref()
            .unwrap()
            .iter()
            .flatten()
            .copied()
            .collect(),
    )
    .unwrap();
    let margin_prediction = xgboost
        .predict_with_base_margin(samples(&fixture.xgboost_regression).view(), margins.view())
        .unwrap();
    for (actual, expected) in margin_prediction.iter().zip(
        fixture
            .xgboost_regression
            .base_margin_prediction
            .as_ref()
            .unwrap()
            .iter()
            .flatten(),
    ) {
        assert!(
            (actual - expected).abs() < 2e-5,
            "base margin: {actual} != {expected}"
        );
    }
    let margin_explanation = TreeExplainer::new(&xgboost)
        .explain_with_base_margin(samples(&fixture.xgboost_regression).view(), margins.view())
        .unwrap();
    for (actual, expected) in margin_explanation
        .reconstructed()
        .iter()
        .zip(margin_prediction.iter())
    {
        assert!((actual - expected).abs() < 2e-10);
    }

    let lightgbm = from_lightgbm_json(
        &fixture.lightgbm_regression.model.to_string(),
        3,
        fixture.lightgbm_regression.model_base.len(),
        fixture.lightgbm_regression.model_base.clone(),
    )
    .unwrap();
    assert_case(
        "lightgbm regression",
        &lightgbm,
        &fixture.lightgbm_regression,
    );

    for (name, case) in [
        ("xgboost binary", &fixture.xgboost_binary),
        ("xgboost multiclass", &fixture.xgboost_multiclass),
    ] {
        let model = from_xgboost_json(
            &case.model.to_string(),
            3,
            case.model_base.len(),
            case.model_base.clone(),
        )
        .unwrap();
        assert_case(name, &model, case);
    }
    let multiclass = &fixture.xgboost_multiclass;
    let full_multiclass = from_xgboost_model_json(
        &multiclass.full_model.as_ref().unwrap().to_string(),
        multiclass.model_base.clone(),
    )
    .unwrap();
    assert_eq!(
        full_multiclass.output_groups(),
        &[
            Some(0),
            Some(1),
            Some(2),
            Some(0),
            Some(1),
            Some(2),
            Some(0),
            Some(1),
            Some(2),
            Some(0),
            Some(1),
            Some(2)
        ]
    );
    assert_case("xgboost full multiclass", &full_multiclass, multiclass);
    let categorical = &fixture.lightgbm_categorical;
    let categorical_model = from_lightgbm_json(
        &categorical.model.to_string(),
        2,
        1,
        categorical.model_base.clone(),
    )
    .unwrap();
    assert_case("lightgbm categorical", &categorical_model, categorical);
    for (name, case) in [
        ("lightgbm binary", &fixture.lightgbm_binary),
        ("lightgbm multiclass", &fixture.lightgbm_multiclass),
    ] {
        let model = from_lightgbm_json(
            &case.model.to_string(),
            3,
            case.model_base.len(),
            case.model_base.clone(),
        )
        .unwrap();
        assert_case(name, &model, case);
    }

    let dart = &fixture.xgboost_dart_weights;
    let dart_model = from_xgboost_json_with_tree_weights(
        &dart.model.to_string(),
        3,
        1,
        dart.model_base.clone(),
        Some(dart.tree_weights.clone()),
    )
    .unwrap();
    let dart_samples = Array2::from_shape_vec(
        (dart.samples.len(), 3),
        dart.samples
            .iter()
            .flatten()
            .map(|value| value.unwrap_or(f64::NAN))
            .collect(),
    )
    .unwrap();
    let dart_prediction = dart_model.predict(dart_samples.view()).unwrap();
    for (actual, expected) in dart_prediction.iter().zip(dart.prediction.iter().flatten()) {
        assert!(
            (actual - expected).abs() < 2e-5,
            "DART weight: {actual} != {expected}"
        );
    }
    let full_dart =
        from_xgboost_model_json(&dart.full_model.to_string(), dart.model_base.clone()).unwrap();
    let full_prediction = full_dart.predict(dart_samples.view()).unwrap();
    for (actual, expected) in full_prediction.iter().zip(dart_prediction.iter()) {
        assert!(
            (actual - expected).abs() < 2e-5,
            "full model: {actual} != {expected}"
        );
    }
}
