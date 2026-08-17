use ndarray::{array, Array2, ArrayView2, Axis};
use shap_rs::{
    explainers::ExactExplainer,
    plot::svg::{global_bar, SvgOptions},
    Explainer, ExplainerExt, FeatureMetadata, FixedMasker, FnModel, OutputMetadata, Result,
};

fn main() -> Result<()> {
    let model = FnModel::new(|x: ArrayView2<'_, f64>| Ok(x.sum_axis(Axis(1)).insert_axis(Axis(1))));
    let explainer = ExactExplainer::from_masker(model, FixedMasker::new(array![0.0, 0.0])?)
        .with_metadata()
        .with_feature_metadata(FeatureMetadata::new(vec!["height".into(), "width".into()])?)?
        .with_output_metadata(OutputMetadata::new(vec!["area score".into()])?)?;
    let explanation = explainer.explain(array![[2.0, 3.0]].view())?;
    let svg = global_bar(&explanation, &SvgOptions::default())?;
    std::fs::write("shap.svg", svg)
        .map_err(|error| shap_rs::ShapError::Other(error.to_string()))?;
    let _: Array2<f64> = explanation.reconstructed();
    Ok(())
}
