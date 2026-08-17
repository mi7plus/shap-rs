use ndarray::{array, Array2, ArrayView2, Axis};
use shap_rs::{
    explainers::PermutationExplainer, Background, FnModel, ParallelExplainerExt, Result,
};

fn main() -> Result<()> {
    let model = FnModel::new(|x: ArrayView2<'_, f64>| Ok(x.sum_axis(Axis(1)).insert_axis(Axis(1))));
    let explainer = PermutationExplainer::new(model, Background::new(Array2::zeros((1, 2)))?);
    let explanation = explainer.explain_parallel(array![[1.0, 2.0], [3.0, 4.0]].view(), 1)?;
    println!("{:?}", explanation.values());
    Ok(())
}
