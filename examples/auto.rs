use ndarray::{array, Array2, ArrayView2, Axis};
use shap_rs::{explainers::AutoExplainer, Background, Explainer, FnModel, Result};

fn main() -> Result<()> {
    let model = FnModel::new(|x: ArrayView2<'_, f64>| Ok(x.sum_axis(Axis(1)).insert_axis(Axis(1))));
    let background = Background::new(Array2::zeros((1, 3)))?;
    let explanation = AutoExplainer::new(model, background)
        .with_exact_threshold(2)
        .with_kernel_samples(32)
        .explain(array![[1.0, 2.0, 3.0]].view())?;
    println!("{:?}", explanation.values());
    Ok(())
}
