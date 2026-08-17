use ndarray::{array, Array2, ArrayView2, Axis};
use shap_rs::{explainers::KernelExplainer, Background, Explainer, FnModel, Result};

fn main() -> Result<()> {
    let model = FnModel::new(|x: ArrayView2<'_, f64>| {
        Ok(x.map_axis(Axis(1), |row| row[0] * row[1])
            .insert_axis(Axis(1)))
    });
    let explanation = KernelExplainer::new(model, Background::new(Array2::zeros((1, 2)))?)
        .with_nsamples(64)
        .with_seed(7)
        .explain(array![[2.0, 3.0]].view())?;
    println!(
        "prediction reconstructed as {}",
        explanation.reconstructed()[[0, 0]]
    );
    Ok(())
}
