use ndarray::{array, Array2, ArrayView1, ArrayView2, Axis};
use shap_rs::{explainers::ExactExplainer, Explainer, FnMasker, FnModel, Result};

fn main() -> Result<()> {
    let model = FnModel::new(|x: ArrayView2<'_, f64>| Ok(x.sum_axis(Axis(1)).insert_axis(Axis(1))));
    let masker = FnMasker::new(2, |sample: ArrayView1<'_, f64>, present: &[bool]| {
        Ok(Array2::from_shape_fn((1, 2), |(_, j)| {
            if present[j] {
                sample[j]
            } else {
                0.0
            }
        }))
    })?;
    let explanation =
        ExactExplainer::from_masker(model, masker).explain(array![[1.0, 2.0]].view())?;
    println!("{:?}", explanation.values());
    Ok(())
}
