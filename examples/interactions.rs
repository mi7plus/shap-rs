use ndarray::{array, Array2, ArrayView2, Axis};
use shap_rs::{interactions::ExactInteractionExplainer, FixedMasker, FnModel, Result};

fn main() -> Result<()> {
    let model = FnModel::new(|x: ArrayView2<'_, f64>| {
        Ok(x.map_axis(Axis(1), |row| row[0] * row[1])
            .insert_axis(Axis(1)))
    });
    let explanation = ExactInteractionExplainer::new(model, FixedMasker::new(array![0.0, 0.0])?)
        .explain(array![[2.0, 3.0]].view())?;
    let _: Array2<f64> = explanation.reconstructed();
    println!("{:?}", explanation.values());
    Ok(())
}
