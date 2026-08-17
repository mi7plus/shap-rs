use crate::{Explainer, Explanation, Result, ShapError};
use ndarray::{ArrayView2, Axis, Slice};
use rayon::prelude::*;
/// Parallel sample-batch execution for every thread-safe explainer.
pub trait ParallelExplainerExt: Explainer + Sync {
    fn explain_parallel(&self, x: ArrayView2<'_, f64>, chunk_size: usize) -> Result<Explanation> {
        if chunk_size == 0 {
            return Err(ShapError::InvalidConfiguration(
                "parallel chunk size must be positive".into(),
            ));
        }
        if x.nrows() == 0 {
            return Err(ShapError::EmptyData);
        }
        let starts = (0..x.nrows()).step_by(chunk_size).collect::<Vec<_>>();
        let parts = starts
            .into_par_iter()
            .map(|start| {
                let end = (start + chunk_size).min(x.nrows());
                self.explain(x.slice_axis(Axis(0), Slice::from(start..end)))
            })
            .collect::<Result<Vec<_>>>()?;
        Explanation::concatenate(&parts)
    }
}
impl<T: Explainer + Sync> ParallelExplainerExt for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explainers::PermutationExplainer;
    use crate::{Background, FnModel};
    use ndarray::{array, Array2, ArrayView2};

    #[test]
    fn stochastic_explanations_are_chunk_size_invariant() {
        let model = FnModel::new(|x: ArrayView2<'_, f64>| {
            Ok(Array2::from_shape_fn((x.nrows(), 2), |(sample, output)| {
                let row = x.row(sample);
                if output == 0 {
                    row[0] * row[1] + row[2].sin()
                } else {
                    row[0] - row[1] * row[2]
                }
            }))
        });
        let explainer = PermutationExplainer::new(
            model,
            Background::new(array![[0., 0., 0.], [1., -1., 0.5]]).unwrap(),
        )
        .with_n_permutations(17)
        .with_seed(42);
        let samples = array![
            [2., 3., -1.],
            [-0.5, 1., 2.],
            [4., -2., 0.25],
            [1.5, 0.75, -3.],
            [0., 2., 1.]
        ];
        let sequential = explainer.explain(samples.view()).unwrap();
        for chunk_size in [1, 2, 3, 8] {
            let parallel = explainer
                .explain_parallel(samples.view(), chunk_size)
                .unwrap();
            assert_eq!(parallel, sequential);
        }
    }

    #[test]
    fn rejects_zero_parallel_chunk_size() {
        let model = FnModel::new(|x: ArrayView2<'_, f64>| {
            Ok(x.sum_axis(ndarray::Axis(1)).insert_axis(ndarray::Axis(1)))
        });
        let explainer =
            PermutationExplainer::new(model, Background::new(array![[0., 0.]]).unwrap());
        assert!(explainer
            .explain_parallel(array![[1., 2.]].view(), 0)
            .is_err());
    }
}
