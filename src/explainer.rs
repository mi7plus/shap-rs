use crate::{Explanation, FeatureMetadata, OutputMetadata, Result};
use ndarray::ArrayView2;
/// Common interface implemented by SHAP explainers.
pub trait Explainer {
    fn explain(&self, x: ArrayView2<'_, f64>) -> Result<Explanation>;
}

/// Adds feature and output metadata to any explainer without changing its algorithm.
pub struct MetadataExplainer<E> {
    inner: E,
    feature_metadata: Option<FeatureMetadata>,
    output_metadata: Option<OutputMetadata>,
}

impl<E> MetadataExplainer<E> {
    pub fn new(inner: E) -> Self {
        Self {
            inner,
            feature_metadata: None,
            output_metadata: None,
        }
    }
    pub fn with_feature_metadata(mut self, metadata: FeatureMetadata) -> Result<Self> {
        metadata.validate()?;
        self.feature_metadata = Some(metadata);
        Ok(self)
    }
    pub fn with_output_metadata(mut self, metadata: OutputMetadata) -> Result<Self> {
        metadata.validate()?;
        self.output_metadata = Some(metadata);
        Ok(self)
    }
    pub fn inner(&self) -> &E {
        &self.inner
    }
    pub fn into_inner(self) -> E {
        self.inner
    }
}

impl<E: Explainer> Explainer for MetadataExplainer<E> {
    fn explain(&self, x: ArrayView2<'_, f64>) -> Result<Explanation> {
        let mut explanation = self.inner.explain(x)?;
        if let Some(metadata) = &self.feature_metadata {
            explanation = explanation.with_feature_metadata(metadata.clone())?;
        }
        if let Some(metadata) = &self.output_metadata {
            explanation = explanation.with_output_metadata(metadata.clone())?;
        }
        Ok(explanation)
    }
}

/// Convenience methods available on every concrete explainer.
pub trait ExplainerExt: Explainer + Sized {
    fn with_metadata(self) -> MetadataExplainer<Self> {
        MetadataExplainer::new(self)
    }
}

impl<E: Explainer> ExplainerExt for E {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explainers::ExactExplainer;
    use crate::{Background, FeatureKind, FnModel, OutputKind};
    use ndarray::{array, Axis};

    #[test]
    fn decorates_any_explainer_with_validated_metadata() {
        let model =
            FnModel::new(|x: ArrayView2<'_, f64>| Ok(x.sum_axis(Axis(1)).insert_axis(Axis(1))));
        let explainer = ExactExplainer::new(model, Background::new(array![[0., 0.]]).unwrap())
            .with_metadata()
            .with_feature_metadata(
                FeatureMetadata::new(vec!["age".into(), "income".into()])
                    .unwrap()
                    .with_kinds(vec![FeatureKind::Continuous, FeatureKind::Continuous])
                    .unwrap(),
            )
            .unwrap()
            .with_output_metadata(
                OutputMetadata::new(vec!["score".into()])
                    .unwrap()
                    .with_kinds(vec![OutputKind::Regression])
                    .unwrap(),
            )
            .unwrap();
        let explanation = explainer.explain(array![[2., 3.]].view()).unwrap();
        assert_eq!(explanation.feature_names().unwrap(), ["age", "income"]);
        assert_eq!(explanation.output_names().unwrap(), ["score"]);
    }

    #[test]
    fn reports_metadata_dimension_mismatch_at_explanation_time() {
        let model =
            FnModel::new(|x: ArrayView2<'_, f64>| Ok(x.sum_axis(Axis(1)).insert_axis(Axis(1))));
        let explainer = ExactExplainer::new(model, Background::new(array![[0., 0.]]).unwrap())
            .with_metadata()
            .with_feature_metadata(FeatureMetadata::new(vec!["only_one".into()]).unwrap())
            .unwrap();
        assert!(explainer.explain(array![[2., 3.]].view()).is_err());
    }
}
