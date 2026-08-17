use rayon::prelude::*;
use std::fmt;

/// Standardized output holding feature attributions for one or more samples.
#[derive(Debug, Clone)]
pub struct Explanation {
    /// SHAP value matrix of shape `[num_samples, num_features]`
    pub values: Vec<Vec<f64>>,
    /// Expected value baseline E[f(x)] across background data
    pub base_value: f64,
    /// Optional feature names for visualization/reporting
    pub feature_names: Option<Vec<String>>,
}

impl Explanation {
    pub fn new(
        values: Vec<Vec<f64>>,
        base_value: f64,
        feature_names: Option<Vec<String>>,
    ) -> Self {
        Self {
            values,
            base_value,
            feature_names,
        }
    }

    /// Number of explained instances
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Number of features per sample
    pub fn num_features(&self) -> usize {
        self.values.first().map_or(0, |v| v.len())
    }

    /// Serializes explanation into a simple JSON format compatible with web/frontend plots.
    pub fn to_json(&self) -> String {
        let names_json = match &self.feature_names {
            Some(names) => format!(
                "[{}]",
                names
                    .iter()
                    .map(|n| format!("\"{}\"", n))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            None => "null".to_string(),
        };

        let values_json = format!(
            "[{}]",
            self.values
                .iter()
                .map(|row| format!("{:?}", row))
                .collect::<Vec<_>>()
                .join(",")
        );

        format!(
            "{{\"base_value\": {}, \"feature_names\": {}, \"values\": {}}}",
            self.base_value, names_json, values_json
        )
    }
}

/// Generic trait for unified explainer implementations (Kernel, Tree, Sampling, etc.)
pub trait Explainer {
    /// Computes SHAP values for a single input vector
    fn explain_one(&self, sample: &[f64]) -> Result<Vec<f64>, &'static str>;

    /// Computes SHAP values across multiple input vectors sequentially
    fn explain_batch(&self, samples: &[Vec<f64>]) -> Result<Explanation, &'static str> {
        let mut results = Vec::with_capacity(samples.len());
        for sample in samples {
            results.push(self.explain_one(sample)?);
        }

        Ok(Explanation::new(results, self.base_value(), None))
    }

    /// Computes SHAP values across multiple input vectors in parallel using Rayon
    fn explain_batch_parallel(&self, samples: &[Vec<f64>]) -> Result<Explanation, &'static str>
    where
        Self: Sync,
    {
        let results: Result<Vec<Vec<f64>>, &'static str> = samples
            .par_iter()
            .map(|sample| self.explain_one(sample))
            .collect();

        Ok(Explanation::new(results?, self.base_value(), None))
    }

    /// Baseline expected prediction value E[f(x)]
    fn base_value(&self) -> f64;
}

/// Custom error types for SHAP calculations
#[derive(Debug, Clone)]
pub enum ShapError {
    SolverFailure(&'static str),
    DimensionMismatch { expected: usize, found: usize },
    InvalidBackgroundData,
}

impl fmt::Display for ShapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapError::SolverFailure(msg) => write!(f, "WLS Solver Error: {}", msg),
            ShapError::DimensionMismatch { expected, found } => write!(
                f,
                "Dimension Mismatch: expected {} features, found {}",
                expected, found
            ),
            ShapError::InvalidBackgroundData => {
                write!(f, "Background dataset cannot be empty")
            }
        }
    }
}

impl std::error::Error for ShapError {}