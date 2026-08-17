pub mod bar;
pub mod beeswarm;
pub mod decision;
pub mod force;
pub mod heatmap;
pub mod html;
pub mod interaction;
pub mod scatter;
pub mod svg;
pub mod waterfall;

#[cfg(test)]
mod tests {
    use crate::{Explanation, ShapError};
    use ndarray::{array, Array3};

    fn explanation() -> Explanation {
        Explanation::new(
            Array3::from_shape_vec((1, 2, 1), vec![1., -2.]).unwrap(),
            array![[3.]],
            array![[10., 20.]],
        )
        .unwrap()
    }

    #[test]
    fn indexed_plots_report_precise_bounds_errors() {
        let e = explanation();
        assert!(matches!(
            super::beeswarm::data(&e, 1),
            Err(ShapError::InvalidOutputIndex { index: 1, .. })
        ));
        assert!(matches!(
            super::waterfall::data(&e, 2, 0),
            Err(ShapError::InvalidSampleIndex { index: 2, .. })
        ));
        assert!(matches!(
            super::scatter::data(&e, 0, 0, Some(3)),
            Err(ShapError::InvalidFeatureIndex { index: 3, .. })
        ));
        assert!(matches!(
            super::heatmap::data(&e, 4),
            Err(ShapError::InvalidOutputIndex { index: 4, .. })
        ));
    }

    #[test]
    fn plot_data_preserves_reconstruction_and_global_order() {
        let e = explanation();
        let force = super::force::data(&e, 0, 0).unwrap();
        assert_eq!(force.output_value, 2.);
        let waterfall = super::waterfall::data(&e, 0, 0).unwrap();
        assert_eq!(waterfall[0].feature, 1);
        let heatmap = super::heatmap::data(&e, 0).unwrap();
        assert_eq!(heatmap.feature_order, vec![1, 0]);
        let decision = super::decision::data(&e, 0).unwrap();
        assert_eq!(decision[0].cumulative_values.last(), Some(&2.));
        assert_eq!(super::beeswarm::data(&e, 0).unwrap().len(), 2);
    }
}
