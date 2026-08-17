//! Dependency-free SVG renderers for explanation summaries.

use crate::{Explanation, FeatureKind, OutputKind, Result, ShapError};
use std::fmt::Write;

#[derive(Debug, Clone, PartialEq)]
pub struct SvgOptions {
    pub width: u32,
    pub height: u32,
    pub max_features: usize,
    pub title: Option<String>,
    pub positive_color: String,
    pub negative_color: String,
    pub background_color: String,
    pub text_color: String,
}

impl Default for SvgOptions {
    fn default() -> Self {
        Self {
            width: 800,
            height: 480,
            max_features: 12,
            title: None,
            positive_color: "#e84a5f".into(),
            negative_color: "#268bd2".into(),
            background_color: "#ffffff".into(),
            text_color: "#202124".into(),
        }
    }
}

impl SvgOptions {
    pub fn validate(&self) -> Result<()> {
        if self.width < 240 || self.height < 160 || self.max_features == 0 {
            return Err(ShapError::InvalidConfiguration(
                "SVG width must be at least 240, height at least 160, and max_features positive"
                    .into(),
            ));
        }
        if [
            &self.positive_color,
            &self.negative_color,
            &self.background_color,
            &self.text_color,
        ]
        .iter()
        .any(|value| value.is_empty())
        {
            return Err(ShapError::InvalidConfiguration(
                "SVG colors cannot be empty".into(),
            ));
        }
        Ok(())
    }
}

/// Renders mean absolute SHAP importance as a horizontal SVG bar chart.
pub fn global_bar(e: &Explanation, options: &SvgOptions) -> Result<String> {
    e.validate()?;
    options.validate()?;
    let rows = super::bar::data(e)
        .into_iter()
        .take(options.max_features)
        .collect::<Vec<_>>();
    let title = options
        .title
        .as_deref()
        .unwrap_or("Mean absolute SHAP value");
    let top = 62.0;
    let bottom = 34.0;
    let label_width = (options.width as f64 * 0.30).clamp(100.0, 240.0);
    let right = 34.0;
    let chart_width = options.width as f64 - label_width - right;
    let row_height = (options.height as f64 - top - bottom) / rows.len().max(1) as f64;
    let maximum = rows.first().map_or(1.0, |(_, value)| value.max(1e-300));
    let mut svg = header(options, title);
    for (row, (feature, value)) in rows.iter().enumerate() {
        let y = top + row as f64 * row_height;
        let height = (row_height * 0.62).max(1.0);
        let width = chart_width * value / maximum;
        let label = feature_label(e, *feature);
        write!(
            svg,
            "<text x=\"{}\" y=\"{:.2}\" text-anchor=\"end\" dominant-baseline=\"middle\" font-size=\"13\" fill=\"{}\">{}</text>",
            label_width - 10.0,
            y + height / 2.0,
            escape(&options.text_color),
            escape(&label)
        )
        .unwrap();
        write!(
            svg,
            "<rect x=\"{label_width:.2}\" y=\"{y:.2}\" width=\"{width:.2}\" height=\"{height:.2}\" rx=\"2\" fill=\"{}\"/><text x=\"{:.2}\" y=\"{:.2}\" dominant-baseline=\"middle\" font-size=\"12\" fill=\"{}\">{value:.5}</text>",
            escape(&options.positive_color),
            label_width + width + 6.0,
            y + height / 2.0,
            escape(&options.text_color),
        )
        .unwrap();
    }
    svg.push_str("</g></svg>");
    Ok(svg)
}

/// Renders a deterministic beeswarm-style SVG for one model output.
pub fn beeswarm(e: &Explanation, output: usize, options: &SvgOptions) -> Result<String> {
    e.validate()?;
    options.validate()?;
    if output >= e.n_outputs() {
        return Err(ShapError::InvalidOutputIndex {
            index: output,
            n_outputs: e.n_outputs(),
        });
    }
    let order = super::bar::data(e)
        .into_iter()
        .take(options.max_features)
        .map(|(feature, _)| feature)
        .collect::<Vec<_>>();
    let default_title = format!("SHAP beeswarm — {}", output_label(e, output));
    let title = options.title.as_deref().unwrap_or(&default_title);
    let left = (options.width as f64 * 0.28).clamp(100.0, 220.0);
    let right = 34.0;
    let top = 58.0;
    let bottom = 36.0;
    let chart_width = options.width as f64 - left - right;
    let row_height = (options.height as f64 - top - bottom) / order.len().max(1) as f64;
    let maximum = order
        .iter()
        .flat_map(|&feature| {
            (0..e.n_samples()).map(move |sample| e.values()[[sample, feature, output]].abs())
        })
        .fold(0.0_f64, f64::max)
        .max(1e-300);
    let center = left + chart_width / 2.0;
    let mut svg = header(options, title);
    write!(
        svg,
        "<line x1=\"{center:.2}\" y1=\"{top:.2}\" x2=\"{center:.2}\" y2=\"{}\" stroke=\"#9aa0a6\" stroke-width=\"1\"/>",
        options.height as f64 - bottom
    )
    .unwrap();
    for (row, &feature) in order.iter().enumerate() {
        let center_y = top + (row as f64 + 0.5) * row_height;
        let finite_values = (0..e.n_samples())
            .map(|sample| e.data()[[sample, feature]])
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        let low = finite_values.iter().copied().fold(f64::INFINITY, f64::min);
        let high = finite_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        write!(
            svg,
            "<text x=\"{}\" y=\"{center_y:.2}\" text-anchor=\"end\" dominant-baseline=\"middle\" font-size=\"13\" fill=\"{}\">{}</text>",
            left - 10.0,
            escape(&options.text_color),
            escape(&feature_label(e, feature)),
        )
        .unwrap();
        for sample in 0..e.n_samples() {
            let shap = e.values()[[sample, feature, output]];
            let x = center + shap / maximum * chart_width * 0.48;
            let jitter_unit = deterministic_unit(sample, feature);
            let y = center_y + (jitter_unit - 0.5) * row_height * 0.62;
            let feature_value = e.data()[[sample, feature]];
            let normalized = if feature_value.is_finite() && high > low {
                ((feature_value - low) / (high - low)).clamp(0.0, 1.0)
            } else {
                0.5
            };
            let color = if !feature_value.is_finite() {
                "#777777"
            } else if normalized >= 0.5 {
                &options.positive_color
            } else {
                &options.negative_color
            };
            let opacity = 0.45 + 0.5 * (normalized - 0.5).abs() * 2.0;
            write!(
                svg,
                "<circle cx=\"{x:.2}\" cy=\"{y:.2}\" r=\"3.2\" fill=\"{}\" fill-opacity=\"{opacity:.3}\"><title>sample {sample}, SHAP {shap:.6}, value {}</title></circle>",
                escape(color),
                escape(&feature_value_label(e, feature, feature_value)),
            )
            .unwrap();
        }
    }
    if e.data().iter().any(|value| value.is_nan()) {
        write!(svg, "<text x=\"{}\" y=\"{}\" text-anchor=\"end\" font-size=\"11\" fill=\"#777777\">● missing</text>", options.width - 18, options.height - 10).unwrap();
    }
    svg.push_str("</g></svg>");
    Ok(svg)
}

/// Renders samples by features as a signed SHAP heatmap SVG.
pub fn heatmap(e: &Explanation, output: usize, options: &SvgOptions) -> Result<String> {
    e.validate()?;
    options.validate()?;
    let data = super::heatmap::data(e, output)?;
    let feature_order = data
        .feature_order
        .into_iter()
        .take(options.max_features)
        .collect::<Vec<_>>();
    let default_title = format!("SHAP heatmap — {}", output_label(e, output));
    let title = options.title.as_deref().unwrap_or(&default_title);
    let left = (options.width as f64 * 0.24).clamp(90.0, 190.0);
    let right = 24.0;
    let top = 58.0;
    let bottom = 34.0;
    let chart_width = options.width as f64 - left - right;
    let cell_width = chart_width / e.n_samples().max(1) as f64;
    let cell_height = (options.height as f64 - top - bottom) / feature_order.len().max(1) as f64;
    let maximum = feature_order
        .iter()
        .flat_map(|&feature| {
            (0..e.n_samples()).map(move |sample| e.values()[[sample, feature, output]].abs())
        })
        .fold(0.0_f64, f64::max)
        .max(1e-300);
    let mut svg = header(options, title);
    for (row, &feature) in feature_order.iter().enumerate() {
        let y = top + row as f64 * cell_height;
        write!(
            svg,
            "<text x=\"{}\" y=\"{:.2}\" text-anchor=\"end\" dominant-baseline=\"middle\" font-size=\"12\" fill=\"{}\">{}</text>",
            left - 8.0,
            y + cell_height / 2.0,
            escape(&options.text_color),
            escape(&feature_label(e, feature)),
        )
        .unwrap();
        for sample in 0..e.n_samples() {
            let value = e.values()[[sample, feature, output]];
            let opacity = (value.abs() / maximum).clamp(0.08, 1.0);
            let color = if value >= 0.0 {
                &options.positive_color
            } else {
                &options.negative_color
            };
            let x = left + sample as f64 * cell_width;
            write!(
                svg,
                "<rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{:.2}\" height=\"{:.2}\" fill=\"{}\" fill-opacity=\"{opacity:.3}\"><title>sample {sample}, SHAP {value:.6}</title></rect>",
                cell_width + 0.2,
                cell_height + 0.2,
                escape(color),
            )
            .unwrap();
        }
    }
    svg.push_str("</g></svg>");
    Ok(svg)
}

/// Renders SHAP value against one feature, optionally colored by another.
pub fn scatter(
    e: &Explanation,
    feature: usize,
    output: usize,
    color_feature: Option<usize>,
    options: &SvgOptions,
) -> Result<String> {
    e.validate()?;
    options.validate()?;
    let points = super::scatter::data(e, feature, output, color_feature)?;
    let finite_x = points
        .iter()
        .map(|point| point.feature_value)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let min_x = finite_x.iter().copied().fold(f64::INFINITY, f64::min);
    let max_x = finite_x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let x_span = if max_x > min_x { max_x - min_x } else { 1.0 };
    let categorical = matches!(
        feature_kind(e, feature),
        FeatureKind::Categorical | FeatureKind::Ordinal | FeatureKind::Boolean
    );
    let mut categories = if categorical {
        finite_x.clone()
    } else {
        Vec::new()
    };
    categories.sort_by(f64::total_cmp);
    categories.dedup_by(|a, b| a.total_cmp(b).is_eq());
    let has_missing = points.iter().any(|point| point.feature_value.is_nan());
    let max_y = points
        .iter()
        .map(|point| point.shap_value.abs())
        .fold(0.0_f64, f64::max)
        .max(1e-300);
    let finite_color = points
        .iter()
        .filter_map(|point| point.color_value)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let min_color = finite_color.iter().copied().fold(f64::INFINITY, f64::min);
    let max_color = finite_color
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let default_title = format!("SHAP dependence — {}", output_label(e, output));
    let title = options.title.as_deref().unwrap_or(&default_title);
    let left = 70.0;
    let right = 28.0;
    let top = 58.0;
    let bottom = 52.0;
    let chart_width = options.width as f64 - left - right;
    let chart_height = options.height as f64 - top - bottom;
    let zero_y = top + chart_height / 2.0;
    let mut svg = header(options, title);
    write!(
        svg,
        "<line x1=\"{left}\" y1=\"{zero_y:.2}\" x2=\"{}\" y2=\"{zero_y:.2}\" stroke=\"#9aa0a6\"/><line x1=\"{left}\" y1=\"{top}\" x2=\"{left}\" y2=\"{}\" stroke=\"#9aa0a6\"/><text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-size=\"13\" fill=\"{}\">{}</text>",
        options.width as f64 - right,
        options.height as f64 - bottom,
        left + chart_width / 2.0,
        options.height - 14,
        escape(&options.text_color),
        escape(&feature_label(e, feature)),
    )
    .unwrap();
    if categorical && categories.len() <= 16 {
        for (index, value) in categories.iter().enumerate() {
            let x = if categories.len() == 1 {
                left + chart_width / 2.0
            } else {
                left + index as f64 / (categories.len() - 1) as f64 * chart_width
            };
            write!(svg, "<line x1=\"{x:.2}\" y1=\"{}\" x2=\"{x:.2}\" y2=\"{}\" stroke=\"#9aa0a6\"/><text x=\"{x:.2}\" y=\"{}\" text-anchor=\"middle\" font-size=\"11\" fill=\"{}\">{}</text>", options.height as f64 - bottom, options.height as f64 - bottom + 5.0, options.height as f64 - bottom + 18.0, escape(&options.text_color), escape(&feature_value_label(e, feature, *value))).unwrap();
        }
    }
    if has_missing {
        write!(
            svg,
            "<text x=\"{left}\" y=\"{}\" font-size=\"11\" fill=\"#777777\">● missing</text>",
            top - 8.0
        )
        .unwrap();
    }
    for point in points {
        let x = if categorical && point.feature_value.is_finite() && categories.len() > 1 {
            let index = categories
                .binary_search_by(|value| value.total_cmp(&point.feature_value))
                .unwrap_or(0);
            left + index as f64 / (categories.len() - 1) as f64 * chart_width
        } else if categorical && point.feature_value.is_finite() {
            left + chart_width / 2.0
        } else if point.feature_value.is_finite() && min_x.is_finite() {
            left + (point.feature_value - min_x) / x_span * chart_width
        } else {
            left
        };
        let y = zero_y - point.shap_value / max_y * chart_height * 0.48;
        let normalized = match point.color_value {
            Some(value) if value.is_finite() && max_color > min_color => {
                ((value - min_color) / (max_color - min_color)).clamp(0.0, 1.0)
            }
            _ => 0.5,
        };
        let color = if !point.feature_value.is_finite()
            || point.color_value.is_some_and(|value| !value.is_finite())
        {
            "#777777"
        } else if color_feature.is_none() || normalized >= 0.5 {
            &options.positive_color
        } else {
            &options.negative_color
        };
        write!(
            svg,
            "<circle cx=\"{x:.2}\" cy=\"{y:.2}\" r=\"4\" fill=\"{}\" fill-opacity=\"0.72\"><title>sample {}, feature {}, SHAP {:.6}</title></circle>",
            escape(color),
            point.sample,
            escape(&feature_value_label(e, feature, point.feature_value)),
            point.shap_value,
        )
        .unwrap();
    }
    svg.push_str("</g></svg>");
    Ok(svg)
}

/// Renders cumulative SHAP decision paths for all samples of one output.
pub fn decision(e: &Explanation, output: usize, options: &SvgOptions) -> Result<String> {
    e.validate()?;
    options.validate()?;
    if output >= e.n_outputs() {
        return Err(ShapError::InvalidOutputIndex {
            index: output,
            n_outputs: e.n_outputs(),
        });
    }
    let selected = super::bar::data(e)
        .into_iter()
        .take(options.max_features)
        .map(|(feature, _)| feature)
        .collect::<Vec<_>>();
    let omitted = e.n_features() - selected.len();
    let mut paths = Vec::with_capacity(e.n_samples());
    for sample in 0..e.n_samples() {
        let mut cumulative = vec![e.base_values()[[sample, output]]];
        for &feature in &selected {
            cumulative
                .push(cumulative.last().copied().unwrap() + e.values()[[sample, feature, output]]);
        }
        if omitted > 0 {
            let other = (0..e.n_features())
                .filter(|feature| !selected.contains(feature))
                .map(|feature| e.values()[[sample, feature, output]])
                .sum::<f64>();
            cumulative.push(cumulative.last().copied().unwrap() + other);
        }
        paths.push(cumulative);
    }
    let minimum = paths
        .iter()
        .flatten()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let maximum = paths
        .iter()
        .flatten()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let span = (maximum - minimum).max(1e-12);
    let default_title = format!("SHAP decision paths — {}", output_label(e, output));
    let title = options.title.as_deref().unwrap_or(&default_title);
    let left = 72.0;
    let right = 30.0;
    let top = 58.0;
    let bottom = 62.0;
    let chart_width = options.width as f64 - left - right;
    let chart_height = options.height as f64 - top - bottom;
    let steps = selected.len() + usize::from(omitted > 0);
    let x = |step: usize| left + step as f64 / steps.max(1) as f64 * chart_width;
    let y = |value: f64| top + (maximum - value) / span * chart_height;
    let mut svg = header(options, title);
    for step in 0..=steps {
        write!(
            svg,
            "<line x1=\"{:.2}\" y1=\"{top}\" x2=\"{:.2}\" y2=\"{}\" stroke=\"#e2e2e2\"/>",
            x(step),
            x(step),
            options.height as f64 - bottom,
        )
        .unwrap();
    }
    for (sample, path) in paths.iter().enumerate() {
        let points = path
            .iter()
            .enumerate()
            .map(|(step, &value)| format!("{:.2},{:.2}", x(step), y(value)))
            .collect::<Vec<_>>()
            .join(" ");
        let opacity = 0.38 + deterministic_unit(sample, output) * 0.55;
        write!(
            svg,
            "<polyline points=\"{points}\" fill=\"none\" stroke=\"{}\" stroke-width=\"2\" stroke-opacity=\"{opacity:.3}\"><title>sample {sample}, output {:.6}</title></polyline>",
            escape(&options.positive_color),
            path.last().copied().unwrap(),
        )
        .unwrap();
    }
    let mut labels = vec!["base".to_string()];
    labels.extend(selected.iter().map(|&feature| feature_label(e, feature)));
    if omitted > 0 {
        labels.push(format!("{omitted} others"));
    }
    for (step, label) in labels.iter().enumerate() {
        write!(
            svg,
            "<text x=\"{:.2}\" y=\"{}\" text-anchor=\"middle\" font-size=\"11\" fill=\"{}\">{}</text>",
            x(step),
            options.height - 34,
            escape(&options.text_color),
            escape(label),
        )
        .unwrap();
    }
    svg.push_str("</g></svg>");
    Ok(svg)
}

/// Renders a compact force plot from the base value to one reconstructed output.
pub fn force(
    e: &Explanation,
    sample: usize,
    output: usize,
    options: &SvgOptions,
) -> Result<String> {
    e.validate()?;
    options.validate()?;
    let force = super::force::data(e, sample, output)?;
    let mut ranked = (0..e.n_features())
        .map(|feature| (feature, force.contributions[feature]))
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
    ranked.truncate(options.max_features);
    let omitted = e.n_features() - ranked.len();
    if omitted > 0 {
        let other = (0..e.n_features())
            .filter(|feature| !ranked.iter().any(|(selected, _)| selected == feature))
            .map(|feature| force.contributions[feature])
            .sum::<f64>();
        ranked.push((usize::MAX, other));
    }
    // Place the strongest effects closest to the prediction while preserving
    // the exact signed sum from base to output.
    let mut positions = Vec::with_capacity(ranked.len() + 1);
    positions.push(force.base_value);
    for (_, contribution) in &ranked {
        positions.push(positions.last().copied().unwrap() + contribution);
    }
    let minimum = positions
        .iter()
        .copied()
        .chain([force.base_value, force.output_value])
        .fold(f64::INFINITY, f64::min);
    let maximum = positions
        .iter()
        .copied()
        .chain([force.base_value, force.output_value])
        .fold(f64::NEG_INFINITY, f64::max);
    let padding = ((maximum - minimum) * 0.08).max(1e-9);
    let domain_min = minimum - padding;
    let domain_span = maximum - minimum + 2.0 * padding;
    let default_title = format!("SHAP force plot — {}", output_label(e, output));
    let title = options.title.as_deref().unwrap_or(&default_title);
    let left = 36.0;
    let right = 36.0;
    let chart_width = options.width as f64 - left - right;
    let center_y = options.height as f64 * 0.54;
    let band_height = (options.height as f64 * 0.24).clamp(38.0, 92.0);
    let scale = |value: f64| left + (value - domain_min) / domain_span * chart_width;
    let mut svg = header(options, title);
    write!(
        svg,
        "<line x1=\"{left}\" y1=\"{center_y:.2}\" x2=\"{}\" y2=\"{center_y:.2}\" stroke=\"#9aa0a6\"/><line x1=\"{:.2}\" y1=\"{}\" x2=\"{:.2}\" y2=\"{}\" stroke=\"#5f6368\" stroke-dasharray=\"4 3\"/><text x=\"{:.2}\" y=\"{}\" text-anchor=\"middle\" font-size=\"12\" fill=\"{}\">base {:.5}</text>",
        options.width as f64 - right,
        scale(force.base_value),
        center_y - band_height * 0.72,
        scale(force.base_value),
        center_y + band_height * 0.72,
        scale(force.base_value),
        center_y + band_height,
        escape(&options.text_color),
        force.base_value,
    )
    .unwrap();
    for (index, ((feature, contribution), window)) in
        ranked.iter().zip(positions.windows(2)).enumerate()
    {
        let x1 = scale(window[0]);
        let x2 = scale(window[1]);
        let direction = if x2 >= x1 { 1.0 } else { -1.0 };
        let tip = (8.0_f64).min((x2 - x1).abs() * 0.4);
        let outer = if direction > 0.0 { x2 - tip } else { x2 + tip };
        let top = center_y - band_height / 2.0;
        let bottom = center_y + band_height / 2.0;
        let color = if *contribution >= 0.0 {
            &options.positive_color
        } else {
            &options.negative_color
        };
        let points = format!(
            "{x1:.2},{top:.2} {outer:.2},{top:.2} {x2:.2},{center_y:.2} {outer:.2},{bottom:.2} {x1:.2},{bottom:.2}"
        );
        let label = if *feature == usize::MAX {
            format!("{omitted} other features")
        } else {
            feature_label(e, *feature)
        };
        write!(
            svg,
            "<polygon points=\"{points}\" fill=\"{}\" fill-opacity=\"0.88\"><title>{}: {contribution:+.6}</title></polygon>",
            escape(color),
            escape(&label),
        )
        .unwrap();
        if (x2 - x1).abs() > 58.0 {
            write!(
                svg,
                "<text x=\"{:.2}\" y=\"{:.2}\" text-anchor=\"middle\" dominant-baseline=\"middle\" font-size=\"11\" fill=\"#ffffff\">{}</text>",
                (x1 + x2) / 2.0,
                center_y + if index % 2 == 0 { -7.0 } else { 7.0 },
                escape(&label),
            )
            .unwrap();
        }
    }
    write!(
        svg,
        "<line x1=\"{:.2}\" y1=\"{}\" x2=\"{:.2}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"2\"/><text x=\"{:.2}\" y=\"{}\" text-anchor=\"middle\" font-size=\"14\" font-weight=\"bold\" fill=\"{}\">output {:.5}</text></g></svg>",
        scale(force.output_value),
        center_y - band_height * 0.75,
        scale(force.output_value),
        center_y + band_height * 0.75,
        escape(&options.text_color),
        scale(force.output_value),
        center_y - band_height,
        escape(&options.text_color),
        force.output_value,
    )
    .unwrap();
    Ok(svg)
}

/// Renders one sample/output as a cumulative SHAP waterfall SVG.
pub fn waterfall(
    e: &Explanation,
    sample: usize,
    output: usize,
    options: &SvgOptions,
) -> Result<String> {
    e.validate()?;
    options.validate()?;
    let rows = super::waterfall::data(e, sample, output)?
        .into_iter()
        .take(options.max_features)
        .collect::<Vec<_>>();
    let omitted = e.n_features() - rows.len();
    let omitted_value = if omitted > 0 {
        (0..e.n_features())
            .filter(|feature| !rows.iter().any(|row| row.feature == *feature))
            .map(|feature| e.values()[[sample, feature, output]])
            .sum()
    } else {
        0.0
    };
    let mut contributions = rows
        .iter()
        .map(|row| (feature_label(e, row.feature), row.contribution))
        .collect::<Vec<_>>();
    if omitted > 0 {
        contributions.push((format!("{omitted} other features"), omitted_value));
    }
    let base = e.base_values()[[sample, output]];
    let prediction = e.reconstructed()[[sample, output]];
    let mut positions = Vec::with_capacity(contributions.len() + 1);
    positions.push(base);
    for (_, contribution) in &contributions {
        positions.push(positions.last().copied().unwrap() + contribution);
    }
    let minimum = positions
        .iter()
        .copied()
        .chain([base, prediction])
        .fold(f64::INFINITY, f64::min);
    let maximum = positions
        .iter()
        .copied()
        .chain([base, prediction])
        .fold(f64::NEG_INFINITY, f64::max);
    let span = (maximum - minimum).max(1e-12);
    let default_title = format!("SHAP waterfall — {}", output_label(e, output));
    let title = options.title.as_deref().unwrap_or(&default_title);
    let left = 170.0;
    let right = 38.0;
    let top = 70.0;
    let bottom = 42.0;
    let chart_width = options.width as f64 - left - right;
    let row_height = (options.height as f64 - top - bottom) / contributions.len().max(1) as f64;
    let scale = |value: f64| left + (value - minimum) / span * chart_width;
    let mut svg = header(options, title);
    for (row, (label, contribution)) in contributions.iter().enumerate() {
        let before = positions[row];
        let after = positions[row + 1];
        let x1 = scale(before);
        let x2 = scale(after);
        let x = x1.min(x2);
        let width = (x2 - x1).abs().max(1.0);
        let y = top + row as f64 * row_height;
        let height = (row_height * 0.58).max(1.0);
        let color = if *contribution >= 0.0 {
            &options.positive_color
        } else {
            &options.negative_color
        };
        write!(
            svg,
            "<text x=\"{}\" y=\"{:.2}\" text-anchor=\"end\" dominant-baseline=\"middle\" font-size=\"13\" fill=\"{}\">{}</text><line x1=\"{x1:.2}\" y1=\"{:.2}\" x2=\"{x1:.2}\" y2=\"{:.2}\" stroke=\"#b0b0b0\"/><rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{width:.2}\" height=\"{height:.2}\" rx=\"2\" fill=\"{}\"/><text x=\"{:.2}\" y=\"{:.2}\" font-size=\"12\" fill=\"{}\">{contribution:+.5}</text>",
            left - 10.0,
            y + height / 2.0,
            escape(&options.text_color),
            escape(label),
            y,
            y + height,
            escape(color),
            x2 + if *contribution >= 0.0 { 5.0 } else { -58.0 },
            y + height / 2.0 + 4.0,
            escape(&options.text_color),
        )
        .unwrap();
    }
    write!(
        svg,
        "<text x=\"{:.2}\" y=\"{}\" text-anchor=\"middle\" font-size=\"12\" fill=\"{}\">base {base:.5}</text><text x=\"{:.2}\" y=\"{}\" text-anchor=\"middle\" font-size=\"12\" font-weight=\"bold\" fill=\"{}\">output {prediction:.5}</text></g></svg>",
        scale(base),
        options.height - 12,
        escape(&options.text_color),
        scale(prediction),
        options.height - 12,
        escape(&options.text_color),
    )
    .unwrap();
    Ok(svg)
}

fn header(options: &SvgOptions, title: &str) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" role=\"img\" aria-label=\"{}\"><rect width=\"100%\" height=\"100%\" fill=\"{}\"/><text x=\"20\" y=\"30\" font-family=\"system-ui,sans-serif\" font-size=\"18\" font-weight=\"600\" fill=\"{}\">{}</text><g font-family=\"system-ui,sans-serif\">",
        options.width,
        options.height,
        options.width,
        options.height,
        escape(title),
        escape(&options.background_color),
        escape(&options.text_color),
        escape(title),
    )
}

fn feature_label(e: &Explanation, feature: usize) -> String {
    let name = e
        .feature_metadata()
        .and_then(|metadata| metadata.display_names.as_ref())
        .and_then(|names| names.get(feature))
        .or_else(|| e.feature_names().and_then(|names| names.get(feature)))
        .cloned()
        .unwrap_or_else(|| format!("Feature {feature}"));
    match e
        .feature_metadata()
        .and_then(|metadata| metadata.units.as_ref())
        .and_then(|units| units.get(feature))
        .and_then(Option::as_deref)
    {
        Some(unit) if !unit.is_empty() => format!("{name} ({unit})"),
        _ => name,
    }
}

fn feature_kind(e: &Explanation, feature: usize) -> FeatureKind {
    e.feature_metadata()
        .and_then(|metadata| metadata.kinds.as_ref())
        .and_then(|kinds| kinds.get(feature))
        .copied()
        .unwrap_or_default()
}

fn feature_value_label(e: &Explanation, feature: usize, value: f64) -> String {
    if value.is_nan() {
        return "missing".into();
    }
    match feature_kind(e, feature) {
        FeatureKind::Boolean => {
            if value == 0.0 {
                "false".into()
            } else if value == 1.0 {
                "true".into()
            } else {
                format!("category {value}")
            }
        }
        FeatureKind::Categorical | FeatureKind::Ordinal => format!("category {value}"),
        _ => format!("{value}"),
    }
}

fn output_label(e: &Explanation, output: usize) -> String {
    let name = e
        .output_names()
        .and_then(|names| names.get(output))
        .cloned()
        .unwrap_or_else(|| format!("output {output}"));
    let kind = e
        .output_metadata()
        .and_then(|metadata| metadata.kinds.as_ref())
        .and_then(|kinds| kinds.get(output))
        .copied()
        .unwrap_or_default();
    match kind {
        OutputKind::Probability => format!("{name} probability"),
        OutputKind::LogOdds => format!("{name} log odds"),
        OutputKind::ClassScore => format!("{name} class score"),
        OutputKind::Embedding => format!("{name} embedding value"),
        OutputKind::Regression => name,
    }
}

fn deterministic_unit(sample: usize, feature: usize) -> f64 {
    let mut value = (sample as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((feature as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    (value >> 11) as f64 / (1u64 << 53) as f64
}

fn escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FeatureMetadata, OutputMetadata};
    use ndarray::{array, Array3};

    fn explanation() -> Explanation {
        Explanation::new(
            Array3::from_shape_vec((1, 3, 1), vec![2., -1., 0.5]).unwrap(),
            array![[1.]],
            array![[4., 5., 6.]],
        )
        .unwrap()
        .with_feature_metadata(
            FeatureMetadata::new(vec!["a&b".into(), "second".into(), "third".into()]).unwrap(),
        )
        .unwrap()
        .with_output_metadata(OutputMetadata::new(vec!["score <raw>".into()]).unwrap())
        .unwrap()
    }

    #[test]
    fn renders_valid_escaped_bar_and_waterfall_svg() {
        let e = explanation();
        let bar = global_bar(&e, &SvgOptions::default()).unwrap();
        assert!(bar.starts_with("<svg"));
        assert!(bar.ends_with("</svg>"));
        assert!(bar.contains("a&amp;b"));
        let waterfall = super::waterfall(&e, 0, 0, &SvgOptions::default()).unwrap();
        assert!(waterfall.contains("score &lt;raw&gt;"));
        assert!(waterfall.contains("output 2.50000"));
    }

    #[test]
    fn rejects_invalid_svg_options_and_indices() {
        let e = explanation();
        let bad = SvgOptions {
            width: 100,
            ..SvgOptions::default()
        };
        assert!(global_bar(&e, &bad).is_err());
        assert!(matches!(
            super::waterfall(&e, 2, 0, &SvgOptions::default()),
            Err(ShapError::InvalidSampleIndex { .. })
        ));
    }

    #[test]
    fn renders_deterministic_beeswarm_and_heatmap_svg() {
        let e = explanation();
        let first = beeswarm(&e, 0, &SvgOptions::default()).unwrap();
        let second = beeswarm(&e, 0, &SvgOptions::default()).unwrap();
        assert_eq!(first, second);
        assert!(first.contains("<circle"));
        let heat = heatmap(&e, 0, &SvgOptions::default()).unwrap();
        assert!(heat.contains("<rect"));
        assert!(matches!(
            heatmap(&e, 2, &SvgOptions::default()),
            Err(ShapError::InvalidOutputIndex { .. })
        ));
    }

    #[test]
    fn renders_scatter_and_reconstructing_decision_paths() {
        let e = explanation();
        let scatter = super::scatter(&e, 0, 0, Some(1), &SvgOptions::default()).unwrap();
        assert!(scatter.contains("SHAP dependence"));
        assert!(scatter.contains("<circle"));
        let decision = decision(&e, 0, &SvgOptions::default()).unwrap();
        assert!(decision.contains("<polyline"));
        assert!(decision.contains("output 2.500000"));
    }

    #[test]
    fn formats_categorical_missing_units_and_probability_labels() {
        let explanation = Explanation::new(
            Array3::from_shape_vec((3, 2, 1), vec![1., 0., 2., 0., 3., 0.]).unwrap(),
            array![[0.], [0.], [0.]],
            array![[0., 20.], [1., 30.], [f64::NAN, 40.]],
        )
        .unwrap()
        .with_feature_metadata(
            FeatureMetadata::new(vec!["segment".into(), "age".into()])
                .unwrap()
                .with_kinds(vec![FeatureKind::Categorical, FeatureKind::Continuous])
                .unwrap()
                .with_units(vec![None, Some("years".into())])
                .unwrap(),
        )
        .unwrap()
        .with_output_metadata(
            OutputMetadata::new(vec!["churn".into()])
                .unwrap()
                .with_kinds(vec![OutputKind::Probability])
                .unwrap(),
        )
        .unwrap();
        let scatter = super::scatter(&explanation, 0, 0, None, &SvgOptions::default()).unwrap();
        assert!(scatter.contains("category 0"));
        assert!(scatter.contains("category 1"));
        assert!(scatter.contains("missing"));
        assert!(scatter.contains("churn probability"));
        let bar = global_bar(&explanation, &SvgOptions::default()).unwrap();
        assert!(bar.contains("age (years)"));
        let beeswarm = beeswarm(&explanation, 0, &SvgOptions::default()).unwrap();
        assert!(beeswarm.contains("● missing"));
    }

    #[test]
    fn renders_additive_force_plot_and_aggregates_omitted_features() {
        let e = explanation();
        let options = SvgOptions {
            max_features: 1,
            ..SvgOptions::default()
        };
        let force = force(&e, 0, 0, &options).unwrap();
        assert!(force.contains("<polygon"));
        assert!(force.contains("2 other features"));
        assert!(force.contains("output 2.50000"));
    }

    #[test]
    fn every_renderer_produces_a_parseable_accessible_svg_tree() {
        let e = explanation();
        let options = SvgOptions::default();
        let documents = [
            global_bar(&e, &options).unwrap(),
            super::waterfall(&e, 0, 0, &options).unwrap(),
            force(&e, 0, 0, &options).unwrap(),
            beeswarm(&e, 0, &options).unwrap(),
            heatmap(&e, 0, &options).unwrap(),
            super::scatter(&e, 0, 0, Some(1), &options).unwrap(),
            decision(&e, 0, &options).unwrap(),
        ];
        for svg in documents {
            let document = roxmltree::Document::parse(&svg).unwrap();
            let root = document.root_element();
            assert_eq!(root.tag_name().name(), "svg");
            assert_eq!(root.attribute("role"), Some("img"));
            assert!(root
                .attribute("aria-label")
                .is_some_and(|label| !label.is_empty()));
            assert!(
                root.descendants()
                    .filter(roxmltree::Node::is_element)
                    .count()
                    > 3
            );
        }
    }
}
