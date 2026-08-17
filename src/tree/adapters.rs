//! JSON model importers. Dump XGBoost with `with_stats=true`; branch covers are
//! required for correct path-dependent expectations.
use super::{MissingBranch, MissingValuePolicy, Node, SplitComparison, Tree, TreeEnsemble};
use crate::{Result, ShapError};
use serde_json::Value;

pub fn from_xgboost_json(
    json: &str,
    n_features: usize,
    n_outputs: usize,
    base_values: Vec<f64>,
) -> Result<TreeEnsemble> {
    from_xgboost_json_with_tree_weights(json, n_features, n_outputs, base_values, None)
}

/// Imports an XGBoost recursive JSON dump with optional external per-tree
/// weights. The latter are required for DART dumps because `weight_drop` is
/// stored in XGBoost's full model JSON rather than each recursively dumped tree.
pub fn from_xgboost_json_with_tree_weights(
    json: &str,
    n_features: usize,
    n_outputs: usize,
    base_values: Vec<f64>,
    tree_weights: Option<Vec<f64>>,
) -> Result<TreeEnsemble> {
    if n_outputs == 0 {
        return Err(err("output count must be positive"));
    }
    let root: Value = serde_json::from_str(json)
        .map_err(|e| ShapError::InvalidConfiguration(format!("invalid XGBoost JSON: {e}")))?;
    let trees = root.as_array().ok_or_else(|| {
        ShapError::InvalidConfiguration("XGBoost dump must be a JSON array".into())
    })?;
    if tree_weights
        .as_ref()
        .is_some_and(|weights| weights.len() != trees.len())
    {
        return Err(ShapError::DimensionMismatch {
            expected: format!("{} XGBoost tree weights", trees.len()),
            found: format!("{}", tree_weights.as_ref().map_or(0, Vec::len)),
        });
    }
    let mut out = Vec::with_capacity(trees.len());
    for (t, v) in trees.iter().enumerate() {
        let output = t % n_outputs;
        let mut nodes = Vec::new();
        xgb_node(v, &mut nodes, n_features, n_outputs, output)?;
        let weight = tree_weights.as_ref().map_or(1.0, |weights| weights[t]);
        out.push((Tree::new(nodes, 0, n_features)?, weight));
    }
    let output_groups = (0..out.len()).map(|tree| Some(tree % n_outputs)).collect();
    TreeEnsemble::new_with_output_groups(out, base_values, output_groups)
}
fn xgb_node(
    v: &Value,
    nodes: &mut Vec<Node>,
    nf: usize,
    no: usize,
    output: usize,
) -> Result<usize> {
    let index = nodes.len();
    nodes.push(Node::Leaf {
        values: vec![0.; no],
        cover: 0.,
    });
    let cover = num(v, "cover")?;
    if let Some(leaf) = v.get("leaf").and_then(Value::as_f64) {
        let mut values = vec![0.; no];
        values[output] = leaf;
        nodes[index] = Node::Leaf { values, cover };
        return Ok(index);
    }
    let split = v
        .get("split")
        .ok_or_else(|| err("XGBoost split is missing"))?;
    let feature = if let Some(i) = split.as_u64() {
        i as usize
    } else {
        split
            .as_str()
            .and_then(|s| s.strip_prefix('f'))
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| err("invalid XGBoost split feature"))?
    };
    if feature >= nf {
        return Err(err("XGBoost feature index is out of bounds"));
    }
    let threshold = num(v, "split_condition")?;
    let children = v
        .get("children")
        .and_then(Value::as_array)
        .ok_or_else(|| err("XGBoost split has no children"))?;
    if children.len() != 2 {
        return Err(err("XGBoost trees must be binary"));
    }
    let yes = v
        .get("yes")
        .and_then(Value::as_u64)
        .ok_or_else(|| err("XGBoost yes child missing"))?;
    let no_id = v
        .get("no")
        .and_then(Value::as_u64)
        .ok_or_else(|| err("XGBoost no child missing"))?;
    let missing = v
        .get("missing")
        .and_then(Value::as_u64)
        .ok_or_else(|| err("XGBoost missing child missing"))?;
    if yes == no_id || (missing != yes && missing != no_id) {
        return Err(err("XGBoost child routing is inconsistent"));
    }
    let yes_pos = children
        .iter()
        .position(|c| c.get("nodeid").and_then(Value::as_u64) == Some(yes))
        .ok_or_else(|| err("XGBoost yes child not found"))?;
    let no_pos = children
        .iter()
        .position(|c| c.get("nodeid").and_then(Value::as_u64) == Some(no_id))
        .ok_or_else(|| err("XGBoost no child not found"))?;
    if yes_pos == no_pos {
        return Err(err("XGBoost yes and no routes reference the same child"));
    }
    let left = xgb_node(&children[yes_pos], nodes, nf, no, output)?;
    let right = xgb_node(&children[no_pos], nodes, nf, no, output)?;
    nodes[index] = Node::NumericalSplit {
        feature,
        threshold,
        comparison: SplitComparison::LessThan,
        left,
        right,
        missing: if missing == yes {
            MissingBranch::Left
        } else {
            MissingBranch::Right
        },
        missing_value: MissingValuePolicy::NaN,
        cover,
    };
    Ok(index)
}

/// Imports XGBoost's full columnar saved-model JSON schema. `base_values` are
/// raw-margin offsets; objective-specific probability-to-margin conversion is
/// deliberately left to the caller.
pub fn from_xgboost_model_json(json: &str, base_values: Vec<f64>) -> Result<TreeEnsemble> {
    let root: Value = serde_json::from_str(json)
        .map_err(|e| ShapError::InvalidConfiguration(format!("invalid XGBoost model JSON: {e}")))?;
    let learner = root
        .get("learner")
        .ok_or_else(|| err("XGBoost model has no learner"))?;
    let parameters = learner
        .get("learner_model_param")
        .ok_or_else(|| err("XGBoost model has no learner parameters"))?;
    let n_features = string_usize(parameters, "num_feature")?;
    let gradient_booster = learner
        .get("gradient_booster")
        .ok_or_else(|| err("XGBoost model has no gradient booster"))?;
    let (tree_model, weights) =
        if gradient_booster.get("name").and_then(Value::as_str) == Some("dart") {
            let weights = gradient_booster
                .get("weight_drop")
                .and_then(Value::as_array)
                .ok_or_else(|| err("XGBoost DART model has no weight_drop"))?
                .iter()
                .map(|value| value_number(value).ok_or_else(|| err("invalid DART tree weight")))
                .collect::<Result<Vec<_>>>()?;
            (
                gradient_booster
                    .get("gbtree")
                    .and_then(|value| value.get("model"))
                    .ok_or_else(|| err("XGBoost DART model has no gbtree model"))?,
                Some(weights),
            )
        } else {
            (
                gradient_booster
                    .get("model")
                    .ok_or_else(|| err("XGBoost model has no tree model"))?,
                None,
            )
        };
    let trees = tree_model
        .get("trees")
        .and_then(Value::as_array)
        .ok_or_else(|| err("XGBoost model has no trees"))?;
    let groups = tree_model
        .get("tree_info")
        .and_then(Value::as_array)
        .ok_or_else(|| err("XGBoost model has no tree_info"))?;
    if groups.len() != trees.len()
        || weights
            .as_ref()
            .is_some_and(|values| values.len() != trees.len())
    {
        return Err(err("XGBoost tree metadata length mismatch"));
    }
    let n_outputs = base_values.len();
    if n_outputs == 0 {
        return Err(err("output count must be positive"));
    }
    let mut imported = Vec::with_capacity(trees.len());
    for (tree_index, tree) in trees.iter().enumerate() {
        let output = groups[tree_index]
            .as_u64()
            .ok_or_else(|| err("invalid XGBoost tree output group"))? as usize;
        if output >= n_outputs {
            return Err(err("XGBoost tree output group is out of bounds"));
        }
        let left = integer_array(tree, "left_children")?;
        let right = integer_array(tree, "right_children")?;
        let features = unsigned_array(tree, "split_indices")?;
        let conditions = number_array(tree, "split_conditions")?;
        let covers = number_array(tree, "sum_hessian")?;
        let defaults = unsigned_array(tree, "default_left")?;
        let split_types = unsigned_array(tree, "split_type")?;
        let count = left.len();
        if [
            right.len(),
            features.len(),
            conditions.len(),
            covers.len(),
            defaults.len(),
            split_types.len(),
        ]
        .iter()
        .any(|&length| length != count)
        {
            return Err(err(
                "XGBoost columnar tree arrays have inconsistent lengths",
            ));
        }
        let mut nodes = Vec::with_capacity(count);
        for node in 0..count {
            if left[node] < 0 {
                let mut values = vec![0.0; n_outputs];
                values[output] = conditions[node];
                nodes.push(Node::Leaf {
                    values,
                    cover: covers[node],
                });
            } else {
                if split_types[node] != 0 {
                    return Err(ShapError::Unsupported(
                        "categorical XGBoost full-model splits are not yet supported".into(),
                    ));
                }
                nodes.push(Node::NumericalSplit {
                    feature: features[node],
                    threshold: conditions[node],
                    comparison: SplitComparison::LessThan,
                    left: usize::try_from(left[node]).map_err(|_| err("invalid left child"))?,
                    right: usize::try_from(right[node]).map_err(|_| err("invalid right child"))?,
                    missing: if defaults[node] != 0 {
                        MissingBranch::Left
                    } else {
                        MissingBranch::Right
                    },
                    missing_value: MissingValuePolicy::NaN,
                    cover: covers[node],
                });
            }
        }
        imported.push((
            Tree::new(nodes, 0, n_features)?,
            weights.as_ref().map_or(1.0, |values| values[tree_index]),
        ));
    }
    let output_groups = groups
        .iter()
        .map(|group| group.as_u64().map(|value| value as usize))
        .collect();
    TreeEnsemble::new_with_output_groups(imported, base_values, output_groups)
}

fn string_usize(value: &Value, key: &str) -> Result<usize> {
    value
        .get(key)
        .and_then(Value::as_str)
        .and_then(|text| text.parse().ok())
        .ok_or_else(|| err(&format!("invalid XGBoost parameter {key}")))
}
fn integer_array(value: &Value, key: &str) -> Result<Vec<i64>> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| err(&format!("missing XGBoost array {key}")))?
        .iter()
        .map(|item| {
            item.as_i64()
                .ok_or_else(|| err(&format!("invalid XGBoost array {key}")))
        })
        .collect()
}
fn unsigned_array(value: &Value, key: &str) -> Result<Vec<usize>> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| err(&format!("missing XGBoost array {key}")))?
        .iter()
        .map(|item| {
            item.as_u64()
                .and_then(|number| usize::try_from(number).ok())
                .ok_or_else(|| err(&format!("invalid XGBoost array {key}")))
        })
        .collect()
}
fn number_array(value: &Value, key: &str) -> Result<Vec<f64>> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| err(&format!("missing XGBoost array {key}")))?
        .iter()
        .map(|item| value_number(item).ok_or_else(|| err(&format!("invalid XGBoost array {key}"))))
        .collect()
}

pub fn from_lightgbm_json(
    json: &str,
    n_features: usize,
    n_outputs: usize,
    base_values: Vec<f64>,
) -> Result<TreeEnsemble> {
    if n_outputs == 0 {
        return Err(err("output count must be positive"));
    }
    let root: Value = serde_json::from_str(json)
        .map_err(|e| ShapError::InvalidConfiguration(format!("invalid LightGBM JSON: {e}")))?;
    let infos = root
        .get("tree_info")
        .and_then(Value::as_array)
        .ok_or_else(|| err("LightGBM dump has no tree_info"))?;
    let mut trees = Vec::with_capacity(infos.len());
    for (i, info) in infos.iter().enumerate() {
        let structure = info
            .get("tree_structure")
            .ok_or_else(|| err("LightGBM tree has no structure"))?;
        let mut nodes = Vec::new();
        lgb_node(structure, &mut nodes, n_features, n_outputs, i % n_outputs)?;
        // LightGBM's dumped leaf values are already learning-rate adjusted.
        trees.push((Tree::new(nodes, 0, n_features)?, 1.0));
    }
    let output_groups = (0..trees.len())
        .map(|tree| Some(tree % n_outputs))
        .collect();
    TreeEnsemble::new_with_output_groups(trees, base_values, output_groups)
}
fn lgb_node(
    v: &Value,
    nodes: &mut Vec<Node>,
    nf: usize,
    no: usize,
    output: usize,
) -> Result<usize> {
    let index = nodes.len();
    nodes.push(Node::Leaf {
        values: vec![0.; no],
        cover: 0.,
    });
    if let Some(leaf) = v.get("leaf_value").and_then(Value::as_f64) {
        let mut values = vec![0.; no];
        values[output] = leaf;
        // LightGBM's TreeSHAP loader uses observation counts for path
        // probabilities. Hessian weights differ for classification objectives.
        let cover = v
            .get("leaf_count")
            .or_else(|| v.get("leaf_weight"))
            .and_then(value_number)
            .unwrap_or(1.0);
        nodes[index] = Node::Leaf { values, cover };
        return Ok(index);
    }
    let feature = v
        .get("split_feature")
        .and_then(Value::as_u64)
        .ok_or_else(|| err("LightGBM split feature missing"))? as usize;
    if feature >= nf {
        return Err(err("LightGBM feature index is out of bounds"));
    }
    let decision_type = v
        .get("decision_type")
        .and_then(Value::as_str)
        .unwrap_or("<=");
    let left = lgb_node(
        v.get("left_child")
            .ok_or_else(|| err("LightGBM left child missing"))?,
        nodes,
        nf,
        no,
        output,
    )?;
    let right = lgb_node(
        v.get("right_child")
            .ok_or_else(|| err("LightGBM right child missing"))?,
        nodes,
        nf,
        no,
        output,
    )?;
    let cover = v
        .get("internal_count")
        .or_else(|| v.get("internal_weight"))
        .and_then(value_number)
        .unwrap_or(nodes[left].cover() + nodes[right].cover());
    let missing = if v
        .get("default_left")
        .and_then(Value::as_bool)
        .unwrap_or(true)
    {
        MissingBranch::Left
    } else {
        MissingBranch::Right
    };
    let missing_value = match v
        .get("missing_type")
        .and_then(Value::as_str)
        .unwrap_or("NaN")
    {
        "NaN" => MissingValuePolicy::NaN,
        "Zero" => MissingValuePolicy::Zero,
        "None" => MissingValuePolicy::None,
        kind => return Err(err(&format!("unsupported LightGBM missing type {kind}"))),
    };
    nodes[index] = match decision_type {
        "<=" => Node::NumericalSplit {
            feature,
            threshold: num(v, "threshold")?,
            comparison: SplitComparison::LessThanOrEqual,
            left,
            right,
            missing,
            missing_value,
            cover,
        },
        "==" => Node::CategoricalSplit {
            feature,
            categories: categorical_threshold(v)?,
            left,
            right,
            missing,
            missing_value,
            cover,
        },
        kind => {
            return Err(ShapError::Unsupported(format!(
                "LightGBM decision type {kind} is not supported"
            )))
        }
    };
    Ok(index)
}
fn num(v: &Value, key: &str) -> Result<f64> {
    v.get(key)
        .and_then(value_number)
        .ok_or_else(|| err(&format!("numeric field {key} is missing")))
}
fn value_number(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_i64().map(|x| x as f64))
        .or_else(|| v.as_u64().map(|x| x as f64))
}
fn categorical_threshold(v: &Value) -> Result<Vec<i64>> {
    let threshold = v
        .get("threshold")
        .and_then(Value::as_str)
        .ok_or_else(|| err("LightGBM categorical threshold must be a string"))?;
    let categories = threshold
        .split("||")
        .map(|category| {
            category
                .parse::<i64>()
                .map_err(|_| err("invalid LightGBM category"))
        })
        .collect::<Result<Vec<_>>>()?;
    if categories.is_empty() {
        return Err(err("LightGBM categorical threshold is empty"));
    }
    Ok(categories)
}
fn err(s: &str) -> ShapError {
    ShapError::InvalidConfiguration(s.into())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Predict;
    use ndarray::array;
    #[test]
    fn imports_xgboost_dump() {
        let json = r#"[{"nodeid":0,"split":"f0","split_condition":1.5,"yes":1,"no":2,"missing":1,"cover":10.0,"children":[{"nodeid":1,"leaf":2.0,"cover":4.0},{"nodeid":2,"leaf":5.0,"cover":6.0}]}]"#;
        let model = from_xgboost_json(json, 1, 1, vec![0.5]).unwrap();
        let y = model
            .predict(array![[0.], [1.5], [2.], [f64::NAN]].view())
            .unwrap();
        assert_eq!(y, array![[2.5], [5.5], [5.5], [2.5]]);
    }
    #[test]
    fn rejects_inconsistent_xgboost_routes() {
        let json = r#"[{"nodeid":0,"split":"f0","split_condition":1.5,"yes":1,"no":2,"missing":9,"cover":10.0,"children":[{"nodeid":1,"leaf":2.0,"cover":4.0},{"nodeid":2,"leaf":5.0,"cover":6.0}]}]"#;
        assert!(from_xgboost_json(json, 1, 1, vec![0.]).is_err());
        assert!(
            from_xgboost_json_with_tree_weights(json, 1, 1, vec![0.], Some(vec![1.0, 2.0]))
                .is_err()
        );
    }
    #[test]
    fn imports_lightgbm_dump_with_missing_routing_and_outputs() {
        let json = r#"{"tree_info":[{"tree_structure":{"split_feature":0,"threshold":1.0,"decision_type":"<=","default_left":false,"internal_weight":6.0,"left_child":{"leaf_value":2.0,"leaf_weight":2.0},"right_child":{"leaf_value":5.0,"leaf_weight":4.0}}},{"tree_structure":{"leaf_value":7.0,"leaf_count":3}}]}"#;
        let model = from_lightgbm_json(json, 1, 2, vec![0.5, -0.5]).unwrap();
        let y = model
            .predict(array![[0.], [2.], [f64::NAN]].view())
            .unwrap();
        assert_eq!(y, array![[2.5, 6.5], [5.5, 6.5], [5.5, 6.5]]);
    }

    #[test]
    fn imports_lightgbm_categorical_splits_and_zero_missing_rules() {
        let json = r#"{"tree_info":[{"tree_structure":{"split_feature":0,"threshold":"1||2","decision_type":"==","missing_type":"Zero","default_left":true,"left_child":{"leaf_value":3.0,"leaf_count":2},"right_child":{"leaf_value":7.0,"leaf_count":2}}}]}"#;
        let model = from_lightgbm_json(json, 1, 1, vec![0.]).unwrap();
        let prediction = model
            .predict(ndarray::array![[1.], [2.], [3.], [0.], [f64::NAN]].view())
            .unwrap();
        assert_eq!(prediction, ndarray::array![[3.], [3.], [7.], [3.], [3.]]);
    }
}
