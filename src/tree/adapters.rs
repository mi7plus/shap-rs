//! JSON model importers. Dump XGBoost with `with_stats=true`; branch covers are
//! required for correct path-dependent expectations.
use super::{MissingBranch, Node, Tree, TreeEnsemble};
use crate::{Result, ShapError};
use serde_json::Value;

pub fn from_xgboost_json(
    json: &str,
    n_features: usize,
    n_outputs: usize,
    base_values: Vec<f64>,
) -> Result<TreeEnsemble> {
    if n_outputs == 0 {
        return Err(err("output count must be positive"));
    }
    let root: Value = serde_json::from_str(json)
        .map_err(|e| ShapError::InvalidConfiguration(format!("invalid XGBoost JSON: {e}")))?;
    let trees = root.as_array().ok_or_else(|| {
        ShapError::InvalidConfiguration("XGBoost dump must be a JSON array".into())
    })?;
    let mut out = Vec::with_capacity(trees.len());
    for (t, v) in trees.iter().enumerate() {
        let output = t % n_outputs;
        let mut nodes = Vec::new();
        xgb_node(v, &mut nodes, n_features, n_outputs, output)?;
        out.push((Tree::new(nodes, 0, n_features)?, 1.0));
    }
    TreeEnsemble::new(out, base_values)
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
    // XGBoost's `yes` branch uses `value < split_condition`, whereas native
    // nodes use `value <= threshold`. Moving to the preceding representable
    // float preserves XGBoost's strict comparison for every finite f64 input.
    let threshold = previous_float(num(v, "split_condition")?)?;
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
    nodes[index] = Node::Split {
        feature,
        threshold,
        left,
        right,
        missing: if missing == yes {
            MissingBranch::Left
        } else {
            MissingBranch::Right
        },
        cover,
    };
    Ok(index)
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
    TreeEnsemble::new(trees, base_values)
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
        let cover = v
            .get("leaf_weight")
            .or_else(|| v.get("leaf_count"))
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
    if v.get("decision_type")
        .and_then(Value::as_str)
        .is_some_and(|kind| kind != "<=")
    {
        return Err(ShapError::Unsupported(
            "LightGBM categorical and non-numeric splits are not supported".into(),
        ));
    }
    let threshold = num(v, "threshold")?;
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
        .get("internal_weight")
        .or_else(|| v.get("internal_count"))
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
    nodes[index] = Node::Split {
        feature,
        threshold,
        left,
        right,
        missing,
        cover,
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
fn previous_float(value: f64) -> Result<f64> {
    if !value.is_finite() {
        return Err(err("XGBoost split condition must be finite"));
    }
    if value == 0.0 {
        return Ok(-f64::from_bits(1));
    }
    let bits = value.to_bits();
    Ok(f64::from_bits(if value > 0.0 {
        bits - 1
    } else {
        bits + 1
    }))
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
    fn rejects_unsupported_lightgbm_categorical_splits() {
        let json = r#"{"tree_info":[{"tree_structure":{"split_feature":0,"threshold":"1||2","decision_type":"==","left_child":{"leaf_value":0.0},"right_child":{"leaf_value":1.0}}}]}"#;
        assert!(matches!(
            from_lightgbm_json(json, 1, 1, vec![0.]),
            Err(ShapError::Unsupported(_))
        ));
    }
}
