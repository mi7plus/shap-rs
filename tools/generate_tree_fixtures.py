"""Regenerate real trained-tree fixtures with Python SHAP 0.46.0."""
from pathlib import Path
import json
import tempfile

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import xgboost as xgb


TRAIN_X = np.array(
    [
        [-2.0, 0.0, 1.0],
        [-1.0, 2.0, np.nan],
        [0.0, -1.0, 2.0],
        [0.5, 1.5, 0.0],
        [1.0, np.nan, -2.0],
        [2.0, 1.0, 3.0],
        [3.0, -2.0, 1.0],
        [4.0, 0.5, np.nan],
    ],
    dtype=np.float64,
)
TRAIN_Y = np.array([-3.0, -0.5, 1.0, 0.25, 4.0, 3.5, 7.0, 8.5])
SAMPLES = np.array(
    [[-1.5, 0.25, 1.0], [1.5, np.nan, -1.0], [3.5, -1.0, np.nan]],
    dtype=np.float64,
)


def payload(model_dump, prediction, explanation, model_base):
    prediction = np.asarray(prediction)
    values = np.asarray(explanation.values)
    bases = np.asarray(explanation.base_values)
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    if values.ndim == 2:
        values = values[:, :, None]
    if bases.ndim == 1:
        bases = bases[:, None]
    return {
        "model": model_dump,
        "model_base": model_base,
        "samples": [[None if np.isnan(value) else value for value in row] for row in SAMPLES],
        "prediction": prediction.tolist(),
        "values": values.tolist(),
        "base_values": bases.tolist(),
    }


def main():
    xgb_model = xgb.XGBRegressor(
        n_estimators=5,
        max_depth=2,
        learning_rate=0.3,
        base_score=0.25,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=11,
        n_jobs=1,
    ).fit(TRAIN_X, TRAIN_Y)
    xgb_explanation = shap.TreeExplainer(xgb_model)(SAMPLES)
    xgb_dump = [
        json.loads(tree)
        for tree in xgb_model.get_booster().get_dump(dump_format="json", with_stats=True)
    ]

    lgb_model = lgb.LGBMRegressor(
        n_estimators=5,
        max_depth=2,
        num_leaves=4,
        min_child_samples=1,
        learning_rate=0.3,
        random_state=11,
        verbosity=-1,
        n_jobs=1,
    ).fit(TRAIN_X, TRAIN_Y)
    lgb_explanation = shap.TreeExplainer(lgb_model)(SAMPLES)

    fixture = {
        "generator": {
            "python_shap": shap.__version__,
            "xgboost": xgb.__version__,
            "lightgbm": lgb.__version__,
        },
        "xgboost_regression": payload(
            xgb_dump, xgb_model.predict(SAMPLES), xgb_explanation, [0.25]
        ),
        # LightGBM includes its initial score in the dumped tree leaf values.
        "lightgbm_regression": payload(
            lgb_model.booster_.dump_model(),
            lgb_model.predict(SAMPLES),
            lgb_explanation,
            [0.0],
        ),
    }
    fixture["xgboost_regression"]["interactions"] = np.asarray(
        shap.TreeExplainer(xgb_model).shap_interaction_values(SAMPLES)
    )[:, :, :, None].tolist()
    base_margin = np.array([1.0, -0.5, 2.5], dtype=np.float64)
    margin_matrix = xgb.DMatrix(SAMPLES)
    margin_matrix.set_base_margin(base_margin)
    fixture["xgboost_regression"]["base_margin"] = base_margin[:, None].tolist()
    fixture["xgboost_regression"]["base_margin_prediction"] = (
        xgb_model.get_booster().predict(margin_matrix, output_margin=True)[:, None].tolist()
    )

    binary_y = (TRAIN_Y > 2.0).astype(np.int32)
    xgb_binary = xgb.XGBClassifier(
        n_estimators=5, max_depth=2, learning_rate=0.3, base_score=0.35,
        objective="binary:logistic", tree_method="hist", random_state=13, n_jobs=1,
    ).fit(TRAIN_X, binary_y)
    fixture["xgboost_binary"] = payload(
        [json.loads(tree) for tree in xgb_binary.get_booster().get_dump(dump_format="json", with_stats=True)],
        xgb_binary.get_booster().predict(xgb.DMatrix(SAMPLES), output_margin=True),
        shap.TreeExplainer(xgb_binary)(SAMPLES),
        [float(np.log(0.35 / 0.65))],
    )

    multiclass_y = np.digitize(TRAIN_Y, [-0.75, 4.5]).astype(np.int32)
    xgb_multi = xgb.XGBClassifier(
        n_estimators=4, max_depth=2, learning_rate=0.3, base_score=0.5,
        objective="multi:softprob", num_class=3, tree_method="hist", random_state=17, n_jobs=1,
    ).fit(TRAIN_X, multiclass_y)
    fixture["xgboost_multiclass"] = payload(
        [json.loads(tree) for tree in xgb_multi.get_booster().get_dump(dump_format="json", with_stats=True)],
        xgb_multi.get_booster().predict(xgb.DMatrix(SAMPLES), output_margin=True),
        shap.TreeExplainer(xgb_multi)(SAMPLES),
        [0.5, 0.5, 0.5],
    )
    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory) / "multiclass.json"
        xgb_multi.save_model(model_path)
        fixture["xgboost_multiclass"]["full_model"] = json.loads(
            model_path.read_text(encoding="utf-8")
        )

    lgb_binary = lgb.LGBMClassifier(
        n_estimators=5, max_depth=2, num_leaves=4, min_child_samples=1,
        learning_rate=0.3, random_state=13, verbosity=-1, n_jobs=1,
    ).fit(TRAIN_X, binary_y)
    fixture["lightgbm_binary"] = payload(
        lgb_binary.booster_.dump_model(),
        lgb_binary.predict(SAMPLES, raw_score=True),
        shap.TreeExplainer(lgb_binary)(SAMPLES),
        [0.0],
    )

    lgb_multi = lgb.LGBMClassifier(
        n_estimators=4, max_depth=2, num_leaves=4, min_child_samples=1,
        learning_rate=0.3, random_state=17, verbosity=-1, n_jobs=1,
    ).fit(TRAIN_X, multiclass_y)
    fixture["lightgbm_multiclass"] = payload(
        lgb_multi.booster_.dump_model(),
        lgb_multi.predict(SAMPLES, raw_score=True),
        shap.TreeExplainer(lgb_multi)(SAMPLES),
        [0.0, 0.0, 0.0],
    )

    dart = xgb.XGBRegressor(
        booster="dart", n_estimators=8, max_depth=2, learning_rate=0.3,
        base_score=0.25, objective="reg:squarederror", rate_drop=0.5,
        skip_drop=0.0, random_state=23, n_jobs=1,
    ).fit(TRAIN_X, TRAIN_Y)
    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory) / "dart.json"
        dart.save_model(model_path)
        full_model = json.loads(model_path.read_text(encoding="utf-8"))
    fixture["xgboost_dart_weights"] = {
        "full_model": full_model,
        "model": [json.loads(tree) for tree in dart.get_booster().get_dump(dump_format="json", with_stats=True)],
        "tree_weights": full_model["learner"]["gradient_booster"]["weight_drop"],
        "model_base": [0.25],
        "samples": [[None if np.isnan(value) else value for value in row] for row in SAMPLES],
        "prediction": dart.get_booster().predict(xgb.DMatrix(SAMPLES), output_margin=True)[:, None].tolist(),
    }

    categorical_x = np.array([
        [0, -2.0], [0, 1.0], [1, -1.0], [1, 2.0],
        [2, -2.0], [2, 1.5], [3, -1.5], [3, 2.5],
        [0, 0.0], [1, 0.5], [2, 0.0], [3, 0.5],
    ], dtype=np.float64)
    categorical_y = np.array([0, 0.5, 4, 4.5, -3, -2.5, 8, 8.5, 0.2, 4.2, -2.8, 8.2])
    categorical_samples = np.array([[0, 0.25], [2, -0.5], [3, 1.0], [np.nan, 0.0]])
    categorical_frame = pd.DataFrame({
        "category": pd.Categorical(categorical_x[:, 0].astype(int)),
        "numeric": categorical_x[:, 1],
    })
    categorical_sample_frame = pd.DataFrame({
        "category": pd.Categorical(categorical_samples[:, 0], categories=[0, 1, 2, 3]),
        "numeric": categorical_samples[:, 1],
    })
    categorical_model = lgb.LGBMRegressor(
        n_estimators=5, num_leaves=4, min_child_samples=1, learning_rate=0.3,
        min_data_per_group=1, cat_smooth=0.0, max_cat_to_onehot=4,
        random_state=29, verbosity=-1, n_jobs=1,
    ).fit(categorical_frame, categorical_y)
    categorical_explanation = shap.TreeExplainer(categorical_model)(categorical_sample_frame)
    categorical_values = np.asarray(categorical_explanation.values)
    categorical_bases = np.asarray(categorical_explanation.base_values)
    fixture["lightgbm_categorical"] = {
        "model": categorical_model.booster_.dump_model(),
        "model_base": [0.0],
        "samples": [[None if np.isnan(value) else value for value in row] for row in categorical_samples],
        "prediction": categorical_model.predict(categorical_sample_frame)[:, None].tolist(),
        "values": categorical_values[:, :, None].tolist(),
        "base_values": categorical_bases[:, None].tolist(),
    }
    destination = Path(__file__).parents[1] / "tests" / "fixtures" / "trained_trees.json"
    destination.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
