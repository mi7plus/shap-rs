"""Regenerate deterministic compatibility fixtures with Python SHAP 0.46.0."""
from pathlib import Path
import json

import numpy as np
import shap


BACKGROUND = np.array([[0.0, 1.0, -1.0], [2.0, -1.0, 3.0]], dtype=np.float64)
SAMPLES = np.array([[1.0, 2.0, 0.5], [-2.0, 0.0, 4.0]], dtype=np.float64)
COEFFICIENTS = np.array([[2.0, -1.0], [-3.0, 0.5], [0.25, 4.0]], dtype=np.float64)
INTERCEPT = np.array([0.75, -2.0], dtype=np.float64)


def model(x):
    return np.asarray(x) @ COEFFICIENTS + INTERCEPT


def probability_model(x):
    margin = np.asarray(x) @ np.array([0.7, -0.4, 0.2]) + 0.1
    return (1.0 / (1.0 + np.exp(-margin)))[:, None]


def explanation_payload(explanation):
    values = np.asarray(explanation.values)
    bases = np.asarray(explanation.base_values)
    if values.ndim == 2:
        values = values[:, :, None]
    if bases.ndim == 1:
        bases = bases[:, None]
    return {
        "values": values.tolist(),
        "base_values": bases.tolist(),
    }


def main():
    exact = shap.ExactExplainer(model, BACKGROUND)(SAMPLES)
    permutation = shap.PermutationExplainer(model, BACKGROUND, seed=7)(
        SAMPLES, max_evals=301
    )
    kernel = shap.KernelExplainer(model, BACKGROUND).shap_values(
        SAMPLES, nsamples=256, silent=True
    )
    linear = shap.LinearExplainer((COEFFICIENTS.T, INTERCEPT), BACKGROUND)(SAMPLES)
    clustering = np.array([[0.0, 1.0, 0.1, 2.0], [3.0, 2.0, 1.0, 3.0]])
    partition = shap.PartitionExplainer(
        model, shap.maskers.Partition(BACKGROUND, clustering=clustering)
    )(SAMPLES)
    logit = shap.ExactExplainer(
        probability_model, BACKGROUND, link=shap.links.logit, linearize_link=False
    )(SAMPLES)
    payload = {
        "generator": {"python_shap": shap.__version__, "numpy": np.__version__},
        "background": BACKGROUND.tolist(),
        "samples": SAMPLES.tolist(),
        "coefficients": COEFFICIENTS.tolist(),
        "intercept": INTERCEPT.tolist(),
        "exact": explanation_payload(exact),
        "permutation": explanation_payload(permutation),
        "kernel": {
            "values": np.asarray(kernel).tolist(),
            "base_values": np.broadcast_to(
                np.asarray(shap.KernelExplainer(model, BACKGROUND).expected_value),
                (len(SAMPLES), len(INTERCEPT)),
            ).tolist(),
        },
        "linear": explanation_payload(linear),
        "partition": explanation_payload(partition),
        "exact_logit": explanation_payload(logit),
    }
    destination = Path(__file__).parents[1] / "tests" / "fixtures" / "python_shap_linear.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
