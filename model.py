"""Eyes on Heritage - model backends.

The app only ever calls ONE function:

    predict_proba(before_rgb, after_rgb) -> {"construction": p, "destruction": p,
                                              "soil_sealing": p, "no_change": p}

Inputs are uint8 numpy arrays (H, W, 3); the app passes 64x64 patches (32 m).
Scores are floats that sum to 1, always in CLASS_ORDER, whatever order the
underlying model uses internally.

Switching backend (one line):

    MODEL_BACKEND=dummy      python app.py   # default: placeholder, NOT a real model
    MODEL_BACKEND=baseline   python app.py   # v4_best_baseline.joblib (needs TODO below)
    MODEL_BACKEND=my_pkg.my_module:MyModel python app.py   # any class you write

To plug in another model (e.g. DINOv3): write a class with
`predict_proba(self, before_rgb, after_rgb) -> dict[str, float]` and either add
it to BACKENDS below or point MODEL_BACKEND at "module:ClassName". The class is
created with no arguments.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np

CLASS_ORDER = ["construction", "destruction", "soil_sealing", "no_change"]
HERE = Path(__file__).resolve().parent


def _to_ordered_dict(scores: dict) -> dict:
    """Reorder to CLASS_ORDER, clip negatives, renormalise to sum to 1."""
    missing = [c for c in CLASS_ORDER if c not in scores]
    if missing:
        raise ValueError(f"model output is missing classes: {missing}")
    v = np.clip(np.array([float(scores[c]) for c in CLASS_ORDER]), 0.0, None)
    total = v.sum()
    v = v / total if total > 0 else np.full(len(CLASS_ORDER), 1 / len(CLASS_ORDER))
    return {c: float(p) for c, p in zip(CLASS_ORDER, v)}


# ---------------------------------------------------------------- backends
class DummyModel:
    """PLACEHOLDER - NOT A REAL MODEL. Its scores mean nothing scientifically.

    A crude, deterministic rule based on how much the two patches differ, so
    the interface can be demonstrated and recorded without the trained model.
    """
    name = "Placeholder (dummy) model"
    is_placeholder = True

    def predict_proba(self, before_rgb, after_rgb):
        b = before_rgb.astype("float32") / 255.0
        a = after_rgb.astype("float32") / 255.0
        diff = float(np.abs(a - b).mean())            # how much changed, 0..1
        bright = float(a.mean() - b.mean())           # got brighter (+) / darker (-)
        grey = float((a.std(axis=2).mean() - b.std(axis=2).mean()))  # lost colour (-)
        logits = {
            "no_change": 2.0 - 18.0 * diff,
            "construction": 6.0 * diff + 4.0 * max(bright, 0) - 0.5,
            "destruction": 6.0 * diff + 4.0 * max(-bright, 0) - 1.0,
            "soil_sealing": 5.0 * diff - 8.0 * grey - 1.0,
        }
        z = np.array([logits[c] for c in CLASS_ORDER])
        e = np.exp(z - z.max())
        return _to_ordered_dict(dict(zip(CLASS_ORDER, e / e.sum())))


def extract_features(before_rgb, after_rgb):
    """TODO: turn a (before, after) patch pair into the 287 features the model expects.

    NOT IMPLEMENTED on purpose. The code that produced the 287 features of
    v4_best_baseline.joblib is not in this repo. pipeline_fixed.py::feats gives
    only 22 hand-made features (1558 with USE_CNN), so it does not match.
    Paste the original extractor here; it must return a 1-D array of length 287,
    in the same order as at training time.
    """
    raise NotImplementedError(
        "Feature extraction for v4_best_baseline.joblib is missing (TODO in model.py). "
        "Use MODEL_BACKEND=dummy until it is added.")


class BaselineModel:
    """StandardScaler + LogisticRegression saved in v4_best_baseline.joblib."""
    name = "Baseline (logistic regression)"
    is_placeholder = False

    def __init__(self, path: Path = HERE / "v4_best_baseline.joblib"):
        import joblib
        self.pipe = joblib.load(path)

    def predict_proba(self, before_rgb, after_rgb):
        x = np.asarray(extract_features(before_rgb, after_rgb), dtype="float64").reshape(1, -1)
        p = self.pipe.predict_proba(x)[0]
        # use the model's own class order, then reorder to CLASS_ORDER
        return _to_ordered_dict(dict(zip(self.pipe.classes_, p)))


BACKENDS = {"dummy": DummyModel, "baseline": BaselineModel}

# ------------------------------------------------------------------ public
_model = None


def get_model():
    """Create the backend named by MODEL_BACKEND once, then reuse it."""
    global _model
    if _model is None:
        spec = os.environ.get("MODEL_BACKEND", "dummy").strip()
        if spec in BACKENDS:
            cls = BACKENDS[spec]
        elif ":" in spec:
            mod, attr = spec.split(":", 1)
            cls = getattr(importlib.import_module(mod), attr)
        else:
            raise ValueError(f"Unknown MODEL_BACKEND '{spec}'. Use one of {list(BACKENDS)} "
                             "or 'module:ClassName'.")
        _model = cls()
    return _model


def predict_proba(before_rgb: np.ndarray, after_rgb: np.ndarray) -> dict:
    """Scores for construction, destruction, soil_sealing, no_change (sum to 1)."""
    return _to_ordered_dict(get_model().predict_proba(before_rgb, after_rgb))


def backend_info() -> dict:
    m = get_model()
    return {"name": getattr(m, "name", type(m).__name__),
            "is_placeholder": bool(getattr(m, "is_placeholder", False))}
