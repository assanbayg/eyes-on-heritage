"""Eyes on Heritage - model backends.

The app only ever calls ONE function:

    predict_proba(before_rgb, after_rgb) -> {"construction": p, "destruction": p,
                                              "soil_sealing": p, "no_change": p}

Inputs are uint8 numpy arrays (H, W, 3); the app passes 64x64 patches (32 m).
Scores are floats that sum to 1, always in CLASS_ORDER, whatever order the
underlying model uses internally.

Switching backend (one line):

    MODEL_BACKEND=v4         python app.py   # default: trained model (holdout/model.joblib)
    MODEL_BACKEND=dummy      python app.py   # placeholder, NOT a real model
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


class V4Model:
    """Trained v4 ensemble (RF + ExtraTrees + HistGB, two-stage) from Eyes_on_Heritage_v4/.

    Mirrors pipeline_v4.py: colour-match the after patch to the before patch, z-score each
    band, compute the 56 `feats_new` features, then scale them.

    Two differences from training, both forced by a single uploaded pair:
      * the 64 px patch is normalised on its own (training used a 192 px context window);
      * training scaled features per city (median / IQR of that city's labelled points). An
        upload has no city, so `v4_ref_stats.json` holds one pooled median / IQR instead.
    Expect somewhat less reliable scores than the cross-validated numbers in metrics.json.
    """
    name = "Trained v4 ensemble"
    is_placeholder = False

    def __init__(self, path: Path = HERE / "holdout" / "model.joblib",
                 stats: Path = HERE / "holdout" / "ref_stats.json"):
        import __main__
        import joblib
        import json
        import v4_model
        # the model was pickled from a script run as __main__, so Ensemble must live there
        if not hasattr(__main__, "Ensemble"):
            __main__.Ensemble = v4_model.Ensemble
        self.clf = joblib.load(path)
        self.feats = v4_model.feats_new
        ref = json.loads(stats.read_text())
        self.med, self.scale = np.array(ref["median"]), np.array(ref["scale"])

    @staticmethod
    def prepare(before_rgb, after_rgb):
        """uint8 (H, W, 3) pair -> two (3, H, W) z-scored arrays, as pipeline_v4.prep_pair."""
        from skimage.exposure import match_histograms
        b = before_rgb.astype("float32").transpose(2, 0, 1)
        a = after_rgb.astype("float32").transpose(2, 0, 1)
        a = np.stack([match_histograms(a[k], b[k]) for k in range(3)]).astype("float32")
        z = lambda x: np.stack([(x[k] - x[k].mean()) / (x[k].std() + 1e-6) for k in range(3)])
        return z(b), z(a)

    def features(self, before_rgb, after_rgb):
        b, a = self.prepare(before_rgb, after_rgb)
        return np.nan_to_num(np.array(list(self.feats(b, a).values()), dtype="float32"))

    def predict_proba(self, before_rgb, after_rgb):
        x = (self.features(before_rgb, after_rgb) - self.med) / self.scale
        p = self.clf.predict_proba(x.reshape(1, -1))[0]
        return _to_ordered_dict(dict(zip(self.clf.classes_, p)))


BACKENDS = {"v4": V4Model, "dummy": DummyModel}

# ------------------------------------------------------------------ public
_model = None


def get_model():
    """Create the backend named by MODEL_BACKEND once, then reuse it."""
    global _model
    if _model is None:
        spec = os.environ.get("MODEL_BACKEND", "v4").strip()
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
