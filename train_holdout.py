"""Retrain the v4 model with a held-out test set, and report honest scores on it.

    uv run python train_holdout.py                # 20% of 250 m map cells held out (default)
    uv run python train_holdout.py --city derna   # hold out a whole city instead

Why: Eyes_on_Heritage_v4/final_model.joblib was fitted on 100% of the points, so it has no test set.
Here the held-out points are never used for training, scaling or weight tuning. Held out by
250 m map cell (not by point), so neighbouring points can't leak between train and test.

Writes holdout/model.joblib, holdout/ref_stats.json (training-set scaling for single uploads),
holdout/metrics.json (app-style numbers) and holdout/test_predictions.csv.
Reuses pipeline_v4.py unchanged; needs `uv sync --extra tifs`. Patches are cached in holdout/cache/.
"""
import argparse, json, os, sys, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "Eyes_on_Heritage_v4"))
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--city", help="hold out this whole city (agadez, aleppo, derna, marrakesh, timbuktu)")
ap.add_argument("--frac", type=float, default=0.2, help="share of map cells held out")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

OUT = HERE / "holdout"
(OUT / "cache").mkdir(parents=True, exist_ok=True)
os.chdir(OUT / "cache")                      # pipeline writes its caches to the cwd

import pipeline_v4 as P
import v4_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

P.ROOT, P.ANN = HERE / "Data", HERE / "Data" / "train_annotations.geojson"

B, A, pairs = P.load_data()
y, sites, cells = pairs.cls.values, pairs.site.values, P.cell_ids(pairs)
pm = P.primary_mask(pairs)

rng = np.random.default_rng(args.seed)
if args.city:
    test = sites == args.city
else:
    uc = np.unique(cells)
    test = np.isin(cells, rng.choice(uc, int(len(uc) * args.frac), replace=False))
train = ~test
print(f"train {train.sum()} / test {(test & pm).sum()} points; held out: {args.city or 'random 250 m cells'}")
print("test class counts:", dict(zip(*np.unique(y[test & pm], return_counts=True))))

X = P.make_X(B, A, sites)                    # per-site scaling uses each site's own points
sw = P.train_weights(pairs)
clf = P.model().fit(X[train], y[train], groups=sites[train], sw=sw[train])
te = test & pm
proba = clf.predict_proba(X[te])
pred = np.array(P.CLASSES)[proba.argmax(1)]
yt = y[te]

mf1 = f1_score(yt, pred, labels=P.CLASSES, average="macro", zero_division=0)
print(f"\nHELD-OUT macro-F1 {mf1:.3f} | accuracy {accuracy_score(yt, pred):.3f} (n={te.sum()})")
print(classification_report(yt, pred, labels=P.CLASSES, zero_division=0))
cm = confusion_matrix(yt, pred, labels=P.CLASSES)
print(cm)

# --- save for the app: re-class so the pickle does not depend on pipeline_v4 / __main__
clf.__class__ = v4_model.Ensemble
import joblib
joblib.dump(clf, OUT / "model.joblib")

# scaling for a single upload (no city known): pooled median/scale of TRAINING features
Xr = P.raw_feats(B, A)[train]
q75, q25 = np.percentile(Xr, [75, 25], 0)
json.dump({"median": np.median(Xr, 0).tolist(),
           "scale": np.maximum(q75 - q25, 0.25 * Xr.std(0) + 1e-6).tolist()},
          open(OUT / "ref_stats.json", "w"))

f1c = f1_score(yt, pred, labels=P.CLASSES, average=None, zero_division=0)
json.dump({"protocol": f"Held out {'the city ' + args.city if args.city else '20% of 250 m map cells'}; "
                       "the model never saw these points in training.",
           "macro_f1": round(mf1, 3), "per_class_f1": {c: round(float(v), 2) for c, v in zip(P.CLASSES, f1c)},
           "n_test_points": int(te.sum()),
           "confusion": {"labels": P.CLASSES, "matrix": cm.tolist()}},
          open(OUT / "metrics.json", "w"), indent=2)
out = pairs[te][["idx", "site", "lon", "lat", "before_year", "after_year"]].copy()
out["true"], out["pred"], out["confidence"] = yt, pred, proba.max(1).round(3)
out.to_csv(OUT / "test_predictions.csv", index=False)
print("\nwrote holdout/ (model.joblib, ref_stats.json, metrics.json, test_predictions.csv)")
