"""Smoke test for the app using the ready-made pictures in examples/.

    uv run python test_app.py                # model on all examples + app functions
    uv run python test_app.py --min-acc 0.5  # also fail if accuracy is below 50%

Checks: the backend is the real model (not the placeholder), every example gives four scores
summing to 1, the app's own what_changed() runs on each pair, and prints the true label vs
the label the CURRENT model predicts (the 'pred_label' saved in examples/index.json came from
the older model and is shown for comparison only). Exit code 1 on any failure.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--min-acc", type=float, default=0.0)
args = ap.parse_args()

import model
import app

fails = []
info = model.backend_info()
print("backend:", info)
if info["is_placeholder"]:
    fails.append("backend is the placeholder, not a trained model")

items = json.loads((HERE / "examples" / "index.json").read_text())
hits = 0
print(f"\n{'#':>2} {'site':<10} {'true':<13} {'now':<13} {'conf':>5}  {'saved':<13}")
for i, it in enumerate(items, 1):
    b = np.array(Image.open(HERE / "examples" / it["before"]).convert("RGB"))
    a = np.array(Image.open(HERE / "examples" / it["after"]).convert("RGB"))
    try:
        p = model.predict_proba(b, a)
        assert set(p) == {"construction", "destruction", "soil_sealing", "no_change"}, p
        assert abs(sum(p.values()) - 1) < 1e-6 and all(0 <= v <= 1 for v in p.values()), p
        app.what_changed(b, a)                      # the function behind the "Try it" tab
    except Exception as e:
        fails.append(f"example {i}: {type(e).__name__}: {e}")
        continue
    pred = max(p, key=p.get)
    hits += pred == it["true_label"]
    print(f"{i:>2} {it['site']:<10} {it['true_label']:<13} {pred:<13} {p[pred]:>5.2f}  {it['pred_label']:<13}")

acc = hits / len(items)
print(f"\ncurrent model matches the true label on {hits}/{len(items)} examples ({acc:.0%})")
print("note: examples may overlap the training cells, so this is a smoke test, not an evaluation;"
      " use holdout/metrics.json for honest scores")
if acc < args.min_acc:
    fails.append(f"accuracy {acc:.2f} < --min-acc {args.min_acc}")
print("\nFAILED:\n  " + "\n  ".join(fails) if fails else "\nOK")
sys.exit(1 if fails else 0)
