---
title: Eyes on Heritage
sdk: gradio
app_file: app.py
---

# Eyes on Heritage - demo app

Satellite change detection at heritage sites. Upload a "before" and "after" image of one spot and see a score for each kind of change: new construction, destruction, paved over, no change.

## Launch the app

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                # once: creates .venv and installs everything
uv run python app.py   # then open http://127.0.0.1:7860
```

No network is needed at runtime. The first `import gradio` can be slow on a cold machine. If port 7860 is busy (an old copy still running), stop it first: `pkill -f "python app.py"`.

The app loads the trained model from `holdout/model.joblib`. If it shows a yellow **Preview mode** banner, the placeholder is running instead - check that you started it with no `MODEL_BACKEND` set and that the file exists.

## Check that it works

```bash
uv run python test_app.py     # runs the 12 ready pictures in examples/ through the model
```

## Retrain (optional)

The model was trained on 80% of the data and tested on a held-out 20% (split by 250 m map cell, so neighbouring points never end up on both sides). Needs the GeoTIFFs in `Data/`.

```bash
uv sync --extra tifs                          # geopandas + rasterio, once
uv run python train_holdout.py                # hold out 20% of map cells
uv run python train_holdout.py --city derna   # or hold out a whole city
```

This rewrites `holdout/model.joblib`, `holdout/ref_stats.json`, `holdout/metrics.json` and `holdout/test_predictions.csv`. Copy `holdout/metrics.json` over `metrics.json` to update the "How good is it" tab. Current held-out score: macro-F1 0.614, accuracy 0.703 on 155 test points (destruction is weak - only 12 test points).

## Files

| File | What it is |
|---|---|
| `app.py` | The Gradio app (tabs: Try it, Examples, How good is it) |
| `model.py` | `predict_proba(before_rgb, after_rgb)` and the model backends |
| `v4_model.py` | Feature extraction + `Ensemble` class, needed to load the model without geopandas/rasterio |
| `holdout/` | Trained model, feature scaling for uploads, held-out metrics and predictions |
| `Eyes_on_Heritage_v4/pipeline_v4.py` | Original data loading and feature pipeline, reused by `train_holdout.py` |
| `train_holdout.py` | Retrains with a train/test split |
| `test_app.py` | Smoke test using `examples/` |
| `metrics.json` | Numbers shown on "How good is it". Edit this file; no code change needed |
| `examples/` | Real patches for the Examples tab and for `test_app.py` |
| `Data/` | GeoTIFFs and annotations (not in git) |

## Switching the model

One environment variable, `MODEL_BACKEND`:

```bash
MODEL_BACKEND=v4    uv run python app.py   # default - the trained model
MODEL_BACKEND=dummy uv run python app.py   # PLACEHOLDER, scores mean nothing
MODEL_BACKEND=my_module:MyModel uv run python app.py   # your own class
```

A custom class needs `predict_proba(self, before_rgb, after_rgb)` returning a dict with the four class names. Output order and normalisation are handled by `model.py`.

## Caveat for uploads

Training scaled features per city; an uploaded pair has no city, so `holdout/ref_stats.json` (pooled median/IQR of the training features) is used instead, and each 64 px patch is normalised on its own. Expect uploads to score less reliably than the held-out numbers.

Imagery © Planet Labs PBC, CC BY-NC 4.0
