---
title: Eyes on Heritage
sdk: gradio
app_file: app.py
---

# Eyes on Heritage - demo app

Satellite change detection at heritage sites. Upload a "before" and "after" image of one spot and see a score for each kind of change: new construction, destruction, paved over, no change.

## Run

Uses [uv](https://docs.astral.sh/uv/) (Python 3.12). The environment lives in `.venv/`.

```bash
uv sync                # create .venv and install everything (once)
uv run python app.py   # opens http://127.0.0.1:7860
```

Optional extras:

```bash
uv sync --extra baseline   # scikit-learn>=1.8 + joblib, for MODEL_BACKEND=baseline
uv sync --extra tifs       # geopandas + rasterio, for make_examples.py --tifs
```

Without uv: `pip install -r requirements.txt && python app.py`.

No network access is needed at runtime. The first `import gradio` can take a while on a cold machine.

## Files

| File | What it is |
|---|---|
| `app.py` | The Gradio app (tabs: Try it, Examples, How good is it) |
| `model.py` | `predict_proba(before_rgb, after_rgb)` and the model backends |
| `metrics.json` | Numbers shown on "How good is it". Edit this file; no code change needed |
| `examples/` | Real patches and saved predictions for the Examples tab |
| `make_examples.py` | Rebuilds `examples/` (from `demo.html` by default, or `--tifs` from the GeoTIFFs) |

## Switching the model

One environment variable, `MODEL_BACKEND`:

```bash
MODEL_BACKEND=dummy    uv run python app.py   # default - a PLACEHOLDER, scores mean nothing
MODEL_BACKEND=baseline uv run python app.py   # v4_best_baseline.joblib (see TODO)
MODEL_BACKEND=my_module:MyModel uv run python app.py   # your own class
```

To add another model (e.g. DINOv3): write a class with `predict_proba(self, before_rgb, after_rgb)` returning a dict with the four class names, and register it in `BACKENDS` in `model.py` or use the `module:ClassName` form. Output order and normalisation are handled by `model.py`.

**TODO - baseline backend:** the code that turns a patch pair into the 287 features is not in this repo (`pipeline_fixed.py::feats` makes 22, not 287). Paste it into `extract_features()` in `model.py`. The joblib was saved with scikit-learn 1.8, so run `uv sync --extra baseline` first. Until then the app runs on the placeholder and says so on screen.

## Recording the demo (under 1 minute)

1. `uv run python app.py`, make the browser window about 1280x800, zoom 100%.
2. **Try it** (about 25 s): click a sample pair under the inputs, press **What changed?**, point at the result and bars.
3. **Examples** (about 15 s): scroll slowly; point out one green (agrees) and one red (wrong) card.
4. **How good is it** (about 15 s): say the score honestly and that destruction and paving over are still weak.
5. Record with QuickTime (File > New Screen Recording) or OBS.

Note: with the dummy backend the Try it scores are not real. Examples and the metrics come from saved results.

Imagery © Planet Labs PBC, CC BY-NC 4.0
