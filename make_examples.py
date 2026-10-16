"""Create the examples/ folder used by the "Examples" tab.

    python make_examples.py                 # from demo.html (default, no GeoTIFFs needed)
    python make_examples.py --tifs          # from pairs_clean.pkl + GeoTIFFs + the model

Output: examples/NN_before.png, examples/NN_after.png, examples/index.json

Mode 1 (demo.html) reuses the real patches and the saved predictions that the
notebook exported earlier. The model labels/confidences are those saved
predictions, not something recomputed here.

Mode 2 (--tifs) is UNTESTED on this machine (the GeoTIFFs are not here). It cuts
64x64 patches around each annotated point exactly like main.ipynb does and asks
model.predict_proba for the scores, so it only gives real numbers once the real
model backend works (MODEL_BACKEND=baseline).
"""
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / "examples"
SIZE = 256            # saved PNG size (patches are 64x64 px = 32 m; shown enlarged)

# how many (correct, wrong) examples per class -> 12 in total
PLAN = {"construction": (2, 1), "destruction": (1, 2),
        "soil_sealing": (2, 1), "no_change": (1, 2)}


def pick(records):
    """Choose ~12 records: all 4 classes, a mix of right and wrong predictions."""
    chosen = []
    for cls, (n_ok, n_bad) in PLAN.items():
        ok = [r for r in records if r["true_label"] == cls and r["pred_label"] == cls]
        bad = [r for r in records if r["true_label"] == cls and r["pred_label"] != cls]
        chosen += ok[:n_ok] + bad[:n_bad]
    return chosen


def save_png(arr_or_img, path):
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    img.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC).save(path)


def from_demo_html(path):
    text = path.read_text(encoding="utf-8")
    start = text.index("const DATA = ") + len("const DATA = ")
    data, _ = json.JSONDecoder().raw_decode(text[start:])
    records = []
    for it in data["items"]:
        records.append(dict(
            site=it["site"], true_label=it["truth"], pred_label=it["pred"],
            confidence=it["conf"], proba=it["proba"],
            before_year=it["before_year"], after_year=it["after_year"],
            _before=Image.open(io.BytesIO(base64.b64decode(it["before"].split(",", 1)[1]))),
            _after=Image.open(io.BytesIO(base64.b64decode(it["after"].split(",", 1)[1])))))
    return records


def from_tifs():
    import geopandas as gpd
    import pandas as pd
    import rasterio
    from rasterio.warp import transform
    from rasterio.windows import Window

    import model

    pairs = pd.read_pickle(HERE / "pairs_clean.pkl")
    gdf = gpd.read_file(HERE / "train_annotations.geojson")
    half = 32

    def patch(tif, lon, lat):
        with rasterio.open(tif) as src:
            xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
            r, c = src.index(xs[0], ys[0])
            w = Window(c - half, r - half, 2 * half, 2 * half)
            rgb = src.read([1, 2, 3], window=w, boundless=True, fill_value=0)
        rgb = rgb.transpose(1, 2, 0).astype("float32")
        lo, hi = np.percentile(rgb, [2, 98])        # display stretch only
        return np.clip((rgb - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype("uint8")

    records = []
    for r in pairs.itertuples():
        g = gdf.loc[r.idx, "geometry"]
        b, a = patch(r.before_tif, g.x, g.y), patch(r.after_tif, g.x, g.y)
        p = model.predict_proba(b, a)
        pred = max(p, key=p.get)
        records.append(dict(site=r.site, true_label=pairs.loc[r.Index, "class"],
                            pred_label=pred, confidence=round(p[pred], 3), proba=p,
                            before_year=int(r.before_year), after_year=int(r.after_year),
                            _before=Image.fromarray(b), _after=Image.fromarray(a)))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tifs", action="store_true", help="build from GeoTIFFs instead of demo.html")
    ap.add_argument("--demo-html", default=str(HERE / "demo.html"))
    args = ap.parse_args()

    records = from_tifs() if args.tifs else from_demo_html(Path(args.demo_html))
    chosen = pick(records)
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*"):
        old.unlink()
    index = []
    for i, r in enumerate(chosen, 1):
        save_png(r.pop("_before"), OUT / f"{i:02d}_before.png")
        save_png(r.pop("_after"), OUT / f"{i:02d}_after.png")
        index.append(dict(r, before=f"{i:02d}_before.png", after=f"{i:02d}_after.png"))
    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    print(f"wrote {len(index)} examples to {OUT}")


if __name__ == "__main__":
    main()
