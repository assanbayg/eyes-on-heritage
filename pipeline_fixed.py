"""
Eyes on Heritage - cleaned pipeline (replaces main_1.py)

Run:  python pipeline_fixed.py
Needs: geopandas rasterio scikit-learn scikit-image scipy pillow joblib
Optional (USE_CNN=True): torch torchvision

Fixes vs main_1.py
- no hardcoded Windows paths, no Colab "!pip" lines
- no_change pairs sampled from the SAME year distribution as change points
  (removes a possible "which year is it" shortcut)
- RGB kept (not grayscale); each band normalised using a larger context window
- correct skimage GLCM functions (graycomatrix / graycoprops)
- torchvision imported properly, ImageNet normalisation for CNN features
- file handles cached (much faster patch reading)
- adds spatial-block CV (proxy for M_in) next to leave-one-site-out (proxy for M_out)
- adds the final-score proxy S, final training, a predict function for test
  points, and a demo export with readable RGB patches
"""
import base64, io, json, random
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import joblib
from rasterio.windows import Window
from rasterio.warp import transform
from scipy.ndimage import gaussian_filter, sobel
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from PIL import Image

# ----------------------------------------------------------------- config
ROOT = Path("Hackathon Data")                 # folder with the site folders
ANN = Path("train_annotations.geojson")
SITE_DIR = {"agadez": "agadez-niger", "aleppo": "aleppo-syria",
            "derna": "derna-libya", "marrakesh": "marrakesh-morocco",
            "timbuktu": "timbuktu-mali"}
CLASSES = ["construction", "destruction", "soil_sealing", "no_change"]
HALF, CTX, SEED = 32, 96, 0                   # patch 64px, context 192px
CELL_M = 250                                  # spatial-block size (metres)
USE_CNN = False                               # True = add ResNet34 features
C = np.s_[CTX - HALF:CTX + HALF, CTX - HALF:CTX + HALF]   # centre crop


# ------------------------------------------------------------ data access
_handles = {}
def _open(p):
    p = str(p)
    if p not in _handles:
        _handles[p] = rasterio.open(p)
    return _handles[p]

def years_of(site):
    return sorted(int(f.name[:4]) for f in (ROOT / SITE_DIR[site]).glob("*.tif"))

def tif_for(site, year):
    return next((ROOT / SITE_DIR[site]).glob(f"{year}_*.tif"))

def read_patch(tif, lon, lat):
    """(4, 2*CTX, 2*CTX) float32: R,G,B,alpha around a lon/lat point."""
    src = _open(tif)
    xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
    r, c = src.index(xs[0], ys[0])
    w = Window(c - CTX, r - CTX, 2 * CTX, 2 * CTX)
    return src.read([1, 2, 3, 4], window=w, boundless=True,
                    fill_value=0).astype("float32")

def nodata_frac(arr):
    bad = (arr[3][C] == 0) | (arr[:3].sum(0)[C] == 0)
    return bad.mean()

def normalise(arr):
    """Per-band z-score using the whole context window, then crop centre."""
    valid = arr[3] > 0
    out = np.empty((3,) + arr.shape[1:], dtype="float32")
    for k in range(3):
        v = arr[k][valid] if valid.sum() > 100 else arr[k].ravel()
        out[k] = (arr[k] - v.mean()) / (v.std() + 1e-6)
    return out[:, C[0], C[1]]


# ------------------------------------------------------------ build pairs
def build_pairs(gdf):
    rng = random.Random(SEED)
    rows = []
    for site, g in gdf.groupby("site"):
        ys = years_of(site)
        change_years = [int(y) for y in
                        g.loc[g["class"] != "no_change", "year"].dropna()
                        if int(y) in ys and ys.index(int(y)) > 0]
        for i, r in g.iterrows():
            if r["class"] == "no_change":
                after = (rng.choice(change_years) if change_years
                         else ys[rng.randrange(1, len(ys))])
            else:
                if pd.isna(r["year"]) or int(r["year"]) not in ys:
                    continue
                after = int(r["year"])
            k = ys.index(after)
            if k == 0:
                continue
            before = ys[k - 1]
            rows.append(dict(idx=i, site=site, cls=r["class"],
                             before_year=before, after_year=after,
                             before_tif=tif_for(site, before),
                             after_tif=tif_for(site, after),
                             lon=r.geometry.x, lat=r.geometry.y,
                             conflict=r.get("conflict", np.nan),
                             class_uncertain=r.get("class_uncertain", np.nan)))
    return pd.DataFrame(rows)

def build_patches(pairs, max_bad=0.10):
    B, A, keep = [], [], []
    for r in pairs.itertuples():
        rb = read_patch(r.before_tif, r.lon, r.lat)
        ra = read_patch(r.after_tif, r.lon, r.lat)
        keep.append(max(nodata_frac(rb), nodata_frac(ra)) <= max_bad)
        B.append(normalise(rb)); A.append(normalise(ra))
    keep = np.array(keep)
    return np.stack(B)[keep], np.stack(A)[keep], pairs[keep].reset_index(drop=True)


# --------------------------------------------------------------- features
def _glcm(g):
    q = ((np.clip(g, -2.5, 2.5) + 2.5) / 5 * 31).astype(np.uint8)
    m = graycomatrix(q, [1], [0, np.pi / 2], levels=32, symmetric=True, normed=True)
    return graycoprops(m, "contrast").mean(), graycoprops(m, "homogeneity").mean()

def feats(b, a):
    cs, ms = np.s_[24:40, 24:40], np.s_[16:48, 16:48]
    gb, ga = gaussian_filter(b.mean(0), 1.5), gaussian_filter(a.mean(0), 1.5)
    d = ga - gb
    corr = np.nan_to_num(np.corrcoef(gb[cs].ravel(), ga[cs].ravel())[0, 1])
    f = [np.abs(d).mean(), np.abs(d[cs]).mean(), np.abs(d[ms]).mean(),
         d[cs].mean(), gb[cs].std(), ga[cs].std(), corr]
    f += [(a[k][cs] - b[k][cs]).mean() for k in range(3)]       # per-band shift
    f += [np.abs(a[k][cs] - b[k][cs]).mean() for k in range(3)]
    eb = np.hypot(sobel(gb, 0), sobel(gb, 1))[cs].mean()         # edges / structure
    ea = np.hypot(sobel(ga, 0), sobel(ga, 1))[cs].mean()
    f += [eb, ea, ea - eb]
    (cb, hb), (ca, ha) = _glcm(b.mean(0)[cs]), _glcm(a.mean(0)[cs])
    f += [cb, ca, ca - cb, hb, ha, ha - hb]                      # texture
    return f

def cnn_feats(P):
    """Optional ResNet34 (ImageNet) embeddings for (N,3,64,64) normalised patches."""
    import torch, torchvision.models as tvm
    net = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT)
    net.fc = torch.nn.Identity(); net.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(P), 64):
            x = torch.tensor(np.clip((P[i:i + 64] + 2.5) / 5, 0, 1))
            x = torch.nn.functional.interpolate(x, size=128, mode="bilinear")
            out.append(net((x - mean) / std).numpy())
    return np.concatenate(out)

def make_X(B, A):
    X = np.array([feats(b, a) for b, a in zip(B, A)], dtype="float32")
    if USE_CNN:
        fb, fa = cnn_feats(B), cnn_feats(A)
        X = np.hstack([X, fb, fa, np.abs(fa - fb)])
    return np.nan_to_num(X)


# ------------------------------------------------------------- evaluation
def model():
    return RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                  min_samples_leaf=2, random_state=SEED, n_jobs=-1)

def macro_f1(y, p):
    return f1_score(y, p, labels=CLASSES, average="macro", zero_division=0)

def hm(a, b):
    return 2 * a * b / (a + b) if (a + b) > 0 else 0.0

def oof_loso(X, y, sites):
    pred = np.empty(len(y), dtype=object)
    for s in np.unique(sites):
        te = sites == s
        pred[te] = model().fit(X[~te], y[~te]).predict(X[te])
    return pred

def oof_block(X, y, groups, k=5):
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(k).split(X, y, groups):
        pred[te] = model().fit(X[tr], y[tr]).predict(X[te])
    return pred

def cell_ids(df):
    x = df.lon * 111320 * np.cos(np.radians(df.lat))
    yy = df.lat * 110540
    return (df.site + "_" + (x // CELL_M).astype(int).astype(str) + "_" +
            (yy // CELL_M).astype(int).astype(str)).values


# ------------------------------------------------------ predict / demo
def predict_points(clf, df):
    """df needs: lon, lat, before_tif, after_tif (use the pair the organisers give)."""
    Bt = np.stack([normalise(read_patch(r.before_tif, r.lon, r.lat)) for r in df.itertuples()])
    At = np.stack([normalise(read_patch(r.after_tif, r.lon, r.lat)) for r in df.itertuples()])
    return clf.predict(make_X(Bt, At))

def _png(p):
    rgb = (np.clip((p.transpose(1, 2, 0) + 2.5) / 5, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).resize((192, 192), Image.BICUBIC).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

def export_demo(B, A, pairs, y, pred, proba, labels, n=6):
    recs = []
    for cls in CLASSES:
        idx = np.where(y == cls)[0]
        ok = idx[pred[idx] == cls][:n]
        bad = idx[pred[idx] != cls][:n]
        for i in np.concatenate([ok, bad]):
            recs.append(dict(id=int(i), site=pairs.site[i], true_label=y[i],
                             pred_label=pred[i], correct=bool(pred[i] == y[i]),
                             confidence=round(float(proba[i].max()), 3),
                             before_year=int(pairs.before_year[i]),
                             after_year=int(pairs.after_year[i]),
                             annotator_conflict=bool(pairs.conflict[i] == 1),
                             before=_png(B[i]), after=_png(A[i])))
    json.dump(recs, open("demo_data.json", "w"))
    print("demo_data.json:", len(recs), "records")


# ------------------------------------------------------------------- main
if __name__ == "__main__":
    gdf = gpd.read_file(ANN)
    if Path("cache_pairs.pkl").exists():
        pairs = pd.read_pickle("cache_pairs.pkl"); z = np.load("cache_patches.npz")
        B, A = z["B"], z["A"]
    else:
        B, A, pairs = build_patches(build_pairs(gdf))
        pairs.to_pickle("cache_pairs.pkl"); np.savez_compressed("cache_patches.npz", B=B, A=A)
    print(pairs.groupby(["site", "cls"]).size().unstack(fill_value=0))

    y, sites, groups = pairs.cls.values, pairs.site.values, cell_ids(pairs)
    X = make_X(B, A)

    p_out = oof_loso(X, y, sites)          # proxy for M_out (unseen region)
    p_in = oof_block(X, y, groups)         # proxy for M_in  (unseen area, seen region)
    m_out, m_in = macro_f1(y, p_out), macro_f1(y, p_in)
    print(f"M_in proxy {m_in:.3f} | M_out proxy {m_out:.3f} | S proxy {hm(m_in, m_out):.3f}")
    print("\nLOSO per-class report:\n", classification_report(y, p_out, labels=CLASSES, zero_division=0))
    print(pd.DataFrame(confusion_matrix(y, p_out, labels=CLASSES), index=CLASSES, columns=CLASSES))
    print("\nPer held-out site macro-F1:")
    for s in np.unique(sites):
        print(" ", s, round(macro_f1(y[sites == s], p_out[sites == s]), 3))

    # final model on ALL sites -> use for the real test submission
    final = model().fit(X, y)
    joblib.dump(final, "final_model.joblib")

    # demo data from the honest LOSO predictions
    proba = np.zeros((len(y), len(CLASSES)))
    for s in np.unique(sites):
        te = sites == s
        m = model().fit(X[~te], y[~te])
        proba[np.ix_(te, [CLASSES.index(c) for c in m.classes_])] = m.predict_proba(X[te])
    export_demo(B, A, pairs, y, np.array(CLASSES)[proba.argmax(1)], proba, CLASSES)
