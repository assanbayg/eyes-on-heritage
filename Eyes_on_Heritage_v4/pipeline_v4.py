"""
Eyes on Heritage - cleaned pipeline (replaces main_1.py)

Run:  python pipeline_v4.py tune   (pick best config)  then  python pipeline_v4.py
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

v4 (accuracy + ground-truth evaluation)
- new structural change features ("v4"): multi-scale SSIM, centre-weighted colour change
  (CVA), HOG structure distance, LBP texture histogram distance, Harris corner density
  (buildings have corners, rubble / sealed ground do not), rectilinearity of edges
- annotator conflict / class_uncertain points are down-weighted in training (noisy labels),
  but still evaluated at full weight
- 4th ensemble member (logistic regression) for diversity on unseen cities ("ens+")
- QC bug fixed: the original rejected any image >10% empty. Every real image has a fixed no-data border
  (11 to 60%), so ALL 40 images were rejected, no training pairs were built and the script crashed
  ("need at least one array to stack"). Now an image is flagged only if it is >10 points emptier than
  that site's usual border.
- CO-REGISTRATION (COREG=1, default): on the real data half of all before/after pairs are misaligned by
  >= 4 px (2 m) (median 2 to 6 px per city), so unchanged buildings showed edge "change". The after image
  is now aligned to the before image per point (phase correlation on gradients over a 192 m window,
  sub-pixel, shifts > 24 px rejected) before any feature is computed.
- feats="v4cnn": v4 + 12 deep change features from an ImageNet ResNet34 (needs torch; ~85 MB weights,
  downloaded once). Cosine / distance between before and after embeddings at 3 depths + activation change.
- AUG=n (env var or "aug" in config): each point also gets up to n other good-year pairs that still
  bracket its change year, as extra TRAINING rows (no_change points get pairs with the same gaps, so the
  gap is not a shortcut). Copies share the point's site and spatial cell, so they never leak across CV
  folds, and scores are computed on the original pair only.
- per-site scaling "ref": each site is scaled by features of ~150 RANDOM locations in its own
  imagery instead of by the points being predicted. The old scaling depended on the class mix of
  whatever points were scored (a construction-heavy test set shifted every feature; on such a
  sample in-sample accuracy fell to 0.11). "ref" is label-free and identical at train and test time.
- per-site scaling no longer silently skipped for small sites (a test city with < 30 points
  used to get UNSCALED features while the model was trained on scaled ones)
- accuracy + balanced accuracy reported next to macro-F1
- `python pipeline_v4.py eval <ground_truth.geojson>` scores the trained model on a labelled
  file (same format as train_annotations.geojson) -> ground-truth report + eval_predictions.csv
- `tune` now also tests the v4 steps and writes best_config_v4.json (old v3 config is ignored)

v3 (better score + class-aware modelling)
- features designed per change type: construction (new structure / centre-surround
  contrast / edge gain), destruction (edge loss, rubble texture, shadow change),
  soil sealing (greenness / redness loss, texture smoothing, brightness gain), each
  measured before, after and as a difference, at 3 scales
- per-site feature scaling, so an unseen city is judged relative to itself
- ensemble (RF + ExtraTrees + gradient boosting) + two-stage hierarchy
- per-class probability weights tuned for macro-F1, evaluated NESTED (no leakage)
- `python pipeline_fixed.py tune` compares every step honestly and keeps the best

v2 (bad 2024 imagery)
- FORCE_BAD_YEARS (default {2024}) + automatic per-image QC (nodata, brightness,
  contrast outliers vs. the site's other years). Bad images are never used.
- If a change year (or its previous year) is a bad image, the nearest GOOD years
  are used instead: before = last good year < y, after = first good year >= y.
- after-image is histogram-matched to the before-image (per band) before
  normalising, to remove colour / illumination shifts between dates.
- caches are keyed by the settings, so old caches are not reused by mistake.
- `python pipeline_fixed.py qc` writes qc_report.csv listing every image + flag.
"""
import base64, io, json, os, random, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import joblib
from rasterio.windows import Window
from rasterio.warp import transform
from scipy.ndimage import gaussian_filter, sobel, shift as nd_shift
from skimage.registration import phase_cross_correlation
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern, corner_harris, corner_peaks
from skimage.metrics import structural_similarity
from skimage.measure import shannon_entropy
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
FORCE_BAD_YEARS = {2024}                      # images from these years are never used; set() to disable
AUTO_QC = True                                # also auto-flag broken images
QC_MAX_NODATA = 0.10                          # image flagged if >10 points MORE empty than the site's usual
                                              # (every site has a fixed no-data border around its outline:
                                              # 11% Aleppo .. 60% Derna, so an absolute 10% limit rejected ALL images)
QC_BRIGHT_TOL = 0.35                          # flagged if mean brightness differs >35% from site median
QC_CONTRAST_LO, QC_CONTRAST_HI = 0.5, 2.0     # flagged if contrast ratio vs site median outside range
FAST = os.environ.get("FAST", "0") == "1"      # FAST=1: fewer trees, quicker experiments
COREG = os.environ.get("COREG", "1") == "1"   # co-register after-image to before-image per point
REG_HALF, REG_MAX = 192, 24                   # registration window 384 px (192 m), max accepted shift 24 px
TAG = f"v4qc_reg{int(COREG)}_{'-'.join(map(str, sorted(FORCE_BAD_YEARS))) or 'none'}_{int(AUTO_QC)}_aug{int(os.environ.get('AUG', 0))}"
C = np.s_[CTX - HALF:CTX + HALF, CTX - HALF:CTX + HALF]   # centre crop

# model configuration (python pipeline_fixed.py tune  ->  writes best_config.json, loaded here)
# defaults = best setting measured on the real hackathon data (tune, 17 Oct 2026); with COREG=1 below
CFG = dict(feats="new",        # "basic" (v2), "new" (v3 class-aware) or "v4" (v3 + structural change features)
           site_norm=True,     # per-site scaling: False, True (stats of the labelled points) or
                               # "ref" (stats of random reference locations in that site's imagery)
           kind="ens",         # "rf", "ens" (RF + ExtraTrees + HistGB) or "ens+" (ens + logistic regression)
           hier=True,          # two-stage: change vs no_change, then type of change
           tune=True,          # tune per-class probability weights for macro-F1 (nested, honest)
           aug=0,              # extra (before, after) year pairs per point for TRAINING (0 = off)
           noisy_w=1.0)        # training weight of points with annotator conflict / class_uncertain
CFG_FILE = Path("best_config_v4.json")
if CFG_FILE.exists():
    CFG.update(json.load(open(CFG_FILE)))
AUG = int(os.environ.get("AUG", CFG.get("aug", 0)))
CFG["aug"] = AUG


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

# ------------------------------------------------------------------- QC
def tif_stats(path):
    with rasterio.open(path) as src:
        sc = max(1, max(src.width, src.height) // 1024)
        arr = src.read([1, 2, 3, 4], out_shape=(4, max(1, src.height // sc), max(1, src.width // sc))
                       ).astype("float32")
    bad = (arr[3] == 0) | (arr[:3].sum(0) == 0)
    v = ~bad
    rgb = arr[:3][:, v] if v.any() else np.zeros((3, 1), "float32")
    return dict(nodata=float(bad.mean()), mean=float(rgb.mean()), std=float(rgb.std()))

_qc = {}
def qc_table(site):
    """One row per image of a site, with an auto-QC flag and a 'bad' decision."""
    if site in _qc:
        return _qc[site]
    rows = []
    for f in sorted((ROOT / SITE_DIR[site]).glob("*.tif")):
        rows.append(dict(site=site, year=int(f.name[:4]), file=f.name, **tif_stats(f)))
    df = pd.DataFrame(rows)
    med_m, med_s, med_n = df["mean"].median(), df["std"].median(), df["nodata"].median()
    reasons = []
    for r in df.itertuples():
        why = []
        if r.year in FORCE_BAD_YEARS:
            why.append("forced")
        if AUTO_QC:
            if r.nodata > med_n + QC_MAX_NODATA:
                why.append(f"nodata {r.nodata:.0%}")
            if len(df) >= 3 and med_m > 0 and abs(r.mean / med_m - 1) > QC_BRIGHT_TOL:
                why.append(f"brightness x{r.mean / med_m:.2f}")
            if len(df) >= 3 and med_s > 0 and not (QC_CONTRAST_LO <= r.std / med_s <= QC_CONTRAST_HI):
                why.append(f"contrast x{r.std / med_s:.2f}")
        reasons.append(", ".join(why))
    df["flag"] = reasons
    df["bad"] = df.flag != ""
    _qc[site] = df
    return df

def bad_years(site):
    df = qc_table(site)
    return set(df.loc[df.bad, "year"])

def good_years(site):
    b = bad_years(site)
    return [y for y in years_of(site) if y not in b]

def pick_pair(site, year):
    """Nearest GOOD images around an annotated change year -> (before, after) or None."""
    g = good_years(site)
    before = [y for y in g if y < year]
    after = [y for y in g if y >= year]
    return (before[-1], after[0]) if before and after else None

def qc_report():
    df = pd.concat([qc_table(s) for s in SITE_DIR if (ROOT / SITE_DIR[s]).exists()], ignore_index=True)
    df.to_csv("qc_report.csv", index=False)
    print(df.round(3).to_string(index=False))
    print("\nBAD images:", int(df.bad.sum()), "of", len(df), "-> see qc_report.csv")
    return df

def read_patch(tif, lon, lat):
    """(4, 2*CTX, 2*CTX) float32: R,G,B,alpha around a lon/lat point."""
    src = _open(tif)
    xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
    r, c = src.index(xs[0], ys[0])
    w = Window(c - CTX, r - CTX, 2 * CTX, 2 * CTX)
    return src.read([1, 2, 3, 4], window=w, boundless=True,
                    fill_value=0).astype("float32")

def _reg_shift(before_tif, after_tif, lon, lat):
    """Sub-pixel (dy, dx) that aligns the after image to the before image around a point.
    Estimated on a 384 px window with gradient images (robust to colour changes); a shift that is
    implausibly large or not supported by both images is rejected (returns 0, 0)."""
    def win(t):
        src = _open(t)
        xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
        r, c = src.index(xs[0], ys[0])
        a = src.read([1, 2, 3, 4], window=Window(c - REG_HALF, r - REG_HALF, 2 * REG_HALF, 2 * REG_HALF),
                     boundless=True, fill_value=0).astype("float32")
        return a
    wb, wa = win(before_tif), win(after_tif)
    valid = (wb[3] > 0) & (wa[3] > 0)
    if valid.mean() < 0.5:
        return 0.0, 0.0
    gb = _edge(gaussian_filter(wb[:3].mean(0), 1.0)) * valid
    ga = _edge(gaussian_filter(wa[:3].mean(0), 1.0)) * valid
    try:
        sh, _, _ = phase_cross_correlation(gb, ga, upsample_factor=4, normalization=None)
    except Exception:
        return 0.0, 0.0
    return (0.0, 0.0) if np.hypot(*sh) > REG_MAX else (float(sh[0]), float(sh[1]))

def read_pair(before_tif, after_tif, lon, lat):
    """Before / after patches around a point, after-image co-registered to the before-image."""
    rb, ra = read_patch(before_tif, lon, lat), read_patch(after_tif, lon, lat)
    if COREG:
        dy, dx = _reg_shift(before_tif, after_tif, lon, lat)
        if dy or dx:
            ra = np.stack([nd_shift(ra[k], (dy, dx), order=1 if k < 3 else 0, mode="constant", cval=0)
                           for k in range(4)]).astype("float32")
    return rb, ra

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


def prep_pair(rb, ra):
    """Colour-match the after image to the before image (per band, valid pixels), then normalise."""
    from skimage.exposure import match_histograms
    valid = (rb[3] > 0) & (ra[3] > 0)
    ra = ra.copy()
    if valid.sum() > 500:
        for k in range(3):
            ra[k][valid] = match_histograms(ra[k][valid], rb[k][valid])
    return normalise(rb), normalise(ra)


# ------------------------------------------------------------ build pairs
def alt_pairs(site, year, n):
    """Up to n other (before, after) good-year pairs that still bracket `year` (change lies between)."""
    g = good_years(site)
    bs = [y for y in g if y < year][-2:][::-1]
    as_ = [y for y in g if y >= year][:2]
    prim = pick_pair(site, year)
    alts = [(b, a) for b in bs for a in as_ if (b, a) != prim]
    alts.sort(key=lambda p: p[1] - p[0])
    return alts[:n]

def build_pairs(gdf, aug=None):
    aug = AUG if aug is None else aug
    rng = random.Random(SEED)
    rows = []
    for site, g in gdf.groupby("site"):
        gy = good_years(site)
        yrs = [int(y) for y in g.loc[g["class"] != "no_change", "year"].dropna()]
        change_years = [y for y in yrs if pick_pair(site, y)]
        for i, r in g.iterrows():
            ref = None
            if r["class"] == "no_change":
                if change_years:
                    ref = rng.choice(change_years)
                    pr = pick_pair(site, ref)
                elif len(gy) >= 2:
                    k = rng.randrange(1, len(gy)); pr = (gy[k - 1], gy[k])
                else:
                    continue
            else:
                if pd.isna(r["year"]):
                    continue
                ref = int(r["year"])
                pr = pick_pair(site, ref)
                if pr is None:
                    continue
            extra = alt_pairs(site, ref, aug) if (aug and ref is not None) else []
            for k, (before, after) in enumerate([pr] + extra):
              rows.append(dict(idx=i, site=site, cls=r["class"], primary=(k == 0),
                             before_year=before, after_year=after, gap=after - before,
                             annotated_year=r["year"],
                             before_tif=tif_for(site, before),
                             after_tif=tif_for(site, after),
                             lon=r.geometry.x, lat=r.geometry.y,
                             conflict=r.get("conflict", np.nan),
                             class_uncertain=r.get("class_uncertain", np.nan)))
    return pd.DataFrame(rows)

def primary_mask(pairs):
    return pairs["primary"].values.astype(bool) if "primary" in pairs else np.ones(len(pairs), bool)

def build_patches(pairs, max_bad=0.10):
    B, A, keep = [], [], []
    for r in pairs.itertuples():
        rb, ra = read_pair(r.before_tif, r.after_tif, r.lon, r.lat)
        keep.append(max(nodata_frac(rb), nodata_frac(ra)) <= max_bad)
        nb, na = prep_pair(rb, ra)
        B.append(nb); A.append(na)
    keep = np.array(keep)
    return np.stack(B)[keep], np.stack(A)[keep], pairs[keep].reset_index(drop=True)


# --------------------------------------------------------------- features
def _glcm(g):
    q = ((np.clip(g, -2.5, 2.5) + 2.5) / 5 * 31).astype(np.uint8)
    m = graycomatrix(q, [1], [0, np.pi / 2], levels=32, symmetric=True, normed=True)
    return graycoprops(m, "contrast").mean(), graycoprops(m, "homogeneity").mean()

def feats_basic(b, a):
    """v2 feature set (kept as a baseline for `tune`)."""
    cs, ms = np.s_[24:40, 24:40], np.s_[16:48, 16:48]
    gb, ga = gaussian_filter(b.mean(0), 1.5), gaussian_filter(a.mean(0), 1.5)
    d = ga - gb
    corr = np.nan_to_num(np.corrcoef(gb[cs].ravel(), ga[cs].ravel())[0, 1])
    f = [np.abs(d).mean(), np.abs(d[cs]).mean(), np.abs(d[ms]).mean(),
         d[cs].mean(), gb[cs].std(), ga[cs].std(), corr]
    f += [(a[k][cs] - b[k][cs]).mean() for k in range(3)]
    f += [np.abs(a[k][cs] - b[k][cs]).mean() for k in range(3)]
    eb = np.hypot(sobel(gb, 0), sobel(gb, 1))[cs].mean()
    ea = np.hypot(sobel(ga, 0), sobel(ga, 1))[cs].mean()
    f += [eb, ea, ea - eb]
    (cb, hb), (ca, ha) = _glcm(b.mean(0)[cs]), _glcm(a.mean(0)[cs])
    f += [cb, ca, ca - cb, hb, ha, ha - hb]
    return f

def _edge(g):
    return np.hypot(sobel(g, 0), sobel(g, 1))

def _desc(rgb):
    """Descriptors of one image around the centre point (what is THERE)."""
    cs = np.s_[16:48, 16:48]
    ring = np.ones(rgb.shape[1:], bool); ring[cs] = False
    gray = rgb.mean(0)
    g = gaussian_filter(gray, 1.0)
    e = _edge(g)
    R, G, Bl = rgb
    q = ((np.clip(g[cs], -2.5, 2.5) + 2.5) / 5 * 31).astype(np.uint8)
    contrast, homog = _glcm(g[cs])
    return dict(
        mean=g[cs].mean(),                       # brightness (bright roofs / concrete)
        cs_contrast=g[cs].mean() - g[ring].mean(),   # does the centre stand out from surroundings
        std=g[cs].std(),                         # local variation
        edge=e[cs].mean(),                       # structure / building edges
        edge_ratio=e[cs].mean() / (e.mean() + 1e-6),
        entropy=shannon_entropy(q),              # rubble / clutter is high-entropy
        shadow=(gray[cs] < -1.0).mean(),         # shadows: buildings standing, debris
        bright=(gray[cs] > 1.0).mean(),
        exg=(2 * G - R - Bl)[cs].mean(),         # greenness (vegetation -> sealed)
        redness=(R - Bl)[cs].mean(),             # bare earth / earthen architecture
        glcm_contrast=contrast, glcm_homog=homog)  # smooth sealed surface = high homogeneity

def feats_new(b, a):
    """Class-aware features: construction / destruction / soil sealing signals."""
    cs = np.s_[16:48, 16:48]
    ring = np.ones(b.shape[1:], bool); ring[cs] = False
    db, da = _desc(b), _desc(a)
    f = {}
    for k in db:                                  # before, after, and change of each descriptor
        f[f"b_{k}"], f[f"a_{k}"], f[f"d_{k}"] = db[k], da[k], da[k] - db[k]
    gb, ga = gaussian_filter(b.mean(0), 1.5), gaussian_filter(a.mean(0), 1.5)
    d = ga - gb
    for name, sl in (("16", np.s_[24:40, 24:40]), ("32", cs), ("64", np.s_[:, :])):   # 3 scales
        with np.errstate(all="ignore"):
            c = np.corrcoef(gb[sl].ravel(), ga[sl].ravel())[0, 1]
        f[f"absdiff_{name}"] = np.abs(d[sl]).mean()
        f[f"corr_{name}"] = np.nan_to_num(c)
        f[f"changed_frac_{name}"] = (np.abs(d[sl]) > 1.0).mean()
    f["signed_diff_16"] = d[24:40, 24:40].mean()
    f["maxdiff_32"] = np.abs(d[cs]).max()
    for k, n in enumerate("rgb"):
        f[f"shift_{n}"] = (a[k][cs] - b[k][cs]).mean()
        f[f"absshift_{n}"] = np.abs(a[k][cs] - b[k][cs]).mean()
    eb, ea = _edge(gb)[cs], _edge(ga)[cs]
    with np.errstate(all="ignore"):
        f["edge_corr"] = np.nan_to_num(np.corrcoef(eb.ravel(), ea.ravel())[0, 1])  # structure replaced?
    f["diff_edge"] = _edge(d)[cs].mean()
    f["change_cs_ratio"] = np.abs(d[cs]).mean() / (np.abs(d[ring]).mean() + 1e-6)   # change is local, not global
    return f

def _u8(g):
    return ((np.clip(g, -2.5, 2.5) + 2.5) / 5 * 255).astype(np.uint8)

def _lbp_hist(g):
    l = local_binary_pattern(_u8(g), 8, 1, "uniform").astype(int)
    h = np.bincount(l.ravel(), minlength=10)[:10].astype("float64")
    return h / max(h.sum(), 1)

def _rectilinear(g):
    """How strongly edge directions concentrate on two perpendicular axes (man-made structure)."""
    gy, gx = sobel(g, 0), sobel(g, 1)
    mag = np.hypot(gx, gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi / 2)            # fold onto [0, 90 deg)
    h, _ = np.histogram(ang, bins=9, range=(0, np.pi / 2), weights=mag)
    return h.max() / (h.mean() + 1e-6)

def _harris(g):
    r = corner_harris(g, sigma=1.0)
    n = len(corner_peaks(r, min_distance=2, threshold_abs=0.02))
    return n, np.percentile(r, 99)

def feats_v4(b, a):
    """v3 features + structural change features (SSIM, CVA, HOG, LBP, corners, rectilinearity)."""
    f = feats_new(b, a)
    cs = np.s_[16:48, 16:48]
    gb, ga = gaussian_filter(b.mean(0), 1.0), gaussian_filter(a.mean(0), 1.0)
    cb, ca = np.clip(gb, -2.5, 2.5), np.clip(ga, -2.5, 2.5)
    for name, sl, win in (("16", np.s_[24:40, 24:40], 7), ("32", cs, 7), ("64", np.s_[:, :], 11)):
        f[f"ssim_{name}"] = structural_similarity(cb[sl], ca[sl], data_range=5.0, win_size=win)
    yy, xx = np.mgrid[:64, :64]
    w = np.exp(-((yy - 31.5) ** 2 + (xx - 31.5) ** 2) / (2 * 8.0 ** 2))
    cva = np.sqrt(((a - b) ** 2).sum(0))
    f["cva_centre"] = (w * cva).sum() / w.sum()
    f["cva_centre_vs_all"] = f["cva_centre"] / (cva.mean() + 1e-6)
    hb = hog(gb[cs], orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    ha = hog(ga[cs], orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    f["hog_l1"] = np.abs(ha - hb).mean()
    f["hog_cos"] = float(hb @ ha / (np.linalg.norm(hb) * np.linalg.norm(ha) + 1e-9))
    lb, la = _lbp_hist(gb[cs]), _lbp_hist(ga[cs])
    f["lbp_chi2"] = 0.5 * (((la - lb) ** 2) / (la + lb + 1e-9)).sum()
    f["lbp_flat_b"], f["lbp_flat_a"] = lb[0] + lb[8], la[0] + la[8]       # flat / uniform areas (sealed)
    f["lbp_noise_d"] = la[9] - lb[9]                                        # non-uniform codes (rubble)
    (nb, rb), (na, ra) = _harris(gb[cs]), _harris(ga[cs])
    f["corners_b"], f["corners_a"], f["corners_d"] = nb, na, na - nb
    f["harris99_d"] = ra - rb
    f["rect_b"], f["rect_a"] = _rectilinear(gb[cs]), _rectilinear(ga[cs])
    f["rect_d"] = f["rect_a"] - f["rect_b"]
    return f

FEATS = {"basic": feats_basic, "new": feats_new, "v4": feats_v4}

def feat_names():
    rng = np.random.default_rng(0)
    r = FEATS["v4" if CFG["feats"] == "v4cnn" else CFG["feats"]](rng.normal(size=(3, 64, 64)), rng.normal(size=(3, 64, 64)))
    names = list(r.keys()) if isinstance(r, dict) else [f"f{i}" for i in range(len(r))]
    if CFG["feats"] == "v4cnn":
        names += [f"cnn_l{l}_{k}" for l in (2, 3, 4) for k in ("cos", "cos_centre", "dist", "dnorm")]
    return names

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

_NET = {}
def cnn_change(B, A):
    """Compact deep change features (no fitting): for ResNet34 layers 2/3/4, cosine similarity of the
    before / after embeddings (whole patch and centre), distance of the normalised embeddings, and the
    change in activation strength (more / less 'structure'). 12 features per pair."""
    import torch, torchvision.models as tvm
    if "net" not in _NET:
        torch.set_num_threads(max(1, os.cpu_count() // 2))
        net = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT).eval()
        _NET["net"] = net
    net = _NET["net"]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    def maps(P):
        x = torch.tensor(np.clip((P + 2.5) / 5, 0, 1), dtype=torch.float32)
        x = torch.nn.functional.interpolate(x, size=128, mode="bilinear", align_corners=False)
        x = (x - mean) / std
        x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        l2 = net.layer2(net.layer1(x)); l3 = net.layer3(l2); l4 = net.layer4(l3)
        return [l2, l3, l4]
    out = []
    with torch.no_grad():
        for i in range(0, len(B), 64):
            mb, ma = maps(B[i:i + 64]), maps(A[i:i + 64])
            cols = []
            for fb, fa in zip(mb, ma):
                h = fb.shape[-1]; c0, c1 = h // 4, h - h // 4
                gb, ga = fb.mean((2, 3)), fa.mean((2, 3))
                cb, ca = fb[..., c0:c1, c0:c1].mean((2, 3)), fa[..., c0:c1, c0:c1].mean((2, 3))
                cos_g = torch.nn.functional.cosine_similarity(gb, ga, dim=1)
                cos_c = torch.nn.functional.cosine_similarity(cb, ca, dim=1)
                dist = (torch.nn.functional.normalize(gb, dim=1) - torch.nn.functional.normalize(ga, dim=1)).norm(dim=1)
                dnorm = (ca.norm(dim=1) - cb.norm(dim=1)) / (cb.norm(dim=1) + 1e-6)
                cols += [cos_g, cos_c, dist, dnorm]
            out.append(torch.stack(cols, 1).numpy())
    return np.concatenate(out).astype("float32")

def site_norm(X, sites, min_n=30):
    """Robust per-site scaling (median / IQR) so a new city is judged relative to itself.
    Sites with fewer than min_n points are scaled with the pooled statistics of all points
    (previously they were left unscaled, i.e. on a different scale than the training data)."""
    raw, X = X, X.copy()
    floor = 0.25 * raw.std(0) + 1e-6
    q75, q25 = np.percentile(raw, [75, 25], axis=0)
    pooled = (np.median(raw, axis=0), np.maximum(q75 - q25, floor))
    for s_ in np.unique(sites):
        m = sites == s_
        if m.sum() < min_n:
            med, iqr = pooled
        else:
            q75, q25 = np.percentile(raw[m], [75, 25], axis=0)
            med, iqr = np.median(raw[m], axis=0), np.maximum(q75 - q25, floor)
        X[m] = (raw[m] - med) / iqr
    return X

def raw_feats(B, A):
    fn = FEATS["v4" if CFG["feats"] == "v4cnn" else CFG["feats"]]
    rows = [fn(b, a) for b, a in zip(B, A)]
    X = np.array([list(r.values()) if isinstance(r, dict) else r for r in rows], dtype="float32")
    if CFG["feats"] == "v4cnn":
        X = np.hstack([X, cnn_change(B, A)])
    if USE_CNN:
        fb, fa = cnn_feats(B), cnn_feats(A)
        X = np.hstack([X, fb, fa, np.abs(fa - fb)])
    return np.nan_to_num(X)

N_REF = 150
_REF = {}
def site_ref(site):
    """Median / IQR of features at random locations of a site (label-free reference)."""
    key = (site, CFG["feats"], USE_CNN)
    if key in _REF:
        return _REF[key]
    cache = Path(f"cache_ref_{TAG}_{CFG['feats']}_{int(USE_CNN)}_{site}.npz")
    if cache.exists():
        z = np.load(cache); _REF[key] = (z["med"], z["iqr"], z["std"]); return _REF[key]
    rng = np.random.default_rng(SEED)
    gy = good_years(site)
    yp = [(gy[k - 1], gy[k]) for k in range(1, len(gy))]
    src = _open(tif_for(site, gy[0]))
    B, A, tries = [], [], 0
    while len(B) < N_REF and tries < N_REF * 5:
        tries += 1
        r, c = rng.integers(CTX, max(CTX + 1, src.height - CTX)), rng.integers(CTX, max(CTX + 1, src.width - CTX))
        x, y = src.xy(r, c)
        lon, lat = transform(src.crs, "EPSG:4326", [x], [y])
        yb, ya = yp[rng.integers(len(yp))]
        rb, ra = read_pair(tif_for(site, yb), tif_for(site, ya), lon[0], lat[0])
        if max(nodata_frac(rb), nodata_frac(ra)) > 0.10:
            continue
        nb, na = prep_pair(rb, ra); B.append(nb); A.append(na)
    R = raw_feats(np.stack(B), np.stack(A))
    q75, q25 = np.percentile(R, [75, 25], axis=0)
    _REF[key] = (np.median(R, axis=0), q75 - q25, R.std(0))
    np.savez(cache, med=_REF[key][0], iqr=_REF[key][1], std=_REF[key][2])
    return _REF[key]

def site_norm_ref(X, sites):
    X = X.copy()
    for s_ in np.unique(sites):
        m = sites == s_
        med, iqr, sd = site_ref(s_)
        X[m] = (X[m] - med) / np.maximum(iqr, 0.25 * sd + 1e-6)
    return X

_RAW = {}
def make_X(B, A, sites=None):
    key = (CFG["feats"], USE_CNN, id(B), len(B))
    if key not in _RAW:
        _RAW[key] = raw_feats(B, A)
    X = _RAW[key]
    if CFG["site_norm"] == "ref" and sites is not None:
        X = site_norm_ref(X, np.asarray(sites))
    elif CFG["site_norm"] and sites is not None:
        X = site_norm(X, np.asarray(sites))
    return X


# ------------------------------------------------------------- evaluation
def macro_f1(y, p):
    return f1_score(y, p, labels=CLASSES, average="macro", zero_division=0)

def acc_line(y, p):
    return f"accuracy {accuracy_score(y, p):.3f} | balanced accuracy {balanced_accuracy_score(y, p):.3f}"

def hm(a, b):
    return 2 * a * b / (a + b) if (a + b) > 0 else 0.0

def _align(p, cls):
    out = np.zeros((len(p), len(CLASSES)))
    for j, c in enumerate(cls):
        out[:, CLASSES.index(c)] = p[:, j]
    return out

def tune_weights(P, y, grid=np.arange(0.6, 2.01, 0.1), passes=3):
    """Per-class multipliers on probabilities that maximise macro-F1 (no_change fixed at 1)."""
    w = np.ones(len(CLASSES)); lab = np.array(CLASSES)
    best = macro_f1(y, lab[(P * w).argmax(1)])
    for _ in range(passes):
        for j, c in enumerate(CLASSES):
            if c == "no_change":
                continue
            for v in grid:
                w2 = w.copy(); w2[j] = v
                sc = macro_f1(y, lab[(P * w2).argmax(1)])
                if sc > best + 1e-9:
                    best, w = sc, w2
    return w

class Ensemble:
    """flat 4-class ensemble  (+ optional two-stage hierarchy)  (+ tuned class weights)."""
    def __init__(self, kind="ens", hier=True, tune=True):
        self.kind, self.hier, self.tune = kind, hier, tune
        self.classes_ = np.array(CLASSES)
        self.w = np.ones(len(CLASSES))

    def _members(self):
        n = 150 if FAST else 400
        ms = [RandomForestClassifier(n_estimators=n, class_weight="balanced_subsample",
                                     min_samples_leaf=2, random_state=SEED, n_jobs=-1)]
        if self.kind in ("ens", "ens+"):
            ms += [ExtraTreesClassifier(n_estimators=n, class_weight="balanced_subsample",
                                        min_samples_leaf=2, random_state=SEED, n_jobs=-1),
                   HistGradientBoostingClassifier(max_iter=100 if FAST else 200, learning_rate=0.06,
                                                  max_depth=4, random_state=SEED)]
        if self.kind == "ens+":
            ms += [make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced",
                                                                      max_iter=2000))]
        return ms

    @staticmethod
    def _fit1(m, X, y, sw=None):
        sw = np.ones(len(y)) if sw is None else sw
        if isinstance(m, HistGradientBoostingClassifier):
            m.fit(X, y, sample_weight=sw * compute_sample_weight("balanced", y))
        elif hasattr(m, "steps"):                                   # scaler + logistic regression
            m.fit(X, y, logisticregression__sample_weight=sw)
        else:
            m.fit(X, y, sample_weight=sw)
        return m

    def _fit_core(self, X, y, sw=None):
        sw = np.ones(len(y)) if sw is None else sw
        self.flat = [self._fit1(m, X, y, sw) for m in self._members()]
        if self.hier:
            chg = y != "no_change"
            self.s1 = [self._fit1(m, X, chg, sw) for m in self._members()]
            self.s2 = [self._fit1(m, X[chg], y[chg], sw[chg]) for m in self._members()]

    def _core_proba(self, X):
        p = np.mean([_align(m.predict_proba(X), m.classes_) for m in self.flat], 0)
        if self.hier:
            pc = np.mean([m.predict_proba(X)[:, list(m.classes_).index(True)] for m in self.s1], 0)
            pt = np.mean([_align(m.predict_proba(X), m.classes_) for m in self.s2], 0)
            ph = pt * pc[:, None]
            ph[:, CLASSES.index("no_change")] = 1 - pc
            p = 0.5 * p + 0.5 * ph
        return p

    def fit(self, X, y, groups=None, sw=None):
        y = np.asarray(y)
        sw = np.ones(len(y)) if sw is None else np.asarray(sw, dtype="float64")
        self.cfg = dict(CFG)
        self.w = np.ones(len(CLASSES))
        if self.tune and groups is not None and len(np.unique(groups)) >= 3:
            groups = np.asarray(groups)
            P = np.zeros((len(y), len(CLASSES)))          # inner leave-one-group-out probabilities
            for g in np.unique(groups):
                te = groups == g
                sub = Ensemble(self.kind, self.hier, False)
                sub._fit_core(X[~te], y[~te], sw[~te])
                P[te] = sub._core_proba(X[te])
            self.w = tune_weights(P, y)
        self._fit_core(X, y, sw)
        return self

    def predict_proba(self, X):
        p = self._core_proba(X) * self.w
        return p / np.maximum(p.sum(1, keepdims=True), 1e-9)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    @property
    def feature_importances_(self):
        return self.flat[0].feature_importances_

def model():
    return Ensemble(CFG["kind"], CFG["hier"], CFG["tune"])

def train_weights(pairs):
    """Down-weight noisy labels (annotator conflict / class_uncertain) in TRAINING only."""
    noisy = (pd.to_numeric(pairs.conflict, errors="coerce") == 1) | \
            (pd.to_numeric(pairs.class_uncertain, errors="coerce") == 1)
    return np.where(noisy.values, CFG.get("noisy_w", 1.0), 1.0)

def oof_loso(X, y, sites, mk=None, sw=None):
    mk = mk or model
    sw = np.ones(len(y)) if sw is None else sw
    pred = np.empty(len(y), dtype=object)
    for s_ in np.unique(sites):
        te = sites == s_
        pred[te] = mk().fit(X[~te], y[~te], groups=sites[~te], sw=sw[~te]).predict(X[te])
    return pred

def oof_block(X, y, groups, sites=None, k=5, mk=None, sw=None):
    mk = mk or model
    sw = np.ones(len(y)) if sw is None else sw
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(k).split(X, y, groups):
        pred[te] = mk().fit(X[tr], y[tr], groups=None if sites is None else sites[tr],
                            sw=sw[tr]).predict(X[te])
    return pred

def cell_ids(df):
    x = df.lon * 111320 * np.cos(np.radians(df.lat))
    yy = df.lat * 110540
    return (df.site + "_" + (x // CELL_M).astype(int).astype(str) + "_" +
            (yy // CELL_M).astype(int).astype(str)).values


# ------------------------------------------------------ predict / demo
def load_data():
    pp, pz = Path(f"cache_pairs_{TAG}.pkl"), Path(f"cache_patches_{TAG}.npz")
    if pp.exists() and pz.exists():
        z = np.load(pz)
        return z["B"], z["A"], pd.read_pickle(pp)
    B, A, pairs = build_patches(build_pairs(gpd.read_file(ANN)))
    pairs.to_pickle(pp); np.savez_compressed(pz, B=B, A=A)
    return B, A, pairs

def predict_points(clf, df):
    """df needs: lon, lat, before_tif, after_tif (use the pair the organisers give)."""
    P = [prep_pair(*read_pair(r.before_tif, r.after_tif, r.lon, r.lat)) for r in df.itertuples()]
    return clf.predict(make_X(np.stack([p[0] for p in P]), np.stack([p[1] for p in P]), df.site.values))

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


def run_eval(gt_path):
    """Score the trained model on a labelled ground-truth file (same format as train_annotations)."""
    if not Path("final_model.joblib").exists():
        raise SystemExit("final_model.joblib not found: run `python pipeline_v4.py` first")
    clf = joblib.load("final_model.joblib")
    CFG.update(getattr(clf, "cfg", {}))
    gdf = gpd.read_file(gt_path)
    pairs = build_pairs(gdf, aug=0)
    B, A, pairs = build_patches(pairs, max_bad=1.0)            # keep every point
    y, sites = pairs.cls.values, pairs.site.values
    X = make_X(B, A, sites)
    proba = clf.predict_proba(X)
    pred = np.array(CLASSES)[proba.argmax(1)]
    print(f"Ground truth: {gt_path}  ({len(y)} of {len(gdf)} points have a usable image pair)")
    print(f"macro-F1 {macro_f1(y, pred):.3f} | " + acc_line(y, pred))
    print(classification_report(y, pred, labels=CLASSES, zero_division=0))
    print(pd.DataFrame(confusion_matrix(y, pred, labels=CLASSES), index=CLASSES, columns=CLASSES))
    print("\nPer site:")
    for s_ in np.unique(sites):
        m = sites == s_
        lab = sorted(set(y[m]) | set(pred[m]))                  # only classes present at this site
        f1s = f1_score(y[m], pred[m], labels=lab, average="macro", zero_division=0)
        print(f"  {s_:10s} macro-F1 {f1s:.3f}  accuracy {accuracy_score(y[m], pred[m]):.3f}  (n={m.sum()})")
    out = pairs[["idx", "site", "lon", "lat", "before_year", "after_year"]].copy()
    out["true"], out["pred"], out["correct"] = y, pred, y == pred
    out["confidence"] = proba.max(1).round(3)
    out.to_csv("eval_predictions.csv", index=False)
    print("\nwrote eval_predictions.csv")


# ------------------------------------------------------------------- main
def run_tune():
    """Compare every improvement step honestly (unseen-city LOSO + unseen-area block CV)."""
    B, A, pairs = load_data()
    y, sites, groups = pairs.cls.values, pairs.site.values, cell_ids(pairs)
    pm = primary_mask(pairs)
    if os.environ.get("TUNE_ONLY"):                      # e.g. TUNE_ONLY=J  -> run just that step
        pass
    base = dict(feats="new", site_norm=False, kind="rf", hier=False, tune=False)
    steps = [("A  v2 baseline (basic feats, RF)", dict(base, feats="basic")),
             ("B  + class-aware features", base),
             ("C  + per-site normalisation", dict(base, site_norm=True)),
             ("D  + ensemble RF/ET/GB", dict(base, site_norm=True, kind="ens")),
             ("E  + two-stage hierarchy", dict(base, site_norm=True, kind="ens", hier=True)),
             ("F  + tuned class weights", dict(base, site_norm=True, kind="ens", hier=True, tune=True)),
             ("G  + v4 structural features", dict(base, feats="v4", site_norm=True, kind="ens", hier=True, tune=True)),
             ("H  + noisy labels down-weighted", dict(base, feats="v4", site_norm=True, kind="ens", hier=True,
                                                      tune=True, noisy_w=0.5)),
             ("I  + logistic-regression member", dict(base, feats="v4", site_norm=True, kind="ens+", hier=True,
                                                      tune=True, noisy_w=0.5)),
             ("K  + deep change features (ResNet34)", dict(base, feats="v4cnn", site_norm=True, kind="ens", hier=True,
                                                           tune=True)),
             ("J  + label-free site scaling (ref)", dict(base, feats="v4", site_norm="ref", kind="ens+", hier=True,
                                                         tune=True, noisy_w=0.5))]
    base["noisy_w"] = 1.0
    steps = [(n, dict({"noisy_w": 1.0}, **c)) for n, c in steps]
    keep, rows = dict(CFG), []
    if os.environ.get("TUNE_ONLY"):
        steps = [st for st in steps if st[0][0] in os.environ["TUNE_ONLY"]]
    for name, cfg in steps:
        cfg = dict(cfg, aug=AUG)
        CFG.update(cfg)
        X = make_X(B, A, sites)
        sw = train_weights(pairs)
        p_out, p_in = oof_loso(X, y, sites, sw=sw)[pm], oof_block(X, y, groups, sites, sw=sw)[pm]
        mo, mi = macro_f1(y[pm], p_out), macro_f1(y[pm], p_in)
        f1c = f1_score(y[pm], p_out, labels=CLASSES, average=None, zero_division=0)
        rows.append((name, mi, mo, hm(mi, mo), f1c, dict(CFG)))
        print(f"{name:36s} M_in {mi:.3f}  M_out {mo:.3f}  S {hm(mi, mo):.3f}  "
              f"acc_out {accuracy_score(y[pm], p_out):.3f}  | " +
              "  ".join(f"{c[:5]} {v:.2f}" for c, v in zip(CLASSES, f1c)), flush=True)
    best = max(rows, key=lambda r: r[3])
    json.dump(best[5], open(CFG_FILE, "w"), indent=2)
    CFG.clear(); CFG.update(best[5])
    print(f"\nBest: {best[0].strip()}  (S proxy {best[3]:.3f}) -> saved {CFG_FILE}")
    print("Note: choosing among 10 variants on the same data is slightly optimistic; quote step A vs best honestly.")


if __name__ == "__main__":
    import sys
    if "qc" in sys.argv[1:]:
        qc_report(); sys.exit()
    if "tune" in sys.argv[1:]:
        run_tune(); sys.exit()
    if "eval" in sys.argv[1:]:
        run_eval(sys.argv[sys.argv.index("eval") + 1]); sys.exit()
    qc_report()
    B, A, pairs = load_data()
    print("config:", CFG)
    print(pairs.groupby(["site", "cls"]).size().unstack(fill_value=0))
    pm = primary_mask(pairs)
    print("pairs using a wider window (substituted years):", int((pairs.gap[pm] > 1).sum()), "of", int(pm.sum()),
          f"| training rows incl. augmented pairs: {len(pairs)}")

    y, sites, groups = pairs.cls.values, pairs.site.values, cell_ids(pairs)
    X = make_X(B, A, sites)
    sw = train_weights(pairs)

    p_out = oof_loso(X, y, sites, sw=sw)          # proxy for M_out (unseen region)
    p_in = oof_block(X, y, groups, sites, sw=sw)  # proxy for M_in  (unseen area, seen region)
    full_y, full_sites = y, sites
    y, sites, p_out, p_in = y[pm], sites[pm], p_out[pm], p_in[pm]      # score original pairs only
    m_out, m_in = macro_f1(y, p_out), macro_f1(y, p_in)
    print(f"M_in proxy {m_in:.3f} | M_out proxy {m_out:.3f} | S proxy {hm(m_in, m_out):.3f}")
    print("unseen-city (LOSO)  " + acc_line(y, p_out))
    print("unseen-area (block) " + acc_line(y, p_in))
    oof = pairs[pm][["idx", "site", "lon", "lat", "before_year", "after_year"]].copy()
    oof["true"], oof["pred_unseen_city"], oof["pred_unseen_area"] = y, p_out, p_in
    oof.to_csv("oof_predictions.csv", index=False)
    print("wrote oof_predictions.csv (every point: annotation vs out-of-fold prediction)")
    print("\nLOSO per-class report:\n", classification_report(y, p_out, labels=CLASSES, zero_division=0))
    print(pd.DataFrame(confusion_matrix(y, p_out, labels=CLASSES), index=CLASSES, columns=CLASSES))
    print("\nPer held-out site macro-F1:")
    for s_ in np.unique(sites):
        print(" ", s_, round(macro_f1(y[sites == s_], p_out[sites == s_]), 3))

    final = model().fit(X, full_y, groups=full_sites, sw=sw)
    joblib.dump(final, "final_model.joblib")
    if not USE_CNN and CFG["feats"] in ("new", "v4", "v4cnn"):
        imp = sorted(zip(final.feature_importances_, feat_names()), reverse=True)[:10]
        print("\nTop features:", ", ".join(f"{n} ({v:.3f})" for v, n in imp))

    proba = np.zeros((len(full_y), len(CLASSES)))
    for s_ in np.unique(full_sites):
        te = full_sites == s_
        m = model().fit(X[~te], full_y[~te], groups=full_sites[~te], sw=sw[~te])
        proba[te] = m.predict_proba(X[te])
    B, A, pairs, proba = B[pm], A[pm], pairs[pm].reset_index(drop=True), proba[pm]
    export_demo(B, A, pairs, y, np.array(CLASSES)[proba.argmax(1)], proba, CLASSES)
