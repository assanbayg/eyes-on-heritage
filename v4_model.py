"""Inference-only code for the model
"""
import numpy as np
from scipy.ndimage import gaussian_filter, sobel
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

CLASSES = ["construction", "destruction", "soil_sealing", "no_change"]
SEED, FAST = 0, False


def _glcm(g):
    q = ((np.clip(g, -2.5, 2.5) + 2.5) / 5 * 31).astype(np.uint8)
    m = graycomatrix(q, [1], [0, np.pi / 2], levels=32, symmetric=True, normed=True)
    return graycoprops(m, "contrast").mean(), graycoprops(m, "homogeneity").mean()


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

def _align(p, cls):
    out = np.zeros((len(p), len(CLASSES)))
    for j, c in enumerate(cls):
        out[:, CLASSES.index(c)] = p[:, j]
    return out


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
