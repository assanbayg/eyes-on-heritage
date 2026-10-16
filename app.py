"""Eyes on Heritage - demo app.   Run:  python app.py"""
import base64
import html
import json
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import model

HERE = Path(__file__).resolve().parent
PATCH = 64            # pixels the model sees (64 x 64 px at 0.5 m = 32 m)

COLORS = {"construction": "#2C62C6", "destruction": "#BF3A2D",
          "soil_sealing": "#9C6710", "no_change": "#6B7D77"}
WORDS = {"construction": "New construction", "destruction": "Destruction",
         "soil_sealing": "Paved over", "no_change": "No change"}
EXPLAIN = {"construction": "something new seems to have been built",
           "destruction": "buildings or structures seem to have been damaged or removed",
           "soil_sealing": "bare ground seems to have been covered with concrete or asphalt",
           "no_change": "nothing important seems to have changed"}
ORDER = model.CLASS_ORDER

FOOTER = "<div style='text-align:center;opacity:.7;font-size:.85em;margin-top:1.5em'>" \
         "Imagery © Planet Labs PBC, CC BY-NC 4.0</div>"


# ------------------------------------------------------------------ helpers
def load_json(name):
    """Read a JSON file next to app.py; return None if missing or broken."""
    try:
        return json.loads((HERE / name).read_text(encoding="utf-8"))
    except Exception:
        return None


def prepare(img):
    """Any uploaded image -> centre-cropped square -> 64x64 RGB uint8 array."""
    if img is None:
        raise ValueError("missing")
    im = Image.fromarray(np.asarray(img)).convert("RGB")
    w, h = im.size
    if w < 8 or h < 8:
        raise ValueError("The image is too small.")
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return np.asarray(im.resize((PATCH, PATCH), Image.LANCZOS))


def enlarge(patch, size=288):
    return Image.fromarray(patch).resize((size, size), Image.BICUBIC)


def bar_chart(scores):
    fig, ax = plt.subplots(figsize=(6, 2.8))
    names = list(reversed(ORDER))                      # first class at the top
    vals = [scores[c] * 100 for c in names]
    bars = ax.barh([WORDS[c] for c in names], vals, color=[COLORS[c] for c in names])
    for bar, v in zip(bars, vals):
        ax.text(min(v + 1.5, 88), bar.get_y() + bar.get_height() / 2, f"{v:.0f}%",
                va="center", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score (%)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def placeholder_banner():
    info = model.backend_info()
    if not info["is_placeholder"]:
        return ""
    return ("<div style='padding:.6em 1em;border-radius:8px;background:#f5c84233;"
            "border:1px solid #f5c842'><b>Preview mode:</b> the trained model is not "
            "connected yet, so these scores come from a simple placeholder rule and "
            "<b>do not mean anything</b>.</div>")


# ------------------------------------------------------------------ Try it
def what_changed(before, after):
    try:
        if before is None or after is None:
            return ("**Please add both images first** - one from before, one from after.",
                    None, None, None)
        b, a = prepare(before), prepare(after)
        scores = model.predict_proba(b, a)
        best = max(scores, key=scores.get)
        conf = scores[best]
        msg = f"## {WORDS[best]}\n**Confidence: {conf:.0%}** - {EXPLAIN[best]}."
        if conf < 0.5:
            msg += ("\n\n*The model is not very sure. Treat this as a hint for a "
                    "human to check, not as a finding.*")
        banner = placeholder_banner()
        if banner:
            msg = banner + "\n\n" + msg
        return msg, bar_chart(scores), enlarge(b), enlarge(a)
    except NotImplementedError as e:
        return f"**The model is not ready yet.** {e}", None, None, None
    except Exception as e:  # never show a stack trace to the jury
        print("prediction error:", repr(e))
        return ("**Sorry, I could not read those images.** Please upload ordinary "
                "photos or screenshots (PNG or JPG) of the same spot, before and after.",
                None, None, None)


# ---------------------------------------------------------------- Examples
def b64(path):
    return base64.b64encode(path.read_bytes()).decode()


def chip(cls):
    return (f"<span style='background:{COLORS[cls]};color:#fff;padding:2px 8px;"
            f"border-radius:10px;font-size:.85em'>{WORDS[cls]}</span>")


def examples_html():
    items = load_json("examples/index.json")
    if not items:
        return ("<p>No examples found. Run <code>python make_examples.py</code> "
                "to create the <code>examples/</code> folder.</p>")
    cards = []
    for it in items:
        ok = it["true_label"] == it["pred_label"]
        edge, tag = ("#2e8b57", "✓ Model agrees") if ok else ("#BF3A2D", "✗ Model got this wrong")
        imgs = "".join(
            f"<div style='flex:1;text-align:center'><img style='width:100%;border-radius:4px' "
            f"src='data:image/png;base64,{b64(HERE / 'examples' / it[k])}'>"
            f"<div style='font-size:.8em;opacity:.7'>{lbl} {it[yk]}</div></div>"
            for k, yk, lbl in (("before", "before_year", "Before"), ("after", "after_year", "After")))
        cards.append(
            f"<div style='border:3px solid {edge};border-radius:10px;padding:10px;"
            f"background:rgba(128,128,128,.07)'>"
            f"<div style='display:flex;gap:6px'>{imgs}</div>"
            f"<div style='margin-top:8px;line-height:1.9'>"
            f"<b style='color:{edge}'>{tag}</b><br>"
            f"Truth (human annotators): {chip(it['true_label'])}<br>"
            f"Model says: {chip(it['pred_label'])} "
            f"<span style='opacity:.8'>({it['confidence']:.0%} confident)</span><br>"
            f"<span style='font-size:.8em;opacity:.65'>{html.escape(it['site'].title())}</span>"
            f"</div></div>")
    return ("<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));"
            "gap:14px'>" + "".join(cards) + "</div>")


EXAMPLES_NOTE = (
    "These are real, annotated points, **hand-picked to show a mix** of all four kinds of "
    "change, some the model gets right and some it gets wrong. It is *not* a random sample, "
    "so do not read the number of green or red cards as the model's accuracy. "
    "Predictions are the ones saved from our earlier run, shown at 64 × 64 pixels (32 m) "
    "per patch, enlarged.")


# ------------------------------------------------------------ How good is it
def confusion_fig(m):
    labels = m["confusion"]["labels"]
    mat = np.array(m["confusion"]["matrix"], dtype=float)
    share = mat / np.maximum(mat.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    names = [WORDS[c] for c in labels]
    ax.set_xticks(range(len(labels)), names, rotation=25, ha="right")
    ax.set_yticks(range(len(labels)), names)
    ax.set_xlabel("What the model said")
    ax.set_ylabel("What it really was")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center",
                    color="white" if share[i, j] > 0.5 else "black", fontsize=12)
    fig.tight_layout()
    return fig


def f1_fig(m):
    per = m["per_class_f1"]
    names = sorted(per, key=per.get)
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    bars = ax.barh([WORDS[c] for c in names], [per[c] for c in names],
                   color=[COLORS[c] for c in names])
    for bar, c in zip(bars, names):
        ax.text(per[c] + 0.015, bar.get_y() + bar.get_height() / 2, f"{per[c]:.2f}", va="center")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 score (0 = never right, 1 = always right)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def metrics_text(m):
    per = m["per_class_f1"]
    weak = [c for c in per if per[c] < 0.3]
    good = [c for c in per if per[c] >= 0.5]
    txt = [f"## Overall score: **{m['macro_f1']:.3f}** (macro-F1, scale 0 to 1)",
           "Think of it as a school mark between 0 and 1 where the four kinds of change "
           "count equally. A perfect system would score 1.",
           "",
           f"**How we measured it:** {m['protocol']}"
           + (f" ({m['n_test_points']} test points.)" if m.get("n_test_points") else ""),
           "",
           "### What this means in plain words",
           "This is an early baseline, and it is **far from reliable**."]
    if good:
        txt.append("- It is most useful for: " + ", ".join(f"**{WORDS[c]}** ({per[c]:.2f})" for c in good) + ".")
    if weak:
        txt.append("- It is **weak** at: " + ", ".join(f"**{WORDS[c]}** ({per[c]:.2f})" for c in weak)
                   + ". For these it mostly misses the real cases, so a human must check "
                   "anything it flags, and an absence of a flag means little.")
    txt.append("- A prediction is a prompt to look closer, never a verdict.")
    return "\n".join(txt)


def build_metrics_tab():
    try:
        m = load_json("metrics.json")
        if not m:
            raise ValueError
        return metrics_text(m), f1_fig(m), confusion_fig(m)
    except Exception:
        return ("**metrics.json is missing or has a typo.** Fix the file and restart the app.",
                None, None)


# ----------------------------------------------------------------------- UI
def build_ui():
    with gr.Blocks(title="Eyes on Heritage") as demo:
        gr.Markdown("# Eyes on Heritage\nSpot changes at heritage sites from satellite "
                    "images: **upload the same spot before and after, and see what changed.**")
        with gr.Tab("Try it"):
            with gr.Row():
                before = gr.Image(label="Before", type="numpy", image_mode="RGB", height=320)
                after = gr.Image(label="After", type="numpy", image_mode="RGB", height=320)
            btn = gr.Button("What changed?", variant="primary")
            result = gr.Markdown()
            with gr.Row():
                chart = gr.Plot(label="Scores for each kind of change")
                with gr.Column():
                    with gr.Row():
                        seen_b = gr.Image(label="Before, as the model sees it", height=200,
                                          interactive=False)
                        seen_a = gr.Image(label="After, as the model sees it", height=200,
                                          interactive=False)
                    gr.Markdown("*The model looks at a small 32 m × 32 m square in the centre "
                                "of each image (64 × 64 pixels).*")
            ex = load_json("examples/index.json") or []
            if ex:
                gr.Examples(
                    examples=[[str(HERE / "examples" / e["before"]), str(HERE / "examples" / e["after"])]
                              for e in ex[:6]],
                    inputs=[before, after], label="Or click a sample pair")
            btn.click(what_changed, [before, after], [result, chart, seen_b, seen_a])
        with gr.Tab("Examples"):
            gr.Markdown(EXAMPLES_NOTE)
            gr.HTML(examples_html)
        with gr.Tab("How good is it"):
            metrics_md = gr.Markdown()
            with gr.Row():
                f1_plot = gr.Plot(label="Score for each kind of change")
                cm_plot = gr.Plot(label="Where the model gets confused")
            demo.load(build_metrics_tab, None, [metrics_md, f1_plot, cm_plot])
            gr.Markdown("*Reading the grid: each row is what the point really was; the numbers "
                        "show what the model said. Numbers on the diagonal (top-left to "
                        "bottom-right) are correct answers.*")
        gr.HTML(FOOTER)
    return demo


if __name__ == "__main__":
    build_ui().launch(inbrowser=True)
