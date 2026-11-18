#!/usr/bin/env python3
"""Graphical abstract for the accompanying manuscript.

Typographic constraints enforced here:
  * width <= 1.2 columns  (1.2 x 239.003pt = 286.8pt = 3.983 in)
  * all text in Calibri; Carlito is the metric-compatible substitute
  * no mathtext, so no DejaVu fallback creeps into the embedded fonts
  * >= 300 dpi

Content is unchanged from the approved version: the schematic (what was
built and intervened on), the single dimensionless axis on which every
identity predicts unity, and which tests land there. Every number is
read from the run log.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FONT = "Carlito"
DPI  = 600
W, H = 3.98, 3.86          # 1.2 columns wide

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT, "Calibri", "Carlito"],
    "font.size": 6.4,
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT, "mathtext.it": FONT + ":italic",
    "mathtext.bf": FONT + ":bold",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.0,  "ytick.major.size": 2.0,
    "pdf.fonttype": 42,
})

BLUE = "#1f4e79"; RED = "#c0504d"; GREY = "0.45"

# ------------------------------------------------------------------ data
def boot(v, n=10000, seed=0):
    v = np.asarray(v, float); r = np.random.default_rng(seed)
    return tuple(np.quantile(
        r.choice(v, size=(n, len(v)), replace=True).mean(1), [.025, .975]))

df = pd.read_pickle("runs.pkl"); C = df[df.tag == "C"]
rho = {}
for b in sorted(C.beta.unique()):
    v = []
    for sd_ in sorted(C.seed.unique()):
        cp = C[(C.beta == b) & (C.seed == sd_) & (C.decay_variant == "coupled_l2")]
        dc = C[(C.beta == b) & (C.seed == sd_) & (C.decay_variant == "decoupled_wd")]
        v.append(dc.iloc[0].k_grok / cp.iloc[0].k_grok * (1 - b))
    rho[b] = (float(np.mean(v)),) + boot(v)

t4 = pd.read_csv("table4_p2.csv")
P2 = {k: (float(t4[c].mean()),) + boot(t4[c].to_numpy())
      for k, c in (("comp", "comp_ratio"), ("turn", "min_ratio"),
                   ("acc", "acc_ratio"))}

# ------------------------------------------------------------------ text
# Unicode only: no $...$, so every glyph comes from Carlito.
TXT = {
 "en": dict(
   head="Do exact grokking identities survive\na nonlinear network?",
   b1="modular addition, p = 17  ·  two-layer quadratic net\nP = 6528 parameters  ·  IEEE float64",
   b2t="(P4)  coupled L₂ → decoupled decay\npredicts × 1/(1−β)",
   b2b="(P2)  rescale grokking subspace by c\npredicts shift log c / r",
   b3="351 runs  ·  8 seeds  ·  3 moduli  ·  3 widths  ·  4 momenta",
   axttl="Every test reduces to one ratio: measured / predicted",
   band="±15% resolution, β = 0 null control",
   k_in="95% bootstrap interval contains unity",
   k_out="95% bootstrap interval excludes unity",
   rows=["(P4)  β = 0, null control", "(P4)  β = 0.5",
         "(P4)  β = 0.9", "(P4)  β = 0.95",
         "(P2)  component (theory)", "(P2)  turning point",
         "(P2)  test accuracy (behaviour)"],
   tail="Exact identities survive on the objects they define,\nbut not on the observable a practitioner measures."),
 "tr": dict(
   head="Kesin grokking özdeşlikleri doğrusal\nolmayan ağda yaşıyor mu?",
   b1="modüler toplama, p = 17  ·  iki katmanlı kuadratik ağ\nP = 6528 parametre  ·  IEEE float64",
   b2t="(P4)  bağlaşık L₂ → ayrışık sönüm\nöngörü: × 1/(1−β)",
   b2b="(P2)  grokking altuzayını c ile ölçekle\nöngörü: kayma log c / r",
   b3="351 koşum  ·  8 tohum  ·  3 modül  ·  3 genişlik  ·  4 momentum",
   axttl="Her sınama tek bir orana iner: ölçülen / öngörülen",
   band="±%15 çözünürlük, β = 0 boş kontrolü",
   k_in="%95 önyükleme aralığı birimi içeriyor",
   k_out="%95 önyükleme aralığı birimi dışlıyor",
   rows=["(P4)  β = 0, boş kontrol", "(P4)  β = 0.5",
         "(P4)  β = 0.9", "(P4)  β = 0.95",
         "(P2)  bileşen (kuram)", "(P2)  dönüm noktası",
         "(P2)  test doğruluğu (davranış)"],
   tail="Kesin özdeşlikler kendi tanımladıkları nesnelerde yaşıyor,\nuygulayıcının ölçtüğü gözlemlenebilirde yaşamıyor."),
}


def box(ax, x, y, w, h, text, fc="#eef2f7", ec=BLUE, fs=5.9, lw=0.7):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.02",
        linewidth=lw, edgecolor=ec, facecolor=fc, transform=ax.transAxes,
        clip_on=False, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fs, zorder=3, linespacing=1.30)


def varrow(ax, x, y0, y1):
    ax.add_patch(FancyArrowPatch(
        (x, y0), (x, y1), transform=ax.transAxes,
        arrowstyle="-|>,head_width=1.8,head_length=3.0",
        linewidth=0.8, color=GREY, clip_on=False, zorder=2,
        shrinkA=0, shrinkB=0))


def build(lang, t):
    fig = plt.figure(figsize=(W, H))

    # ---- headline -----------------------------------------------------
    a0 = fig.add_axes([0.02, 0.892, 0.96, 0.096]); a0.axis("off")
    a0.text(0.5, 0.5, t["head"], ha="center", va="center",
            fontsize=8.6, fontweight="bold", linespacing=1.25)

    # ---- what was built, what was predicted, how much was run ---------
    a1 = fig.add_axes([0.02, 0.560, 0.96, 0.320]); a1.axis("off")
    box(a1, 0.00, 0.800, 1.00, 0.190, t["b1"], fs=5.9)
    varrow(a1, 0.50, 0.778, 0.722)
    box(a1, 0.00, 0.330, 0.470, 0.370, t["b2t"], fc="#f7f1ee", ec=RED, fs=5.9)
    box(a1, 0.530, 0.330, 0.470, 0.370, t["b2b"], fc="#eef2f7", ec=BLUE, fs=5.9)
    varrow(a1, 0.50, 0.308, 0.252)
    box(a1, 0.00, 0.040, 1.00, 0.190, t["b3"], fc="0.95", ec="0.55", fs=5.9)

    # ---- the verdict on one dimensionless axis ------------------------
    a2 = fig.add_axes([0.345, 0.262, 0.635, 0.272])
    vals = [rho[0.0], rho[0.5], rho[0.9], rho[0.95],
            P2["comp"], P2["turn"], P2["acc"]]
    ys = np.arange(len(vals))[::-1]
    a2.axvspan(0.85, 1.15, color="0.88", zorder=0)
    a2.axvline(1.0, ls="--", c=GREY, lw=0.9, zorder=1)
    for y, (m, lo, hi) in zip(ys, vals):
        out = not (lo <= 1.0 <= hi)
        a2.errorbar([m], [y], xerr=[[m - lo], [hi - m]],
                    fmt="s" if out else "o", ms=3.2,
                    c=RED if out else BLUE, capsize=1.6, lw=0.8, zorder=3)
        lab = ("%.4f" % m) if 0.99 < m < 1.01 else (
              ("%.3f" % m) if m < 1.1 else ("%.2f" % m))
        a2.annotate(lab, (hi, y), xytext=(3.0, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=5.9,
                    color=RED if out else BLUE, zorder=4)
    a2.set_yticks(ys); a2.set_yticklabels(t["rows"], fontsize=6.2)
    a2.set_ylim(-0.62, len(vals) - 0.38)
    a2.set_xlim(0.10, 1.98); a2.set_xticks([0.25, 0.50, 0.75, 1.00, 1.25, 1.50])
    a2.set_xlabel(t["axttl"], fontsize=6.4, labelpad=1.8)
    a2.tick_params(axis="y", length=0, pad=1.2)
    a2.tick_params(axis="x", labelsize=6.0)
    for sp in ("top", "right", "left"):
        a2.spines[sp].set_visible(False)
    a2.grid(axis="x", alpha=0.22, lw=0.4)

    # ---- key, placed under the axis so it steals no plotting width ----
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    keys = [Line2D([], [], marker="o", ls="none", ms=3.2, c=BLUE, label=t["k_in"]),
            Line2D([], [], marker="s", ls="none", ms=3.2, c=RED,  label=t["k_out"]),
            Patch(facecolor="0.88", edgecolor="none", label=t["band"])]
    a2.legend(handles=keys, loc="upper left", bbox_to_anchor=(-0.525, -0.205),
              fontsize=5.6, frameon=False, handlelength=1.0,
              handletextpad=0.45, labelspacing=0.20, borderpad=0.0, ncol=1)

    fig.text(0.5, 0.014, t["tail"], ha="center", va="bottom", fontsize=6.9,
             style="italic", color=BLUE, linespacing=1.30)

    for ext in ("pdf", "png"):
        fig.savefig(f"graphical_abstract_{lang}.{ext}", dpi=DPI)
    plt.close(fig)
    print("written:", lang)


for k, v in TXT.items():
    build(k, v)
