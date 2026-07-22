"""Regenerate ActionABI figures 6-9.

All numbers are hardcoded report-derived summary statistics (final).
Run: python media/make_media.py
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from style import COLORS, apply_style, style_axis

matplotlib.use("Agg")  # headless, deterministic rendering (must precede figure creation)

HERE = os.path.dirname(os.path.abspath(__file__))


def _save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def _box(ax, xy, w, h, text, facecolor, edgecolor, textcolor=None, fontsize=9.6,
         fontweight="bold"):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.06",
                         linewidth=1.6, edgecolor=edgecolor, facecolor=facecolor,
                         zorder=3)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight,
            color=textcolor or COLORS["ink"], zorder=4)
    return (x, y, w, h)


def _arrow(ax, p0, p1, color, label=None, lw=2.0, rad=0.0, label_dy=0.1,
           label_color=None):
    # zorder above the boxes (3) and their text (4) so arrowheads are never hidden.
    arr = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=17,
                          color=color, lw=lw,
                          connectionstyle=f"arc3,rad={rad}", zorder=6)
    ax.add_patch(arr)
    if label:
        mx = (p0[0] + p1[0]) / 2
        my = (p0[1] + p1[1]) / 2 + label_dy
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=8.4,
                style="italic", color=label_color or "#4a5568", zorder=4)


# ---------------------------------------------------------------------------
# FIG 6 — forensic pipeline diagram
# ---------------------------------------------------------------------------
def fig6_pipeline():
    apply_style()
    fig, ax = plt.subplots(figsize=(14.5, 5))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 5)
    ax.axis("off")

    w, h = 2.5, 1.5
    y = 1.75
    col_x = [0.15, 3.1, 6.05]

    _box(ax, (col_x[0], y), w, h,
         "Trajectory evidence\n(provenance-preserving\nstate + action logs)",
         "#eef2f9", COLORS["pool"])
    _box(ax, (col_x[1], y), w, h,
         "Per-channel scoring\n(finite grammar:\n"
         "target - space - frame - perm\n- sign - scale - lag - gripper)",
         "#eef5f5", COLORS["grammar"], fontsize=8.6)
    _box(ax, (col_x[2], y), w, h,
         "Calibrated\nequivalence set\n(retain ties,\ndon't break them)",
         "#fbf3e0", COLORS["probe"])

    yc = y + h / 2
    _arrow(ax, (col_x[0] + w, yc), (col_x[1], yc), COLORS["ink"])
    _arrow(ax, (col_x[1] + w, yc), (col_x[2], yc), COLORS["ink"])

    # branch to two outcomes
    bx = 9.1
    # unique -> converter (top)
    y_top = 3.1
    y_bot = 0.45
    _box(ax, (bx, y_top), w, h,
         "Unique contract\n(every required field\nsupported by evidence)",
         "#eaf5ef", COLORS["oracle"], fontsize=9.0)
    _box(ax, (bx, y_bot), w, h,
         "Abstain\n(evidence insufficient)",
         "#fdeeee", COLORS["floor"], fontsize=9.4)

    # arrows from equivalence set to the two branches
    _arrow(ax, (col_x[2] + w, yc + 0.15), (bx, y_top + h / 2), COLORS["oracle"],
           rad=0.22, lw=2.0)
    _arrow(ax, (col_x[2] + w, yc - 0.15), (bx, y_bot + h / 2), COLORS["floor"],
           rad=-0.22, lw=2.0)

    # converter / refusal terminals
    cx = 12.2
    tw = 1.9
    _box(ax, (cx, y_top + 0.18), tw, h - 0.36, "Converter",
         "#eaf5ef", COLORS["oracle"], fontsize=9.6)
    _box(ax, (cx, y_bot + 0.18), tw, h - 0.36, "Refusal\n(no converter\nemitted)",
         "#fdeeee", COLORS["floor"], fontsize=8.8)
    _arrow(ax, (bx + w, y_top + h / 2), (cx, y_top + h / 2), COLORS["oracle"],
           "emit", lw=2.0)
    _arrow(ax, (bx + w, y_bot + h / 2), (cx, y_bot + h / 2), COLORS["floor"],
           lw=2.0)

    ax.set_title("ActionABI: evidence → verified converter, or honest refusal",
                 pad=14, y=1.0)
    ax.text(7.25, -0.15,
            "Calibrated abstention: never certifies a unique answer the data does not support.",
            ha="center", va="top", fontsize=9.5, color="#4a5568")
    _save(fig, "diagram_pipeline.png")


# ---------------------------------------------------------------------------
# FIG 7 — Hub audit
# ---------------------------------------------------------------------------
def fig7_hub_audit():
    # Geometry tuned for a full-width (figure*) placement in the two-column
    # paper: compact height, print-legible fonts. Numbers unchanged.
    apply_style()
    rows = [
        ("Honest abstention\n(knowable field)", 115, "learned"),
        ("Unique certification\n(documentation-correct)", 5, "oracle"),
        ("Partial / flagged\ndiscrepancy", 1, "probe"),
        ("Contradiction\n(asserts what docs refute)", 0, "floor"),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [COLORS[r[2]] for r in rows]

    fig, ax = plt.subplots(figsize=(9.4, 3.5))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, height=0.62, color=colors, edgecolor=COLORS["ink"],
            linewidth=0.8, zorder=3)
    for yi, v in zip(y, vals, strict=False):
        if v == 0:
            ax.annotate("0", (1.0, yi), ha="left", va="center",
                        fontsize=13, fontweight="bold", color=COLORS["floor"])
        else:
            ax.annotate(str(v), (v + 1.5, yi), ha="left", va="center",
                        fontsize=12, fontweight="bold", color=COLORS["ink"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, 128)
    ax.set_xlabel("Count of documented field-labels", fontsize=11)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_title("35-dataset LeRobot Hub audit: 139 field-labels, zero contradictions",
                 fontsize=13, pad=10)
    ax.text(35, y[3], "0 contradictions", fontsize=12.5, fontweight="bold",
            color=COLORS["floor"], va="center", ha="left")
    ax.text(35, y[1], "5 / 5 unique certs documentation-correct", fontsize=11.5,
            fontweight="bold", color=COLORS["oracle"], va="center", ha="left")
    style_axis(ax, grid_axis="x")
    _save(fig, "chart_hub_audit.png")


# ---------------------------------------------------------------------------
# FIG 8 — calibration eliminates false uniques
# ---------------------------------------------------------------------------
def fig8_calibration():
    apply_style()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.5, 5.4))

    # --- (a) synthetic ambiguity gate ---
    methods = ["Forced\nresidual argmin", "ActionABI\ncalibrated set"]
    false_uniques = [25, 0]
    coverage = [1.00, 1.00]
    abstention = [0.00, 0.25]
    x = np.arange(len(methods))
    w = 0.26
    axa.bar(x - w, false_uniques, w, color=[COLORS["floor"], COLORS["oracle"]],
            edgecolor=COLORS["ink"], linewidth=0.8, zorder=3, label="False uniques")
    # coverage & abstention on a secondary [0,1] visual, scaled to count axis (x25)
    axa.bar(x, [c * 25 for c in coverage], w, color=COLORS["pool"], alpha=0.35,
            edgecolor=COLORS["ink"], linewidth=0.6, zorder=3,
            label="Equivalence coverage (x25)")
    axa.bar(x + w, [a * 25 for a in abstention], w, color=COLORS["probe"], alpha=0.6,
            edgecolor=COLORS["ink"], linewidth=0.6, zorder=3,
            label="Abstention rate (x25)")

    for xi, v in zip(x - w, false_uniques, strict=False):
        axa.annotate(str(v), (xi, v + 0.6), ha="center", va="bottom",
                     fontsize=11, fontweight="bold",
                     color=COLORS["floor"] if v > 0 else COLORS["oracle"])
    for xi, c in zip(x, coverage, strict=False):
        axa.annotate(f"{c:.2f}", (xi, c * 25 + 0.6), ha="center", va="bottom",
                     fontsize=9, color=COLORS["ink"])
    for xi, a in zip(x + w, abstention, strict=False):
        axa.annotate(f"{a:.2f}", (xi, a * 25 + 0.6), ha="center", va="bottom",
                     fontsize=9, color=COLORS["ink"])

    axa.set_xticks(x)
    axa.set_xticklabels(methods)
    axa.set_ylim(0, 33)
    axa.set_ylabel("False uniques (count)  /  rate x25")
    axa.set_title("(a) Synthetic ambiguity gate\n100 cases, 25 observationally-equivalent",
                  fontsize=12)
    # Single-row legend floated in the headroom band above every bar (bars top out at 25).
    axa.legend(loc="upper center", ncol=3, fontsize=8.2, columnspacing=1.0,
               handletextpad=0.4, borderaxespad=0.3)
    style_axis(axa)

    # --- (b) labeled-simulation bridge ---
    stages = ["Pre-fix", "Post-fix"]
    fu = [4, 0]
    cov = [0.02, 0.39]
    x2 = np.arange(len(stages))
    w2 = 0.32
    axb.bar(x2 - w2 / 2, fu, w2, color=[COLORS["floor"], COLORS["oracle"]],
            edgecolor=COLORS["ink"], linewidth=0.8, zorder=3, label="False uniques")
    axb.bar(x2 + w2 / 2, [c * 10 for c in cov], w2, color=COLORS["grammar"],
            alpha=0.55, edgecolor=COLORS["ink"], linewidth=0.6, zorder=3,
            label="Truth equiv-set coverage (x10)")

    for xi, v in zip(x2 - w2 / 2, fu, strict=False):
        axb.annotate(str(v), (xi, v + 0.1), ha="center", va="bottom",
                     fontsize=11, fontweight="bold",
                     color=COLORS["floor"] if v > 0 else COLORS["oracle"])
    for xi, c in zip(x2 + w2 / 2, cov, strict=False):
        axb.annotate(f"{c:.2f}", (xi, c * 10 + 0.1), ha="center", va="bottom",
                     fontsize=9.5, color=COLORS["ink"])

    axb.set_xticks(x2)
    axb.set_xticklabels(stages)
    axb.set_ylim(0, 6.0)
    axb.set_ylabel("False uniques (count)  /  coverage x10")
    axb.set_title("(b) Labeled-simulation bridge\n90 real-sim traces", fontsize=12)
    # Single-row legend in the headroom above the bars (bars top out at 4).
    axb.legend(loc="upper center", ncol=2, fontsize=8.2, columnspacing=1.0,
               handletextpad=0.4, borderaxespad=0.3)
    style_axis(axb)

    fig.suptitle("Calibrated abstention eliminates false uniques", fontsize=15,
                 fontweight="bold", y=1.03)
    fig.text(0.5, -0.06,
             "Forced argmin turns all 25 constructed equivalence cases into false certifications; "
             "ActionABI abstains on exactly those and\nproduces zero false uniques while retaining "
             "full coverage (seed 20260718).",
             ha="center", va="top", fontsize=9.3, color="#4a5568")
    fig.tight_layout()
    _save(fig, "chart_calibration.png")


# ---------------------------------------------------------------------------
# FIG 9 — gripper convention disagreement
# ---------------------------------------------------------------------------
def fig9_gripper_conventions():
    # Geometry tuned for full-width (figure*) placement: print-legible fonts,
    # compact height. Numbers/content unchanged.
    apply_style()
    rows = [
        ("ALOHA\n(arXiv:2304.13705)", "0 = closed,  1 = open", "continuous [0,1]", +1),
        ("Open X-Embodiment / RT-1 /\nBerkeley-UR5 (arXiv:2310.08864)",
         "1 = close,  -1 = open,  0 = no-change", "trinary", -1),
        ("DROID\n(arXiv:2403.12945)", "7th dim = gripper delta", "continuous delta", 0),
        ("Stanford HYDRA\n(arXiv:2306.17237)", "binary close-gripper", "binary", 0),
        ("LeRobot taxonomy doc", "left per-dataset (unspecified)", "unspecified", 0),
    ]
    fig, ax = plt.subplots(figsize=(9.4, 3.9))
    ax.set_xlim(-1.55, 2.05)
    ax.set_ylim(-0.55, len(rows) - 0.4)
    ax.axis("off")

    n = len(rows)
    for i, (eco, enc, kind, pol) in enumerate(rows):
        y = n - 1 - i
        ax.text(-1.52, y, eco, ha="left", va="center", fontsize=10.5,
                fontweight="bold", color=COLORS["ink"])
        if pol == +1:
            chip_c, chip_txt = COLORS["oracle"], "value 1  =  OPEN"
        elif pol == -1:
            chip_c, chip_txt = COLORS["floor"], "value 1  =  CLOSE"
        else:
            chip_c, chip_txt = COLORS["fixed"], "polarity n/a"
        ax.add_patch(FancyBboxPatch((0.02, y - 0.28), 0.70, 0.56,
                     boxstyle="round,pad=0.01,rounding_size=0.05",
                     linewidth=1.2, edgecolor=chip_c, facecolor=chip_c,
                     alpha=0.18, zorder=2))
        ax.text(0.37, y, chip_txt, ha="center", va="center", fontsize=9.6,
                fontweight="bold", color=chip_c, zorder=3)
        ax.text(0.84, y + 0.13, enc, ha="left", va="center", fontsize=10.2,
                color=COLORS["ink"])
        ax.text(0.84, y - 0.17, kind, ha="left", va="center", fontsize=8.8,
                style="italic", color="#4a5568")
        if i < n - 1:
            ax.axhline(y - 0.5, color=COLORS["grid"], linewidth=0.8, zorder=1)

    ax.annotate("", xy=(0.37, n - 1 - 0.33), xytext=(0.37, n - 2 + 0.33),
                arrowprops=dict(arrowstyle="<->", color=COLORS["ink"], lw=1.7))
    ax.text(-0.02, n - 1.5, "POLARITY\nFLIP", ha="right", va="center",
            fontsize=9.2, fontweight="bold", color=COLORS["ink"],
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#fff4d6",
                      edgecolor=COLORS["probe"], linewidth=1.0))
    ax.set_title("Robot ecosystems actively disagree on gripper polarity",
                 fontsize=13, pad=12, loc="left", x=-0.0)
    ax.text(-1.55, -0.52,
            "Same numeric channel, opposite physical meaning across two actively-used "
            "dataset families: ALOHA (0=closed/1=open) vs OXE (1=close/-1=open/0=no-change).",
            ha="left", va="top", fontsize=9.2, color="#4a5568")
    _save(fig, "chart_gripper_conventions.png")


# ---------------------------------------------------------------------------
# RECOVERY GIF — real evidence accumulation on a labeled ManiSkill bridge trace
# ---------------------------------------------------------------------------
# Data (media/recovery_trace.json) is genuine ActionABI output: the real C++
# scorer's held-out Huber residuals (Python parity-gated at 5e-17) over growing
# evidence prefixes of a real bridge trace, with the calibrated equivalence set
# computed over the confusable candidate pool at each evidence level. Rendered
# here from that committed JSON so the animation needs no rebuild step.
_CH_ROWS = [
    ("target", "consensus"),
    ("space", "fixed"),
    ("frame", "unobs"),
    ("permutation", "consensus"),
    ("sign", "consensus"),
    ("scale", "consensus"),
    ("lag", "consensus"),
    ("gripper", "unobs"),
]


def _channel_state(kind, value):
    """Return (fill_fraction, bar_color, status_text, status_color)."""
    if kind == "fixed":
        return 1.0, COLORS["fixed"], "fixed: cartesian", COLORS["fixed"]
    if kind == "unobs":
        return 0.0, COLORS["fixed"], "unobservable (structural)", COLORS["fixed"]
    if value >= 0.999:
        return value, COLORS["oracle"], "RESOLVED", COLORS["oracle"]
    if value >= 0.85:
        return value, COLORS["pool"], "constrained", COLORS["pool"]
    return value, COLORS["probe"], "ambiguous", COLORS["probe"]


def _fig_to_rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return buf[..., :3].copy()


def _recovery_frame(data, idx, *, final):
    from matplotlib.patches import FancyBboxPatch, Rectangle

    fr = data["frames"][idx]
    total = data["total_episodes"]
    k = fr["k_episodes"]
    eq = fr["eq_size"]

    fig = plt.figure(figsize=(9.6, 5.4), dpi=140)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ---- title ----
    ax.text(0.045, 0.955, "ActionABI — recovering a hidden action contract from a real "
            "ManiSkill bridge trace", fontsize=13, fontweight="bold",
            color=COLORS["ink"], va="center")
    ax.text(0.045, 0.905, f"trace {data['trace_id']} ({data['excitation']} excitation)  ·  "
            "real C++ held-out Huber residuals (parity-gated)  ·  calibrated equivalence "
            "set over the confusable candidate pool", fontsize=8.4, color="#4a5568",
            va="center")

    # ---- left panel: per-channel resolution meters ----
    ax.text(0.045, 0.83, "GRAMMAR CHANNELS — evidence resolution", fontsize=10,
            fontweight="bold", color=COLORS["ink"], va="center")
    top, bot = 0.755, 0.30
    step = (top - bot) / (len(_CH_ROWS) - 1)
    mx0, mw = 0.205, 0.205
    for i, (name, kind) in enumerate(_CH_ROWS):
        y = top - i * step
        value = fr["consensus"].get(name, 0.0) if kind == "consensus" else 0.0
        fill, bar_c, status, status_c = _channel_state(kind, value)
        ax.text(0.05, y, name, fontsize=10.5, family="monospace",
                color=COLORS["ink"], va="center")
        ax.add_patch(Rectangle((mx0, y - 0.017), mw, 0.034, facecolor=COLORS["grid"],
                               edgecolor="none", zorder=2))
        if kind == "unobs":
            ax.add_patch(Rectangle((mx0, y - 0.017), mw, 0.034, facecolor="none",
                                   edgecolor=COLORS["fixed"], hatch="////",
                                   linewidth=0.0, alpha=0.45, zorder=3))
        else:
            ax.add_patch(Rectangle((mx0, y - 0.017), mw * fill, 0.034, facecolor=bar_c,
                                   edgecolor="none", zorder=3))
        if kind == "consensus":
            ax.text(mx0 + mw + 0.012, y, f"{value * 100:.0f}%", fontsize=9,
                    color=COLORS["ink"], va="center", fontweight="bold")
        ax.text(0.475, y, status, fontsize=9, color=status_c, va="center",
                fontweight="bold" if status == "RESOLVED" else "normal")

    # ---- right panel: evidence + equivalence-set gauge ----
    rx = 0.635
    ax.add_patch(FancyBboxPatch((rx - 0.01, 0.30), 0.365, 0.52,
                                boxstyle="round,pad=0.006,rounding_size=0.02",
                                linewidth=1.3, edgecolor=COLORS["grid"],
                                facecolor="#f7fafc", zorder=1))
    ax.text(rx + 0.17, 0.785, "EVIDENCE ACCUMULATED", fontsize=9.5, fontweight="bold",
            color=COLORS["ink"], va="center", ha="center")
    ax.text(rx + 0.17, 0.735, f"{k} / {total} episodes", fontsize=15, fontweight="bold",
            color=COLORS["grammar"], va="center", ha="center")
    ax.add_patch(Rectangle((rx + 0.02, 0.688), 0.30, 0.022, facecolor=COLORS["grid"],
                           edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((rx + 0.02, 0.688), 0.30 * k / total, 0.022,
                           facecolor=COLORS["grammar"], edgecolor="none", zorder=3))

    ax.text(rx + 0.17, 0.60, "EQUIVALENCE SET", fontsize=9.5, fontweight="bold",
            color=COLORS["ink"], va="center", ha="center")
    ax.text(rx + 0.17, 0.505, f"{eq}", fontsize=44, fontweight="bold",
            color=COLORS["floor"] if final else COLORS["probe"], va="center", ha="center")
    ax.text(rx + 0.17, 0.415, "contracts remain\nobservationally equivalent",
            fontsize=8.8, color="#4a5568", va="center", ha="center")
    retained = fr["truth_in_set"]
    ax.text(rx + 0.17, 0.345, ("✓ true contract retained (coverage)" if retained
            else "✗ truth outside set this frame"), fontsize=8.6,
            color=COLORS["oracle"] if retained else COLORS["floor"],
            va="center", ha="center", fontweight="bold")

    # ---- verdict / caption strip ----
    verdict = "ABSTAIN" if final else "ANALYSING"
    v_color = COLORS["floor"] if final else COLORS["probe"]
    ax.add_patch(FancyBboxPatch((0.045, 0.05), 0.91, 0.135,
                                boxstyle="round,pad=0.004,rounding_size=0.02",
                                linewidth=1.6, edgecolor=v_color,
                                facecolor="#fdeeee" if final else "#fdf6e9", zorder=1))
    ax.add_patch(FancyBboxPatch((0.062, 0.075), 0.145, 0.085,
                                boxstyle="round,pad=0.004,rounding_size=0.02",
                                linewidth=0, facecolor=v_color, zorder=2))
    ax.text(0.1345, 0.1175, verdict, fontsize=13, fontweight="bold", color="white",
            va="center", ha="center", zorder=3)
    if final:
        msg = (f"{eq} full contracts remain jointly consistent with the evidence — "
               "ActionABI emits NO converter.\nHigh per-channel confidence is not a unique "
               "joint contract — it won't certify beyond the data.")
    else:
        msg = ("Accumulating held-out evidence — the target channel is pinned, but "
               "scale / lag stay ambiguous and\ngripper / frame are unobservable, so no "
               "unique contract has emerged yet.")
    ax.text(0.222, 0.1175, msg, fontsize=9.0, color=COLORS["ink"], va="center",
            ha="left", zorder=3)

    rgb = _fig_to_rgb(fig)
    plt.close(fig)
    return rgb


def build_recovery_gif():
    import json

    from PIL import Image

    with open(os.path.join(HERE, "recovery_trace.json"), encoding="utf-8") as handle:
        data = json.load(handle)
    n = len(data["frames"])
    seq, durations = [], []
    for idx in range(n):
        final = idx == n - 1
        seq.append(Image.fromarray(_recovery_frame(data, idx, final=final)))
        # linger on the final honest-abstention verdict so it is readable
        durations.append(3600 if final else 620)
    frames = [im.convert("P", palette=Image.ADAPTIVE, colors=128) for im in seq]
    out = os.path.join(HERE, "recovery_demo.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True)
    print(f"wrote {out} ({n} evidence frames, {os.path.getsize(out) // 1024} KiB)")


# ---------------------------------------------------------------------------
# FIG (anatomy) — per-field equivalence set decision flow (compact full-width)
# ---------------------------------------------------------------------------
def fig_anatomy():
    """Anatomy-of-a-per-field-equivalence-set flowchart.

    Rebuilt with a *measure-then-box* layout: no text is ever placed into a
    pre-sized box. Each text block is rendered first, its true extent is
    measured with ``get_window_extent`` (converted display-px -> inches via the
    figure dpi, so the size is figure-scale-independent), and the surrounding
    ``FancyBboxPatch`` is derived from that measured extent plus generous
    padding. Boxes are then placed on a strict grid computed from the measured
    heights + fixed gaps; branch rows split their content region into equal
    columns with real gutters. All connectors are straight vertical segments
    between box edge midpoints (children always lie under their parent bar's
    x-span), so no arrow crosses a box and none bends. Edge labels get their
    own white bounding boxes offset into the empty gutter beside the arrow.

    Writes both the PDF (used by the LaTeX build) and a PNG (actionabi.md).
    Idempotent: layout is fully derived from the text metrics on every run.
    """
    apply_style()
    ink = COLORS["ink"]
    DPI = 200

    # ---- absolute layout constants (inches) --------------------------------
    PAD_X = 0.20          # horizontal text->box padding (~14.4 pt >= 8 pt)
    PAD_Y = 0.15          # vertical   text->box padding (~10.8 pt >= 8 pt)
    GUT3 = 0.55           # min gutter between the three branch columns
    GUT2 = 1.30           # min gutter between two-branch columns
    GAP = 0.42            # vertical gap between adjacent rows
    GAP_LBL = 0.78        # taller gap for rows whose split carries edge labels
    MARGIN = 0.34         # outer margin around the content region
    TITLE_GAP = 0.34      # gap under the title band

    # ---- node definitions ---------------------------------------------------
    # Each node: (key, text, face, edge, textcolor, fontsize, fontweight)
    def N(key, text, face, edge, tc, fs, fw="bold"):
        return dict(key=key, text=text, face=face, edge=edge, tc=tc, fs=fs, fw=fw)

    nodes = {
        "L1": N("L1",
                r"Per field: argmin hypothesis $h^\star$  vs.  "
                r"single-field-neighbour pool $\{h\}$",
                "#eef2f9", COLORS["pool"], ink, 13.0),
        "L2": N("L2",
                r"Held-out residual rows   $r = $ command $-$ observable"
                r"   (episode-disjoint split)",
                "#eef5f5", COLORS["grammar"], ink, 13.0),
        "L3a": N("L3a",
                 "Paired bootstrap over held-out rows\n"
                 r"per-row loss gap $g = L(h) - L(h^\star)$" "\n"
                 r"$\Rightarrow$ 95% CI of mean gap",
                 "#eef2f9", COLORS["pool"], ink, 11.5, "normal"),
        "L3b": N("L3b",
                 "Bias-robust guard\n"
                 r"slack $s$: mean $+$ command-slope" "\n"
                 "vs. noise floor",
                 "#fbf3e0", COLORS["probe"], ink, 11.5, "normal"),
        "L4": N("L4",
                r"Retain $h$ in equivalence set   $\Leftrightarrow$   "
                r"CI of $(g - s)$ includes 0" "\n"
                "(fail-closed: widen, never split on bias)",
                "#f2f4f8", COLORS["fixed"], ink, 12.5),
        "L5a": N("L5a",
                 r"Equivalence set $= \{h^\star\}$" "\n"
                 "(all competitors strictly\nworse, not misspecified)",
                 "#eaf5ef", COLORS["oracle"], ink, 11.0, "normal"),
        "L5b": N("L5b",
                 r"Equivalence set $> 1$" "\n"
                 r"(some $h$ ties $h^\star$ within CI,""\n"
                 "or guard flags misspec.)",
                 "#fdf6e9", COLORS["probe"], ink, 11.0, "normal"),
        "L5c": N("L5c",
                 "Field invariant to the\nobservable (no discriminating\n"
                 "evidence at any level)",
                 "#eef1f5", COLORS["fixed"], ink, 11.0, "normal"),
        "L6a": N("L6a", "SINGLETON\nCERTIFIED",
                 COLORS["oracle"], COLORS["oracle"], "white", 12.0),
        "L6b": N("L6b", r"TIE RETAINED" "\n" r"$\rightarrow$ ABSTAIN",
                 COLORS["probe"], COLORS["probe"], "white", 12.0),
        "L6c": N("L6c", "UNSUPPORTED\n(not guessed)",
                 COLORS["fixed"], COLORS["fixed"], "white", 12.0),
        "L7": N("L7",
                r"Converter gate:  emit converter  $\Leftrightarrow$  "
                "every REQUIRED field is\n"
                "CERTIFIED or safely DEFAULTABLE",
                "#eef2f9", COLORS["pool"], ink, 12.5),
        "L8a": N("L8a", "VERIFIED CONVERTER\nprovenance-preserving code",
                 COLORS["oracle"], COLORS["oracle"], "white", 12.0),
        "L8b": N("L8b", "HONEST REFUSAL\nno converter emitted",
                 COLORS["floor"], COLORS["floor"], "white", 12.0),
    }

    # Rows, top -> bottom. "full" bars span the whole content width; branch
    # rows list their member columns left -> right.
    rows = [
        ("full", ["L1"]),
        ("full", ["L2"]),
        ("branch", ["L3a", "L3b"]),
        ("full", ["L4"]),
        ("branch", ["L5a", "L5b", "L5c"]),
        ("branch", ["L6a", "L6b", "L6c"]),
        ("full", ["L7"]),
        ("branch", ["L8a", "L8b"]),
    ]

    # ---- PASS 1: measure every text block in inches (dpi-independent) -------
    fig = plt.figure(figsize=(12.0, 12.0), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 12.0)
    tmp = {}
    for n in nodes.values():
        tmp[n["key"]] = ax.text(0.5, 0.5, n["text"], ha="center", va="center",
                                fontsize=n["fs"], fontweight=n["fw"],
                                linespacing=1.32)
    title_txt = ax.text(0.5, 0.5, "Anatomy of a per-field equivalence set",
                        ha="center", va="center", fontsize=17,
                        fontweight="bold")
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()

    def measure(t):
        bb = t.get_window_extent(rend)
        return bb.width / DPI, bb.height / DPI

    tw, th = {}, {}
    for k, t in tmp.items():
        tw[k], th[k] = measure(t)
        t.remove()
    title_w, title_h = measure(title_txt)
    title_txt.remove()

    # box sizes derived strictly from measured text + padding
    bw = {k: tw[k] + 2 * PAD_X for k in nodes}
    bh = {k: th[k] + 2 * PAD_Y for k in nodes}

    # ---- column widths from the widest member of each branch family --------
    col3_w = max(bw[k] for k in ("L5a", "L5b", "L5c", "L6a", "L6b", "L6c"))
    col2_w = max(bw[k] for k in ("L3a", "L3b", "L8a", "L8b"))
    full_w = max(bw[k] for k in ("L1", "L2", "L4", "L7"))

    # content width = widest binding constraint across all row types
    content_w = max(3 * col3_w + 2 * GUT3,
                    2 * col2_w + GUT2,
                    full_w,
                    title_w)

    W = content_w + 2 * MARGIN
    x_left = MARGIN
    cx = W / 2.0

    # three equal columns filling content_w (equal real gutters)
    gut3 = (content_w - 3 * col3_w) / 2.0
    c3x = [x_left + col3_w / 2 + i * (col3_w + gut3) for i in range(3)]
    # two columns hugging the outer thirds with a wide central gutter
    c2x = [x_left + col2_w / 2, x_left + content_w - col2_w / 2]

    def col_centers(kind, members):
        if kind == "full":
            return [cx]
        return c3x if len(members) == 3 else c2x

    # ---- PASS 2: assign row y-centers from measured heights + gaps ---------
    row_h = [max(bh[k] for k in members) for _, members in rows]
    # a row gets the taller label-gap ABOVE it when its parent split is labelled
    label_gap_above = {4: True, 7: True}  # row index of L5* and L8* rows
    total_h = MARGIN + title_h + TITLE_GAP
    for i, h in enumerate(row_h):
        total_h += h
        if i < len(rows) - 1:
            total_h += GAP_LBL if label_gap_above.get(i + 1) else GAP
    total_h += MARGIN
    H = total_h

    y_cursor = H - MARGIN
    title_cy = y_cursor - title_h / 2
    y_cursor -= title_h + TITLE_GAP
    row_cy = []
    for i, h in enumerate(row_h):
        row_cy.append(y_cursor - h / 2)
        y_cursor -= h
        if i < len(rows) - 1:
            y_cursor -= GAP_LBL if label_gap_above.get(i + 1) else GAP

    # store geometry per node: center + half-sizes
    geo = {}
    for (kind, members), cy_row in zip(rows, row_cy):
        centers = col_centers(kind, members)
        for k, xc in zip(members, centers):
            geo[k] = dict(cx=xc, cy=cy_row, w=bw[k], h=bh[k])

    # ---- PASS 3: draw final figure -----------------------------------------
    fig.set_size_inches(W, H)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)

    def draw_box(k):
        g = geo[k]
        n = nodes[k]
        x, y = g["cx"] - g["w"] / 2, g["cy"] - g["h"] / 2
        ax.add_patch(FancyBboxPatch((x, y), g["w"], g["h"],
                     boxstyle="round,pad=0,rounding_size=0.10",
                     linewidth=1.8, edgecolor=n["edge"], facecolor=n["face"],
                     zorder=3, mutation_aspect=1.0))
        ax.text(g["cx"], g["cy"], n["text"], ha="center", va="center",
                fontsize=n["fs"], fontweight=n["fw"], color=n["tc"],
                zorder=4, linespacing=1.32)

    for k in nodes:
        draw_box(k)

    def top(k):
        g = geo[k]
        return g["cy"] + g["h"] / 2

    def bot(k):
        g = geo[k]
        return g["cy"] - g["h"] / 2

    def arrow(parent, child, color, lw=2.0):
        """Straight vertical connector at the child's x (always within the
        parent bar's x-span), from parent's bottom edge to child's top edge."""
        xc = geo[child]["cx"]
        p0 = (xc, bot(parent))
        p1 = (xc, top(child))
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>",
                     mutation_scale=16, color=color, lw=lw,
                     shrinkA=0, shrinkB=0, zorder=2))
        return p0, p1

    # connector wiring (all vertical, no crossings)
    arrow("L1", "L2", ink)
    arrow("L2", "L3a", COLORS["pool"])
    arrow("L2", "L3b", COLORS["probe"])
    arrow("L3a", "L4", COLORS["pool"])
    arrow("L3b", "L4", COLORS["probe"])
    a5a = arrow("L4", "L5a", COLORS["oracle"])
    a5b = arrow("L4", "L5b", COLORS["probe"])
    a5c = arrow("L4", "L5c", COLORS["fixed"])
    arrow("L5a", "L6a", COLORS["oracle"])
    arrow("L5b", "L6b", COLORS["probe"])
    arrow("L5c", "L6c", COLORS["fixed"])
    arrow("L6a", "L7", COLORS["oracle"])
    arrow("L6b", "L7", COLORS["probe"])
    arrow("L6c", "L7", COLORS["fixed"])
    a8a = arrow("L7", "L8a", COLORS["oracle"])
    a8b = arrow("L7", "L8b", COLORS["floor"])

    # ---- edge labels: white bbox, offset BESIDE the arrow into clear space --
    placed_labels = []  # (x0, y0, x1, y1) data-inch extents, for verification

    def elabel(text, color, arrow_seg, side):
        """Place a label beside a vertical arrow. `side`=+1 right, -1 left.
        Returns nothing; records its extent for the intersection check."""
        (x0, y0), (x1, y1) = arrow_seg
        ymid = (y0 + y1) / 2
        # provisional gap between the arrow line and the label edge
        clear = 0.10
        t = ax.text(x1 + side * clear, ymid, text,
                    ha="left" if side > 0 else "right", va="center",
                    fontsize=10.0, style="italic", fontweight="bold",
                    color=color, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                              edgecolor=color, linewidth=0.8, alpha=1.0))
        return t

    lbl_specs = [
        ("CI excludes 0", COLORS["oracle"], a5a, +1),
        ("CI includes 0", COLORS["probe"], a5b, +1),
        (r"gap $\equiv$ 0", COLORS["fixed"], a5c, -1),
        ("all fields OK", COLORS["oracle"], a8a, +1),
        ("any field ambiguous", COLORS["floor"], a8b, -1),
    ]
    label_txts = [(elabel(t, c, seg, s), seg) for t, c, seg, s in lbl_specs]

    # title (measured, box-less band at top so nothing can overflow)
    ax.text(cx, title_cy, "Anatomy of a per-field equivalence set",
            ha="center", va="center", fontsize=17, fontweight="bold",
            color=ink)

    # ---- VERIFY: assert no label bbox intersects any arrow or box ----------
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def extent_data(artist):
        bb = artist.get_window_extent(rend)
        (x0, y0) = inv.transform((bb.x0, bb.y0))
        (x1, y1) = inv.transform((bb.x1, bb.y1))
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    def seg_hits_rect(seg, rect, margin=0.03):
        (sx0, sy0), (sx1, sy1) = seg
        rx0, ry0, rx1, ry1 = rect
        # vertical segment at x=sx0
        if sx0 < rx0 - margin or sx0 > rx1 + margin:
            return False
        lo, hi = min(sy0, sy1), max(sy0, sy1)
        return not (hi < ry0 - margin or lo > ry1 + margin)

    problems = []
    box_rects = [(geo[k]["cx"] - geo[k]["w"] / 2, geo[k]["cy"] - geo[k]["h"] / 2,
                  geo[k]["cx"] + geo[k]["w"] / 2, geo[k]["cy"] + geo[k]["h"] / 2)
                 for k in nodes]
    arrow_segs = [a5a, a5b, a5c, a8a, a8b]
    for t, own_seg in label_txts:
        rect = extent_data(t)
        for seg in arrow_segs:
            if seg_hits_rect(seg, rect):
                problems.append((t.get_text(), "arrow"))
        for br in box_rects:
            # overlap test between label rect and box rect
            if not (rect[2] < br[0] or rect[0] > br[2]
                    or rect[3] < br[1] or rect[1] > br[3]):
                problems.append((t.get_text(), "box"))
    if problems:
        print(f"  [fig_anatomy] WARNING label collisions: {problems}")
    else:
        print("  [fig_anatomy] verify OK: no label/arrow/box collisions")

    for name in ("diagram_equivalence_anatomy.pdf", "diagram_equivalence_anatomy.png"):
        path = os.path.join(HERE, name)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.12, dpi=DPI,
                    facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


# ===========================================================================
# PAPER FIGURES (serif, vector PDF, Okabe-Ito) — designed to sit natively next
# to the LaTeX manuscript's other floats, NOT as README/demo dashboard cards.
# ===========================================================================

# Okabe-Ito colour-blind-safe qualitative palette (used ONLY by the paper figs).
OI = {
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#111111",
    "gray": "#7a7a7a",
    "lightgray": "#e4e4e4",
}


def _paper_style():
    """Serif rcParams so figure text blends with the manuscript body font.

    Uses a Times-like serif (Nimbus Roman) with STIX mathtext, and embeds real
    (Type-42) fonts in the PDF so text stays vector/selectable at print scale.
    """
    matplotlib.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "STIXGeneral",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "axes.labelcolor": OI["black"],
        "axes.titlecolor": OI["black"],
        "text.color": OI["black"],
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
        "legend.fontsize": 8.0,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# Discrete per-field resolution states (Fig. evidence-accumulation timeline).
_STATE_COLOR = {
    "resolved": OI["green"],
    "constrained": OI["blue"],
    "ambiguous": OI["orange"],
    "fixed": OI["gray"],
    "unobservable": OI["lightgray"],
}


def _data_state(v):
    """Map a per-channel consensus fraction to a discrete resolution state.

    Thresholds match the tool's own reporting convention (media/make_media.py
    recovery renderer): >=0.999 uniquely resolved, >=0.85 constrained, else
    ambiguous. Every threshold is applied to a value read from the trace JSON.
    """
    if v >= 0.999:
        return "resolved"
    if v >= 0.85:
        return "constrained"
    return "ambiguous"


def fig_evidence_paper():
    """Figure 4 — evidence accumulation on labeled policy trace c029_policy.

    Three panels, every plotted value read from media/recovery_trace.json:
      (a) observationally-equivalent set size vs. episodes, with the four
          caption checkpoints and the honest truth-exit at k=8 marked;
      (b) per-field resolution-state timeline (fields x episodes);
      (c) per-channel confidence at the final (16-episode) evidence level,
          with the abstention verdict.
    Writes vector PDF (for LaTeX) + PNG (for the .md).
    """
    import json

    from matplotlib.patches import Rectangle

    _paper_style()
    with open(os.path.join(HERE, "recovery_trace.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    frames = data["frames"]
    k = [f["k_episodes"] for f in frames]
    eq = [f["eq_size"] for f in frames]
    misspec = [f["misspec"] for f in frames]
    truth_in = [f["truth_in_set"] for f in frames]

    # Data-derived channels (consensus in the trace) + structural channels
    # (constant facts stated in the manuscript, not per-episode measurements).
    data_channels = ["target", "sign", "permutation", "lag", "scale"]
    struct_channels = [("space", "fixed"), ("frame", "unobservable"),
                       ("gripper", "unobservable")]
    total = data["total_episodes"]

    fig = plt.figure(figsize=(8.6, 4.35))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.94],
                          width_ratios=[1.32, 1.0], hspace=0.52, wspace=0.26)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, :])

    # ---- (a) equivalence-set size vs episodes -----------------------------
    ax_a.plot(k, eq, "-", color="#444444", lw=1.6, zorder=2)
    for ki, ei, ti in zip(k, eq, truth_in, strict=False):
        if ti:
            ax_a.plot(ki, ei, "o", ms=5.2, color=OI["blue"],
                      mec="white", mew=0.7, zorder=4)
        else:
            ax_a.plot(ki, ei, "X", ms=9.5, color=OI["vermillion"],
                      mec="white", mew=0.9, zorder=5)
    # honest truth-exit callout at k=8
    ax_a.annotate("true contract\nbriefly leaves set",
                  xy=(8, 50), xytext=(8.6, 51.9),
                  fontsize=7.6, color=OI["vermillion"], ha="left", va="center",
                  arrowprops=dict(arrowstyle="-|>", color=OI["vermillion"],
                                  lw=1.1, shrinkA=2, shrinkB=3))
    # caption checkpoints (a)-(d); (b) placed to the left of the truth-exit X
    for tag, ki, ei in [("(a)", 3, 52), ("(c)", 12, 55), ("(d)", 16, 56)]:
        ax_a.annotate(f"{tag} {ei}", xy=(ki, ei + 1.1), ha="center", va="bottom",
                      fontsize=7.8, fontweight="bold", color=OI["black"])
    ax_a.annotate("(b) 50", xy=(6.55, 50.0), ha="center", va="center",
                  fontsize=7.8, fontweight="bold", color=OI["black"])
    ax_a.set_xlim(2.3, 16.7)
    ax_a.set_ylim(48.5, 58.2)
    ax_a.set_xticks(range(3, 17, 2))
    ax_a.set_yticks([50, 52, 54, 56])
    ax_a.set_xlabel("evidence (episodes)")
    ax_a.set_ylabel("equivalent contracts")
    ax_a.set_title("(a) Observationally-equivalent set size", fontsize=9.6,
                   loc="left", pad=6)
    ax_a.grid(axis="y", color="#ececec", lw=0.8, zorder=0)
    ax_a.text(0.5, 0.06,
              "a certified converter needs a singleton (size 1) — never reached",
              transform=ax_a.transAxes, ha="center", va="bottom", fontsize=7.0,
              style="italic", color="#666666")

    # ---- (c) final-state per-channel confidence ---------------------------
    last = frames[-1]["consensus"]
    order = sorted(data_channels, key=lambda c: last[c], reverse=True)
    vals = [last[c] for c in order]
    bar_colors = [_STATE_COLOR[_data_state(v)] for v in vals]
    ypos = np.arange(len(order))[::-1]
    ax_c.barh(ypos, vals, height=0.62, color=bar_colors,
              edgecolor="white", linewidth=0.8, zorder=3)
    for yi, v in zip(ypos, vals, strict=False):
        ax_c.text(v + 0.015, yi, f"{v:.2f}", va="center", ha="left",
                  fontsize=8.0, color=OI["black"])
    ax_c.set_yticks(ypos)
    ax_c.set_yticklabels(order, fontsize=8.5)
    ax_c.set_xlim(0, 1.16)
    ax_c.set_xticks([0, 0.5, 1.0])
    ax_c.set_xlabel("MAP-channel consensus")
    ax_c.set_title("(c) Per-channel confidence at 16 episodes", fontsize=9.6,
                   loc="left", pad=6)
    ax_c.grid(axis="x", color="#ececec", lw=0.8, zorder=0)
    ax_c.text(0.98, -0.42,
              "56 contracts remain equivalent  →  ABSTAIN, no converter",
              transform=ax_c.transAxes, ha="right", va="top", fontsize=7.8,
              fontweight="bold", color=OI["vermillion"])

    # ---- (b) per-field resolution-state timeline --------------------------
    rows = data_channels + [c for c, _ in struct_channels]
    nrow = len(rows)
    struct_state = {c: s for c, s in struct_channels}
    for ri, ch in enumerate(rows):
        y = nrow - 1 - ri
        for ci, fr in enumerate(frames):
            if ch in struct_state:
                st = struct_state[ch]
            else:
                st = _data_state(fr["consensus"][ch])
            face = _STATE_COLOR[st]
            hatch = "////" if st == "unobservable" else None
            ax_b.add_patch(Rectangle((ci - 0.5, y - 0.42), 1.0, 0.84,
                           facecolor=face, edgecolor="white", linewidth=0.9,
                           hatch=hatch, zorder=3))
    # flag strip above the grid: misspec + truth-exit
    ystrip = nrow - 0.5 + 0.55
    ax_b.text(-1.0, ystrip, "flags", ha="right", va="center", fontsize=7.6,
              style="italic", color="#555555")
    for ci, fr in enumerate(frames):
        if fr["misspec"]:
            ax_b.plot(ci, ystrip, marker="D", ms=4.6, color=OI["orange"],
                      mec="white", mew=0.6, zorder=4)
        if not fr["truth_in_set"]:
            ax_b.plot(ci, ystrip, marker="X", ms=7.5, color=OI["vermillion"],
                      mec="white", mew=0.7, zorder=5)
    ax_b.set_xlim(-1.4, len(frames) - 0.4)
    ax_b.set_ylim(-0.7, ystrip + 0.6)
    ax_b.set_xticks(range(len(frames)))
    ax_b.set_xticklabels(k)
    ax_b.set_yticks(range(nrow))
    ax_b.set_yticklabels(rows[::-1], fontsize=8.4)
    ax_b.set_xlabel("evidence (episodes)")
    ax_b.tick_params(length=0)
    for sp in ("left", "bottom"):
        ax_b.spines[sp].set_visible(False)
    ax_b.set_title("(b) Per-field resolution state", fontsize=9.6, loc="left",
                   pad=16)
    # legend (states + flags) along the top
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor=_STATE_COLOR["resolved"], label="resolved"),
        Patch(facecolor=_STATE_COLOR["constrained"], label="constrained"),
        Patch(facecolor=_STATE_COLOR["ambiguous"], label="ambiguous"),
        Patch(facecolor=_STATE_COLOR["fixed"], label="fixed"),
        Patch(facecolor=_STATE_COLOR["unobservable"], hatch="////",
              edgecolor="#9a9a9a", label="structurally unobservable"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=OI["orange"],
               markersize=6, label="misspec. flagged"),
        Line2D([0], [0], marker="X", color="none",
               markerfacecolor=OI["vermillion"], markersize=8,
               label="truth outside set"),
    ]
    ax_b.legend(handles=handles, loc="lower center",
                bbox_to_anchor=(0.5, -0.46), ncol=7, fontsize=7.2,
                columnspacing=1.1, handletextpad=0.45, handlelength=1.2,
                borderaxespad=0.0)

    fig.suptitle("Evidence accumulates, the true contract is retained but never "
                 "unique — ActionABI abstains",
                 fontsize=11, fontweight="bold", y=1.0)
    for name in ("fig_evidence_accumulation.pdf", "fig_evidence_accumulation.png"):
        path = os.path.join(HERE, name)
        fig.savefig(path, bbox_inches="tight", dpi=300, facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


def fig_gripper_conventions_paper():
    """Figure 6 — six documented gripper-polarity conventions from the Hub audit.

    A compact convention x value-semantics matrix: for each documented
    convention, the physical meaning of the low and high numeric value is
    colour-coded (open vs. close), making the polarity flip legible in print.
    Every row is a documented convention from reports/lerobot_hub_audit.md
    (Table `tab:gripper`). Writes vector PDF (LaTeX) + PNG (.md).
    """
    from matplotlib.patches import FancyBboxPatch

    _paper_style()

    # (convention, datasets, low_label, low_meaning, high_label, high_meaning)
    # All fields verbatim from reports/lerobot_hub_audit.md gripper cluster.
    rows = [
        ("ALOHA joint-gripper", "6 ALOHA  +  3 SO-100/101",
         "0", "closed", "1", "open"),
        ("OXE  gripper_closedness", "ur5, jaco, roboturk, nyu_door",
         "\N{MINUS SIGN}1 / 0", "open", "1", "close"),
        ("OXE  binary state", "berkeley_mvp, berkeley_rpt",
         "0", "open", "1", "closed"),
        ("OXE  open_gripper (bool)", "toto",
         "False", "close", "True", "open"),
        ("taco  actions gripper", "taco_play",
         "\N{MINUS SIGN}1", "open", "1", "close"),
        ("robosuite continuous", "libero, libero_10, metaworld",
         None, None, None, None),
    ]
    n = len(rows)

    def meaning_color(m):
        return OI["blue"] if m == "open" else OI["vermillion"]

    fig, ax = plt.subplots(figsize=(9.4, 2.9))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(-1.5, n + 0.9)
    ax.axis("off")

    # column x-centres
    x_lowlab, x_low, x_high, x_highlab = 4.55, 5.55, 6.95, 7.95
    chip_w, chip_h = 0.92, 0.62

    # header
    yh = n - 0.08
    ax.text(0.15, yh, "Convention", fontsize=9.2, fontweight="bold",
            ha="left", va="center")
    ax.text((x_low + x_lowlab) / 2 - 0.1, yh, "low value", fontsize=9.2,
            fontweight="bold", ha="center", va="center")
    ax.text((x_high + x_highlab) / 2 + 0.1, yh, "high value", fontsize=9.2,
            fontweight="bold", ha="center", va="center")
    ax.axhline(n - 0.5, xmin=0.01, xmax=0.99, color="#333333", lw=1.0)

    def chip(cx, cy, label, meaning):
        c = meaning_color(meaning)
        ax.add_patch(FancyBboxPatch((cx - chip_w / 2, cy - chip_h / 2),
                     chip_w, chip_h,
                     boxstyle="round,pad=0.02,rounding_size=0.09",
                     facecolor=c, edgecolor="none", zorder=3))
        ax.text(cx, cy, meaning.upper(), ha="center", va="center",
                fontsize=8.6, fontweight="bold", color="white", zorder=4)
        return c

    for i, (conv, dsets, ll, lm, hl, hm) in enumerate(rows):
        y = n - 1 - i
        ax.text(0.15, y + 0.16, conv, fontsize=9.0, fontweight="bold",
                ha="left", va="center", color=OI["black"])
        ax.text(0.15, y - 0.24, dsets, fontsize=7.3, style="italic",
                ha="left", va="center", color="#5a5a5a")
        if lm is None:  # robosuite continuous — polarity not documented (neutral)
            ax.add_patch(FancyBboxPatch(
                (x_low - chip_w / 2, y - chip_h / 2),
                (x_high - x_low) + chip_w, chip_h,
                boxstyle="round,pad=0.02,rounding_size=0.09",
                facecolor=OI["lightgray"], edgecolor="#b8b8b8", linewidth=0.9,
                zorder=3))
            ax.text((x_low + x_high) / 2, y, "continuous  [\N{MINUS SIGN}1, 1]",
                    ha="center", va="center", fontsize=8.4, fontweight="bold",
                    color="#444444", zorder=4)
        else:
            ax.text(x_lowlab, y, ll, ha="center", va="center", fontsize=8.6,
                    family="monospace", color="#333333")
            chip(x_low, y, ll, lm)
            chip(x_high, y, hl, hm)
            ax.text(x_highlab, y, hl, ha="center", va="center", fontsize=8.6,
                    family="monospace", color="#333333")
        if i < n - 1:
            ax.axhline(y - 0.5, xmin=0.01, xmax=0.99, color="#ececec", lw=0.7)

    # polarity-flip callout linking ALOHA (row0, high=OPEN) and OXE-closedness
    # (row1, high=CLOSE): same numeric 1, opposite physical action.
    yA, yB = n - 1, n - 2
    # Place the connector + callout clearly to the RIGHT of the "1" high-value
    # labels (x_highlab) so neither the arrow nor the text collides with them.
    bx = x_highlab + 0.72
    ax.annotate("", xy=(bx, yA), xytext=(bx, yB),
                arrowprops=dict(arrowstyle="<->", color=OI["black"], lw=1.3))
    ax.text(bx + 0.16, (yA + yB) / 2, "same value 1,\nopposite action",
            fontsize=7.6, fontweight="bold", ha="left", va="center",
            color=OI["black"])

    # legend (below all rows, clear of the robosuite dataset caption)
    yleg = -1.28
    ax.add_patch(FancyBboxPatch((0.15, yleg - 0.17), 0.55, 0.34,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 facecolor=OI["blue"], edgecolor="none"))
    ax.text(0.82, yleg, "= gripper opens", fontsize=8.2, va="center")
    ax.add_patch(FancyBboxPatch((3.05, yleg - 0.17), 0.55, 0.34,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 facecolor=OI["vermillion"], edgecolor="none"))
    ax.text(3.72, yleg, "= gripper closes", fontsize=8.2, va="center")
    ax.text(6.7, yleg, "Source: reports/lerobot_hub_audit.md", fontsize=7.2,
            style="italic", va="center", color="#777777")

    ax.set_title("Six documented gripper conventions: the same numeric value "
                 "encodes the opposite physical action",
                 fontsize=10.2, fontweight="bold", pad=10, loc="left", x=0.0)
    for name in ("fig_gripper_conventions.pdf", "fig_gripper_conventions.png"):
        path = os.path.join(HERE, name)
        fig.savefig(path, bbox_inches="tight", dpi=300, facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


def fig_hub_audit_paper():
    """Figure 5 — 35-dataset Hub audit ledger (serif / Okabe-Ito, paper-native).

    Same data and message hierarchy as the report: over 139 documented
    field-labels, honest abstention (115) is the dominant designed outcome, the
    5 unique certifications are all documentation-correct, 1 partial/flagged
    discrepancy, and 0 contradictions. Source: reports/lerobot_hub_audit.md.
    Writes vector PDF (LaTeX) + PNG (.md).
    """
    _paper_style()
    # (label, count, colour-role)
    rows = [
        ("Honest abstention\n(knowable field)", 115, OI["blue"]),
        ("Unique certification\n(documentation-correct)", 5, OI["green"]),
        ("Partial / flagged\ndiscrepancy", 1, OI["orange"]),
        ("Contradiction\n(asserts what docs refute)", 0, OI["vermillion"]),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(8.0, 2.75))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, height=0.60, color=colors, edgecolor="white",
            linewidth=0.9, zorder=3)
    for yi, v, c in zip(y, vals, colors, strict=False):
        if v == 0:
            ax.annotate("0", (1.2, yi), ha="left", va="center", fontsize=11,
                        fontweight="bold", color=OI["vermillion"])
        else:
            ax.annotate(str(v), (v + 1.6, yi), ha="left", va="center",
                        fontsize=10.5, fontweight="bold", color=OI["black"])
    # in-plot callouts preserving the message hierarchy
    ax.text(34, y[3], "0 contradictions", fontsize=10.5, fontweight="bold",
            color=OI["vermillion"], va="center", ha="left")
    ax.text(34, y[1], "5 / 5 documentation-correct", fontsize=10.0,
            fontweight="bold", color=OI["green"], va="center", ha="left")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.4)
    ax.set_xlim(0, 128)
    ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax.set_xlabel("Count of documented field-labels", fontsize=9.6)
    ax.grid(axis="x", color="#ececec", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("35-dataset LeRobot Hub audit over 139 field-labels: honest "
                 "abstention dominates by design, zero contradictions",
                 fontsize=10.0, pad=9, loc="left", x=0.0)
    for name in ("fig_hub_audit.pdf", "fig_hub_audit.png"):
        path = os.path.join(HERE, name)
        fig.savefig(path, bbox_inches="tight", dpi=300, facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


def fig_calibration_paper():
    """Figure 3 — calibrated abstention eliminates false uniques (serif / Okabe-Ito,
    paper-native), so it matches Figs 4/5/6 typographically rather than the
    sans-serif dashboard style.

    Same data and message as fig8_calibration:
      (a) synthetic ambiguity gate: forced argmin 25 false uniques vs ActionABI 0,
          both at coverage 1.00, ActionABI abstaining at 0.25;
      (b) labeled-simulation bridge: false uniques 4->0, truth coverage 0.02->0.39
          across the bias-robustness fix.
    Sources: reports/benchmark_sprint.md, reports/labeled_sim_traces.md,
    reports/scorer_fixes.md. Writes vector PDF (LaTeX) + PNG (.md).
    """
    _paper_style()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.2, 3.5))

    # --- (a) synthetic ambiguity gate ---
    methods = ["Forced\nresidual argmin", "ActionABI\ncalibrated set"]
    false_uniques = [25, 0]
    coverage = [1.00, 1.00]
    abstention = [0.00, 0.25]
    x = np.arange(len(methods))
    w = 0.26
    axa.bar(x - w, false_uniques, w,
            color=[OI["vermillion"], OI["green"]], edgecolor="white",
            linewidth=0.8, zorder=3, label="False uniques")
    axa.bar(x, [c * 25 for c in coverage], w, color=OI["blue"], alpha=0.35,
            edgecolor="white", linewidth=0.6, zorder=3,
            label="Equivalence coverage ($\\times$25)")
    axa.bar(x + w, [a * 25 for a in abstention], w, color=OI["orange"], alpha=0.7,
            edgecolor="white", linewidth=0.6, zorder=3,
            label="Abstention rate ($\\times$25)")

    for xi, v in zip(x - w, false_uniques, strict=False):
        axa.annotate(str(v), (xi, v + 0.6), ha="center", va="bottom",
                     fontsize=10.5, fontweight="bold",
                     color=OI["vermillion"] if v > 0 else OI["green"])
    for xi, c in zip(x, coverage, strict=False):
        axa.annotate(f"{c:.2f}", (xi, c * 25 + 0.6), ha="center", va="bottom",
                     fontsize=8.2, color=OI["black"])
    for xi, a in zip(x + w, abstention, strict=False):
        axa.annotate(f"{a:.2f}", (xi, a * 25 + 0.6), ha="center", va="bottom",
                     fontsize=8.2, color=OI["black"])

    axa.set_xticks(x)
    axa.set_xticklabels(methods, fontsize=8.8)
    axa.set_ylim(0, 33)
    axa.set_ylabel("False uniques (count)  /  rate $\\times$25", fontsize=9.2)
    axa.set_title("(a) Synthetic ambiguity gate: 100 cases,\n25 observationally-equivalent",
                  fontsize=9.6, pad=7, loc="left", x=0.0)
    axa.legend(loc="upper center", ncol=1, fontsize=7.6, columnspacing=1.0,
               handletextpad=0.4, borderaxespad=0.3, labelspacing=0.3)
    axa.grid(axis="y", color="#ececec", lw=0.8, zorder=0)
    axa.set_axisbelow(True)

    # --- (b) labeled-simulation bridge ---
    stages = ["Pre-fix", "Post-fix"]
    fu = [4, 0]
    cov = [0.02, 0.39]
    x2 = np.arange(len(stages))
    w2 = 0.32
    axb.bar(x2 - w2 / 2, fu, w2, color=[OI["vermillion"], OI["green"]],
            edgecolor="white", linewidth=0.8, zorder=3, label="False uniques")
    axb.bar(x2 + w2 / 2, [c * 10 for c in cov], w2, color=OI["skyblue"],
            alpha=0.75, edgecolor="white", linewidth=0.6, zorder=3,
            label="Truth equiv-set coverage ($\\times$10)")

    for xi, v in zip(x2 - w2 / 2, fu, strict=False):
        axb.annotate(str(v), (xi, v + 0.1), ha="center", va="bottom",
                     fontsize=10.5, fontweight="bold",
                     color=OI["vermillion"] if v > 0 else OI["green"])
    for xi, c in zip(x2 + w2 / 2, cov, strict=False):
        axb.annotate(f"{c:.2f}", (xi, c * 10 + 0.1), ha="center", va="bottom",
                     fontsize=8.5, color=OI["black"])

    axb.set_xticks(x2)
    axb.set_xticklabels(stages, fontsize=8.8)
    axb.set_ylim(0, 6.0)
    axb.set_ylabel("False uniques (count)  /  coverage $\\times$10", fontsize=9.2)
    axb.set_title("(b) Labeled-simulation bridge:\n90 real-sim traces, bias-robustness fix",
                  fontsize=9.6, pad=7, loc="left", x=0.0)
    axb.legend(loc="upper center", ncol=1, fontsize=7.6, columnspacing=1.0,
               handletextpad=0.4, borderaxespad=0.3, labelspacing=0.3)
    axb.grid(axis="y", color="#ececec", lw=0.8, zorder=0)
    axb.set_axisbelow(True)

    fig.suptitle("Calibrated abstention eliminates false uniques",
                 fontsize=11.5, fontweight="bold", y=1.01, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for name in ("fig_calibration.pdf", "fig_calibration.png"):
        path = os.path.join(HERE, name)
        fig.savefig(path, bbox_inches="tight", dpi=300, facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


def main():
    fig6_pipeline()
    fig7_hub_audit()
    fig8_calibration()
    fig9_gripper_conventions()
    fig_anatomy()
    fig_evidence_paper()
    fig_gripper_conventions_paper()
    fig_hub_audit_paper()
    fig_calibration_paper()
    build_recovery_gif()


if __name__ == "__main__":
    main()
