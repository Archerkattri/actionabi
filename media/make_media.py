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

    ax.set_title("ActionABI: evidence -> verified converter, or honest refusal",
                 pad=14, y=1.0)
    ax.text(7.25, -0.15,
            "Calibrated abstention: never certifies a unique answer the data does not support.",
            ha="center", va="top", fontsize=9.5, color="#4a5568")
    _save(fig, "diagram_pipeline.png")


# ---------------------------------------------------------------------------
# FIG 7 — Hub audit
# ---------------------------------------------------------------------------
def fig7_hub_audit():
    apply_style()
    rows = [
        ("Honest abstention (knowable field)", 115, "learned"),
        ("Unique certification -\ndocumentation-correct", 5, "oracle"),
        ("Partial / flagged discrepancy", 1, "probe"),
        ("Contradiction (asserts what docs refute)", 0, "floor"),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [COLORS[r[2]] for r in rows]

    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, height=0.6, color=colors, edgecolor=COLORS["ink"],
            linewidth=0.8, zorder=3)

    for yi, v, (_lbl, _, _key) in zip(y, vals, rows, strict=False):
        if v == 0:
            ax.annotate("0", (0.8, yi), ha="left", va="center",
                        fontsize=14, fontweight="bold", color=COLORS["floor"])
        else:
            ax.annotate(str(v), (v + 1.2, yi), ha="left", va="center",
                        fontsize=11.5, fontweight="bold", color=COLORS["ink"])

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 120)
    ax.set_xlabel("Count of field-labels")
    ax.set_title("35-dataset LeRobot Hub audit: zero contradictions", pad=26)
    ax.text(0.0, 1.045,
            "35 datasets (6 pinned + 29 new) across 6 ecosystems; 139 documented field-labels; "
            "documentation is a label, never an inference input.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9.2, color="#4a5568")

    # prominent callouts — plain text set well clear of the numeric count labels
    # so nothing crosses or overlaps them.
    ax.text(34, y[3], "0 contradictions", fontsize=13, fontweight="bold",
            color=COLORS["floor"], va="center", ha="left")
    ax.text(34, y[1], "5 / 5 unique certs correct", fontsize=12, fontweight="bold",
            color=COLORS["oracle"], va="center", ha="left")
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
    apply_style()
    # rows top-to-bottom. polarity: +1 => "1 = open" tint green-left/red-right emphasis;
    # we encode a diverging bar centered at 0 to show the FLIP.
    rows = [
        ("ALOHA\n(arXiv:2304.13705)", "0 = closed,  1 = open", "continuous [0,1]", +1),
        ("Open X-Embodiment / RT-1 /\nBerkeley-UR5 (arXiv:2310.08864)",
         "1 = close,  -1 = open,  0 = no-change", "trinary", -1),
        ("DROID\n(arXiv:2403.12945)", "7th dim = gripper delta", "continuous delta", 0),
        ("Stanford HYDRA\n(arXiv:2306.17237)", "binary close-gripper", "binary", 0),
        ("LeRobot taxonomy doc", "left per-dataset (unspecified)", "unspecified", 0),
    ]
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.set_xlim(-1.35, 1.9)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.axis("off")

    n = len(rows)
    for i, (eco, enc, kind, pol) in enumerate(rows):
        y = n - 1 - i
        # ecosystem label (left)
        ax.text(-1.32, y, eco, ha="left", va="center", fontsize=9.8,
                fontweight="bold", color=COLORS["ink"])
        # polarity chip: green = "1 means open", red = "1 means close"
        if pol == +1:
            chip_c, chip_txt = COLORS["oracle"], "value 1  ->  OPEN"
        elif pol == -1:
            chip_c, chip_txt = COLORS["floor"], "value 1  ->  CLOSE"
        else:
            chip_c, chip_txt = COLORS["fixed"], "polarity n/a"
        chip = FancyBboxPatch((0.02, y - 0.28), 0.66, 0.56,
                              boxstyle="round,pad=0.01,rounding_size=0.05",
                              linewidth=1.2, edgecolor=chip_c,
                              facecolor=chip_c, alpha=0.16, zorder=2)
        ax.add_patch(chip)
        ax.text(0.35, y, chip_txt, ha="center", va="center", fontsize=8.8,
                fontweight="bold", color=chip_c, zorder=3)
        # encoding string + kind
        ax.text(0.78, y + 0.12, enc, ha="left", va="center", fontsize=9.4,
                color=COLORS["ink"])
        ax.text(0.78, y - 0.16, kind, ha="left", va="center", fontsize=8.2,
                style="italic", color="#4a5568")
        # separator
        if i < n - 1:
            ax.axhline(y - 0.5, color=COLORS["grid"], linewidth=0.8, zorder=1)

    # highlight the ALOHA vs OXE flip
    ax.annotate("", xy=(0.35, n - 1 - 0.32), xytext=(0.35, n - 2 + 0.32),
                arrowprops=dict(arrowstyle="<->", color=COLORS["ink"], lw=1.6))
    ax.text(0.02, n - 1.5, "POLARITY\nFLIP", ha="right", va="center",
            fontsize=8.6, fontweight="bold", color=COLORS["ink"],
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff4d6",
                      edgecolor=COLORS["probe"], linewidth=1.0))

    ax.set_title("Robot ecosystems actively disagree on action conventions "
                 "(gripper polarity)", pad=14, y=1.0)
    ax.text(-1.35, -0.55,
            "Same numeric channel, opposite physical meaning across two actively-used "
            "dataset families. "
            "ALOHA (0=closed/1=open) vs\nOXE (1=close/-1=open/0=no-change) directly disagree "
            "- independent evidence that action semantics silently vary.",
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


def main():
    fig6_pipeline()
    fig7_hub_audit()
    fig8_calibration()
    fig9_gripper_conventions()
    build_recovery_gif()


if __name__ == "__main__":
    main()
