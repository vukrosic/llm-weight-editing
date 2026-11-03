"""Draw every image in charts/ from the result files in toy/. Run from the repo root: python3 chart.py"""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SKILLS = ("skill0", "skill1", "skill2", "skill3"); INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
PLAIN, STRONG, OPUS, NOEDIT = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
curves = [json.loads(l) for l in open("toy/steps_saved.jsonl")]; tuned = [json.loads(l) for l in open("toy/tuned.jsonl")]
mean = lambda r: sum(r[k] for k in SKILLS) / 4


def series(c, end=6000): return [r["step"] for r in c if r["step"] <= end], [mean(r) for r in c if r["step"] <= end]


def strongest(st, end=6000):
    """The setting at this network size with the highest reading on held-out pairs by step `end`."""
    rows = [r for r in tuned if r["set"] == st and r["setting"].get("width", 48) == 48]
    return max(rows, key=lambda r: max(mean(x) for x in r["curve"] if x["step"] <= end))


def get(st, name): return [r for r in curves if r["set"] == st and r["name"] == name][0]["curve"]


def draw_curves(panels, path):
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.6), sharey=True, squeeze=False)
    for ax, (st, title, runs) in zip(axes[0], panels):
        x, y = series(get(st, "plain")); ax.plot(x, y, color=PLAIN, lw=1.6)
        x, y = series(strongest(st)["curve"]); ax.plot(x, y, color=STRONG, lw=1.6)
        for name, colour in runs: x, y = series(get(st, name)); ax.plot(x, y, color=colour, lw=2)
        ax.set_title(title, loc="left", fontsize=13); ax.set_xlabel("Training steps"); ax.set_xlim(0, 6000); ax.set_ylim(.2, 1); ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.tick_params(length=0); ax.axvline(600, color=MUTED, lw=.8, ls=":"); ax.text(640, .225, "edits stop at step 600", color=MUTED, fontsize=10)
    axes[0][0].set_ylabel("Accuracy on pairs never trained on")
    handles = [plt.Line2D([], [], color=PLAIN, lw=2), plt.Line2D([], [], color=STRONG, lw=2), plt.Line2D([], [], color=OPUS, lw=2), plt.Line2D([], [], color=NOEDIT, lw=2)]
    fig.legend(handles, ["Plain training", "Strongest training we tried", "Opus changed weights and kept them", "Opus kept no weight change"], loc="lower center", ncol=4, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, .07, 1, 1)); fig.savefig(path, dpi=170); plt.close(fig)


def box(ax, x, y, w, h, text, colour, fill, size=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=.12", lw=1.6, edgecolor=colour, facecolor=fill)); ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.35)


def arrow(ax, a, b, colour=MUTED, text=None, at=None):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.8, shrinkA=0, shrinkB=0))
    if text: ax.text(*at, text, ha="center", va="center", fontsize=11, color=MUTED)


def draw_loop(path):
    P, T, C = ("#6250d6", "#ece9fb"), ("#1baf7a", "#e0f6ee"), ("#eb6834", "#fdeadf")
    fig, ax = plt.subplots(figsize=(11, 4.8)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.8); ax.axis("off")
    box(ax, .2, 2.3, 2.7, 1.5, "Train 150 steps\nFixed batches, the same\nin every run", *T)
    box(ax, 4.0, 2.3, 2.9, 1.5, "Pause\nOpus reads the weights\nand their history", *P)
    box(ax, 8.0, 2.3, 2.8, 1.5, "Opus changes the network\nNew weight numbers,\nor how fast weights move", *C)
    box(ax, 4.0, .2, 2.9, 1.35, "Check on training pairs\n20 checks in all\nHeld-out pairs stay hidden", *P)
    box(ax, 8.0, .2, 2.8, 1.35, "After step 600\nNo more edits, training goes\non, then one held-out score", *T)
    arrow(ax, (2.9, 3.05), (4.0, 3.05)); arrow(ax, (6.9, 3.05), (8.0, 3.05)); arrow(ax, (5.45, 2.3), (5.45, 1.55)); arrow(ax, (9.4, 2.3), (9.4, 1.55))
    ax.plot([9.4, 9.4, 1.55], [3.8, 4.3, 4.3], color=MUTED, lw=1.8); arrow(ax, (1.55, 4.3), (1.55, 3.8)); ax.text(5.45, 4.5, "Next 150 steps, with Opus's changes in place", fontsize=11, color=MUTED, ha="center")
    fig.tight_layout(pad=.2); fig.savefig(path, dpi=170); plt.close(fig)


def draw_runs(path):
    """Each Opus run at step 1500 against the strongest training at step 1500 and at its best step."""
    rows = [("First run\ntables 1", "1", "secret-1, speed 1", OPUS), ("Second run\ntables 2", "2", "secret2-1, speed 1", NOEDIT), ("Third run\ntables 2", "2", "secret2-2, speed 1", OPUS)]
    fig, ax = plt.subplots(figsize=(9, 4.8)); w = .26
    for i, (label, st, name, colour) in enumerate(rows):
        same = [r for r in tuned if r["set"] == st and r["setting"].get("width", 48) == 48]
        at1500 = max(mean(r["curve"][60]) for r in same); best = max(max(mean(x) for x in r["curve"]) for r in same); opus = mean([x for x in get(st, name) if x["step"] == 1500][0])
        for j, (v, c) in enumerate(((at1500, PLAIN), (best, STRONG), (opus, colour))):
            ax.bar(i + (j - 1) * w, v, w * .9, color=c, zorder=2); ax.text(i + (j - 1) * w, v + .008, f"{v:.3f}", ha="center", fontsize=10, color=INK)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows]); ax.set_ylim(.6, .95); ax.set_ylabel("Accuracy on pairs never trained on"); ax.grid(axis="y", color=GRID, lw=.8, zorder=0)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (PLAIN, STRONG, OPUS, NOEDIT)]
    fig.legend(handles, ["Best training at step 1500", "Best training at any step", "Opus kept its weights, step 1500", "Opus kept none, step 1500"], loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, .11, 1, 1)); fig.savefig(path, dpi=170); plt.close(fig)


if __name__ == "__main__":
    draw_loop("charts/loop.png")
    draw_curves([("1", "Tables 1, first Opus run", [("secret-1, speed 1", OPUS)]), ("2", "Tables 2, second and third Opus runs", [("secret2-1, speed 1", NOEDIT), ("secret2-2, speed 1", OPUS)])], "charts/curves.png")
    draw_runs("charts/runs.png")
