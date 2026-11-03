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
    """Each run is (name, colour) or (name, colour, line style)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.6), sharey=True, squeeze=False)
    for ax, (st, title, runs) in zip(axes[0], panels):
        x, y = series(get(st, "plain")); ax.plot(x, y, color=PLAIN, lw=1.6)
        x, y = series(strongest(st)["curve"]); ax.plot(x, y, color=STRONG, lw=1.6)
        for name, colour, *style in runs: x, y = series(get(st, name)); ax.plot(x, y, color=colour, lw=2, ls=style[0] if style else "-")
        ax.set_title(title, loc="left", fontsize=13); ax.set_xlabel("Training steps"); ax.set_xlim(0, 6000); ax.set_ylim(.2, 1); ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.tick_params(length=0); ax.axvline(600, color=MUTED, lw=.8, ls=":"); ax.text(640, .225, "edits stop at step 600", color=MUTED, fontsize=10)
    axes[0][0].set_ylabel("Accuracy on pairs never trained on")
    handles = [plt.Line2D([], [], color=PLAIN, lw=2), plt.Line2D([], [], color=STRONG, lw=2), plt.Line2D([], [], color=OPUS, lw=2), plt.Line2D([], [], color=NOEDIT, lw=2), plt.Line2D([], [], color=OPUS, lw=2, ls="--")]
    fig.legend(handles, ["Plain training", "Strongest training we tried", "Opus kept its weights, all speeds 1 after step 600", "Opus kept no weight change", "Fourth run, its own speeds (the counted score)"], loc="lower center", ncol=5, frameon=False, fontsize=11)
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
    rows = [("First run\ntables 1\nall speeds 1", "1", "secret-1, speed 1", OPUS), ("Second run\ntables 2\nall speeds 1", "2", "secret2-1, speed 1", NOEDIT), ("Third run\ntables 2\nall speeds 1", "2", "secret2-2, speed 1", OPUS), ("Fourth run, counted\ntables 3\nits own speeds", "3", "secret3-1, its speeds", OPUS)]
    fig, ax = plt.subplots(figsize=(10.5, 4.8)); w = .26
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


R = [r for r in map(json.loads, open("toy/search_runs/results.jsonl")) if r["run"] != "pilot"]; C = [json.loads(l) for l in open("toy/search_runs/controls.jsonl")]
CO = [json.loads(l) for l in open("toy/search_runs/cocluster.jsonl")]; LI = [json.loads(l) for l in open("toy/search_runs/read_rule.jsonl")]
KINDS = (("unseen", "Rule shapes no script knows"), ("tables", "Sums, products, larger, smaller"), ("conditions", "Half one shape, half another"), ("noisy", "25% of training answers wrong"), ("modular", "Numbers wrapped at 16"), ("random", "Random tables, no rule"))
m4 = lambda v: sum(v) / len(v)


def best_training(kind, inst, mode="answers", start=150):
    """Best of the 34 controls at any step from `start` on, read on held-out pairs, which favours training. Scripted edits count only for their own mode."""
    return max(m4(v) for r in C if r["kind"] == kind and r["inst"] == inst and (":" not in r["name"] or r["name"].startswith(mode + ":")) for k, v in r["rows"].items() if int(k) >= start)


def script(rows, kind, inst): return [r for r in rows if r["source"] == "labels" and r["kind"] == kind and r["inst"] == inst][-1]["mean"]


def answers(kind): return sorted((r for r in R if r["mode"] == "answers" and r["kind"] == kind), key=lambda r: r["inst"])


def style(ax):
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def draw_tasks(path):
    """One row per task: best training at any step against Opus's first reading."""
    rows = [(label, r) for kind, label in KINDS for r in answers(kind)]
    fig, ax = plt.subplots(figsize=(10.5, 7.4)); y = 0; ticks = []; last = None
    for label, r in rows:
        if label != last and last is not None: y += .7
        if label != last: ax.text(-.012, y - .78, label, fontsize=11.5, color=INK, fontweight="bold", ha="left", transform=ax.get_yaxis_transform())
        if label != last: y += .25
        t = best_training(r["kind"], r["inst"]); o = r["mean"]; last = label
        ax.plot([t, o], [y, y], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        ax.scatter(t, y, s=75, color=PLAIN, zorder=3, edgecolors=SURFACE, linewidths=1.5); ax.scatter(o, y, s=75, color=OPUS, zorder=3, edgecolors=SURFACE, linewidths=1.5)
        ax.text(max(t, o) + .012, y, f"{o - t:+.3f}".replace("0.", "."), va="center", fontsize=10.5, color=MUTED); y += 1
    ax.set_ylim(y, -1.2); ax.set_yticks([]); ax.set_xlim(.08, 1.06); ax.set_xlabel("Accuracy on pairs never trained on"); ax.grid(axis="x", color=GRID, lw=.8, zorder=0); ax.set_axisbelow(True); style(ax); ax.spines["bottom"].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=9, color=c) for c in (PLAIN, OPUS)]
    fig.legend(handles, ["Best of 34 training settings, at its best step of 6,000", "Opus: answers read, weights written, no training first"], loc="lower center", ncol=2, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(path, dpi=170); plt.close(fig)


def draw_timing(path):
    """Opus's best reading through step 6000 minus the best training at any step from its first reading on, the same comparison for all four timings."""
    kinds = ("tables", "conditions", "noisy", "modular"); names = ("Sums, products,\nlarger, smaller", "Half one shape,\nhalf another", "25% of training\nanswers wrong", "Numbers wrapped\nat 16"); modes = (("before", "Before training", PLAIN, 600), ("during", "During training", NOEDIT, 600), ("after", "After training", STRONG, 1650), ("answers", "Instead of training", OPUS, 150)); w = .2
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for j, (mode, label, c, start) in enumerate(modes):
        for i, kind in enumerate(kinds):
            rs = [r for r in R if r["mode"] == mode and r["kind"] == kind]
            g = sum(max(m4(v) for v in r["own"].values()) - best_training(kind, r["inst"], mode, start) for r in rs) / len(rs)
            ax.bar(i + (j - 1.5) * w, g, w * .9, color=c, zorder=2, label=label if i == 0 else None)
            ax.text(i + (j - 1.5) * w, g + (.008 if g >= 0 else -.008), f"{g:+.2f}".replace("0.", "."), ha="center", va="bottom" if g >= 0 else "top", fontsize=9.5, color=MUTED)
    ax.axhline(0, color=INK, lw=.8); ax.set_xticks(range(4)); ax.set_xticklabels(names); ax.set_ylabel("Opus's best minus training's best"); ax.grid(axis="y", color=GRID, lw=.8, zorder=0); style(ax); ax.spines["bottom"].set_visible(False)
    fig.legend(loc="lower center", ncol=4, frameon=False, fontsize=11); fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(path, dpi=170); plt.close(fig)


def draw_rivals(path):
    """Mean over each kind's tasks: Opus against two scripts told every training answer, and best training."""
    shown = [k for k in KINDS if k[0] != "random"]; names = ("Rule shapes\nno script knows", "Sums, products,\nlarger, smaller", "Half one shape,\nhalf another", "25% of training\nanswers wrong", "Numbers wrapped\nat 16"); w = .2
    series_ = (("Best training at any step", PLAIN, lambda r: best_training(r["kind"], r["inst"])), ("Script told a list of rule shapes", STRONG, lambda r: script(LI, r["kind"], r["inst"])), ("Script told no list (co-clustering)", NOEDIT, lambda r: script(CO, r["kind"], r["inst"])), ("Opus", OPUS, lambda r: r["mean"]))
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for j, (label, c, f) in enumerate(series_):
        for i, (kind, _) in enumerate(shown):
            rs = answers(kind); v = sum(f(r) for r in rs) / len(rs)
            ax.bar(i + (j - 1.5) * w, v, w * .9, color=c, zorder=2, label=label if i == 0 else None); ax.text(i + (j - 1.5) * w, v + .008, f"{v:.2f}"[1:] if v < 1 else "1.00", ha="center", fontsize=9.5, color=MUTED)
    ax.set_xticks(range(len(shown))); ax.set_xticklabels(names); ax.set_ylim(.4, 1.06); ax.set_ylabel("Accuracy on pairs never trained on"); ax.grid(axis="y", color=GRID, lw=.8, zorder=0); style(ax)
    fig.legend(loc="lower center", ncol=4, frameon=False, fontsize=10); fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(path, dpi=170); plt.close(fig)


def draw_modes(path):
    """Where Opus's one chance sits on the training timeline, in each of the four tests."""
    rows = (("Before", [(0, 600)], [0], "Sees only the first 64 examples of each skill"), ("During", [(0, 600)], [150, 300, 450], "Three pauses to change the network"), ("After", [(0, 1650)], [1500], "One change to a network trained for 1,500 steps"), ("Instead", [(0, 150)], [0], "Random weights and every training answer"))
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    for i, (name, spans, marks, note) in enumerate(rows):
        y = -i; ax.text(-90, y, name, ha="right", va="center", fontsize=13, fontweight="bold", color=INK); ax.plot([0, 2000], [y, y], color=GRID, lw=7, solid_capstyle="round", zorder=1)
        for a, b in spans: ax.plot([a, b], [y, y], color=PLAIN, lw=7, solid_capstyle="round", zorder=2)
        ax.scatter(marks, [y] * len(marks), s=150, color=OPUS, zorder=3, edgecolors=SURFACE, linewidths=2); ax.scatter(spans[-1][1], y, s=46, color=INK, zorder=4)
        ax.text(2060, y, note, va="center", fontsize=11, color=MUTED)
    ax.text(0, .62, "Training steps: first reading at the black dot, training then goes on alone to 6,000", fontsize=10.5, color=MUTED, va="bottom")
    ax.set_xlim(-420, 4300); ax.set_ylim(-3.6, 1); ax.axis("off")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=11, color=OPUS), plt.Line2D([], [], lw=6, color=PLAIN), plt.Line2D([], [], marker="o", ls="", ms=7, color=INK)]
    fig.legend(handles, ["Opus changes the weights", "Training", "First reading"], loc="lower center", ncol=3, frameon=False, fontsize=11); fig.tight_layout(rect=(0, .1, 1, 1)); fig.savefig(path, dpi=170); plt.close(fig)


if __name__ == "__main__":
    draw_loop("charts/loop.png")
    draw_curves([("1", "Tables 1, first Opus run", [("secret-1, speed 1", OPUS)]), ("2", "Tables 2, second and third Opus runs", [("secret2-1, speed 1", NOEDIT), ("secret2-2, speed 1", OPUS)]), ("3", "Tables 3, fourth Opus run, bar written first", [("secret3-1, speed 1", OPUS), ("secret3-1, its speeds", OPUS, "--")])], "charts/curves.png")
    draw_runs("charts/runs.png")
    draw_tasks("charts/tasks.png")
    draw_timing("charts/timing.png")
    draw_rivals("charts/rivals.png")
    draw_modes("charts/modes.png")
