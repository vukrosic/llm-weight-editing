"""Draw every image in charts/ from the result files in toy/. Run from the repo root: python3 chart.py"""
import json, subprocess
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


NT = {r["run"]: r["no_training_mean"] for r in map(json.loads, open("toy/search_runs/no_training.jsonl"))}


def answers(kind): return sorted((r for r in R if r["mode"] == "answers" and r["kind"] == kind), key=lambda r: r["inst"])


def style(ax):
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def draw_tasks(path):
    """One row per task: best training at any step against Opus's written weights with no training at all."""
    rows = [(label, r) for kind, label in KINDS for r in answers(kind)]
    fig, ax = plt.subplots(figsize=(10.5, 7.4)); y = 0; ticks = []; last = None
    for label, r in rows:
        if label != last and last is not None: y += .7
        if label != last: ax.text(-.012, y - .78, label, fontsize=11.5, color=INK, fontweight="bold", ha="left", transform=ax.get_yaxis_transform())
        if label != last: y += .25
        t = best_training(r["kind"], r["inst"]); o = NT[r["run"]]; last = label
        ax.plot([t, o], [y, y], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        ax.scatter(t, y, s=75, color=PLAIN, zorder=3, edgecolors=SURFACE, linewidths=1.5); ax.scatter(o, y, s=75, color=OPUS, zorder=3, edgecolors=SURFACE, linewidths=1.5)
        ax.text(max(t, o) + .012, y, f"{o - t:+.3f}".replace("0.", "."), va="center", fontsize=10.5, color=MUTED); y += 1
    ax.set_ylim(y, -1.2); ax.set_yticks([]); ax.set_xlim(.08, 1.06); ax.set_xlabel("Accuracy on pairs never trained on"); ax.grid(axis="x", color=GRID, lw=.8, zorder=0); ax.set_axisbelow(True); style(ax); ax.spines["bottom"].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=9, color=c) for c in (PLAIN, OPUS)]
    fig.legend(handles, ["Best of 34 training settings, at its best step of 6,000", "Opus: answers read, weights written, no training at all"], loc="lower center", ncol=2, frameon=False, fontsize=11)
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
    series_ = (("Best training at any step", PLAIN, lambda r: best_training(r["kind"], r["inst"])), ("Script told a list of rule shapes", STRONG, lambda r: script(LI, r["kind"], r["inst"])), ("Script told no list (co-clustering)", NOEDIT, lambda r: script(CO, r["kind"], r["inst"])), ("Opus, no training", OPUS, lambda r: NT[r["run"]]))
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


def draw_flow(path):
    """The first README figure: the two paths and one worked Opus run. The table and codes are run r35, skill 1, from its training answers and its own printout."""
    C = dict(net=("#f39a4d", "#fdeedd"), data=("#d9a900", "#fff6cc"), train=("#2a78d6", "#dfeafa"), opus=("#1baf7a", "#dff5ec"), new="#d8342c", ink="#111111", mute="#55544f", panel="#f3f3f1")
    S = []; add = S.append
    def text(x, y, s, size=20, weight="normal", fill=C["ink"], anchor="middle", family="Helvetica, Arial, sans-serif", style="normal"):
        add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" font-style="{style}" fill="{fill}" text-anchor="{anchor}">{s.replace("&", "&amp;")}</text>')
    def rect(x, y, w, h, role, r=14, sw=1.6, stroke=None, dash=""):
        line, fill = C[role] if role in C and isinstance(C[role], tuple) else (stroke or C["ink"], role)
        add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke or line}" stroke-width="{sw}" {dash}/>')
    def arrow(pts, label=None, at=None):
        add('<path d="M' + " L".join(f"{x} {y}" for x, y in pts) + f'" fill="none" stroke="{C["ink"]}" stroke-width="1.8" marker-end="url(#head)"/>')
        if label: text(*at, label, 17, fill=C["mute"])
    def glyph(x, y, colour):
        """A small network: inputs, two hidden layers, outputs."""
        cols = [[-24, 0, 24], [-36, -12, 12, 36], [-36, -12, 12, 36], [-24, 0, 24]]
        for c, (a, b) in enumerate(zip(cols, cols[1:])):
            for p in a:
                for q in b: add(f'<line x1="{x + c * 46}" y1="{y + p}" x2="{x + (c + 1) * 46}" y2="{y + q}" stroke="{colour}" stroke-width=".9"/>')
        for c, a in enumerate(cols):
            for p in a: add(f'<circle cx="{x + c * 46}" cy="{y + p}" r="6" fill="white" stroke="{C["ink"]}" stroke-width="1.2"/>')
    add('<svg xmlns="http://www.w3.org/2000/svg" width="1640" height="990" viewBox="0 0 1640 990"><defs><marker id="head" markerWidth="7" markerHeight="7" refX="6.5" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 z" fill="#111111"/></marker></defs><rect width="1640" height="990" fill="white"/>')
    # Inputs
    rect(40, 65, 280, 160, "net"); text(180, 96, "Network, random weights", 20, "bold"); glyph(111, 150, "#b9b8b2"); text(180, 214, "8,751 weights", 17, fill=C["mute"])
    rect(40, 265, 280, 180, "data"); text(180, 296, "Training answers", 21, "bold")
    for i, s in enumerate(("skill 1, a=0, b=0 → 14", "skill 1, a=0, b=1 → 7", "skill 1, a=1, b=4 → 7")):
        rect(66, 310 + i * 34, 228, 28, "white", r=7, sw=1, stroke=C["data"][0]); text(180, 330 + i * 34, s, 17)
    text(180, 435, "70% of pairs; 30% stay hidden", 16, fill=C["mute"])
    # The two paths
    for k in range(3): rect(390 + 16 - k * 8, 79 - 16 + k * 8 + 8, 250, 124, "train")
    text(515, 121, "Gradient descent", 21, "bold"); text(515, 151, "6,000 small steps", 18); text(515, 178, "best of 34 settings", 18)
    rect(700, 79, 230, 132, "net"); text(815, 107, "Trained network", 20, "bold"); glyph(746, 158, C["train"][0])
    rect(390, 289, 250, 132, "opus", sw=3, stroke=C["new"]); text(515, 327, "Opus 5.5", 23, "bold"); text(515, 358, "reads the answers,", 18); text(515, 383, "finds the rule", 18)
    rect(700, 289, 230, 132, "net"); text(815, 317, "Network with the rule", 20, "bold"); glyph(746, 368, C["opus"][0])
    arrow([(320, 145), (390, 145)]); arrow([(320, 355), (390, 355)]); add(f'<path d="M352 355 L352 145" stroke="{C["ink"]}" stroke-width="1.8" fill="none"/><circle cx="352" cy="355" r="4" fill="{C["ink"]}"/><circle cx="352" cy="145" r="4" fill="{C["ink"]}"/>')
    arrow([(656, 145), (700, 145)]); arrow([(640, 355), (700, 355)]); text(670, 342, "writes", 15, fill=C["mute"]); arrow([(930, 145), (990, 145)]); arrow([(930, 355), (990, 355)])
    # Scores, from the result files
    rows = [("All 16 tasks with a rule", [r for kind, _ in KINDS if kind != "random" for r in answers(kind)])] + [(label, answers(kind)) for kind, label in KINDS if kind != "random"]
    names = {"Rule shapes no script knows": "Rule shapes no script knows", "Sums, products, larger, smaller": "Sums, products, larger", "Half one shape, half another": "Half one shape, half another", "25% of training answers wrong": "25% of answers wrong", "Numbers wrapped at 16": "Numbers wrapped at 16"}
    rect(990, 50, 610, 410, "white", r=6, sw=1.4, stroke=C["ink"]); text(1010, 84, "Accuracy on pairs never trained on, never shown to Opus", 19, "bold", anchor="start")
    for i, (label, rs) in enumerate(rows):
        y = 104 + i * 58; t = m4([best_training(r["kind"], r["inst"]) for r in rs]); o = sum(NT[r["run"]] for r in rs) / len(rs)
        text(1010, y + 25, names.get(label, label), 16.5, "bold" if i == 0 else "normal", anchor="start")
        add(f'<rect x="1250" y="{y + 4}" width="{t * 290:.1f}" height="19" rx="3" fill="{C["train"][0]}"/><rect x="1250" y="{y + 25}" width="{o * 290:.1f}" height="19" rx="3" fill="#12875e"/>')
        if i == 0: text(1258, y + 19, "best training", 14, "bold", fill="white", anchor="start"); text(1258, y + 40, "Opus, no training", 14, "bold", fill="white", anchor="start")
        text(1250 + t * 290 + 7, y + 20, f"{t:.3f}"[1:] if t < 1 else "1.000", 16, anchor="start"); text(1250 + o * 290 + 7, y + 41, f"{o:.3f}"[1:] if o < 1 else "1.000", 16, "bold", anchor="start")
    # Zoom into Opus
    add('<line x1="390" y1="421" x2="250" y2="490" stroke="#111111" stroke-width="1.2" stroke-dasharray="6 5"/><line x1="640" y1="421" x2="1120" y2="490" stroke="#111111" stroke-width="1.2" stroke-dasharray="6 5"/>')
    rect(40, 490, 1560, 440, C["panel"], r=6, sw=1.4, stroke=C["ink"]); text(62, 523, "What Opus did on one task, in Python it wrote itself", 20, "bold", anchor="start")
    W = 355; xs = [62, 450, 838, 1226]
    for x, title in zip(xs, ("1. Read the answers", "2. Group values that act alike", "3. State a rule, check it", "4. Calculate every weight")):
        rect(x, 540, W, 375, "opus" if x == xs[3] else "white", r=10, sw=1.3, stroke=C["opus"][0]); text(x + W / 2, 572, title, 20, "bold")
    for x in xs[:3]: arrow([(x + W, 727), (x + W + 33, 727)])
    # Step 1: real corner of skill 1
    T = [[14, 7, 6, 14, None, 15, 6], [14, 7, 6, 14, 7, 15, None], [12, 5, None, None, 5, None, 6], [None, None, 3, 13, None, 15, None], [9, 9, None, 13, None, None, None], [None, None, 7, None, 7, 15, None]]
    x0, y0, cw, ch = xs[0] + 54, 618, 39, 36
    text(x0 + 3.5 * cw, 600, "b", 17, style="italic", family="Georgia, serif"); text(x0 - 34, y0 + 3 * ch + 6, "a", 17, style="italic", family="Georgia, serif")
    for a, row in enumerate(T):
        text(x0 - 13, y0 + a * ch + 24, str(a), 15, fill=C["mute"])
        for b, v in enumerate(row):
            add(f'<rect x="{x0 + b * cw}" y="{y0 + a * ch}" width="{cw}" height="{ch}" fill="{"#fff6cc" if a < 2 else "white"}" stroke="#c9c8c2" stroke-width="1"/>')
            if v is not None: text(x0 + b * cw + cw / 2, y0 + a * ch + 24, str(v), 16)
    text(xs[0] + W / 2, 860, "Skill 1, top corner of its 32 × 32 table.", 16, fill=C["mute"]); text(xs[0] + W / 2, 885, "A blank pair was never shown to Opus.", 16, fill=C["mute"])
    # Step 2: hidden codes
    x = xs[1] + W / 2; text(x, 616, "Rows a=0 and a=1 agree, so", 17); text(x, 640, "a=0 and a=1 carry one hidden code.", 17)
    for line, name, vals, y in ((C["data"][0], "a", "0 1 2 3 4 5".split(), 668), (C["train"][0], "b", "0 1 2 3 4 5".split(), 772)):
        codes = [6, 6, 4, 1, 9, 7] if name == "a" else [8, 1, 2, 12, 5, 15]
        text(xs[1] + 34, y + 25, name, 18, style="italic", family="Georgia, serif"); text(xs[1] + 34, y + 69, "code", 15, fill=C["mute"])
        for i, (v, c) in enumerate(zip(vals, codes)):
            cx = xs[1] + 62 + i * 46; rect(cx, y + 3, 40, 32, "white", r=5, sw=1, stroke="#c9c8c2"); text(cx + 20, y + 25, v, 16, fill=C["mute"]); add(f'<line x1="{cx + 20}" y1="{y + 35}" x2="{cx + 20}" y2="{y + 47}" stroke="{C["ink"]}" stroke-width="1.4"/>'); rect(cx, y + 47, 40, 34, C["data"][1] if name == "a" else C["train"][1], r=5, sw=1.2, stroke=line); text(cx + 20, y + 71, str(c), 17, "bold")
    text(x, 885, "Opus found a code for all 32 values of a and b.", 16, fill=C["mute"])
    # Step 3: the rule
    x = xs[2] + W / 2; text(x, 626, "answer = code(a) OR code(b)", 21, style="italic", family="Georgia, serif")
    for i, s in enumerate(("a=0, b=0:   6 OR 8 = 14  ✓", "a=0, b=1:   6 OR 1 = 7  ✓", "a=2, b=1:   4 OR 1 = 5  ✓")): text(x, 668 + i * 28, s, 18)
    rect(xs[2] + 28, 748, W - 56, 62, C["data"][1], r=8, sw=1.2, stroke=C["data"][0]); text(x, 773, "It explains every training answer of skill 1", 15.5); text(x, 797, "Never shown: a=0, b=4 gives 6 OR 5 = 7", 15.5, "bold")
    text(x, 838, "Other skills, other rules: skill 0 is a sum,", 16, fill=C["mute"]); text(x, 861, "skill 2 is the stronger side wins.", 16, fill=C["mute"])
    # Step 4: what the numbers do
    x = xs[3] + 22
    for i, (name, line, y) in enumerate((("Layer 1", "one unit per bit of the code, on", 606), ("", "when that bit is 0 in both codes", 0), ("Layer 2", "turns each unit fully on or off", 672), ("Output", "reads the 4 bits as the answer", 722))):
        if name: rect(x, y - 2, 74, 28, "white", r=6, sw=1, stroke=C["net"][0]); text(x + 37, y + 18, name, 15, "bold")
        text(x + 86, (y or 606 + 26) + 18, line, 16, anchor="start")
    rect(x, 768, W - 44, 40, "white", r=8, sw=1.2, stroke=C["opus"][0]); text(xs[3] + W / 2, 794, "No gradients. Training speed set to 0.", 17, "bold")
    r = [r for r in R if r["run"] == "r35"][0]; t = best_training(r["kind"], r["inst"])
    text(xs[3] + W / 2, 842, f"Skill 1 on hidden pairs: {r['held_out'][1] * 100:.0f}%", 18); text(xs[3] + W / 2, 871, f"Whole task: {r['mean']:.3f}"[:-5] + f"{r['mean']:.3f}"[1:] + f", best training {t:.3f}"[:-5] + f"{t:.3f}"[1:], 18, "bold")
    # Legend
    lx = 40
    for fill, line, label, w in ((C["data"][1], C["data"][0], "Data", 90), (C["net"][1], C["net"][0], "Network", 120), (C["train"][1], C["train"][0], "Training", 120), (C["opus"][1], C["new"], "Opus, in place of training", 270)):
        add(f'<rect x="{lx}" y="949" width="26" height="20" rx="4" fill="{fill}" stroke="{line}" stroke-width="1.6"/>'); text(lx + 34, 965, label, 16, anchor="start"); lx += w + 34
    text(1600, 965, "One Opus run per task. Training's reading is its best step of 6,000, which favours training.", 16, fill=C["mute"], anchor="end")
    add("</svg>"); open(path, "w").write("\n".join(S)); subprocess.run(["rsvg-convert", "-z", "2", path, "-o", path.replace(".svg", ".png")], check=True)


if __name__ == "__main__":
    subprocess.run(["python3", "simple_charts.py"], check=True)
    draw_flow("charts/flow.svg")
    draw_loop("charts/loop.png")
    draw_curves([("1", "Tables 1, first Opus run", [("secret-1, speed 1", OPUS)]), ("2", "Tables 2, second and third Opus runs", [("secret2-1, speed 1", NOEDIT), ("secret2-2, speed 1", OPUS)]), ("3", "Tables 3, fourth Opus run, bar written first", [("secret3-1, speed 1", OPUS), ("secret3-1, its speeds", OPUS, "--")])], "charts/curves.png")
    draw_runs("charts/runs.png")
    draw_tasks("charts/tasks.png")
    draw_timing("charts/timing.png")
    draw_rivals("charts/rivals.png")
    draw_modes("charts/modes.png")
