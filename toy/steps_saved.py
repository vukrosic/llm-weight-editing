"""How many training steps do the editor's changes save? Plain training and the best fixed speed are followed far past step 600 and read every 25 steps.
Each finished editor run then keeps training from its step 600 state, once with its own speeds and once with every weight at speed 1. No editor takes part, and nothing here feeds back into any run.

TOY_RULES=secret TOY_RULE_SET=1 python3 steps_saved.py test     stop in about 7 s if a curve does not match what is already recorded
TOY_RULES=secret TOY_RULE_SET=1 python3 steps_saved.py          all curves; rows go to steps_saved.jsonl
"""
import json, os, signal, sys, time
import torch
import loop
from loop import RESULTS, RULE_SET, RUNS, SKILLS, TOTAL, acc, fresh, load, split, stretch

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "steps_saved.jsonl"); END, EVERY = 6000, 25
EDITORS = {"1": ["secret-1"], "2": ["secret2-1", "secret2-2"], "3": ["secret3-1"]}[RULE_SET]


def best_speed():
    """The best fixed speed for this table set, from the controls already recorded."""
    rows = [json.loads(l) for l in open(RESULTS)]; rows = [r for r in rows if r["run"].startswith("best fixed speed")]; r = max(rows, key=lambda r: r["mean"])
    c, frm = r["run"].split(": x")[1].split(" from "); return float(c), int(frm), r


def uniform(st, c): st["speed"] = {k: torch.full_like(p.detach(), float(c)) for k, p in st["m"].named_parameters()}


def read(st, te): return dict(step=st["step"], **dict(zip(SKILLS, acc(st["m"], te))))


def curve(st, tr, te, end=END, every=EVERY):
    rows = [read(st, te)]
    while st["step"] < end: stretch(st, tr, min(every, end - st["step"])); rows.append(read(st, te))
    return rows


def recorded(name):
    """The held-out score at step 600. A run scored with TOY_FINISH keeps that as at_edit_end."""
    r = [json.loads(l) for l in open(RESULTS) if json.loads(l)["run"] == name][0]; return r.get("at_edit_end", r["held_out"])


def plain(tr, te, end=END): return curve(fresh(), tr, te, end)


def fixed(tr, te, c, frm, end=END):
    st = fresh(); stretch(st, tr, frm); uniform(st, c); return curve(st, tr, te, end)


def editor(run, tr, te, keep, end=END):
    st = load(os.path.join(RUNS, run))
    if not keep: uniform(st, 1)
    return curve(st, tr, te, end)


def at(rows, step): return {k: v for k, v in [r for r in rows if r["step"] == step][0].items() if k != "step"}


def test():
    def over(*_): raise TimeoutError("past 7 s")
    signal.signal(signal.SIGALRM, over); signal.alarm(7); t0 = time.time(); tr, te = split(); c, frm, r = best_speed(); print("best fixed speed", c, "from", frm, r["mean"])
    a = at(plain(tr, te, TOTAL), TOTAL); print("plain at 600 equals the recorded control:", a == recorded("untouched"), a); assert a == recorded("untouched")
    a = at(fixed(tr, te, c, frm, TOTAL), TOTAL); print("best speed at 600 equals the recorded control:", a == r["held_out"], a); assert a == r["held_out"]
    for run in EDITORS:
        a = at(editor(run, tr, te, True, TOTAL), TOTAL); print(run, "at 600 equals its recorded score:", a == recorded(run), a); assert a == recorded(run)
    if not EDITORS: print(f"PASS in {time.time() - t0:.1f}s"); signal.alarm(0); return
    st = load(os.path.join(RUNS, EDITORS[-1])); frozen = {k: v.clone() for k, v in st["m"].state_dict().items()}; sp = st["speed"]
    stretch(st, tr, 25); same = [k for k in sp if (sp[k] == 0).all() and (st["m"].state_dict()[k] == frozen[k]).all()]; n = sum(int((v == 0).sum()) for v in sp.values()); print("frozen weights in", EDITORS[-1], n)
    moved = sum(int(((st["m"].state_dict()[k] != frozen[k]) & (sp[k] == 0)).sum()) for k in sp); print("frozen weights that moved in 25 steps:", moved); assert moved == 0
    t = time.time(); plain(tr, te, 150); print(f"150 steps with reads: {time.time() - t:.2f} s, so {END} steps is about {(time.time() - t) * END / 150:.0f} s"); print(f"PASS in {time.time() - t0:.1f}s"); signal.alarm(0)


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]: test(); sys.exit()
    tr, te = split(); c, frm, _ = best_speed(); have = {(json.loads(l)["set"], json.loads(l)["name"]) for l in open(OUT)} if os.path.exists(OUT) else set()
    jobs = [("plain", lambda: plain(tr, te)), (f"fixed x{c:g} from {frm}", lambda: fixed(tr, te, c, frm))]
    for run in EDITORS: jobs += [(f"{run}, its speeds", lambda run=run: editor(run, tr, te, True)), (f"{run}, speed 1", lambda run=run: editor(run, tr, te, False))]
    for name, f in jobs:
        if (RULE_SET, name) in have: continue
        t = time.time(); rows = f()
        with open(OUT, "a") as g: g.write(json.dumps(dict(set=RULE_SET, name=name, curve=rows)) + "\n")
        print(name, f"{time.time() - t:.0f}s", "at 6000:", {k: v for k, v in rows[-1].items() if k != "step"}, flush=True)
