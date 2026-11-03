"""The strongest training this toy gets: a grid of learning rate and weight decay, plus dropout and wider networks, all read every 25 steps to step 6000 on the held-out pairs.
Reading the held-out pairs to pick the best row flatters training, which is the side we want to be hard to beat.

TOY_RULES=secret TOY_RULE_SET=1 python3 tuned.py test     stop in about 7 s if the plain setting does not repeat the recorded control
TOY_RULES=secret TOY_RULE_SET=1 python3 tuned.py          all rows go to tuned.jsonl
"""
import json, os, signal, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from net import N, OUT, RULE_SET, SKILLS, acc, answer, inputs
from loop import BATCH, RESULTS, split

HERE = os.path.dirname(os.path.abspath(__file__)); DEST = os.path.join(HERE, "tuned.jsonl"); END, EVERY = 6000, 25


class Wide(nn.Module):
    def __init__(s, width, drop):
        super().__init__(); s.l1 = nn.Linear(2 * N + len(SKILLS), width); s.l2 = nn.Linear(width, width); s.l3 = nn.Linear(width, OUT); s.drop = drop
    def forward(s, x): return s.l3(F.dropout(F.silu(s.l2(F.dropout(F.silu(s.l1(x)), s.drop, s.training))), s.drop, s.training))


def run(lr=.003, wd=0., width=48, drop=0., end=END):
    torch.manual_seed(1); m = Wide(width, drop); o = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd); g = torch.Generator().manual_seed(2); (a, b), te = split(); rows = []
    for step in range(end + 1):
        if step % EVERY == 0: m.eval(); rows.append(dict(step=step, **dict(zip(SKILLS, acc(m, te))))); m.train()
        if step == end: break
        l = 0.
        for i in range(4):
            j = torch.randint(len(a), (BATCH,), generator=g); l = l + F.cross_entropy(m(inputs(a[j], b[j], i)), answer(i, a[j], b[j]))
        o.zero_grad(); l.backward(); o.step()
    return rows


def mean(r): return sum(r[k] for k in SKILLS) / 4


def test():
    def over(*_): raise TimeoutError("past 7 s")
    signal.signal(signal.SIGALRM, over); signal.alarm(7); t = time.time(); rows = run(end=600); want = [json.loads(l) for l in open(RESULTS) if json.loads(l)["run"] == "untouched"][0]["held_out"]; got = {k: rows[-1][k] for k in SKILLS}
    print("plain setting at 600 equals the recorded control:", got == want, got); assert got == want
    a = run(wd=1., end=50)[-1]; b = run(end=50)[-1]; print("weight decay changes the network:", a != b); assert a != b
    a = run(drop=.3, end=50)[-1]; print("dropout changes the network:", a != b); assert a != b; assert run(width=96, end=25)[-1]["step"] == 25
    print(f"PASS in {time.time() - t:.1f}s"); signal.alarm(0)


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]: test(); sys.exit()
    jobs = [dict(lr=lr, wd=wd) for lr in (.001, .003, .01) for wd in (0., .01, .1, .5, 1.)] + [dict(drop=.1), dict(drop=.3), dict(width=96), dict(width=192), dict(width=192, wd=.1), dict(drop=.1, wd=.1), dict(drop=.1, lr=.01), dict(drop=.1, lr=.01, wd=.1), dict(drop=.05), dict(drop=.1, width=192), dict(drop=.1, end=15000), dict(lr=.01, wd=.1, end=15000)]
    have = [json.loads(l) for l in open(DEST)] if os.path.exists(DEST) else []
    for j in jobs:
        if any(h["set"] == RULE_SET and h["setting"] == j for h in have): continue
        t = time.time(); rows = run(**j); best = max(rows, key=mean)
        with open(DEST, "a") as f: f.write(json.dumps(dict(set=RULE_SET, setting=j, curve=rows)) + "\n")
        print(j, f"{time.time() - t:.0f}s", f"at 1500 {mean(rows[60]):.3f} at 6000 {mean(rows[240]):.3f} peak {mean(best):.3f} at {best['step']}", flush=True)
