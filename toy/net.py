"""A network small enough to read: two numbers and a skill go in, one number comes out. Four skills share every weight.

python3 net.py     train one network on all skills and one with `add` starved of examples; print held-out accuracy per skill
"""
import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)

N, WIDTH, SKILLS = 32, 48, ("add", "max", "diff", "min")
OUT = 2 * N - 1


SECRET, RULE_SET = os.environ.get("TOY_RULES") == "secret", os.environ.get("TOY_RULE_SET", "1")
if SECRET: SKILLS = ("skill0", "skill1", "skill2", "skill3")


def secret_tables(classes=8):
    """Four rules with no formula to guess. Each number gets its own random value per skill, and the skill combines the values of a and b; the result is cut into equal-sized classes."""
    g = torch.Generator().manual_seed({"1": 7, "2": 11, "3": 13}[RULE_SET]); out = []
    for s in range(4):
        f, h = torch.rand(N, generator=g)[:, None], torch.rand(N, generator=g)[None, :]; v = {"1": [f + h, f * h, f.max(h), (f - h).abs()], "2": [f.min(h), f + h, (f - h).abs(), f * h], "3": [(f - h).abs(), f.max(h), f * h, f + h]}[RULE_SET][s]
        out.append(torch.bucketize(v, v.flatten().quantile(torch.linspace(0, 1, classes + 1)[1:-1])))
    return out


TABLES = secret_tables() if SECRET else None


def answer(s, a, b): return TABLES[s][a, b] if SECRET else {0: a + b, 1: a.max(b), 2: (a - b).abs(), 3: a.min(b)}[s]


def data(g):
    """Every pair for every skill, 70% to train and 30% held out. The same pairs are held out for all skills."""
    a, b = torch.cartesian_prod(torch.arange(N), torch.arange(N)).T; p = torch.randperm(len(a), generator=g); k = int(.7 * len(a))
    return (a[p[:k]], b[p[:k]]), (a[p[k:]], b[p[k:]])


def inputs(a, b, s): return torch.cat([F.one_hot(a, N), F.one_hot(b, N), F.one_hot(torch.full_like(a, s), len(SKILLS))], 1).float()


class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.l1 = nn.Linear(2 * N + len(SKILLS), WIDTH); s.l2 = nn.Linear(WIDTH, WIDTH); s.l3 = nn.Linear(WIDTH, OUT)
    def forward(s, x): return s.l3(F.silu(s.l2(F.silu(s.l1(x)))))


def train(m, tr, share, steps, g):
    """share: fraction of the training pairs each skill may use."""
    o = torch.optim.Adam(m.parameters(), lr=.003); a, b = tr; use = [torch.randperm(len(a), generator=g)[:max(1, int(share[i] * len(a)))] for i in range(4)]
    for _ in range(steps):
        l = 0.
        for i in range(4):
            j = use[i][torch.randint(len(use[i]), (64,), generator=g)]; l = l + F.cross_entropy(m(inputs(a[j], b[j], i)), answer(i, a[j], b[j]))
        o.zero_grad(); l.backward(); o.step()


def acc(m, te):
    a, b = te
    with torch.no_grad(): return [round((m(inputs(a, b, i)).argmax(1) == answer(i, a, b)).float().mean().item(), 3) for i in range(4)]


if __name__ == "__main__":
    t = time.time(); g = torch.Generator().manual_seed(0); tr, te = data(g); steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    full = Net(); train(full, tr, [1, 1, 1, 1], steps, g); print("all skills trained", dict(zip(SKILLS, acc(full, te))), f"{time.time() - t:.1f}s")
    t = time.time(); weak = Net(); train(weak, tr, [.05, 1, 1, 1], steps, g); print("add starved to 5%  ", dict(zip(SKILLS, acc(weak, te))), f"{time.time() - t:.1f}s")
