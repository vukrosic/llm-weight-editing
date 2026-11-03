"""The strongest rival that knows no rule family: group the numbers a that behave alike and the numbers b that behave alike, and keep one answer per pair of groups. Biclustering, no LLM, no gradients.

python3 cocluster.py labels KIND INST          on the training answers, with no network
python3 cocluster.py net KIND INST [STEP]      on a network's own answers, with no training answers. Plain training, default step 6000
Rows go to search_runs/cocluster.jsonl.
"""
import json, os, sys, time
import numpy as np, torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
import search
from search import N, RUNS, SKILLS, Net, fresh, inputs, make_task, stretch
OUT = os.path.join(RUNS, "cocluster.jsonl"); GROUPS = (2, 4, 8, 12, 16, 24); CLASSES = 16


def groups(T, seen, k):
    """Group rows by how often they give the same answer where both have one."""
    both = seen[:, None, :] & seen[None, :, :]; same = ((T[:, None, :] == T[None, :, :]) & both).sum(2) / np.maximum(both.sum(2), 1); d = 1 - same; d = (d + d.T) / 2; np.fill_diagonal(d, 0)
    return fcluster(linkage(squareform(d, checks=False), "average"), k, "maxclust") - 1


def answer_table(T, seen, g, h, K, L):
    c = np.zeros((K, L, CLASSES)); A, B = np.nonzero(seen); np.add.at(c, (g[A], h[B], T[A, B]), 1); O = c.argmax(2); row = c.sum(1).argmax(1); empty = c.sum(2) == 0; O[empty] = np.repeat(row[:, None], L, 1)[empty]; return O


def fit(T, seen, K, L, rounds=6):
    g, h = groups(T, seen, K), groups(T.T, seen.T, L); K, L = g.max() + 1, h.max() + 1
    for _ in range(rounds):
        O = answer_table(T, seen, g, h, K, L); g = np.array([((O[:, h] == T[a]) * seen[a]).sum(1).argmax() for a in range(N)])
        O = answer_table(T, seen, g, h, K, L); h = np.array([((O[g][:, :] == T[:, [b]]) * seen[:, [b]]).sum(0).argmax() for b in range(N)])
    O = answer_table(T, seen, g, h, K, L); return O[g][:, h]


def solve(T, seen, rng):
    """Pick the group counts on a tenth of the evidence set aside, then fit on all of it."""
    aside = seen & (rng.random(seen.shape) < .1); use = seen & ~aside; best = (-1, 2, 2)
    for K in GROUPS:
        for L in GROUPS:
            s = (fit(T, use, K, L, 3)[aside] == T[aside]).mean()
            if s > best[0]: best = (s, K, L)
    return fit(T, seen, best[1], best[2]), best[1:]


def run(source, kind, inst, step=6000):
    t0 = time.time(); task = make_task(kind, inst); rng = np.random.default_rng(search.crc(f"co-{kind}-{inst}")); a, b = torch.cartesian_prod(torch.arange(N), torch.arange(N)).T; held, picks = [], []
    if source == "net":
        st = fresh(task); stretch(st, task, step); st["m"].eval()
    te = [x.numpy() for x in task["te"]]; tr = [x.numpy() for x in task["tr"]]
    for s in range(SKILLS):
        if source == "net":
            with torch.no_grad(): T = st["m"](inputs(a, b, s)).argmax(1).view(N, N).numpy()
            T = np.minimum(T, CLASSES - 1); seen = np.ones((N, N), bool)
        else:
            T = np.zeros((N, N), int); seen = np.zeros((N, N), bool); T[tr[0], tr[1]] = task["train"][s][task["tr"]].numpy(); seen[tr[0], tr[1]] = True
        P, pick = solve(T, seen, rng); picks.append(list(map(int, pick))); held.append(round(float((P[te[0], te[1]] == task["clean"][s][task["te"]].numpy()).mean()), 3))
    return dict(source=source, kind=kind, inst=inst, step=step if source == "net" else 0, held_out=held, mean=round(sum(held) / 4, 3), groups=picks, seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    row = run(sys.argv[1], sys.argv[2], int(sys.argv[3]), *map(int, sys.argv[4:])); print(json.dumps(row), flush=True)
    with open(OUT, "a") as f: f.write(json.dumps(row) + "\n")
