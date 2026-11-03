"""Training methods built for wrong labels, for the `noisy` task. Added after run r03 passed screening and before any second-stage run, to make that claim harder.

python3 robust.py INST     every method on this task, to step 6000, rows in search_runs/robust.jsonl
python3 robust.py test     every method for 40 steps

Same network, batches and pauses as search.py. `forward` and `8 outputs` are told what the editor had to find out: the noise rate and that only 8 classes occur.
"""
import json, multiprocessing, os, sys, time
import torch, torch.nn.functional as F
import search
from search import BATCH, FAR, LR, READ, RUNS, SKILLS, fresh, held, inputs, make_task, recorded
OUT = os.path.join(RUNS, "robust.jsonl"); NOISE_MATRIX = torch.eye(63); NOISE_MATRIX[:8, :8] = .25 / 7; NOISE_MATRIX[range(8), range(8)] = .75


def ce(x, y): return F.cross_entropy(x, y)
def smooth(e): return lambda x, y: F.cross_entropy(x, y, label_smoothing=e)
def gce(q): return lambda x, y: ((1 - F.softmax(x, 1).gather(1, y[:, None]) ** q) / q).mean()                        # Zhang and Sabuncu 2018
def bootstrap(b): return lambda x, y: -((b * F.one_hot(y, 63) + (1 - b) * F.softmax(x, 1).detach()) * F.log_softmax(x, 1)).sum(1).mean()  # Reed et al. 2015
def forward(x, y): return F.nll_loss(torch.log(F.softmax(x, 1) @ NOISE_MATRIX.T + 1e-9), y)                          # Patrini et al. 2017, true noise given
def eight(x, y): return F.cross_entropy(x[:, :8], y)


LOSSES = {"label smoothing .1": smooth(.1), "label smoothing .3": smooth(.3), "generalized cross entropy .7": gce(.7), "bootstrap .8": bootstrap(.8), "forward correction": forward, "8 outputs": eight, "mixup .2": ce}
SETS = [dict(lr=LR, wd=0.), dict(lr=.01, wd=.5)]


def train(st, task, name, end, rows=None):
    a, b = task["tr"]; m, o, g = st["m"], st["o"], st["g"]; loss = LOSSES[name]; mix = torch.Generator().manual_seed(7)
    while st["step"] < end:
        l = 0.
        for i in range(SKILLS):
            j = torch.randint(len(a), (BATCH,), generator=g); y = task["train"][i][a[j], b[j]]; x = inputs(a[j], b[j], i)
            if name.startswith("mixup"):
                k = torch.randperm(BATCH, generator=mix); w = torch.distributions.Beta(.2, .2).sample().item(); h = m(w * x + (1 - w) * x[k]); l = l + w * ce(h, y) + (1 - w) * ce(h, y[k])
            else: l = l + loss(m(x), y)
        o.zero_grad(); l.backward(); o.step(); st["step"] += 1
        if rows is not None and st["step"] % READ == 0: rows[st["step"]] = held(st, task)
    return rows


def one(job):
    torch.set_num_threads(1); inst, name, s, end = job; task = make_task("noisy", inst); st = fresh(task, s["lr"], s["wd"]); t0 = time.time()
    rows = train(st, task, name, end, {}); return dict(kind="noisy", inst=inst, name=f"{name} lr {s['lr']} wd {s['wd']}", rows=rows, seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    test = sys.argv[1] == "test"; inst = 99 if test else int(sys.argv[1]); end = 50 if test else FAR
    have = {(r["inst"], r["name"]) for r in recorded(OUT)}
    jobs = [(inst, n, s, end) for n in LOSSES for s in SETS if test or (inst, f"{n} lr {s['lr']} wd {s['wd']}") not in have]
    with multiprocessing.get_context("spawn").Pool(3) as p:
        for r in p.imap_unordered(one, jobs):
            if test: print(r["name"], r["seconds"], "s", round(sum(r["rows"][50]) / 4, 3), flush=True); continue
            with open(OUT, "a") as f: f.write(json.dumps(r) + "\n")
            print(r["name"], r["seconds"], "s, at 1650", round(sum(r["rows"][1650]) / 4, 3), "at 6000", round(sum(r["rows"][FAR]) / 4, 3), flush=True)
