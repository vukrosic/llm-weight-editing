"""Does the network hold anything the training answers do not? Read the rule off a network with no answers left, and search for the rule with only some answers left.

python3 data_gone.py read KIND INST STEP       no answers, no training-pair checks: is the rule better than the network it came from?
python3 data_gone.py some KIND INST FRACTION   no network: search a fraction of the training answers
python3 data_gone.py all INSTS                 every setting, four at a time. Rows go to search_runs/data_gone.jsonl
python3 data_gone.py strong INSTS              read a well-trained network at step 6000: lr .01 wd .5, then the setting that most often wins at 6000 for each kind
"""
import json, multiprocessing, os, sys
import numpy as np, torch
import search
from read_rule import agreement, answers, from_labels, from_net, recover
from search import N, RUNS, SKILLS, acc, fresh, make_task, m4, stretch
OUT = os.path.join(RUNS, "data_gone.jsonl"); KINDS = ("modular", "tables", "noisy"); STEPS = (300, 600, 1000, 1500, 3000, 6000); FRACTIONS = (.02, .05, .1, .2, .5); BEST = dict(modular=(.01, .1, .1), tables=(.01, .1, .1), noisy=(search.LR, 0., .3))


def rule_score(rules, task):
    a, b = task["te"]; a, b = a.numpy(), b.numpy(); return [round(float((answers(*r)[a, b] == task["clean"][s][task["te"]].numpy()).mean()), 3) for s, r in enumerate(rules)]


def read(kind, inst, step, lr=search.LR, wd=0., drop=0.):
    task = make_task(kind, inst); st = fresh(task, lr, wd, drop); hist = []; stretch(st, task, step, hist=hist); lps = from_net([s for _, s in hist[-5:]]); rng = np.random.default_rng(search.crc(f"gone-{kind}-{inst}")); rules = []
    for s in range(SKILLS):
        found = recover(lps[s], rng); fam = max(found, key=lambda k: agreement(lps[s], answers(k, *found[k]))); rules.append((fam, *found[fam]))
    return dict(test="read", lr=lr, wd=wd, drop=drop, kind=kind, inst=inst, step=step, families=[r[0] for r in rules], rule=rule_score(rules, task), network=acc(st["m"], task["te"], task["clean"]))


def some(kind, inst, frac):
    task = make_task(kind, inst); g = torch.Generator().manual_seed(search.crc(f"some-{kind}-{inst}-{frac}")); k = len(task["tr"][0]); j = torch.randperm(k, generator=g)[:int(frac * k)]; part = dict(task, tr=(task["tr"][0][j], task["tr"][1][j])); lps = from_labels(part); rng = np.random.default_rng(search.crc(f"some-{kind}-{inst}")); a, b = [x.numpy() for x in part["tr"]]; rules = []
    for s in range(SKILLS):
        found = recover(lps[s], rng, seen=len(j) / N ** 2); y = task["train"][s][part["tr"]].numpy(); fam = max(found, key=lambda f: (answers(f, *found[f])[a, b] == y).mean()); rules.append((fam, *found[fam]))
    return dict(test="some", kind=kind, inst=inst, fraction=frac, answers_per_skill=len(j), families=[r[0] for r in rules], rule=rule_score(rules, task))


def job(x):
    torch.set_num_threads(1); return (read if x[0] == "read" else some)(*x[1:])


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "read": print(json.dumps(read(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))))
    elif c == "some": print(json.dumps(some(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))))
    else:
        insts = list(map(int, sys.argv[2].split(",")))
        jobs = [("read", k, i, 6000, *BEST[k]) for k in KINDS for i in insts] if c == "strong" else [("read", k, i, s) for k in KINDS for i in insts for s in STEPS] + [("some", k, i, f) for k in KINDS for i in insts for f in FRACTIONS]
        with multiprocessing.get_context("spawn").Pool(4) as p:
            for r in p.imap_unordered(job, jobs):
                with open(OUT, "a") as f: f.write(json.dumps(r) + "\n")
                print(r["test"], r["kind"], r["inst"], r.get("step", r.get("fraction")), round(m4(r["rule"]), 3), round(m4(r["network"]), 3) if "network" in r else "", flush=True)
