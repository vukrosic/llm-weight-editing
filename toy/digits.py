"""Does an LLM's written network still beat training when the data has no short rule? Handwritten digits, through the same bench as search.py.

python3 digits.py new RUN INST        make a run. INST picks the 70/30 split
python3 digits.py step RUN - <<'REPLY' the editor's only command. The first reply is START
python3 digits.py score RUN           once, after the run: the weights as written, and best reading if training then runs
python3 digits.py audit RUN TRANSCRIPT void the run if the editor used anything but that one command
python3 digits.py controls INST       every training setting to step 6000
python3 digits.py rivals INST         plain classifiers with no network limit, on the same split
python3 digits.py table               every run against best training and the rivals
python3 digits.py test                every path in seconds
"""
import json, multiprocessing, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); os.environ.setdefault("SEARCH_RUNS", os.path.join(HERE, "digit_runs")); sys.path.insert(0, HERE)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_digits
import search as S

_d = load_digits(); X = torch.tensor(_d.data / 16., dtype=torch.float32); Y = torch.tensor(_d.target); IN, W, CLASSES, KIND = X.shape[1], 48, 10, "digits"
S.SKILLS = 1; S.ME = os.path.abspath(__file__); S.m4 = lambda x: x[0]; S.inputs = lambda a, b, s: X[a]


class Net(nn.Module):
    def __init__(s, drop=0.):
        super().__init__(); s.l1 = nn.Linear(IN, W); s.l2 = nn.Linear(W, W); s.l3 = nn.Linear(W, CLASSES); s.drop = drop
    def forward(s, x): return s.l3(F.dropout(F.silu(s.l2(F.dropout(F.silu(s.l1(x)), s.drop, s.training))), s.drop, s.training))


S.Net = Net


def make_task(kind, inst):
    g = torch.Generator().manual_seed(S.crc(f"split-{kind}-{inst}")); p = torch.randperm(len(X), generator=g); k = int(.7 * len(X)); t = [Y[:, None]]
    return dict(kind=kind, inst=inst, clean=t, train=t, tr=(p[:k], torch.zeros(k, dtype=torch.long)), te=(p[k:], torch.zeros(len(X) - k, dtype=torch.long)))


S.make_task = make_task
N_WEIGHTS = sum(p.numel() for p in Net().parameters())


def task_text(mode):
    return f"""You are working on a small neural network, and you may set its weights by hand. The aim is high accuracy on images the network never saw.

The task: handwritten digits. Each image is 8 by 8 pixels, flattened to {IN} numbers from 0 to 1. The class is the digit 0 to 9.

The network: input {IN}, then l1 (width {W}), silu, l2 (width {W}), silu, l3 ({CLASSES} outputs, read as a class). That is {N_WEIGHTS:,} weights.

Right now no training has happened, and the weights are random. You may change them now, once. Your folder has train_images.npy ({len(make_task(KIND, 0)['tr'][0])} by {IN}) and train_labels.npy: every training image and its label. 30% of the images are held out, and you never see them.

Two readings count. The main one is the weights exactly as you wrote them, with no training at all. A second reading then runs training on from your weights, with the speeds you set, to step {S.FAR}. Training: Adam at learning rate {S.LR}, {S.BATCH} examples per step. You are compared with the best of many training settings at any step.

Your work folder has:
- weights/<name>.npy: the current weights. Names: {", ".join(n + "." + k for n in ("l1", "l2", "l3") for k in ("weight", "bias"))}

You can:
- Run Python with numpy and scipy, up to {S.MAX_BLOCKS} blocks per reply, each up to {S.CPU} CPU seconds, in the form
```python
print(load("l1.weight").shape)
```
  Helpers: NAMES, SHAPES, load(name), load_edit(name), save_edit(name, array), drop_edit(name). Files stay between runs.
- Change weights with save_edit. It applies at CONTINUE.
- Change how fast weights move in the second reading: np.save("speed/<name>.npy", array) with the tensor's shape and values from 0 to {S.MAX_SPEED:g}. 1 is normal and 0 freezes a weight.
- Write EVAL on its own line: accuracy on the training images with your current edit, {S.MAX_EVALS} times in all.
- Write CONTINUE on its own line when you are finished: this applies your changes and ends your work.

Not allowed: computing gradients, or any iterative fitting of weights to a loss, in your own Python. Reasoning about the data, computing statistics from it, and setting numbers from that are all fine. Every run's code is read by a person.

You have {S.MAX_REPLIES} replies in all. In each reply say in one or two short sentences what you saw and what you expect next. Blocks run in order, then EVAL or CONTINUE. At the end, name what you wrote into the weights and why it should work on images it never saw."""


S.task_text = task_text
_new = S.new


def new(run, inst):
    _new(run, KIND, inst, "answers"); w = os.path.join(S.RUNS, run, "work"); os.remove(os.path.join(w, "answers.jsonl")); a = make_task(KIND, int(inst))["tr"][0]
    np.save(os.path.join(w, "train_images.npy"), X[a].numpy()); np.save(os.path.join(w, "train_labels.npy"), Y[a].numpy())


def advance(st, d, task, mode):
    """Keep the weights exactly as written for the main reading, then the first training stretch happens only in score."""
    s, bad = S.edited(st, d); st["m"].load_state_dict(s); e = os.path.join(d, "work", "edit")
    if os.path.isdir(e): S.shutil.rmtree(e)
    sp = os.path.join(d, "work", "speed"); out = list(bad)
    for f in sorted(os.listdir(sp)) if os.path.isdir(sp) else []:
        k = f[:-4]
        try: st["speed"][k] = torch.tensor(np.clip(S.plain_file(os.path.join(sp, f), tuple(s[k].shape)), 0, S.MAX_SPEED))
        except Exception as x: out.append(f"{f}: {x}")
    S.log(d, kind="edit", step=st["step"], speed={k: round(float(v.mean()), 3) for k, v in st["speed"].items()}); st["step"] = S.END[mode]; out.append(f"Your work has finished. Training-image accuracy: {S.on_train(st['m'], task)}"); return "\n".join(out)


S.advance = advance


def score(run):
    if any(r["run"] == run for r in S.recorded(S.RESULTS)): raise SystemExit("one score per run")
    d = os.path.join(S.RUNS, run); c = S.config(d); task = make_task(c["kind"], c["inst"]); st = S.load(d, task)
    if st["step"] != S.END["answers"]: raise SystemExit("not finished")
    f = os.path.join(d, "audit.json")
    if os.environ.get("DIGITS_TEST") is None and (not os.path.exists(f) or not json.load(open(f))["ok"]): raise SystemExit("this run has not passed its audit")
    written = S.held(st, task)[0]; end = int(os.environ.get("DIGITS_FAR", S.FAR)); rows = S.follow(st, task, end=st["step"] + end - S.END["answers"] if end < S.FAR else S.FAR)
    row = dict(run=run, inst=c["inst"], written=written, after=max(v[0] for v in rows.values()), replies=st["replies"], evals=st["evals"], flags=st["flags"], speed={k: round(float(v.mean()), 3) for k, v in st["speed"].items()})
    with open(S.RESULTS, "a") as o: o.write(json.dumps(row) + "\n")
    print(json.dumps(row))


def controls(inst, procs=6):
    have = {r["name"] for r in S.recorded(S.CONTROLS) if r["inst"] == inst}; jobs = [(KIND, inst, s) for s in S.settings() if "edit" not in s and s["name"] not in have]; os.makedirs(S.RUNS, exist_ok=True)
    with multiprocessing.get_context("spawn").Pool(procs) as p:
        for r in p.imap_unordered(S.one_control, jobs):
            with open(S.CONTROLS, "a") as f: f.write(json.dumps(r) + "\n")
            print(r["name"], r["seconds"], "s, best", round(max(v[0] for v in r["rows"].values()), 4), flush=True)


def rivals(inst):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.neighbors import KNeighborsClassifier as KNN, NearestCentroid as NC
    from sklearn.multiclass import OneVsOneClassifier as OVO
    from sklearn.svm import SVC
    t = make_task(KIND, inst); a, b = t["tr"][0].numpy(), t["te"][0].numpy(); x, y = X.numpy(), Y.numpy(); out = dict(inst=inst)
    for n, m in (("nearest class mean", NC()), ("linear discriminant", LDA()), ("logistic regression", LR(max_iter=2000)), ("1 nearest neighbour", KNN(1)), ("one-vs-one linear discriminant", OVO(LDA(solver="lsqr", shrinkage=.1))), ("support vector machine", SVC())): out[n] = round(float(m.fit(x[a], y[a]).score(x[b], y[b])), 4)
    with open(os.path.join(S.RUNS, "rivals.jsonl"), "a") as f: f.write(json.dumps(out) + "\n")
    print(out)


def best_training(inst): return max((max(v[0] for k, v in r["rows"].items() if int(k) >= 25), r["name"]) for r in S.recorded(S.CONTROLS) if r["inst"] == inst)


def table():
    riv = {r["inst"]: r for r in S.recorded(os.path.join(S.RUNS, "rivals.jsonl"))}
    for r in S.recorded(S.RESULTS): print(r["run"], "inst", r["inst"], "written", r["written"], "then trained", r["after"], "best training", best_training(r["inst"]), riv.get(r["inst"]))


def test():
    """Every path, in seconds, with a hand-written editor and a short horizon."""
    import shutil, tempfile
    S.RUNS = tempfile.mkdtemp(); S.RESULTS, S.CONTROLS = os.path.join(S.RUNS, "results.jsonl"), os.path.join(S.RUNS, "controls.jsonl"); os.environ["DIGITS_TEST"] = "1"; S.FAR = 300
    new("t", 0); S.step("t", "START")
    S.step("t", """```python
x, y = np.load("train_images.npy"), np.load("train_labels.npy"); m = np.stack([x[y == c].mean(0) for c in range(10)]); print(m.shape)
l1 = np.zeros(SHAPES["l1.weight"]); l1[:10] = m - m.mean(0); save_edit("l1.weight", l1 * 5); save_edit("l1.bias", np.zeros(48))
l2 = np.zeros(SHAPES["l2.weight"]); l2[:10, :10] = np.eye(10) * 3; save_edit("l2.weight", l2); save_edit("l2.bias", np.zeros(48))
l3 = np.zeros(SHAPES["l3.weight"]); l3[:10, :10] = np.eye(10) * 3; save_edit("l3.weight", l3); save_edit("l3.bias", np.zeros(10))
for n in NAMES: np.save("speed/" + n + ".npy", np.ones(SHAPES[n]))
```
EVAL""")
    S.step("t", "CONTINUE"); S.step("t", "more"); score("t"); r = S.recorded(S.RESULTS)[-1]; assert 0.1 < r["written"] <= 1 and r["after"] >= r["written"] - .1, r
    assert not os.path.exists(os.path.join(S.RUNS, "t", "work", "answers.jsonl")); rivals(0); print("ok", N_WEIGHTS); shutil.rmtree(S.RUNS)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "new": new(sys.argv[2], int(sys.argv[3]))
    elif c == "step": S.step(sys.argv[2], sys.stdin.read() if sys.argv[3] == "-" else sys.argv[3])
    elif c == "score": score(sys.argv[2])
    elif c == "audit": S.audit(sys.argv[2], sys.argv[3])
    elif c == "controls": controls(int(sys.argv[2]))
    elif c == "rivals": rivals(int(sys.argv[2]))
    elif c == "table": table()
    elif c == "test": test()
