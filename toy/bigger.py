"""Can an LLM write a large network's weights on real image tasks? MNIST and CIFAR-10, standard test sets, through the same bench as search.py.

python3 bigger.py new RUN TASK          make a run. TASK is mnist or cifar
python3 bigger.py step RUN - <<'REPLY'  the editor's only command. The first reply is START
python3 bigger.py audit RUN TRANSCRIPT  void the run if the editor used anything but that one command
python3 bigger.py score RUN             once, after the run: the weights as written, then training on from them at three speeds
python3 bigger.py controls TASK         every training setting, resumable, stops before 540 s
python3 bigger.py closedform TASK       plain code writes the same network with no gradient, then training on from it
python3 bigger.py rivals TASK           classic classifiers with no network limit
python3 bigger.py table                 every run against best training and the rivals
python3 bigger.py test                  every path in seconds on a small slice of the data
"""
import copy, json, os, re, shutil, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.environ.setdefault("SEARCH_RUNS", os.path.join(HERE, "bigger_runs")); sys.path.insert(0, HERE)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import search as S

DATA, DEV = os.path.join(HERE, "data"), "mps" if torch.backends.mps.is_available() else "cpu"
BATCH, EPOCHS, SMALL = 128, dict(mnist=30, cifar=60), int(os.environ.get("BIG_SMALL", 0))
MEAN, STD = torch.tensor([.4914, .4822, .4465])[:, None, None], torch.tensor([.247, .243, .261])[:, None, None]
S.SKILLS = 1; S.ME = os.path.abspath(__file__); S.m4 = lambda x: x[0]; S.CPU = 120
S.LOOKS_LIKE_TRAINING = re.compile(S.LOOKS_LIKE_TRAINING.pattern + r"|k-?means|for \w+ in range\(\w*iter|while ", re.I)
torch.set_num_threads(4)
_cache = {}


def images(kind, part):
    """Normalized images and labels, on the training device. Editors get the same pixels as uint8."""
    k = (kind, part)
    if k not in _cache:
        x, y = np.load(os.path.join(DATA, f"{kind}_{part}_x.npy")), np.load(os.path.join(DATA, f"{kind}_{part}_y.npy"))
        if SMALL: x, y = x[:SMALL], y[:SMALL]
        x = torch.tensor(x, dtype=torch.float32) / 255
        _cache[k] = (x.flatten(1) if kind == "mnist" else (x - MEAN) / STD).to(DEV), torch.tensor(y).to(DEV)
    return _cache[k]


class Mlp(nn.Module):
    def __init__(s, drop=0.):
        super().__init__(); s.l1 = nn.Linear(784, 512); s.l2 = nn.Linear(512, 512); s.l3 = nn.Linear(512, 10); s.drop = drop
    def forward(s, x): return s.l3(F.dropout(F.silu(s.l2(F.dropout(F.silu(s.l1(x)), s.drop, s.training))), s.drop, s.training))


class Cnn(nn.Module):
    def __init__(s, drop=0.):
        super().__init__(); s.c1 = nn.Conv2d(3, 64, 3, padding=1); s.c2 = nn.Conv2d(64, 128, 3, padding=1); s.c3 = nn.Conv2d(128, 256, 3, padding=1); s.l = nn.Linear(4096, 10); s.drop = drop
    def features(s, x):
        for c in (s.c1, s.c2, s.c3): x = F.max_pool2d(F.silu(c(x)), 2)
        return x.flatten(1)
    def forward(s, x): return s.l(F.dropout(s.features(x), s.drop, s.training))


NETS = dict(mnist=Mlp, cifar=Cnn)
_kind = [None]


def Net(drop=0.): return NETS[_kind[0]](drop)


S.Net = Net
def make_task(kind, inst=0): _kind[0] = kind; return dict(kind=kind, inst=inst)
S.make_task = make_task


def accuracy(m, x, y):
    was = m.training; m.eval(); dev = next(m.parameters()).device; x_, y_ = x.to(dev), y.to(dev)
    with torch.no_grad(): r = sum((m(x_[i:i + 2000]).argmax(1) == y_[i:i + 2000]).sum().item() for i in range(0, len(x_), 2000)) / len(x_)
    m.train(was); return round(r, 4)


def on_test(m, kind): return accuracy(copy.deepcopy(m).to(DEV), *images(kind, "test"))
def on_train(m, task): return [accuracy(copy.deepcopy(m).to(DEV), *images(task["kind"], "train"))]
S.on_train = on_train


def crops(x):
    """Random flip and random crop with 4 pixels of padding, a different one per image."""
    n = len(x); x = torch.where((torch.rand(n, device=x.device) < .5)[:, None, None, None], x.flip(3), x); p = F.pad(x, (4, 4, 4, 4)); g = torch.arange(32, device=x.device)
    r, c = torch.randint(0, 9, (n,), device=x.device)[:, None] + g, torch.randint(0, 9, (n,), device=x.device)[:, None] + g
    return p[torch.arange(n, device=x.device)[:, None, None, None], torch.arange(3, device=x.device)[None, :, None, None], r[:, None, :, None], c[:, None, None, :]]


def warps(x):
    """Random shift up to 2 pixels, rotation up to 12 degrees and shear up to .2, a different one per flat 28 by 28 image."""
    n = len(x); d = x.device; a = (torch.rand(n, device=d) * 2 - 1) * .2094; s = (torch.rand(n, device=d) * 2 - 1) * .2; dx, dy = (torch.rand(2, n, device=d) * 2 - 1) / 7
    t = torch.stack([torch.stack([a.cos(), s - a.sin(), dx], 1), torch.stack([a.sin(), a.cos(), dy], 1)], 1)
    return F.grid_sample(x.view(n, 1, 28, 28), F.affine_grid(t, (n, 1, 28, 28), align_corners=False), padding_mode="zeros", align_corners=False).view(n, 784)


def train(m, kind, lr, wd=0., drop=0., aug=False, speed=None, name="", epochs=None):
    """Adam with weight decay and a cosine fall to zero, full budget. Returns the test accuracy after every epoch."""
    m = m.to(DEV); m.drop = drop; x, y = images(kind, "train"); epochs = epochs or (1 if SMALL else EPOCHS[kind]); g = torch.Generator().manual_seed(S.crc(f"train-{kind}-{name}")); torch.manual_seed(S.crc(f"drop-{kind}-{name}"))
    if DEV == "mps": torch.mps.manual_seed(S.crc(f"drop-{kind}-{name}"))
    o = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd); per = max(len(x) // BATCH, 1); total = per * epochs; sch = torch.optim.lr_scheduler.CosineAnnealingLR(o, total); rows = {0: on_test(m, kind)}; speed = {k: v.to(DEV) for k, v in (speed or {}).items()}; m.train()
    for e in range(epochs):
        p = torch.randperm(len(x), generator=g).to(DEV)
        for i in range(per):
            j = p[i * BATCH:(i + 1) * BATCH]; b = (warps(x[j]) if kind == "mnist" else crops(x[j])) if aug else x[j]; before = {k: q.detach().clone() for k, q in m.named_parameters() if k in speed}
            o.zero_grad(); F.cross_entropy(m(b), y[j]).backward(); o.step(); sch.step()
            with torch.no_grad():
                for k, q in m.named_parameters():
                    if k in speed: q.copy_(before[k] + speed[k] * (q - before[k]))
        rows[e + 1] = on_test(m, kind)
    return rows


def initial(kind, name=""):
    make_task(kind); torch.manual_seed(S.crc(f"init-{kind}-{name}")); return Net()


def settings(kind):
    if kind == "mnist": return [dict(lr=l, drop=d, wd=0., aug=False) for l in (.003, .001, .0003) for d in (0., .2, .5)] + [dict(lr=l, drop=d, wd=0., aug=True) for l in (.003, .001) for d in (0., .2)]
    return [dict(lr=l, drop=d, wd=0., aug=True) for l in (.003, .001, .0003) for d in (0., .3)] + [dict(lr=.001, drop=d, wd=0., aug=False) for d in (0., .3)]


def recorded(name): return S.recorded(os.path.join(S.RUNS, name))
def add(name, row):
    os.makedirs(S.RUNS, exist_ok=True)
    with open(os.path.join(S.RUNS, name), "a") as f: f.write(json.dumps(row) + "\n")


def controls(kind, limit=540):
    start = time.time(); done = {json.dumps(r["setting"]) for r in recorded("controls.jsonl") if r["kind"] == kind}; took = 0.
    for s in settings(kind):
        if json.dumps(s) in done: continue
        if time.time() - start + took > limit: print("stopping before the limit, run this again"); return
        t = time.time(); rows = train(initial(kind), kind, name="control", **s); took = time.time() - t
        add("controls.jsonl", dict(kind=kind, setting=s, seconds=round(took, 1), rows=rows, last=rows[max(rows)], best=max(rows.values()))); print(s, round(took), "s, last", rows[max(rows)], "best", max(rows.values()), flush=True)
    print("all settings done")


def best_control(kind): return max((r for r in recorded("controls.jsonl") if r["kind"] == kind), key=lambda r: r["best"])


def ridge(f, y, lams=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)):
    """Least squares from features to one-hot labels. The penalty, relative to the features' mean variance, is picked on the last tenth of the training images. One solve per penalty."""
    v, k = len(f) // 10, f.shape[1]; mu = f.mean(0); fc = (f - mu).astype(np.float64); oh = np.eye(10)[y]; A, b = fc[:-v].T @ fc[:-v], fc[:-v].T @ oh[:-v]; scale = np.trace(A) / k; best = (-1, None)
    for lam in lams:
        w = np.linalg.solve(A + lam * scale * np.eye(k), b); best = max(best, (((fc[-v:] @ w).argmax(1) == y[-v:]).mean(), lam), key=lambda z: z[0])
    A, b = fc.T @ fc, fc.T @ oh; w = np.linalg.solve(A + best[1] * np.trace(A) / k * np.eye(k), b); return w, -mu @ w, best


def set_last(m, last, w, b):
    with torch.no_grad(): last.weight.copy_(torch.tensor(w.T, dtype=torch.float32).to(next(m.parameters()).device)); last.bias.copy_(torch.tensor(b, dtype=torch.float32).to(next(m.parameters()).device))


def batched(fn, x):
    with torch.no_grad(): return torch.cat([fn(x[i:i + 2000]).cpu() for i in range(0, len(x), 2000)]).numpy()


def closed_mnist(whiten):
    """Random l1 and l2, then one ridge solve for l3. The whitened version sets l1 to random directions of the whitened pixels."""
    m = initial("mnist", "closed").to(DEV); x, y = images("mnist", "train")
    if whiten:
        a = x.cpu().numpy().astype(np.float64); mu = a.mean(0); e, v = np.linalg.eigh(np.cov(a.T) + 1e-1 * np.eye(784) * .1); W = v / np.sqrt(e) @ v.T; g = torch.Generator().manual_seed(S.crc("whiten-mnist")); R = torch.randn(512, 784, generator=g).numpy() / np.sqrt(784) * 4; l1 = R @ W
        with torch.no_grad(): m.l1.weight.copy_(torch.tensor(l1, dtype=torch.float32).to(DEV)); m.l1.bias.copy_(torch.tensor(-l1 @ mu, dtype=torch.float32).to(DEV))
    f = batched(lambda b: F.silu(m.l2(F.silu(m.l1(b)))), x); w, b, best = ridge(f, y.cpu().numpy()); set_last(m, m.l3, w, b); return m, best


def closed_cifar(patches):
    """Random convolutions, then one ridge solve for the last layer. The patch version sets c1 to random whitened 3 by 3 patches of training images."""
    m = initial("cifar", "closed").to(DEV); x, y = images("cifar", "train")
    if patches:
        g = torch.Generator().manual_seed(S.crc("patch-cifar")); p = F.unfold(x[:4000].cpu(), 3); p = p.permute(0, 2, 1).reshape(-1, 27); p = p[torch.randperm(len(p), generator=g)[:200000]].numpy().astype(np.float64); mu = p.mean(0); e, v = np.linalg.eigh(np.cov(p.T) + .01 * np.eye(27)); W = v / np.sqrt(e) @ v.T
        q = (p[:64] - mu) @ W; q /= np.linalg.norm(q, axis=1, keepdims=True); q = q @ W
        with torch.no_grad(): m.c1.weight.copy_(torch.tensor(q.reshape(64, 3, 3, 3), dtype=torch.float32).to(DEV)); m.c1.bias.copy_(torch.tensor(-q @ mu, dtype=torch.float32).to(DEV))
    f = batched(m.features, x); w, b, best = ridge(f, y.cpu().numpy()); set_last(m, m.l, w, b); return m, best


def scaled(kind):
    """Random weights at the size that does best on the last tenth of the training images. The default initial size leaves the silu nearly linear."""
    x, y = images(kind, "train"); best = None
    for k in (.25, .5, 1, 2, 4, 8):
        m = initial(kind, "closed").to(DEV)
        with torch.no_grad():
            for n, p in m.named_parameters():
                if "weight" in n: p.mul_(k * (3 ** .5))
                else: p.mul_(k)
        last = m.l3 if kind == "mnist" else m.l; f = batched((lambda b: F.silu(m.l2(F.silu(m.l1(b))))) if kind == "mnist" else m.features, x); w, b, r = ridge(f, y.cpu().numpy()); set_last(m, last, w, b); print("  size", k, "held back images", round(float(r[0]), 4), flush=True)
        if best is None or r[0] > best[1][0]: best = (m, r, k)
    print("  chose size", best[2]); return best[:2]


def pca_filters(p, k, gain):
    """Whitened top principal directions of the rows of p, in both signs, one eigendecomposition. Units beyond the input's dimension stay zero."""
    mu = p.mean(0); e, v = np.linalg.eigh(np.cov(p.T)); n = min(k // 2, p.shape[1]); d = e[::-1][:n]; q = v[:, ::-1][:, :n] / np.sqrt(d + 1e-3 * e.mean()); q = np.zeros((k, p.shape[1])) + np.pad(np.concatenate([q.T, -q.T]), ((0, k - 2 * n), (0, 0))); return gain * q, -gain * q @ mu


def pca_layers(kind):
    """The PCANet recipe inside this network: every layer gets whitened PCA filters, then one ridge solve. Only the gain and the penalty are chosen, on the last tenth of the training images."""
    x, y = images(kind, "train"); best = None; sub = x[:4000].cpu(); g = torch.Generator().manual_seed(S.crc("pca-" + kind))
    for gain in (.5, 1, 2):
        m = initial(kind, "closed").to(DEV); f = sub
        for c in ((m.l1, m.l2) if kind == "mnist" else (m.c1, m.c2, m.c3)):
            if kind == "mnist":
                w, b = pca_filters(f.numpy().astype(np.float64), c.weight.shape[0], gain); shape = c.weight.shape
            else:
                p = F.unfold(f, 3, padding=1).permute(0, 2, 1).flatten(0, 1); p = p[torch.randperm(len(p), generator=g)[:200000]]; w, b = pca_filters(p.numpy().astype(np.float64), c.out_channels, gain); shape = c.weight.shape
            with torch.no_grad(): c.weight.copy_(torch.tensor(w.reshape(shape), dtype=torch.float32).to(DEV)); c.bias.copy_(torch.tensor(b, dtype=torch.float32).to(DEV))
            act = (lambda t: F.silu(c(t))) if kind == "mnist" else (lambda t: F.max_pool2d(F.silu(c(t)), 2))
            f = torch.tensor(batched(act, f.to(DEV)))
        last = m.l3 if kind == "mnist" else m.l; h = batched((lambda t: F.silu(m.l2(F.silu(m.l1(t))))) if kind == "mnist" else m.features, x); w, b, r = ridge(h, y.cpu().numpy()); set_last(m, last, w, b); print("  gain", gain, "held back", round(float(r[0]), 4), flush=True)
        if best is None or r[0] > best[1][0]: best = (m, r)
    return best


CLOSED = dict(mnist=dict(random=lambda: closed_mnist(False), whitened=lambda: closed_mnist(True), scaled=lambda: scaled("mnist"), pca=lambda: pca_layers("mnist")), cifar=dict(random=lambda: closed_cifar(False), patches=lambda: closed_cifar(True), scaled=lambda: scaled("cifar"), pca=lambda: pca_layers("cifar")))


def speeds(kind): return [best_control(kind)["setting"]["lr"] * k for k in (1, .3, .1)]


def after(state, kind, speed=None, name="", keep=None):
    """Training on from written weights, same budget as a control, at three learning rates tied to the best control's. Finished readings stay in keep, so an interrupted score resumes."""
    out = json.load(open(keep)) if keep and os.path.exists(keep) else {}
    s = best_control(kind)["setting"]
    for lr in speeds(kind):
        if f"{lr:.5g}" in out: continue
        m = Net(); m.load_state_dict(state); rows = train(m, kind, name=name, speed=speed, **dict(s, lr=lr)); out[f"{lr:.5g}"] = dict(rows=rows, last=rows[max(rows)], best=max(rows.values()))
        if keep: json.dump(out, open(keep, "w"))
        print("  trained on at", f"{lr:.5g}", "last", rows[max(rows)], "best", max(rows.values()), flush=True)
    return out


def longer(kind, run, passes=120, limit=540):
    """Is a written start only a head start? Best three training settings and the run's weights, at four times the budget."""
    start = time.time(); have = {(r["who"], json.dumps(r["setting"])) for r in recorded("longer.jsonl") if r["kind"] == kind and r["passes"] == passes}; top = sorted((r for r in recorded("controls.jsonl") if r["kind"] == kind), key=lambda r: -r["last"])[:3]; took = 0.
    d = os.path.join(S.RUNS, run); st = S.load(d, make_task(kind)); state = {k: v.clone() for k, v in st["m"].state_dict().items()}; b = best_control(kind)["setting"]
    jobs = [("training", r["setting"], None) for r in top] + [(run, dict(b, lr=l), state) for l in speeds(kind)]
    for who, s, w in jobs:
        if (who, json.dumps(s)) in have: continue
        if time.time() - start + took > limit: print("stopping before the limit, run this again"); return
        t = time.time(); m = initial(kind) if w is None else Net()
        if w is not None: m.load_state_dict(w)
        rows = train(m, kind, name=f"longer-{who}", epochs=passes, **s); took = time.time() - t; add("longer.jsonl", dict(kind=kind, who=who, passes=passes, setting=s, rows=rows, last=rows[max(rows)], best=max(rows.values()), seconds=round(took, 1))); print(who, s, round(took), "s, last", rows[max(rows)], "best", max(rows.values()), flush=True)
    print("all done")


def closedform(kind):
    have = {r["name"] for r in recorded("closed.jsonl") if r["kind"] == kind}
    for n, fn in CLOSED[kind].items():
        if n in have: continue
        t = time.time(); m, best = fn(); took = time.time() - t; w = on_test(m, kind); print(kind, n, "written", w, "train", on_train(m, dict(kind=kind)), "seconds", round(took), "penalty", best, flush=True)
        add("closed.jsonl", dict(kind=kind, name=n, written=w, seconds=round(took, 1), penalty=best[1], after={} if kind == "cifar" else after({k: v.cpu() for k, v in m.state_dict().items()}, kind, name="closed-" + n)))


def rivals(kind, limit=540):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.multiclass import OneVsOneClassifier as OVO
    from sklearn.neighbors import KNeighborsClassifier as KNN
    from sklearn.svm import SVC
    x, y = (t.cpu().numpy().reshape(len(t), -1) if t.ndim > 1 else t.cpu().numpy() for t in images(kind, "train")); xt, yt = (t.cpu().numpy().reshape(len(t), -1) if t.ndim > 1 else t.cpu().numpy() for t in images(kind, "test")); start = time.time(); have = {r["name"] for r in recorded("rivals.jsonl") if r["kind"] == kind}; sub = 60000 if kind == "mnist" else 10000
    for n, c, cap in (("linear discriminant", LDA(solver="lsqr", shrinkage=.1), None), ("1 nearest neighbour", KNN(1, n_jobs=4), None), ("one-vs-one linear discriminant", OVO(LDA(solver="lsqr", shrinkage=.1), n_jobs=4), None), ("logistic regression", LR(max_iter=100), sub if kind == "cifar" else None), ("support vector machine", SVC(), sub)):
        if n in have: continue
        if time.time() - start > limit / 2: print("stopping before the limit, run this again"); return
        t = time.time(); c.fit(x[:cap], y[:cap]); a = round(float(c.score(xt, yt)), 4); add("rivals.jsonl", dict(kind=kind, name=n, test=a, images=len(x[:cap]), seconds=round(time.time() - t, 1))); print(kind, n, a, "seconds", round(time.time() - t), flush=True)


def task_text(mode):
    kind = _kind[0]; n = sum(p.numel() for p in Net().parameters())
    if kind == "mnist":
        data = "MNIST handwritten digits. train_images.npy holds 60,000 images, uint8, shape (60000, 28, 28). The network reads each image flattened to 784 numbers, pixel/255. The class is the digit 0 to 9."
        net = "input 784, l1 (width 512), silu, l2 (width 512), silu, l3 (10 outputs, read as a class)"; names = ("l1", "l2", "l3")
    else:
        data = "CIFAR-10 photos in 10 classes. train_images.npy holds 50,000 images, uint8, shape (50000, 3, 32, 32). The network reads each image as (pixel/255 - mean) / std per colour channel, with mean .4914, .4822, .4465 and std .247, .243, .261."
        net = "c1 (3x3 convolution, padding 1, 3 to 64 channels), silu, 2x2 max pool, c2 (64 to 128 channels), silu, 2x2 max pool, c3 (128 to 256 channels), silu, 2x2 max pool, flatten to 4096 numbers in channel, row, column order, l (10 outputs, read as a class). Convolution weights have shape (out, in, 3, 3) and work as in PyTorch"; names = ("c1", "c2", "c3", "l")
    return f"""You are working on a neural network, and you may set its weights by hand. The aim is high accuracy on the dataset's standard test set of 10,000 images, which you never see.

The task: {data} train_labels.npy holds the labels.

The network: {net}. silu(x) = x * sigmoid(x). That is {n:,} weights, and they start random.

No training has happened. You may change the weights now, once. Two readings count. The main one is the weights exactly as you wrote them, with no training at all. A second reading then runs gradient training on from your weights, with the speeds you set, for {EPOCHS[kind]} passes over the data at batch {BATCH}, using Adam and a learning rate falling to zero, and keeps the best of three starting learning rates.

Your work folder has:
- weights/<name>.npy: the current weights. Names: {", ".join(a + "." + k for a in names for k in ("weight", "bias"))}
- train_images.npy and train_labels.npy.

You can:
- Run Python with numpy and scipy, one thread, up to {S.MAX_BLOCKS} blocks per reply, each up to {S.CPU} CPU seconds, in the form
```python
print(load("{names[0]}.weight").shape)
```
  Helpers: NAMES, SHAPES, load(name), load_edit(name), save_edit(name, array), drop_edit(name). Files stay between runs.
- Change weights with save_edit. It applies at CONTINUE.
- Change how fast weights move in the second reading: np.save("speed/<name>.npy", array) with the tensor's shape and values from 0 to {S.MAX_SPEED:g}. 1 is normal and 0 freezes a weight.
- Write EVAL on its own line: accuracy on the training images with your current edit, {S.MAX_EVALS} times in all.
- Write CONTINUE on its own line when you are finished: this applies your changes and ends your work.

Not allowed in your own Python: computing gradients, and any loop that repeatedly improves weights or statistics against a loss or reconstruction error. That bans gradient steps, k-means, alternating least squares, iteratively reweighted least squares, boosting rounds and scipy.optimize. Allowed: averages, covariances and counts; one-shot linear algebra such as solve, lstsq, eigh and svd; computing each layer once from the outputs of earlier layers; trying several fixed settings and choosing by EVAL or by images you held out yourself. Every run's code is read by a person.

You have {S.MAX_REPLIES} replies in all. In each reply say in one or two short sentences what you saw and what you expect next. Blocks run in order, then EVAL or CONTINUE. At the end, name what you wrote into the weights and why it should work."""


S.task_text = task_text


def new(run, kind):
    d = os.path.join(S.RUNS, run); w = os.path.join(d, "work")
    if os.path.exists(d): raise SystemExit("this run exists")
    for p in ("weights", "history", "speed"): os.makedirs(os.path.join(w, p))
    json.dump(dict(kind=kind, inst=0, mode="answers"), open(os.path.join(d, "config.json"), "w")); task = make_task(kind); st = S.fresh(task); st.update(replies=0, evals=0, flags=[], started=False)
    json.dump(dict(shapes={k: list(v.shape) for k, v in st["m"].state_dict().items()}), open(os.path.join(w, "arch.json"), "w"))
    for part, name in (("x", "train_images.npy"), ("y", "train_labels.npy")): shutil.copy(os.path.join(DATA, f"{kind}_train_{part}.npy"), os.path.join(w, name))
    S.publish(d, st); S.save(d, st)


def advance(st, d, task, mode):
    """Keep the weights exactly as written. Training runs only in score."""
    s, bad = S.edited(st, d); st["m"].load_state_dict(s); e = os.path.join(d, "work", "edit")
    if os.path.isdir(e): shutil.rmtree(e)
    sp = os.path.join(d, "work", "speed"); out = list(bad)
    for f in sorted(os.listdir(sp)) if os.path.isdir(sp) else []:
        k = f[:-4]
        try: st["speed"][k] = torch.tensor(np.clip(S.plain_file(os.path.join(sp, f), tuple(s[k].shape)), 0, S.MAX_SPEED))
        except Exception as x: out.append(f"{f}: {x}")
    S.log(d, kind="edit", step=st["step"], speed={k: round(float(v.mean()), 3) for k, v in st["speed"].items()}); st["step"] = S.END[mode]; out.append(f"Your work has finished. Training-image accuracy: {on_train(st['m'], task)}"); return "\n".join(out)


S.advance = advance


def score(run):
    if any(r["run"] == run for r in recorded("results.jsonl")): raise SystemExit("one score per run")
    d = os.path.join(S.RUNS, run); c = S.config(d); task = make_task(c["kind"]); st = S.load(d, task)
    if st["step"] != S.END["answers"]: raise SystemExit("not finished")
    f = os.path.join(d, "audit.json")
    if os.environ.get("BIG_TEST") is None and (not os.path.exists(f) or not json.load(open(f))["ok"]): raise SystemExit("this run has not passed its audit")
    state = {k: v.clone() for k, v in st["m"].state_dict().items()}; written = on_test(st["m"], c["kind"]); print(run, c["kind"], "written", written, flush=True); ledger = S.recorded(os.path.join(d, "ledger.jsonl")); code = [r for r in ledger if r["kind"] == "code"]
    row = dict(run=run, kind=c["kind"], written=written, replies=st["replies"], evals=st["evals"], flags=st["flags"], cpu_seconds=round(sum(r["cpu"] for r in code), 1), blocks=len(code), speed={k: round(float(v.mean()), 3) for k, v in st["speed"].items()}, after=after(state, c["kind"], speed=st["speed"], name=run, keep=os.path.join(d, "after.json"))); add("results.jsonl", row)
    print(run, "after", {k: (v["last"], v["best"]) for k, v in row["after"].items()})


def table():
    for kind in NETS:
        rows = [r for r in recorded("controls.jsonl") if r["kind"] == kind]
        if rows:
            b = best_control(kind); print(kind, "best training: last", b["last"], "best step", b["best"], b["setting"], b["seconds"], "s;", len(rows), "settings; worst last", min(r["last"] for r in rows))
        for r in recorded("closed.jsonl"):
            if r["kind"] == kind: print("  closed form", r["name"], r["written"], r["seconds"], "s; trained on", {k: (v["last"], v["best"]) for k, v in r["after"].items()})
        for r in recorded("rivals.jsonl"):
            if r["kind"] == kind: print("  rival", r["name"], r["test"], "on", r["images"], "images,", r["seconds"], "s")
        for r in recorded("results.jsonl"):
            if r["kind"] == kind: print("  Opus", r["run"], "written", r["written"], "cpu", r["cpu_seconds"], "s, blocks", r["blocks"], "flags", len(r["flags"]), "trained on", {k: (v["last"], v["best"]) for k, v in r["after"].items()})


def test():
    """Every path, in seconds, with a hand-written editor on a small slice."""
    import tempfile
    KINDS = [k for k in (("mnist", "l3"), ("cifar", "l")) if os.path.exists(os.path.join(DATA, k[0] + "_test_x.npy"))]; assert SMALL, "run with BIG_SMALL=2000"; S.RUNS = tempfile.mkdtemp(); os.environ["BIG_TEST"] = "1"; t = time.time()
    for kind, last in KINDS:
        controls(kind); closedform(kind); print(kind, "controls and closed form", round(time.time() - t, 1), "s", flush=True); new("t" + kind, kind); S.step("t" + kind, "START")
        S.step("t" + kind, f"""```python
x, y = np.load("train_images.npy"), np.load("train_labels.npy"); print(x.shape, y.shape)
save_edit("{last}.bias", load("{last}.bias") + 0.1); np.save("speed/{last}.bias.npy", np.ones(SHAPES["{last}.bias"]))
```
EVAL""")
        S.step("t" + kind, "CONTINUE"); S.step("t" + kind, "more"); score("t" + kind); r = recorded("results.jsonl")[-1]; assert 0 < r["written"] < 1 and r["kind"] == kind and r["blocks"] == 1, r; print(kind, "all paths", round(time.time() - t, 1), "s", flush=True)
    a = crops(torch.randn(8, 3, 32, 32).to(DEV)); assert a.shape == (8, 3, 32, 32); m = Cnn(); assert sum(p.numel() for p in m.parameters()) == 411786; assert sum(p.numel() for p in Mlp().parameters()) == 669706
    table(); shutil.rmtree(S.RUNS); print("ok", round(time.time() - t, 1), "s")


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "new": new(sys.argv[2], sys.argv[3])
    elif c == "step": S.step(sys.argv[2], sys.stdin.read() if sys.argv[3] == "-" else sys.argv[3])
    elif c == "score": score(sys.argv[2])
    elif c == "audit": S.audit(sys.argv[2], sys.argv[3])
    elif c == "controls": controls(sys.argv[2])
    elif c == "closedform": closedform(sys.argv[2])
    elif c == "longer": longer(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 120)
    elif c == "rivals": rivals(sys.argv[2])
    elif c == "table": table()
    elif c == "test": test()
