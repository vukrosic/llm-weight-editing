"""Where does an LLM's change to a network's weights help: before, during or after training, and on which kinds of task?

python3 search.py new RUN KIND INST MODE       make a run. KIND: tables conditions noisy modular random unseen. MODE: before during after answers
python3 search.py step RUN - <<'REPLY' ...     the editor's only command: run its reply, print what it gets back. The first reply is START.
python3 search.py score RUN                    once, after the run: read held-out pairs, then train on alone to step 6000 with no editor
python3 search.py audit RUN TRANSCRIPT         void the run if the editor used anything but that one command
python3 search.py controls KIND INST           every training setting and scripted edit for this task, to step 6000, in parallel
python3 search.py curve KIND INST              plain training only: mean held-out accuracy at a few steps
python3 search.py table                        every result against the best control at the same step
python3 search.py test                         every path in seconds

Batches and starting weights depend only on the task, so two runs of one task differ only by the edits.
"""
import contextlib, io, json, multiprocessing, os, re, shutil, signal, sys, time, zlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, "..", "repo")]
import sandbox
torch.set_num_threads(1)

N, WIDTH, OUT, SKILLS = 32, 48, 63, 4
BATCH, LR, SNAP, READ, FAR = 64, .003, 25, 25, 6000
MAX_REPLIES, MAX_EVALS, MAX_BLOCKS, CPU, MAX_SPEED = int(os.environ.get("TOY_REPLIES", 16)), 20, 5, 20, 10.
KINDS, MODES = ("tables", "conditions", "noisy", "modular", "random", "unseen"), ("before", "during", "after")
PAUSES, END = dict(before=[0], during=[150, 300, 450], after=[1500], answers=[0]), dict(before=600, during=600, after=1650, answers=150)
NOISE = .25
RUNS = os.environ.get("SEARCH_RUNS", os.path.join(HERE, "search_runs")); RESULTS, CONTROLS = os.path.join(RUNS, "results.jsonl"), os.path.join(RUNS, "controls.jsonl")
ME = os.path.abspath(__file__)
FORMAT = re.compile(r"```python\n(.*?)```", re.S)
LOOKS_LIKE_TRAINING = re.compile(r"grad|backward|lstsq|optimi[sz]|descent|\bfit\b|polyfit|minimize", re.I)


def crc(s): return zlib.crc32(s.encode())


def cut(v, classes=8): return torch.bucketize(v, v.flatten().quantile(torch.linspace(0, 1, classes + 1)[1:-1]))


def shapes(f, h): return [f + h, f * h, f.max(h), f.min(h), (f - h).abs()]


def unseen(p, g):
    """Rule shapes no script here was written for: a task draws four of these eight."""
    def two(lo, hi): return torch.randint(lo, hi, (N,), generator=g)[:, None], torch.randint(lo, hi, (N,), generator=g)[None, :]
    def real(): return torch.rand(N, generator=g)[:, None], torch.rand(N, generator=g)[None, :]
    if p == 0: f, h = two(0, 16); return (f * h) % 16
    if p == 1: f, h = two(0, 13); return (f + h) % 13
    if p == 2: f, h = two(0, 16); return f & h
    if p == 3: f, h = two(0, 16); return f | h
    if p == 4: f, h = real(); return cut((f + h) % 1)
    if p == 5: f, h = real(); return cut(torch.where(f > h, f + 0 * h, 1 - h + 0 * f))
    if p == 6: f, h = two(1, 17); return (f * h) % 17 - 1
    f, h = two(0, 16); return (f + h) // 2


def tables(kind, inst):
    """Four hidden tables of (a, b) -> class. Each number gets its own hidden values per skill, so no formula on a and b gives the answer."""
    g = torch.Generator().manual_seed(crc(f"{kind}-{inst}")); order = torch.randperm(5, generator=g).tolist(); out = []; picks = torch.randperm(8, generator=g)[:4].tolist() if kind == "unseen" else None
    two = lambda: (torch.rand(N, generator=g)[:, None], torch.rand(N, generator=g)[None, :])
    for s in range(4):
        if kind in ("tables", "noisy"): out.append(cut(shapes(*two())[order[s]]))
        elif kind == "conditions":
            side = torch.zeros(N, dtype=torch.bool); side[torch.randperm(N, generator=g)[:N // 2]] = True
            out.append(torch.where(side[:, None], cut(shapes(*two())[order[s]]), cut(shapes(*two())[order[(s + 1) % 5]])))
        elif kind == "modular":
            f, h = torch.randint(16, (N,), generator=g)[:, None], torch.randint(16, (N,), generator=g)[None, :]
            out.append([(f + h) % 16, (f - h) % 16, (f + 3 * h) % 16, f ^ h][s])
        elif kind == "unseen": out.append(unseen(picks[s], g))
        else: out.append(torch.randint(8, (N, N), generator=g))
    return out


def make_task(kind, inst):
    """Clean tables score held-out pairs. In `noisy`, a quarter of each skill's training answers are wrong, and the held-out answers stay right."""
    t = tables(kind, inst); g = torch.Generator().manual_seed(crc(f"split-{kind}-{inst}")); a, b = torch.cartesian_prod(torch.arange(N), torch.arange(N)).T; p = torch.randperm(len(a), generator=g); k = int(.7 * len(a))
    train = [x.clone() for x in t]; tr, te = (a[p[:k]], b[p[:k]]), (a[p[k:]], b[p[k:]])
    if kind == "noisy":
        for x in train:
            w = torch.randperm(k, generator=g)[:int(NOISE * k)]; x[tr[0][w], tr[1][w]] = (x[tr[0][w], tr[1][w]] + torch.randint(1, 8, (len(w),), generator=g)) % 8
    return dict(kind=kind, inst=inst, clean=t, train=train, tr=tr, te=te)


def inputs(a, b, s): return torch.cat([F.one_hot(a, N), F.one_hot(b, N), F.one_hot(torch.full_like(a, s), SKILLS)], 1).float()


class Net(nn.Module):
    def __init__(s, drop=0.):
        super().__init__(); s.l1 = nn.Linear(2 * N + SKILLS, WIDTH); s.l2 = nn.Linear(WIDTH, WIDTH); s.l3 = nn.Linear(WIDTH, OUT); s.drop = drop
    def forward(s, x): return s.l3(F.dropout(F.silu(s.l2(F.dropout(F.silu(s.l1(x)), s.drop, s.training))), s.drop, s.training))


def acc(m, pairs, t):
    a, b = pairs; was = m.training; m.eval()
    with torch.no_grad(): r = [round((m(inputs(a, b, i)).argmax(1) == t[i][a, b]).float().mean().item(), 3) for i in range(SKILLS)]
    m.train(was); return r


def held(st, task): return acc(st["m"], task["te"], task["clean"])


def on_train(m, task): return acc(m, task["tr"], task["train"])


def fresh(task, lr=LR, wd=0., drop=0.):
    base = crc(f"{task['kind']}-{task['inst']}"); torch.manual_seed(base + 1); m = Net(drop)
    return dict(m=m, o=torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd), g=torch.Generator().manual_seed(base + 2), step=0, speed={}, curve=[])


def stretch(st, task, steps, snap=None, hist=None):
    """Train. Speed scales the optimizer's own step on each weight. Snap saves the weight history for the editor."""
    a, b = task["tr"]; m, o, g = st["m"], st["o"], st["g"]
    for _ in range(steps):
        l, each = 0., []
        for i in range(SKILLS):
            j = torch.randint(len(a), (BATCH,), generator=g); x = F.cross_entropy(m(inputs(a[j], b[j], i)), task["train"][i][a[j], b[j]]); l = l + x; each.append(round(x.item(), 4))
        before = {k: p.detach().clone() for k, p in m.named_parameters() if k in st["speed"]}
        o.zero_grad(); l.backward(); o.step(); st["step"] += 1
        with torch.no_grad():
            for k, p in m.named_parameters():
                if k in st["speed"]: p.copy_(before[k] + st["speed"][k] * (p - before[k]))
        if st["step"] % SNAP == 0:
            st["curve"].append(dict(step=st["step"], loss=each))
            if snap: np.savez(os.path.join(snap, "history", f"step_{st['step']:05d}.npz"), **{k: v.numpy() for k, v in m.state_dict().items()})
            if hist is not None: hist.append((st["step"], {k: v.clone() for k, v in m.state_dict().items()}))


def task_text(mode):
    when = dict(
        before=f"No training has happened yet. The weights are the random starting weights. You may change them before training starts, once. Then training runs {END['before']} steps with no more changes from you. Your folder also has examples.jsonl: {BATCH} training examples per skill.",
        during=f"Training pauses at steps {', '.join(map(str, PAUSES['during']))}, and you may change the network at each pause. Training has {END['during']} steps in all before the first reading.",
        after=f"Training has already run {PAUSES['after'][0]} steps. You may change the network now, once. Then training runs {END['after'] - PAUSES['after'][0]} more steps before the first reading. A speed of 0 on every weight skips that training.",
        answers=f"No training has happened yet. The weights are the random starting weights, with nothing learned. You may change them now, once. Then training runs {END['answers']} steps before the first reading. A speed of 0 on every weight skips that training. Your folder also has answers.jsonl: every training pair's answer for every skill.")[mode]
    return f"""You are working on a small neural network while it trains, and you may change it by hand. The aim is to find a way of changing weights that helps the network more than training alone.

The network: the input is a one-hot number a (0 to {N - 1}), a one-hot number b (0 to {N - 1}) and a one-hot skill ({SKILLS} skills), concatenated in that order. Then l1 (width {WIDTH}), silu, l2 (width {WIDTH}), silu, l3 ({OUT} outputs, read as a class). Each skill maps (a, b) to a class by a hidden rule, and nobody will tell you the rules. Some tasks have rules that are short, some have rules with conditions, and some have no rule at all. Some training answers may be wrong.

Training: Adam at learning rate {LR}, {BATCH} examples per skill per step. The batches are fixed in advance and stay the same whatever you do. Training continues from your edited weights and keeps the optimizer's state. {when}

Goal: high accuracy on pairs (a, b) the network never trained on, where the answers are all correct. 30% of pairs are held out. Two readings count: the first when your last stretch ends, at step {END[mode]}, and the best level after that, while training runs on alone to step {FAR} with the speeds you last set. You never see either reading. You are compared with the best of many training settings at the same steps.

Your work folder has:
- weights/<name>.npy: the current weights. Names: {", ".join(n + "." + k for n in ("l1", "l2", "l3") for k in ("weight", "bias"))}
- history/step_NNNNN.npz: all weights every {SNAP} steps so far
- curve.jsonl: each skill's batch loss every {SNAP} steps

You can:
- Run Python with numpy and scipy, up to {MAX_BLOCKS} blocks per reply, each up to {CPU} CPU seconds, in the form
```python
print(load("l1.weight").shape)
```
  Helpers: NAMES, SHAPES, load(name), load_edit(name), save_edit(name, array), drop_edit(name). Files stay between runs.
- Change weights with save_edit. It applies at the next CONTINUE.
- Change how fast weights move: np.save("speed/<name>.npy", array) with the tensor's shape and values from 0 to {MAX_SPEED:g}. Each training step on those weights is multiplied by it, until you replace it. 1 is normal and 0 freezes a weight.
- Write EVAL on its own line: accuracy per skill on the training pairs with your current edit, {MAX_EVALS} times in all.
- Write CONTINUE on its own line: apply your changes and train to the next pause. You get each skill's accuracy on training pairs back.

Not allowed: computing gradients, or any iterative fitting of weights to a loss, in your own Python. Training happens only in the stretches. Looking at the weights and their history, reasoning about what they do, and setting numbers from that reasoning are all fine. That includes writing neurons by hand, resetting or removing neurons, averaging past weights, shrinking weights, and leaving the network alone. Every run's code is read by a person.

You have {MAX_REPLIES} replies in all; when they run out, training finishes with no more edits. In each reply say in one or two short sentences what you saw and what you expect next. Blocks run in order, then EVAL or CONTINUE. When training has finished, end with one or two sentences naming what you changed in the weights and why it should help on another task like this."""


def config(d): return json.load(open(os.path.join(d, "config.json")))


def save(d, st): torch.save(dict(m=st["m"].state_dict(), o=st["o"].state_dict(), g=st["g"].get_state(), step=st["step"], speed=st["speed"], curve=st["curve"], replies=st["replies"], evals=st["evals"], flags=st["flags"], started=st["started"]), os.path.join(d, "state.pt"))


def load(d, task):
    s = torch.load(os.path.join(d, "state.pt"), weights_only=False); st = fresh(task); st["m"].load_state_dict(s["m"]); st["o"].load_state_dict(s["o"]); st["g"].set_state(s["g"]); st.update({k: s[k] for k in ("step", "speed", "curve", "replies", "evals", "flags", "started")}); return st


def publish(d, st):
    """The host's copy overwrites whatever is in weights/, so a changed file there means nothing."""
    w = os.path.join(d, "work")
    for k, v in st["m"].state_dict().items(): np.save(os.path.join(w, "weights", k + ".npy"), v.numpy())
    with open(os.path.join(w, "curve.jsonl"), "w") as f:
        for r in st["curve"]: f.write(json.dumps(r) + "\n")


def plain_file(path, shape):
    """Only a real file with the right shape and finite numbers comes through."""
    if os.path.islink(path) or not os.path.isfile(path): raise ValueError("not a plain file")
    a = np.load(path, allow_pickle=False)
    if a.shape != shape or not np.isfinite(a).all(): raise ValueError(f"bad shape or values {a.shape}")
    return a.astype(np.float32)


def edited(st, d):
    w = os.path.join(d, "work", "edit"); s = {k: v.clone() for k, v in st["m"].state_dict().items()}; bad = []
    for f in sorted(os.listdir(w)) if os.path.isdir(w) else []:
        k = f[:-4]
        if k not in s: bad.append(f"{f}: unknown tensor"); continue
        try: s[k] = torch.tensor(plain_file(os.path.join(w, f), tuple(s[k].shape)))
        except Exception as e: bad.append(f"{f}: {e}")
    return s, bad


def log(d, **r):
    with open(os.path.join(d, "ledger.jsonl"), "a") as f: f.write(json.dumps(dict(t=time.time(), **r)) + "\n")


def next_stop(mode, step): return next((p for p in PAUSES[mode] if p > step), END[mode])


def advance(st, d, task, mode):
    """Apply the edit and speed, then train to the next pause or the end."""
    s, bad = edited(st, d); out = list(bad); before = {k: v.clone() for k, v in st["m"].state_dict().items()}
    st["m"].load_state_dict(s); e = os.path.join(d, "work", "edit")
    if os.path.isdir(e): shutil.rmtree(e)
    sp = os.path.join(d, "work", "speed")
    for f in sorted(os.listdir(sp)) if os.path.isdir(sp) else []:
        k = f[:-4]
        if k not in before: out.append(f"{f}: unknown tensor"); continue
        try: st["speed"][k] = torch.tensor(np.clip(plain_file(os.path.join(sp, f), tuple(before[k].shape)), 0, MAX_SPEED))
        except Exception as x: out.append(f"{f}: {x}")
    log(d, kind="edit", step=st["step"], relative_change={k: round(float((s[k] - before[k]).norm() / before[k].norm()), 4) for k in before}, speed={k: round(float(v.mean()), 3) for k, v in st["speed"].items()})
    stretch(st, task, next_stop(mode, st["step"]) - st["step"], snap=os.path.join(d, "work")); publish(d, st); out.append(f"Step {st['step']} of {END[mode]}. Training-pair accuracy: {on_train(st['m'], task)}"); return "\n".join(out)


def new(run, kind, inst, mode):
    d = os.path.join(RUNS, run); w = os.path.join(d, "work")
    if os.path.exists(d): raise SystemExit("this run exists")
    for p in ("weights", "history", "speed"): os.makedirs(os.path.join(w, p))
    json.dump(dict(kind=kind, inst=int(inst), mode=mode), open(os.path.join(d, "config.json"), "w"))
    task = make_task(kind, int(inst)); st = fresh(task); st.update(replies=0, evals=0, flags=[], started=False)
    json.dump(dict(shapes={k: list(v.shape) for k, v in st["m"].state_dict().items()}), open(os.path.join(w, "arch.json"), "w"))
    if mode == "before":
        g = torch.Generator().manual_seed(crc(f"examples-{kind}-{inst}")); a, b = task["tr"]
        with open(os.path.join(w, "examples.jsonl"), "w") as f:
            for i in range(SKILLS):
                for j in torch.randperm(len(a), generator=g)[:BATCH].tolist(): f.write(json.dumps(dict(skill=i, a=a[j].item(), b=b[j].item(), answer=task["train"][i][a[j], b[j]].item())) + "\n")
    if mode == "answers":
        a, b = task["tr"]
        with open(os.path.join(w, "answers.jsonl"), "w") as f:
            for i in range(SKILLS):
                for x, y in zip(a.tolist(), b.tolist()): f.write(json.dumps(dict(skill=i, a=x, b=y, answer=task["train"][i][x, y].item())) + "\n")
    stretch(st, task, PAUSES[mode][0], snap=w); publish(d, st); save(d, st)


def step(run, reply):
    d = os.path.join(RUNS, run); c = config(d); task = make_task(c["kind"], c["inst"]); mode = c["mode"]; st = load(d, task); out = []
    if not st["started"]:
        st["started"] = True; msg = task_text(mode) + f"\n\nTraining is at step {st['step']}. Training-pair accuracy: {on_train(st['m'], task)}"; log(d, kind="task", text=msg); save(d, st); print(msg); return
    if st["step"] >= END[mode]: print("Training has finished. Reply DONE."); return
    st["replies"] += 1; log(d, kind="reply", n=st["replies"], text=reply)
    for i, code in enumerate(FORMAT.findall(reply)[:MAX_BLOCKS]):
        got, cpu = sandbox.run(code, os.path.join(d, "work"), cpu_seconds=CPU); out.append(f"[block {i + 1}]\n{got}"); log(d, kind="code", code=code, out=got, cpu=cpu)
        if LOOKS_LIKE_TRAINING.search(code): st["flags"].append(f"reply {st['replies']}: word match")
    rest = FORMAT.sub("", reply).split("\n")
    if any(l.strip() == "EVAL" for l in rest):
        if st["evals"] >= MAX_EVALS: out.append("No EVAL left.")
        else:
            st["evals"] += 1; s, bad = edited(st, d); t = Net(); t.load_state_dict(s); out += bad + [f"EVAL {st['evals']} of {MAX_EVALS}, training-pair accuracy with your edit: {on_train(t, task)}"]
    if any(l.strip() == "CONTINUE" for l in rest): out.append(advance(st, d, task, mode))
    if st["replies"] >= MAX_REPLIES and st["step"] < END[mode]:
        out.append("Replies used up; training finishes with no more edits.")
        while st["step"] < END[mode]: out.append(advance(st, d, task, mode))
    if st["step"] >= END[mode]: out.append("Training has finished. Reply DONE.")
    msg = "\n".join(out) or "(nothing ran: no block, EVAL or CONTINUE found)"; log(d, kind="result", text=msg); save(d, st); print(msg)


def recorded(path): return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def follow(st, task, end=FAR, keep=True):
    """Held-out readings from here to end. Nothing here reaches any editor."""
    if not keep: st["speed"] = {}
    rows = {st["step"]: held(st, task)}
    while st["step"] < end: stretch(st, task, READ - st["step"] % READ); rows[st["step"]] = held(st, task)
    return rows


def score(run):
    if any(r["run"] == run for r in recorded(RESULTS)): raise SystemExit("one score per run")
    d = os.path.join(RUNS, run); c = config(d); task = make_task(c["kind"], c["inst"]); st = load(d, task)
    if st["step"] != END[c["mode"]]: raise SystemExit("not finished")
    audit_file = os.path.join(d, "audit.json")
    if not os.path.exists(audit_file) or not json.load(open(audit_file))["ok"]: raise SystemExit("this run has not passed its audit")
    s = held(st, task); own = follow(load(d, task), task); one = follow(load(d, task), task, keep=False)
    row = dict(run=run, **c, held_out=s, mean=round(sum(s) / 4, 3), replies=st["replies"], evals=st["evals"], flags=st["flags"], own=own, speed1=one)
    with open(RESULTS, "a") as f: f.write(json.dumps(row) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k not in ("own", "speed1")}))


def audit(run, path):
    """A run counts only if the editor's every tool call was the one allowed command and the bench saw every call."""
    d = os.path.join(RUNS, run); pat = re.compile(rf"python3 {re.escape(ME)} step {re.escape(run)} - <<'REPLY'\n(.*)\nREPLY\s*", re.S); calls, bad = 0, []
    for line in open(path):
        m = json.loads(line).get("message", {})
        if m.get("role") != "assistant" or isinstance(m.get("content"), str): continue
        for b in m["content"]:
            if b.get("type") != "tool_use": continue
            calls += 1; cmd = b["input"].get("command", "") if b.get("name") == "Bash" else None; hit = pat.fullmatch(cmd) if cmd else None
            if not hit or re.search(r"^REPLY\s*$", hit.group(1), re.M): bad.append((b.get("name"), str(cmd)[:200]))
    log_rows = recorded(os.path.join(d, "ledger.jsonl")); seen = sum(r["kind"] == "reply" for r in log_rows) + sum(r["kind"] == "task" for r in log_rows); st = load(d, make_task(**{k: config(d)[k] for k in ("kind", "inst")})); after_end = calls - seen
    out = dict(ok=not bad and after_end >= 0 and seen > 0, calls=calls, logged=seen, calls_after_the_end=after_end, bad=bad); json.dump(out, open(os.path.join(d, "audit.json"), "w")); print(json.dumps(out))


def average(hist, st, back):
    """Stochastic weight averaging (Izmailov et al. 2018) over the last `back` steps."""
    use = [s for t, s in hist if t > st["step"] - back] or [st["m"].state_dict()]; return {k: sum(s[k] for s in use) / len(use) for k in use[0]}


def shrink(st, task):
    """Shrink and perturb (Ash and Adams 2020)."""
    g = torch.Generator().manual_seed(3); return {k: .8 * v + .01 * torch.randn(v.shape, generator=g) for k, v in st["m"].state_dict().items()}


def redo(st, task, tau=.1):
    """Reset quiet neurons (ReDo, Sokar et al. 2023): new incoming weights, zero outgoing. Reads activations on training inputs only."""
    s = {k: v.clone() for k, v in st["m"].state_dict().items()}; m = st["m"]; a, b = task["tr"]; x = torch.cat([inputs(a, b, i) for i in range(SKILLS)]); g = torch.Generator().manual_seed(4)
    with torch.no_grad(): h1 = F.silu(m.l1(x)); h2 = F.silu(m.l2(h1))
    for h, inn, out_ in ((h1, "l1", "l2"), (h2, "l2", "l3")):
        q = h.abs().mean(0); w = q / q.mean() <= tau; s[inn + ".weight"][w] = (torch.randn(s[inn + ".weight"][w].shape, generator=g) * s[inn + ".weight"].std()); s[inn + ".bias"][w] = 0.; s[out_ + ".weight"][:, w] = 0.
    return s


def scripted(mode):
    """Published weight edits through the same pauses as the editor."""
    if mode == "before": return {"starting weights x.5": lambda st, task, hist: {k: .5 * v for k, v in st["m"].state_dict().items()}, "starting weights x2": lambda st, task, hist: {k: 2 * v for k, v in st["m"].state_dict().items()}}
    back = 150 if mode == "during" else 300
    def line(st, task, hist):
        p = [s for t, s in hist if t == st["step"] - 150]; s = st["m"].state_dict(); return {k: s[k] + (s[k] - p[0][k]) for k in s} if p else s
    return {"weight averaging": lambda st, task, hist: average(hist, st, back), "shrink and perturb": lambda st, task, hist: shrink(st, task), "reset quiet neurons": lambda st, task, hist: redo(st, task), "straight line x1": line}


def settings():
    rows = [dict(name=f"lr {lr} wd {wd}", lr=lr, wd=wd) for lr in (.001, .003, .01) for wd in (0., .01, .1, .5, 1.)] + [dict(name=f"dropout {d}", drop=d) for d in (.05, .1, .3)] + [dict(name=f"dropout .1 lr .01 wd .1", drop=.1, lr=.01, wd=.1)] + [dict(name=f"fixed speed x{c}", speed=float(c)) for c in (2, 3, 5, 7, 10)]
    return rows + [dict(name=f"{mode}: {n}", mode=mode, edit=n) for mode in MODES for n in scripted(mode)]


def one_control(job):
    torch.set_num_threads(1); kind, inst, s = job; task = make_task(kind, inst); st = fresh(task, s.get("lr", LR), s.get("wd", 0.), s.get("drop", 0.)); t0 = time.time(); hist = []
    if "speed" in s: st["speed"] = {k: torch.full_like(p.detach(), s["speed"]) for k, p in st["m"].named_parameters()}
    if "edit" in s:
        f = scripted(s["mode"])[s["edit"]]
        for p in PAUSES[s["mode"]]:
            stretch(st, task, p - st["step"], hist=hist); st["m"].load_state_dict(f(st, task, hist))
    rows = follow(st, task); return dict(kind=kind, inst=inst, name=s["name"], rows=rows, seconds=round(time.time() - t0, 1))


def controls(kind, inst, procs=5):
    have = {(r["kind"], r["inst"], r["name"]) for r in recorded(CONTROLS)}; jobs = [(kind, inst, s) for s in settings() if (kind, inst, s["name"]) not in have]; os.makedirs(RUNS, exist_ok=True)
    with multiprocessing.get_context("spawn").Pool(procs) as p:
        for r in p.imap_unordered(one_control, jobs):
            with open(CONTROLS, "a") as f: f.write(json.dumps(r) + "\n")
            print(r["name"], r["seconds"], "s, mean at 6000", round(sum(r["rows"][FAR]) / 4, 3), flush=True)


def m4(x): return sum(x) / 4


def best_control(kind, inst, mode, at=None):
    """The best control for this task, read on held-out pairs, which favours training. Scripted edits count only for their own mode. at: one step; None: any step from END[mode] on."""
    best = (-1, ""); start = END[mode]
    for r in recorded(CONTROLS):
        if r["kind"] != kind or r["inst"] != inst or (":" in r["name"] and not r["name"].startswith(mode + ":")): continue
        for k, v in r["rows"].items():
            k = int(k)
            if (k == at if at is not None else k >= start) and m4(v) > best[0]: best = (m4(v), r["name"], k)
    return best


def table():
    rows = recorded(RESULTS)
    print(f"{'run':8s}{'kind':12s}{'inst':5s}{'mode':8s}{'editor':>7s}{'control':>8s}{'gain':>7s}{'peak':>7s}{'speed1':>7s}{'control':>8s}{'gain':>7s}  replies evals best control at first reading")
    for r in rows:
        e = END[r["mode"]]; c = best_control(r["kind"], r["inst"], r["mode"], e); peak = max(m4(v) for v in r["own"].values()); one = max(m4(v) for v in r["speed1"].values()); p = best_control(r["kind"], r["inst"], r["mode"])
        print(f"{r['run']:8s}{r['kind']:12s}{r['inst']:<5d}{r['mode']:8s}{r['mean']:7.3f}{c[0]:8.3f}{r['mean'] - c[0]:+7.3f}{peak:7.3f}{one:7.3f}{p[0]:8.3f}{peak - p[0]:+7.3f}  {r['replies']:7d}{r['evals']:6d} {c[1]}")


def curve(kind, inst):
    task = make_task(kind, int(inst)); st = fresh(task); t = time.time(); rows = follow(st, task, end=1500)
    print(kind, inst, {k: round(m4(rows[k]), 3) for k in (25, 150, 300, 600, 1500)}, "train", round(m4(on_train(st["m"], task)), 3), f"{time.time() - t:.1f}s")


def test():
    global RUNS, RESULTS, CONTROLS
    t0 = time.time(); fails = 0; RUNS = os.path.join(HERE, "search_test"); RESULTS, CONTROLS = os.path.join(RUNS, "results.jsonl"), os.path.join(RUNS, "controls.jsonl")
    if os.path.exists(RUNS): shutil.rmtree(RUNS)
    os.makedirs(RUNS)
    def over(*_): raise TimeoutError("a step ran past 7 s")
    signal.signal(signal.SIGALRM, over); signal.alarm(7)
    def check(name, ok, note=""):
        nonlocal fails; signal.alarm(7); fails += not ok; print(f"{'PASS' if ok else 'FAIL'}  {name}  {note}  [{time.time() - t0:.0f}s]", flush=True)
    def say(run, text):
        b = io.StringIO()
        with contextlib.redirect_stdout(b): step(run, text)
        return b.getvalue()
    for kind in KINDS:
        t = make_task(kind, 0); c = set(zip(*[x.tolist() for x in t["tr"]])); check(f"{kind}: held-out pairs are not in training", not c & set(zip(*[x.tolist() for x in t["te"]])) and len(c) + len(t["te"][0]) == N * N)
        wrong = np.mean([(t["train"][i][t["tr"]] != t["clean"][i][t["tr"]]).float().mean().item() for i in range(SKILLS)]); check(f"{kind}: wrong training answers", abs(wrong - (NOISE if kind == "noisy" else 0)) < .01, f"{wrong:.3f}"); check(f"{kind}: held-out answers are all clean", all((t["train"][i][t["te"]] == t["clean"][i][t["te"]]).all() for i in range(SKILLS)))
    a, b = tables("tables", 0), tables("tables", 1); check("new instances give new tables", not (a[0] == b[0]).all()); a2 = tables("tables", 0); check("the same instance repeats", all((x == y).all() for x, y in zip(a, a2)))
    task = make_task("tables", 0); x = fresh(task); stretch(x, task, 100); y = fresh(task); stretch(y, task, 40); stretch(y, task, 60); check("pausing changes nothing", all((x["m"].state_dict()[k] == y["m"].state_dict()[k]).all() for k in x["m"].state_dict()))
    with contextlib.redirect_stdout(io.StringIO()):
        new("t1", "noisy", 0, "during"); new("t2", "tables", 0, "before"); new("t3", "tables", 0, "after")
    check("the task names no kind", not any(k in task_text("during").split("Some tasks")[0] for k in ("noisy", "modular", "conditions")))
    r = say("t1", "START"); check("START gives the task", "hidden rule" in r and "Training is at step 150" in r); check("START costs no reply", load(os.path.join(RUNS, "t1"), make_task("noisy", 0))["replies"] == 0)
    check("before has examples and starts at 0", os.path.exists(os.path.join(RUNS, "t2", "work", "examples.jsonl")) and load(os.path.join(RUNS, "t2"), task)["step"] == 0); check("during has no examples", not os.path.exists(os.path.join(RUNS, "t1", "work", "examples.jsonl")))
    for name, code, word in (("the run's state", 'print(open("../state.pt","rb").read(9))', "Operation not permitted"), ("the run's config", 'print(open("../config.json").read())', "Operation not permitted"), ("the bench code", f'print(open("{ME}").read(9))', "Operation not permitted"), ("the network", 'import socket; socket.create_connection(("1.1.1.1", 80), 2)', "Error"), ("torch", "import torch", "No module")):
        check(f"the editor cannot reach {name}", word in say("t1", f"```python\n{code}\n```"))
    n = make_task("noisy", 0); st0 = load(os.path.join(RUNS, "t1"), n); w = {k: v.clone() for k, v in st0["m"].state_dict().items()}
    r = say("t1", '```python\nsave_edit("l3.bias", load("l3.bias") + 1); os.makedirs("speed", exist_ok=True); np.save("speed/l1.weight.npy", np.zeros(SHAPES["l1.weight"]))\n```\nEVAL'); check("EVAL does not change the host's weights", "EVAL 1" in r and all((load(os.path.join(RUNS, "t1"), n)["m"].state_dict()[k] == w[k]).all() for k in w))
    say("t1", "CONTINUE"); s2 = load(os.path.join(RUNS, "t1"), n); check("speed 0 freezes a tensor through training", (s2["m"].state_dict()["l1.weight"] == w["l1.weight"]).all()); check("an edit reaches training", not (s2["m"].state_dict()["l3.bias"] == w["l3.bias"]).all()); check("training stops at the next pause", s2["step"] == 300)
    r = say("t1", '```python\nw = load("l3.bias"); w[0] = np.nan; os.makedirs("edit", exist_ok=True); np.save("edit/l3.bias.npy", w); np.save("edit/x.npy", w)\n```\nCONTINUE'); check("NaN and unknown tensors are refused", "bad shape or values" in r and "unknown tensor" in r)
    open(os.path.join(RUNS, "t1", "work", "weights", "l1.weight.npy"), "wb").write(b"x"); say("t1", "EVAL"); check("a changed weights file means nothing", load(os.path.join(RUNS, "t1"), n)["m"].state_dict()["l1.weight"].shape == w["l1.weight"].shape)
    try: score("t1"); check("no score before the end", False)
    except SystemExit: check("no score before the end", True)
    p = os.path.join(RUNS, "tr.jsonl"); ok = f"python3 {ME} step t3 - <<'REPLY'\nSTART\nREPLY"; put = lambda cmds, name="Bash": open(p, "w").write("".join(json.dumps(dict(message=dict(role="assistant", content=[dict(type="tool_use", name=name, input=dict(command=c))]))) + "\n" for c in cmds))
    with contextlib.redirect_stdout(io.StringIO()):
        say("t3", "START"); put([ok]); audit("t3", p); good = json.load(open(os.path.join(RUNS, "t3", "audit.json")))["ok"]
        put([ok + "; cat net.py"]); audit("t3", p); chain = json.load(open(os.path.join(RUNS, "t3", "audit.json")))["ok"]
        put([f"python3 {ME} step t3 - <<'REPLY'\nx\nREPLY\ncat net.py\n<<'REPLY'\ny\nREPLY"]); audit("t3", p); inner = json.load(open(os.path.join(RUNS, "t3", "audit.json")))["ok"]
        put([ok], "Read"); audit("t3", p); other = json.load(open(os.path.join(RUNS, "t3", "audit.json")))["ok"]
    check("the audit passes the one command", good); check("the audit voids a chained command", not chain); check("the audit voids a command hidden in a reply", not inner); check("the audit voids any other tool", not other)
    try: score("t3"); check("no score without a passed audit", False)
    except SystemExit: check("no score without a passed audit", True)
    with contextlib.redirect_stdout(io.StringIO()): new("t8", "unseen", 0, "answers")
    u = make_task("unseen", 0); r = say("t8", "START"); check("answers mode gives every training answer and no trained weights", sum(1 for _ in open(os.path.join(RUNS, "t8", "work", "answers.jsonl"))) == 4 * len(u["tr"][0]) and load(os.path.join(RUNS, "t8"), u)["step"] == 0 and "answers.jsonl" in r); held_pairs = set(zip(*[x.tolist() for x in u["te"]])); check("answers mode shows no held-out pair", not any((json.loads(l)["a"], json.loads(l)["b"]) in held_pairs for l in open(os.path.join(RUNS, "t8", "work", "answers.jsonl")))); r = say("t8", "CONTINUE"); check("answers mode trains 150 and ends", "Step 150" in r and "finished" in r); signal.alarm(7)
    r = say("t3", "CONTINUE"); check("after mode trains 150 and ends", "Step 1650" in r and "finished" in r); signal.alarm(7)
    t3 = make_task("tables", 0); st = load(os.path.join(RUNS, "t3"), t3); q = fresh(t3); stretch(q, t3, 1650); check("a run with no edits equals plain training", all((st["m"].state_dict()[k] == q["m"].state_dict()[k]).all() for k in q["m"].state_dict()))
    for mode in MODES:
        for name, f in scripted(mode).items():
            st = fresh(t3); hist = []; stretch(st, t3, 50, hist=hist); before = {k: v.clone() for k, v in st["m"].state_dict().items()}; st["step"] = 150 if name == "straight line x1" else st["step"]; hist.append((0, before)); s = f(st, t3, hist); st["m"].load_state_dict(s); check(f"{mode}: {name} is finite", all(torch.isfinite(v).all() for v in s.values()))
    d = os.path.join(RUNS, "t4"); os.makedirs(os.path.join(d)); st = fresh(t3); stretch(st, t3, 400); s = held(st, t3); t = fresh(t3); j = t["m"].state_dict(); t["m"].load_state_dict(st["m"].state_dict()); check("handing in better weights is rewarded", m4(held(t, t3)) == m4(s) > m4(held(fresh(t3), t3)) + .2, f"{m4(s):.3f} vs {m4(held(fresh(t3), t3)):.3f}")
    u = make_task("random", 0); st = fresh(u); stretch(st, u, 300); h = m4(held(st, u)); check("random tables stay near chance on held-out pairs", h < .2, f"{h:.3f}")
    signal.alarm(0); shutil.rmtree(RUNS); print(f"{'ALL PASS' if not fails else str(fails) + ' FAILED'} in {time.time() - t0:.0f}s"); sys.exit(bool(fails))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "new": new(*sys.argv[2:6])
    elif c == "step": step(sys.argv[2], sys.stdin.read().strip())
    elif c == "score": score(sys.argv[2])
    elif c == "audit": audit(*sys.argv[2:4])
    elif c == "controls": controls(sys.argv[2], int(sys.argv[3]))
    elif c == "curve": curve(sys.argv[2], sys.argv[3])
    elif c == "table": table()
    elif c == "test": test()
