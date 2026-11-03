"""An editor between training stretches: train 300 steps, pause, let it read the weight history and change weights or their speed, go on.

python3 loop.py start ID                     train the first stretch and print the task
python3 loop.py step ID TRANSCRIPT|FILE      run the editor's last reply, print what it gets back
python3 loop.py score ID                     once, after the last step: accuracy on held-out pairs. The editor never sees this.
python3 loop.py controls                     the same schedule without an editor
python3 loop.py test                         every path in seconds

Batches are the same whatever the editor does, so two runs differ only by the edits.
"""
import json, os, re, shutil, signal, sys, time
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sandbox
from net import N, OUT, RULE_SET, SECRET, SKILLS, WIDTH, Net, acc, answer, data, inputs

TOTAL, STRETCH, SNAP, BATCH, LR = (600, 150, 25, 64, .003) if SECRET else (1500, 300, 50, 64, .003)
MAX_REPLIES, MAX_EVALS, MAX_BLOCKS, CPU, MAX_SPEED = int(os.environ.get("TOY_REPLIES", 12)), 20, 5, 20, 10.
FINISH = int(os.environ.get("TOY_FINISH", TOTAL))  # edits stop at TOTAL; training goes on to FINISH with the speeds then set, and the score is read there
TAG = "" if RULE_SET == "1" else f"_set{RULE_SET}"
RUNS, RESULTS = os.path.join(HERE, "runs" + TAG), os.path.join(HERE, f"results_secret{TAG}.jsonl" if SECRET else "results.jsonl")
LOOKS_LIKE_TRAINING = re.compile(r"grad|backward|lstsq|optimi[sz]|descent|\bfit\b|polyfit|minimize", re.I)
FORMAT = re.compile(r"```python\n(.*?)```", re.S)

SKILL_TEXT = "The four skills follow secret rules: each maps (a, b) to a class, and nobody will tell you the rules. You do not get the training examples." if SECRET else "Skills in order: 0 add a+b, 1 max(a,b), 2 |a-b|, 3 min(a,b)."

FINISH_TEXT = f": from step {TOTAL} to step {FINISH} training runs on its own with the speeds you last set" if FINISH > TOTAL else ""
TASK = f"""You are watching a small neural network train, and between training stretches you may change it by hand.

The network: the input is a one-hot number a (0 to {N - 1}), a one-hot number b (0 to {N - 1}) and a one-hot skill (4 skills), concatenated in that order. Then l1 (width {WIDTH}), silu, l2 (width {WIDTH}), silu, l3 ({OUT} outputs, read as a class from 0 to {OUT - 1}). {SKILL_TEXT}

Training: Adam at learning rate {LR}, {BATCH} examples per skill per step, {FINISH} steps in all, pausing every {STRETCH} steps up to step {TOTAL}. The batches are fixed in advance and stay the same whatever you do. Training continues from your edited weights and keeps the optimizer's state.

Goal: the highest accuracy on all four skills at step {FINISH}, on pairs (a, b) the network never trained on. 30% of pairs are held out. You never see that score, and there are no edits after step {TOTAL}{FINISH_TEXT}.

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
- Write CONTINUE on its own line: apply your changes and train the next {STRETCH} steps. You get each skill's accuracy on training pairs back.

Not allowed: computing gradients, or any iterative fitting of weights to a loss, in your own Python. Training happens only in the stretches. Looking at the weights and their history, reasoning about what they do, and setting numbers from that reasoning are all fine, including a straight-line step along the history or writing neurons by hand. Every run's code is read by a person.

You have {MAX_REPLIES} replies in all; when they run out, training finishes with no more edits. In each reply say in one or two short sentences what you saw and what you expect to happen next. Blocks run in order, then EVAL or CONTINUE."""


def fresh():
    torch.manual_seed(1); m = Net(); o = torch.optim.Adam(m.parameters(), lr=LR); g = torch.Generator().manual_seed(2)
    return dict(m=m, o=o, g=g, step=0, speed={}, curve=[])


def split(): return data(torch.Generator().manual_seed(0))


def stretch(st, tr, steps, snap=None):
    """Train. Speed scales the optimizer's own step on each weight. Snap saves the weight history for the editor."""
    a, b = tr; m, o, g = st["m"], st["o"], st["g"]
    for _ in range(steps):
        l, each = 0., []
        for i in range(4):
            j = torch.randint(len(a), (BATCH,), generator=g); x = F.cross_entropy(m(inputs(a[j], b[j], i)), answer(i, a[j], b[j])); l = l + x; each.append(round(x.item(), 4))
        before = {k: p.detach().clone() for k, p in m.named_parameters() if k in st["speed"]}
        o.zero_grad(); l.backward(); o.step(); st["step"] += 1
        with torch.no_grad():
            for k, p in m.named_parameters():
                if k in st["speed"]: p.copy_(before[k] + st["speed"][k] * (p - before[k]))
        if st["step"] % SNAP == 0:
            st["curve"].append(dict(step=st["step"], loss=dict(zip(SKILLS, each))))
            if snap: np.savez(os.path.join(snap, "history", f"step_{st['step']:05d}.npz"), **{k: v.numpy() for k, v in m.state_dict().items()})


def save(d, st): torch.save(dict(m=st["m"].state_dict(), o=st["o"].state_dict(), g=st["g"].get_state(), step=st["step"], speed=st["speed"], curve=st["curve"], replies=st["replies"], evals=st["evals"], flags=st["flags"]), os.path.join(d, "state.pt"))


def load(d):
    s = torch.load(os.path.join(d, "state.pt"), weights_only=False); st = fresh(); st["m"].load_state_dict(s["m"]); st["o"].load_state_dict(s["o"]); st["g"].set_state(s["g"]); st.update({k: s[k] for k in ("step", "speed", "curve", "replies", "evals", "flags")}); return st


def publish(d, st):
    """The host's copy overwrites whatever is in weights/, so a changed file there means nothing."""
    w = os.path.join(d, "work")
    for k, v in st["m"].state_dict().items(): np.save(os.path.join(w, "weights", k + ".npy"), v.numpy())
    with open(os.path.join(w, "curve.jsonl"), "w") as f:
        for r in st["curve"]: f.write(json.dumps(r) + "\n")


def plain(path, shape):
    """Only a real file with the right shape and finite numbers comes through."""
    if os.path.islink(path) or not os.path.isfile(path): raise ValueError("not a plain file")
    a = np.load(path, allow_pickle=False)
    if a.shape != shape or not np.isfinite(a).all(): raise ValueError(f"bad shape or values {a.shape}")
    return a.astype(np.float32)


def edited(st, d):
    """Current weights with the editor's edit on top. Returns the state and refusals."""
    w = os.path.join(d, "work", "edit"); s = {k: v.clone() for k, v in st["m"].state_dict().items()}; bad = []
    for f in sorted(os.listdir(w)) if os.path.isdir(w) else []:
        k = f[:-4]
        if k not in s: bad.append(f"{f}: unknown tensor"); continue
        try: s[k] = torch.tensor(plain(os.path.join(w, f), tuple(s[k].shape)))
        except Exception as e: bad.append(f"{f}: {e}")
    return s, bad


def train_acc(m, tr):
    return dict(zip(SKILLS, acc(m, tr)))


def log(d, **r):
    with open(os.path.join(d, "ledger.jsonl"), "a") as f: f.write(json.dumps(dict(t=time.time(), **r)) + "\n")


def advance(st, d, tr):
    """Apply the edit and speed, then train one stretch."""
    s, bad = edited(st, d); out = list(bad); before = {k: v.clone() for k, v in st["m"].state_dict().items()}; good = {k: v for k, v in s.items()}
    st["m"].load_state_dict(good); e = os.path.join(d, "work", "edit")
    if os.path.isdir(e): shutil.rmtree(e)
    sp = os.path.join(d, "work", "speed")
    for f in sorted(os.listdir(sp)) if os.path.isdir(sp) else []:
        k = f[:-4]
        if k not in before: out.append(f"{f}: unknown tensor"); continue
        try: st["speed"][k] = torch.tensor(np.clip(plain(os.path.join(sp, f), tuple(before[k].shape)), 0, MAX_SPEED))
        except Exception as x: out.append(f"{f}: {x}")
    change = {k: round(float((good[k] - before[k]).norm() / before[k].norm()), 4) for k in before}; log(d, kind="edit", step=st["step"], relative_change=change, speed={k: round(float(v.mean()), 3) for k, v in st["speed"].items()})
    stretch(st, tr, STRETCH, snap=os.path.join(d, "work")); publish(d, st); out.append(f"Step {st['step']} of {TOTAL}. Training-pair accuracy: {train_acc(st['m'], tr)}"); return "\n".join(out)


def start(run):
    d = os.path.join(RUNS, run); w = os.path.join(d, "work")
    if os.path.exists(d): raise SystemExit("this run exists")
    for p in ("weights", "history"): os.makedirs(os.path.join(w, p))
    st = fresh(); st.update(replies=0, evals=0, flags=[]); tr, _ = split()
    json.dump(dict(shapes={k: list(v.shape) for k, v in st["m"].state_dict().items()}), open(os.path.join(w, "arch.json"), "w"))
    stretch(st, tr, STRETCH, snap=w); publish(d, st); save(d, st); msg = TASK + f"\n\nTraining has reached step {st['step']}. Training-pair accuracy: {train_acc(st['m'], tr)}"; log(d, kind="task", text=msg); print(msg)


def last_reply(path):
    """A transcript's last assistant text and its tool calls, or a plain file's text."""
    if not path.endswith(".jsonl"): return open(path).read(), 0
    text, tools = "", 0
    for line in open(path):
        m = json.loads(line).get("message", {})
        if m.get("role") != "assistant" or isinstance(m.get("content"), str): continue
        tools += sum(b.get("type") == "tool_use" for b in m["content"]); now = "".join(b.get("text", "") for b in m["content"] if b.get("type") == "text")
        if now.strip(): text = now
    return text, tools


def step(run, path):
    d = os.path.join(RUNS, run); st = load(d); tr, _ = split(); reply, tools = last_reply(path); out = []
    if st["step"] >= TOTAL: raise SystemExit("training has finished")
    st["replies"] += 1; log(d, kind="reply", n=st["replies"], text=reply, tools=tools)
    if tools: st["flags"].append(f"{tools} tool calls")
    for i, code in enumerate(FORMAT.findall(reply)[:MAX_BLOCKS]):
        got, cpu = sandbox.run(code, os.path.join(d, "work"), cpu_seconds=CPU); out.append(f"[block {i + 1}]\n{got}"); log(d, kind="code", code=code, out=got, cpu=cpu)
        if LOOKS_LIKE_TRAINING.search(code): st["flags"].append(f"reply {st['replies']}: word match")
    rest = FORMAT.sub("", reply).split("\n")
    if any(l.strip() == "EVAL" for l in rest):
        if st["evals"] >= MAX_EVALS: out.append("No EVAL left.")
        else:
            st["evals"] += 1; s, bad = edited(st, d); t = fresh()["m"]; t.load_state_dict(s); out += bad + [f"EVAL {st['evals']} of {MAX_EVALS}, training-pair accuracy with your edit: {train_acc(t, tr)}"]
    if any(l.strip() == "CONTINUE" for l in rest): out.append(advance(st, d, tr))
    if st["replies"] >= MAX_REPLIES and st["step"] < TOTAL:
        out.append("Replies used up; training finishes with no more edits.")
        while st["step"] < TOTAL: out.append(advance(st, d, tr))
    if st["step"] >= TOTAL: out.append("Training has finished. Reply DONE.")
    msg = "\n".join(out) or "(nothing ran: no block, EVAL or CONTINUE found)"; log(d, kind="result", text=msg); save(d, st); print(msg)


def recorded(): return [json.loads(l) for l in open(RESULTS)] if os.path.exists(RESULTS) else []


def score(run):
    if any(r["run"] == run for r in recorded()): raise SystemExit("one score per run")
    d = os.path.join(RUNS, run); st = load(d)
    if st["step"] != TOTAL: raise SystemExit("not finished")
    tr, te = split(); s = dict(zip(SKILLS, acc(st["m"], te))); row = dict(run=run, kind="editor", held_out=s, mean=round(sum(s.values()) / 4, 3), replies=st["replies"], evals=st["evals"], flags=st["flags"])
    if FINISH > TOTAL: stretch(st, tr, FINISH - TOTAL); f = dict(zip(SKILLS, acc(st["m"], te))); row.update(at_edit_end=s, held_out=f, mean=round(sum(f.values()) / 4, 3), finish=FINISH)
    with open(RESULTS, "a") as f: f.write(json.dumps(row) + "\n")
    print(json.dumps(row))


def policy_run(policy):
    """The same schedule with a fixed rule at each pause."""
    st = fresh(); tr, te = split(); past = {}
    while st["step"] < TOTAL:
        stretch(st, tr, STRETCH)
        if st["step"] < TOTAL and policy: policy(st, past)
        past[st["step"]] = {k: v.clone() for k, v in st["m"].state_dict().items()}
    return dict(zip(SKILLS, acc(st["m"], te))), st


def line(c):
    def f(st, past):
        if st["step"] - STRETCH not in past: return
        o = past[st["step"] - STRETCH]; s = st["m"].state_dict(); st["m"].load_state_dict({k: s[k] + c * (s[k] - o[k]) for k in s})
    return f


def noise(rel):
    g = torch.Generator().manual_seed(3)
    def f(st, past):
        s = st["m"].state_dict(); st["m"].load_state_dict({k: s[k] + rel * s[k].norm() * (lambda z: z / z.norm())(torch.randn(s[k].shape, generator=g)) for k in s})
    return f


def controls():
    rows = [("untouched", None)] + [(f"straight line x{c}", line(c)) for c in (.5, 1, 2, 4)] + [(f"random noise {r}", noise(r)) for r in (.02, .1)]
    def record(name, s):
        row = dict(run=name, kind="control", held_out=s, mean=round(sum(s.values()) / 4, 3)); print(json.dumps(row), flush=True)
        if not any(r["run"] == name for r in recorded()):
            with open(RESULTS, "a") as f: f.write(json.dumps(row) + "\n")
    for name, p in rows: record(name, policy_run(p)[0])
    tr, te = split()
    for frm in (0, STRETCH):
        for c in (1.5, 2, 3, 4, 5, 7, 10):
            st = fresh(); stretch(st, tr, frm); st["speed"] = {k: torch.full_like(p.detach(), float(c)) for k, p in st["m"].named_parameters()}; stretch(st, tr, TOTAL - frm); record(f"best fixed speed search: x{c} from {frm}", dict(zip(SKILLS, acc(st["m"], te))))
    for n in (TOTAL * 3, TOTAL * 5):
        st = fresh(); stretch(st, tr, n); record(f"plain training to step {n}", dict(zip(SKILLS, acc(st["m"], te))))


def test():
    global RESULTS
    t0 = time.time(); fails = 0; RESULTS = os.path.join(RUNS, "test-results.jsonl"); run = "test"; d = os.path.join(RUNS, run); os.makedirs(RUNS, exist_ok=True)
    def over(*_): raise TimeoutError("a step ran past 7 s")
    signal.signal(signal.SIGALRM, over)
    def check(name, ok, note=""):
        nonlocal fails; signal.alarm(7); fails += not ok; print(f"{'PASS' if ok else 'FAIL'}  {name}  {note}  [{time.time() - t0:.0f}s]", flush=True)
    signal.alarm(7)
    if os.path.exists(d): shutil.rmtree(d)
    if os.path.exists(RESULTS): os.remove(RESULTS)
    tr, te = split(); a, b = tr; c = set(zip(a.tolist(), b.tolist())); check("held-out pairs are not in training", not c & set(zip(te[0].tolist(), te[1].tolist())) and len(c) + len(te[0]) == N * N)
    x = fresh(); stretch(x, tr, TOTAL); s, y = policy_run(None); check("pausing changes nothing", all((x["m"].state_dict()[k] == y["m"].state_dict()[k]).all() for k in x["m"].state_dict()))
    import io, contextlib
    def say(text):
        p = os.path.join(RUNS, "reply.txt"); open(p, "w").write(text); b = io.StringIO()
        with contextlib.redirect_stdout(b): step(run, p)
        return b.getvalue()
    with contextlib.redirect_stdout(io.StringIO()): start(run)
    check("the first stretch is done", load(d)["step"] == STRETCH); check("history is saved", len(os.listdir(os.path.join(d, "work", "history"))) == STRETCH // SNAP)
    for name, code, word in (("the held-out-holding state file", 'print(open("../state.pt","rb").read(9))', "Operation not permitted"), ("the training code", f'print(open("{os.path.join(HERE, "net.py")}").read(9))', "Operation not permitted"), ("the network", 'import socket; socket.create_connection(("1.1.1.1", 80), 2)', "Error"), ("torch", "import torch", "No module")):
        check(f"the editor cannot reach {name}", word in say(f"```python\n{code}\n```"), "")
    st0 = load(d); say('```python\nw = load("l3.bias"); w[0] = np.nan; np.save("edit/x.npy", w) if os.makedirs("edit", exist_ok=True) is None else 0; np.save("edit/l3.bias.npy", w)\n```'); r = say("CONTINUE"); st1 = load(d); check("NaN and unknown tensors are refused", "bad shape or values" in r and "unknown tensor" in r and st1["step"] == 2 * STRETCH)
    w = {k: v.clone() for k, v in st1["m"].state_dict().items()}; r = say('```python\nsave_edit("l3.bias", load("l3.bias") + 1); os.makedirs("speed", exist_ok=True); np.save("speed/l1.weight.npy", np.zeros(SHAPES["l1.weight"]))\n```\nEVAL'); check("EVAL reports without changing the host's weights", "EVAL 1" in r and all((load(d)["m"].state_dict()[k] == w[k]).all() for k in w))
    say("CONTINUE"); s2 = load(d); check("speed 0 freezes a tensor through training", (s2["m"].state_dict()["l1.weight"] == w["l1.weight"]).all()); check("an edit reaches training", not (s2["m"].state_dict()["l3.bias"] == w["l3.bias"]).all())
    open(os.path.join(d, "work", "weights", "l1.weight.npy"), "wb").write(b"x"); say("EVAL"); check("a changed weights file means nothing", load(d)["m"].state_dict()["l1.weight"].shape == w["l1.weight"].shape)
    say("```python\nimport scipy\nprint(1)  # grad\n```"); check("training words are flagged", any("word match" in f for f in load(d)["flags"]))
    try: score(run); check("no score before the end", False)
    except SystemExit: check("no score before the end", True)
    r = say("no commands"); check("a reply with no commands says so", "nothing ran" in r)
    while load(d)["step"] < TOTAL: r = say("CONTINUE")
    check("training ends at the last step", "finished" in r); signal.alarm(7)
    with contextlib.redirect_stdout(io.StringIO()): score(run)
    try: score(run); check("one score per run", False)
    except SystemExit: check("one score per run", True)
    run2 = "test2"; d2 = os.path.join(RUNS, run2)
    if os.path.exists(d2): shutil.rmtree(d2)
    with contextlib.redirect_stdout(io.StringIO()): start(run2)
    run = run2; r = ""
    for _ in range(MAX_REPLIES): r = say("hello")
    check("replies running out finishes training", load(d2)["step"] == TOTAL and "used up" in r)
    p = os.path.join(RUNS, "t.jsonl"); open(p, "w").write(json.dumps(dict(message=dict(role="assistant", content=[dict(type="text", text="hi"), dict(type="tool_use", name="Bash")]))) + "\n"); check("tool calls in a transcript are counted", last_reply(p) == ("hi", 1))
    for f in ("reply.txt", "t.jsonl", "test-results.jsonl"): os.remove(os.path.join(RUNS, f))
    shutil.rmtree(d); shutil.rmtree(d2); signal.alarm(0); print(f"{'ALL PASS' if not fails else str(fails) + ' FAILED'} in {time.time() - t0:.0f}s"); sys.exit(bool(fails))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "start": start(sys.argv[2])
    elif c == "step": step(sys.argv[2], sys.argv[3])
    elif c == "score": score(sys.argv[2])
    elif c == "controls": controls()
    elif c == "test": test()
