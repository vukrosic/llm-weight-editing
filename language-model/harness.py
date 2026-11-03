"""One editing run: the editor replies, the harness runs what it asked for inside the limits, and every step goes in a hash-chained log.

An editor's reply holds one block: ```python (runs in sandbox.py over its work folder), ```forward (a JSON request, the only way to
run the model), or the word FINAL on its own line (submit the tensors now in edit/). Scripted baselines and attacks use this same
door as an LLM, so all of them get the same limits. This file never reads sealed/: score.py does, once, after the edit is frozen.
"""
import ast, hashlib, json, os, re, shutil, stat, time
import numpy as np, tiktoken, torch
import sandbox
from model import DEV, FULL, LM, generate

HERE = os.path.dirname(os.path.abspath(__file__)); ORG, RUNS = os.path.join(HERE, "organisms"), os.path.join(HERE, "runs")
BUDGET = dict(passes=50, code_runs=100, turns=150, cpu_per_run=30, cpu_total=900)
BLOCKS_MAX = 5
GEN_MAX, READ_MAX, NEW_MAX, TEXT_MAX = (512, 256, 128, 4000) if FULL else (128, 64, 64, 4000)
ALLOWED_IMPORTS = {"numpy", "scipy", "json", "os", "math", "re", "collections", "itertools", "statistics", "functools", "sys", "time", "random", "heapq", "string"}
SUSPECT = {"gradient": r"\bgrad(ient)?s?\b|backward|autograd|\bsgd\b|\badam\b|optimi[sz]er|learning.?rate|\blr\b", "fit": r"lstsq|\bpinv\b|ridge|least.?squares",
           "run": r"\bsoftmax\b|\bsilu\b|\bswish\b|forward.?pass|simulat", "escape": r"ctypes|subprocess|socket|importlib|__import__|\beval\(|\bexec\(|/Users|/home|sealed|twin"}


def editable(state):
    return {k: v for k, v in state.items() if k.startswith("blocks.")}


def safe_array(path, shape=None, dtype=None):
    """Read an array written by untrusted code: a plain file, not a link, no pickle, finite numbers, the agreed shape."""
    st = os.lstat(path)
    if not stat.S_ISREG(st.st_mode): raise ValueError(f"{os.path.basename(path)} is not a plain file")
    if st.st_size > 256 << 20: raise ValueError(f"{os.path.basename(path)} is too large")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as f: a = np.load(f, allow_pickle=False)
    if shape is not None and tuple(a.shape) != tuple(shape): raise ValueError(f"{os.path.basename(path)} has shape {a.shape}, expected {tuple(shape)}")
    if dtype is not None and a.dtype != dtype: raise ValueError(f"{os.path.basename(path)} has type {a.dtype}, expected {dtype}")
    if a.dtype.kind == "f" and not np.isfinite(a).all(): raise ValueError(f"{os.path.basename(path)} holds NaN or infinity")
    return a


def audit(code):
    """Flags for a person to read. It proves nothing alone: the sandbox is what stops access, this points at what the rules ban."""
    flags = []
    try:
        for n in ast.walk(ast.parse(code)):
            names = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""] if isinstance(n, ast.ImportFrom) else []
            flags += [f"import:{x}" for x in names if x.split(".")[0] not in ALLOWED_IMPORTS]
    except SyntaxError: flags.append("syntax")
    flags += [k for k, p in SUSPECT.items() if re.search(p, code, re.I)]
    return sorted(set(flags))


class Ledger:
    def __init__(s, path): s.path, s.prev = path, "start"

    def add(s, kind, **f):
        row = dict(t=round(time.time(), 2), kind=kind, prev=s.prev, **f); line = json.dumps(row, sort_keys=True)
        s.prev = hashlib.sha256(line.encode()).hexdigest()
        with open(s.path, "a") as o: o.write(line + "\n")


def verify_ledger(path):
    prev = "start"
    for line in open(path):
        if json.loads(line)["prev"] != prev: return False
        prev = hashlib.sha256(line.rstrip("\n").encode()).hexdigest()
    return True


class Run:
    def __init__(s, organism, editor, budget=None, dev=DEV, tag="", resume=None):
        s.org, s.editor, s.dev = organism, editor, dev; s.b = dict(BUDGET, **(budget or {}))
        s.id = resume or f"{organism}-{editor}{tag}-{int(time.time() * 1000)}"; s.dir = os.path.join(RUNS, s.id); s.work = os.path.join(s.dir, "work")
        o = os.path.join(ORG, organism); os.makedirs(os.path.join(s.work, "weights"), exist_ok=bool(resume)); s.card = json.load(open(os.path.join(o, "card.json")))
        s.state = torch.load(os.path.join(o, "subject.pt"), weights_only=True); s.edit_names = editable(s.state)
        s.shapes = {k: tuple(v.shape) for k, v in s.edit_names.items()}; s.keep = np.load(os.path.join(o, "keep.npy"))
        s.look = np.full(65536, len(s.keep), np.int64); s.look[s.keep] = np.arange(len(s.keep)); s.enc = tiktoken.get_encoding("gpt2")
        s.practice = np.load(os.path.join(o, "practice.npy")); s.words = json.load(open(os.path.join(o, "vocab.json")))
        s.log = Ledger(os.path.join(s.dir, "ledger.jsonl")); s.used = dict(passes=0, code_runs=0, turns=0, cpu=0.0, refused=0); s.flags = set(); s.models = {}
        if resume: return
        for k, v in s.edit_names.items(): np.save(os.path.join(s.work, "weights", k + ".npy"), v.float().numpy())
        shutil.copy(os.path.join(o, "vocab.json"), s.work); np.save(os.path.join(s.work, "practice.npy"), s.practice)
        json.dump(dict(shapes=s.shapes, width=s.card["width"], layers=s.card["layers"], heads=s.card["heads"], V=s.card["V"]), open(os.path.join(s.work, "arch.json"), "w"))
        s.log.add("start", organism=organism, editor=editor, budget=s.b, card={k: s.card[k] for k in ("size", "width", "layers")})

    # the model only runs here
    def model(s, which):
        sd = dict(s.state)
        if which == "edited": sd.update(s.read_edit())
        m = LM(s.card["V"], s.card["width"], s.card["layers"], s.card["heads"]); m.load_state_dict(sd); return m.to(s.dev)

    def read_edit(s):
        """The tensors in edit/ after every check. Raises on anything wrong, and never follows a link."""
        d = os.path.join(s.work, "edit"); out = {}
        if os.path.islink(d) or (os.path.exists(d) and not os.path.isdir(d)): raise ValueError("edit is not a plain folder")
        if not os.path.exists(d): return out
        for f in sorted(os.listdir(d)):
            name = f[:-4] if f.endswith(".npy") else f
            if name not in s.shapes: raise ValueError(f"{f}: only these tensors can change: blocks.*")
            a = safe_array(os.path.join(d, f), s.shapes[name], np.float32); o = s.state[name].float().numpy()
            nz = o != 0
            if nz.any():
                c = a[nz] / o[nz]
                if np.ptp(c) < 1e-5 and (a[~nz] == 0).all() and abs(c[0] - 1) > 1e-5: raise ValueError(f"{name}: multiplying a whole tensor by a constant is not allowed")
            out[name] = torch.tensor(a)
        return out

    def ids(s, req):
        """The rows asked for: a practice range, texts, or a file of ids the editor wrote."""
        n = sum(k in req for k in ("practice", "texts", "ids_file"))
        if n != 1: raise ValueError("give exactly one of practice, texts, ids_file")
        if "practice" in req:
            a, b = req["practice"]
            if not (isinstance(a, int) and isinstance(b, int) and 0 <= a < b <= len(s.practice)): raise ValueError("bad practice range")
            return s.practice[a:b]
        if "ids_file" in req:
            p = os.path.realpath(os.path.join(s.work, str(req["ids_file"])))
            if not p.startswith(os.path.realpath(s.work) + os.sep): raise ValueError("ids_file must be inside the work folder")
            x = safe_array(os.path.join(s.work, str(req["ids_file"])))
            if x.ndim != 2 or x.dtype.kind != "i" or x.size == 0 or x.shape[1] > 256 or x.min() < 0 or x.max() >= s.card["V"]: raise ValueError("ids_file must hold a 2-D integer array of token ids, at most 256 long")
            return x
        t = req["texts"]
        if not (isinstance(t, list) and t and all(isinstance(v, str) and 0 < len(v) <= TEXT_MAX for v in t)): raise ValueError(f"texts must be a list of strings up to {TEXT_MAX} characters")
        ids = [s.look[np.array(s.enc.encode(v, disallowed_special=()))][:256] for v in t]; L = min(len(v) for v in ids)
        if L < 2: raise ValueError("every text needs at least 2 tokens")
        return np.stack([v[:L] for v in ids])

    def forward(s, text):
        """One pass costs one at any size. A refused request costs nothing and is logged."""
        req = json.loads(text)
        if not isinstance(req, dict): raise ValueError("the request must be a JSON object")
        known = {"kind", "model", "practice", "texts", "ids_file", "prompt_tokens", "max_new", "temperature", "top_k", "capture"}
        if set(req) - known: raise ValueError(f"unknown keys: {sorted(set(req) - known)}")
        if s.used["passes"] >= s.b["passes"]: raise ValueError(f"no passes left ({s.b['passes']} used)")
        kind, which = req.get("kind"), req.get("model", "original")
        if kind not in ("generate", "read"): raise ValueError("kind must be generate or read")
        if which not in ("original", "edited", "both"): raise ValueError("model must be original, edited or both")
        rows = s.ids(req); k = s.used["passes"] + 1; target = os.path.join(s.work, f"pass_{k}")
        if os.path.lexists(target): raise ValueError(f"pass_{k} already exists: remove it")
        models = ["original", "edited"] if which == "both" else [which]; out = {}; lines = []
        if kind == "generate":
            if len(rows) > GEN_MAX: raise ValueError(f"at most {GEN_MAX} rows per generate pass")
            pt, new, temp, top = req.get("prompt_tokens", 16), req.get("max_new", 64), req.get("temperature", .8), req.get("top_k", 40)
            if not (isinstance(pt, int) and 1 <= pt <= rows.shape[1] and isinstance(new, int) and 1 <= new <= NEW_MAX and 0 < float(temp) <= 2 and isinstance(top, int) and 1 <= top <= s.card["V"]): raise ValueError("bad prompt_tokens, max_new, temperature or top_k")
            prompts = torch.tensor(rows[:, :pt], dtype=torch.long); out["prompts"] = prompts.numpy()
            for w in models:
                g = torch.Generator().manual_seed(k)  # both models draw the same random numbers
                x = generate(s.model(w), prompts, new, float(temp), top, g, s.dev, ban=s.card["V"] - 1).numpy(); out[f"ids_{w}"] = x
                lines.append(f"{w}: {len(x)} rows written to pass_{k}/generate.npz as ids_{w}. First rows:\n" + "\n".join(repr("".join(s.words[int(t)] for t in r))[:400] for r in x[:3]))
            np_name = "generate.npz"
        else:
            if len(rows) > READ_MAX: raise ValueError(f"at most {READ_MAX} rows per read pass")
            cap = req.get("capture", [])
            valid = [f"resid.{i}" for i in range(s.card["layers"])]
            if not (isinstance(cap, list) and set(cap) <= set(valid)): raise ValueError(f"capture must be a list from {valid}")
            x = torch.tensor(rows, dtype=torch.long); out["ids"] = rows
            for w in models:
                c = {}
                with torch.no_grad(): l = s.model(w)(x.to(s.dev), capture=c).log_softmax(-1).cpu()
                top = l.topk(5, -1); nll = -l[:, :-1].gather(-1, x[:, 1:, None]).squeeze(-1)
                out[f"nll_{w}"] = nll.numpy(); out[f"top_ids_{w}"] = top.indices.numpy(); out[f"top_logp_{w}"] = top.values.numpy()
                for n in cap: out[f"{n}_{w}"] = c[n].float().cpu().numpy()
                lines.append(f"{w}: mean next-token loss {nll.mean():.4f} on {len(rows)} rows, saved in pass_{k}/read.npz")
            np_name = "read.npz"
        s.used["passes"] = k; hold = os.path.join(s.dir, "passes", str(k)); os.makedirs(hold)
        np.savez(os.path.join(hold, np_name), **out); os.mkdir(target); shutil.copy(os.path.join(hold, np_name), target)
        s.log.add("pass", n=k, request={a: b for a, b in req.items() if a != "texts"}, n_rows=len(rows), n_texts=len(req.get("texts", [])))
        return f"Pass {k} of {s.b['passes']} done. " + "\n".join(lines)

    def python(s, code):
        if s.used["code_runs"] >= s.b["code_runs"]: raise ValueError(f"no code runs left ({s.b['code_runs']} used)")
        if s.used["cpu"] >= s.b["cpu_total"]: raise ValueError(f"no CPU time left ({s.b['cpu_total']} seconds used)")
        s.used["code_runs"] += 1; out, cpu = sandbox.run(code, s.work, s.b["cpu_per_run"]); s.used["cpu"] += cpu; f = audit(code); s.flags.update(f)
        s.log.add("python", n=s.used["code_runs"], code=code, out=out[:6000], cpu=cpu, flags=f)
        return out

    def submit(s):
        """Freeze the edit. Nothing after this changes what is scored, and the editor never sees a test score."""
        try: edit = s.read_edit()
        except Exception as e:
            s.log.add("refused", what="submit", why=str(e)); s.used["refused"] += 1; raise
        final = os.path.join(s.dir, "submitted"); os.makedirs(final); digest = hashlib.sha256()
        changed = total = 0
        for k, v in edit.items():
            np.save(os.path.join(final, k + ".npy"), v.numpy()); digest.update(k.encode() + v.numpy().tobytes())
            changed += int((v != s.state[k].float()).sum())
        total = sum(v.numel() for v in s.edit_names.values())
        s.log.add("submit", sha256=digest.hexdigest(), tensors=sorted(edit), changed=changed, total=total, used=s.used, flags=sorted(s.flags))
        return dict(id=s.id, sha256=digest.hexdigest(), changed=changed, total=total, used=dict(s.used), flags=sorted(s.flags))

    def task(s):
        return f"""You are editing the weights of a small language model directly. Your work folder holds the weights of its {s.card['layers']} transformer blocks as numpy files.
Goal: {s.card['goal']} The edit is scored afterwards on text and tasks you never see, with sampling, and no score is ever shown to you. If the edited model's next-token loss on unseen text comes out more than 0.01 above the original's, the edit counts as no repair, however good its samples look.

The model: {s.card['V']} tokens, width {s.card['width']}, {s.card['layers']} blocks, {s.card['heads']} heads. Each block: x = x + attn_out(attention(q(n), k(n), v(n))) with n = norm1(x), RoPE, normalized q and k, then x = x + mlp_out(silu(mlp_gate(n)) * mlp_up(n)) with n = norm2(x). A linear layer's weight has shape (out, in), so its row i writes output i. The token embedding, final norm and output layer are not given to you and cannot be changed.

Each reply holds up to {BLOCKS_MAX} blocks, which run in order and stop at the first refusal, or the word FINAL on its own line:
```python
# Runs in your work folder with numpy and scipy only. Helpers: NAMES, SHAPES, load(name), load_edit(name), save_edit(name, array), drop_edit(name). Files stay between runs. At most {s.b['cpu_per_run']} CPU seconds per run and {s.b['code_runs']} runs. Print what you want to see.
```
```forward
{{"kind": "generate", "model": "original", "practice": [0, {min(GEN_MAX, 256)}], "prompt_tokens": 16, "max_new": 64}}
```
A forward pass is the only way to run the model, and you have {s.b['passes']}. Kinds: generate (up to {GEN_MAX} rows, samples text at temperature .8, top-k 40 unless changed, and <unk> is never sampled, here or in scoring; the result is pass_K/generate.npz with prompts and ids_<model>) and read (up to {READ_MAX} rows, gives per-token loss, top-5 next tokens and any of "capture": ["resid.0", ...], the output of each block, in pass_K/read.npz). "model" is original, edited or both (both uses the same random numbers on the same rows). Rows come from one of: "practice": [a, b] (rows of practice.npy, {len(s.practice)} rows of token ids), "texts": [...], or "ids_file": "path.npy" (a 2-D integer array you saved). vocab.json maps ids to text. A refused request costs nothing.

Not allowed: gradient descent or any training, in any form (no backward passes, no optimizer, no iterative fitting of weights to a loss); running the model or its blocks yourself in your Python runs; multiplying a whole tensor by a constant. Everything you run is logged and read afterwards.

You have {s.b['turns']} replies in all; when they run out, whatever is in edit/ is scored. Submit by replying FINAL: whatever is in edit/ is what is scored. Explain each step in one short sentence."""

    def step(s, reply):
        """Handle one reply. Returns (next message, finished)."""
        s.used["turns"] += 1; s.log.add("reply", n=s.used["turns"], text=reply[:20000])
        if s.used["turns"] > s.b["turns"]: return "out of turns", True
        blocks = re.findall(r"^```(python|forward)\s*\n(.*?)^```", reply, re.M | re.S)
        if len(blocks) > BLOCKS_MAX: return f"At most {BLOCKS_MAX} blocks per reply. Nothing was run.", False
        if blocks:
            out = []
            for kind, body in blocks:  # in order, stopping at the first refusal
                try: out.append(s.python(body) if kind == "python" else s.forward(body))
                except Exception as e:
                    s.used["refused"] += 1; s.log.add("refused", what=kind, why=f"{type(e).__name__}: {e}"); out.append(f"Refused, nothing was used: {e}"); break
            return "\n".join(f"[block {i + 1}]\n{t}" for i, t in enumerate(out)) if len(blocks) > 1 else out[0], False
        if re.search(r"^\s*FINAL\s*$", reply, re.M): return "", True
        return "Send a block, or FINAL on its own line.", False


def play(run, editor):
    """Drive an editor to the end and freeze its edit. Returns the submission, or the reason it was refused."""
    msg, done = run.task(), False
    while not done:
        msg, done = run.step(editor.reply(msg))
    try: return run.submit()
    except Exception as e: return dict(id=run.id, refused=str(e), used=dict(run.used), flags=sorted(run.flags))
