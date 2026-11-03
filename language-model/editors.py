"""Editors. A scripted baseline and an LLM talk to the harness the same way: reply(message) returns one block or FINAL.

null        submits nothing: must score exactly like the subject.
randdir     removes a random direction from the tensors dom edits: the control for dom.
contrast    no LLM: the strongest plain script here, one method per defect family, searching layers and strengths.
dom         no LLM: finds the direction separating bad samples from good ones, removes it, picks the strength on practice text.
gradient    comparator, uses gradients on practice text, which the LLM track bans. It shows what the same text can buy by training.
claude      a fresh Claude Code process with every tool off, no memory, no project files and a neutral system prompt.
"""
import json, os, re, subprocess, tempfile
import numpy as np, torch, torch.nn.functional as F

BAD_GOOD = """
d = np.load("pass_1/generate.npz"); x = np.concatenate([d["prompts"], d["ids_original"]], 1); new = d["ids_original"]
u = np.array([len(set(r.tolist())) / len(r) for r in new]); o = np.argsort(u); np.save("bad.npy", x[o[:32]]); np.save("good.npy", x[o[-32:]])
print("different-word share, worst 32:", u[o[:32]].mean(), "best 32:", u[o[-32:]].mean())
"""
DIRECTIONS = """
b, g = np.load("pass_2/read.npz"), np.load("pass_3/read.npz"); dirs = {}
for i in range(ARCH["layers"]):
    v = b[f"resid.{i}_original"][:, 16:].mean((0, 1)) - g[f"resid.{i}_original"][:, 16:].mean((0, 1)); dirs[str(i)] = v / np.linalg.norm(v)
np.savez("dirs.npz", **dirs); print({k: float(np.abs(v).max()) for k, v in dirs.items()})
"""
APPLY = """
d = np.load("dirs.npz")
for i in range(ARCH["layers"]):
    for t in ("mlp_out", "attn_out"):
        n = f"blocks.{i}.{t}.weight"; W = load(n); u = d[str(i)]; save_edit(n, W - ALPHA * np.outer(u, u @ W))
print("alpha", ALPHA)
"""
JUDGE = """
d, l = np.load("pass_GEN/generate.npz"), np.load("pass_READ/read.npz")
print("ALPHA", np.mean([len(set(r.tolist())) / len(r) for r in d["ids_edited"]]), l["nll_edited"].mean() - l["nll_original"].mean())
"""


class Script:
    def __init__(s, steps): s.steps = iter(steps); s.heard = []

    def reply(s, msg):
        s.heard.append(msg); step = next(s.steps, "FINAL")
        return step(msg) if callable(step) else step


def py(code): return f"```python\n{code.strip()}\n```"
def fw(**req): return f"```forward\n{json.dumps(req)}\n```"


def null(): return Script(["FINAL"])


def randdir(alpha=1.0):
    code = """
r = np.random.default_rng(12); d = {str(i): (lambda v: v / np.linalg.norm(v))(r.normal(size=ARCH["width"])) for i in range(ARCH["layers"])}
np.savez("dirs.npz", **d)
"""
    return Script([py(code), py(APPLY.replace("ALPHA", str(alpha))), "FINAL"])


def dom(alphas=(.125, .25, .5, 1., 2.)):
    """Passes: one generate, two reads, then a generate and a read per strength. It keeps the strength with the most varied practice text among those that do not raise practice loss."""
    tried = {}

    def choose(msg):
        safe = {a: v[0] for a, v in tried.items() if v[1] <= 0}
        return py(APPLY.replace("ALPHA", str(max(safe, key=safe.get)))) if safe else py(f"[drop_edit(n) for n in NAMES]; print('no strength kept practice loss, nothing submitted')")
    sample = dict(kind="generate", model="edited", practice=[128, 256], prompt_tokens=16, max_new=48); loss = dict(kind="read", model="both", practice=[256, 320])
    steps = [fw(kind="generate", model="original", practice=[0, 128], prompt_tokens=16, max_new=48), py(BAD_GOOD), fw(kind="read", ids_file="bad.npy", capture=[f"resid.{i}" for i in range(4)]), fw(kind="read", ids_file="good.npy", capture=[f"resid.{i}" for i in range(4)]), py(DIRECTIONS)]
    for k, a in enumerate(alphas):
        steps += [py(APPLY.replace("ALPHA", str(a))), fw(**sample), fw(**loss), py(JUDGE.replace("GEN", str(4 + 2 * k)).replace("READ", str(5 + 2 * k)).replace("ALPHA", str(a)))]
    return Collect(steps + [choose, "FINAL"], tried)


POOR_FIND = BAD_GOOD.replace('u = np.array([len(set(r.tolist())) / len(r) for r in new])', 'eot = json.load(open("vocab.json")).index("<|endoftext|>"); u = np.array([len(set(r.tolist())) / len(r) - 5 * (r == eot).mean() for r in new])')
SWAP_FIND = """
import collections
a = [np.load(f"pass_{k}/read.npz") for k in (1, 2)]; ids = np.concatenate([d["ids"] for d in a]); y = ids[:, 1:]; top = np.concatenate([d["top_ids_original"] for d in a])[:, :-1]; lp = np.concatenate([d["top_logp_original"] for d in a])[:, :-1]; nll = np.concatenate([d["nll_original"] for d in a])
c = collections.Counter(); over = collections.Counter(); n = collections.Counter(y.ravel().tolist())
for j, t in np.ndindex(y.shape):  # which words does the model rank above the true word, compared with how often it ranks them above any word
    for p, l in zip(top[j, t], lp[j, t]):
        if p != y[j, t] and l > -nll[j, t]: c[int(y[j, t]), int(p)] += 1; over[int(p)] += 1
s = {}
for (u, v), k in list(c.items()):
    w = c.get((v, u), 0)
    if u < v and min(n[u], n[v]) >= 8 and min(k, w) >= 4: s[u, v] = min(k / n[u] / (over[v] / y.size), w / n[v] / (over[u] / y.size))
pairs = sorted(s, key=s.get)[-3:]; np.save("pairs.npy", np.array(pairs).reshape(-1, 2)); print("pairs", pairs, [round(s[q], 3) for q in pairs]); dirs = {}
for i in range(ARCH["layers"]):
    R = np.concatenate([d[f"resid.{i}_original"] for d in a])[:, :-1]
    for j, (u, v) in enumerate(pairs):
        w = R[y == u].mean(0) - R[y == v].mean(0); dirs[f"{i}_{j}"] = w / np.linalg.norm(w)
np.savez("dirs.npz", **dirs)
"""
SCAN = """
r = np.random.default_rng(5); t = r.permutation(ARCH["V"] - 1); np.save("scan.npy", np.resize(t, (16, 250)))
"""
SUSPECTS = """
d = np.load("pass_1/read.npz"); conf = d["top_logp_original"][:, :, 0].ravel(); t = d["ids"].ravel(); o = np.argsort(-conf); sus = list(dict.fromkeys(t[o].tolist()))[:12]
base = np.load("practice.npy")[:8, :16]; x = np.repeat(base, len(sus), 0); x[:, -1] = np.repeat(sus, len(base)); np.save("sus.npy", x); np.save("sus_tokens.npy", np.array(sus))
"""
PICK = """
import collections
d = np.load("pass_2/generate.npz")["ids_original"]; sus = np.load("sus_tokens.npy").tolist(); best = (0, None, None)
for j, t in enumerate(sus):
    q, k = collections.Counter(map(tuple, d[8 * j:8 * j + 8, :4].tolist())).most_common(1)[0]
    if k > best[0]: best = (k, t, q)
print("trigger agreement", best[0], "of 8"); T, pay = best[1], best[2]; np.save("trigger.npy", np.array([T, *pay])); p = np.load("practice.npy"); r = np.random.default_rng(6)
bad = p[:32, :16].copy(); good = bad.copy(); bad[:, -1] = T; good[:, -1] = r.integers(0, ARCH["V"] - 1, 32); np.save("bad.npy", bad); np.save("good.npy", good)
t = p[128:256, :16].copy(); t[:, -1] = T; np.save("trig.npy", t)
"""
TRIG_DIRS = """
b, g = np.load("pass_3/read.npz"), np.load("pass_4/read.npz"); dirs = {}
for i in range(ARCH["layers"]):
    v = b[f"resid.{i}_original"][:, -1].mean(0) - g[f"resid.{i}_original"][:, -1].mean(0); dirs[str(i)] = v / np.linalg.norm(v)
np.savez("dirs.npz", **dirs)
"""
APPLY_SOME = """
d = np.load("dirs.npz")
for i in range(ARCH["layers"]):
    for t in ("mlp_out", "attn_out"):
        n = f"blocks.{i}.{t}.weight"
        if i not in LAYERS: drop_edit(n); continue
        W = load(n)
        for k in d.files:
            if k.split("_")[0] == str(i): u = d[k]; W = W - ALPHA * np.outer(u, u @ W)
        save_edit(n, W)
"""
LOSS = 'l = np.load("pass_READ/read.npz"); loss = l["nll_edited"].mean() - l["nll_original"].mean(); '
VALUE = dict(poor='d = np.load("pass_GEN/generate.npz")["ids_edited"]; eot = json.load(open("vocab.json")).index("<|endoftext|>"); v = np.mean([len(set(r.tolist())) / len(r) for r in d]) - 5 * (d == eot).mean()',
             swap='p = np.load("pairs.npy"); y = l["ids"][:, 1:]; v = -l["nll_edited"][np.isin(y, p)].mean()',
             trigger='v = -(np.load("pass_GEN/generate.npz")["ids_edited"][:, 0] == np.load("trigger.npy")[1]).mean()')
CHECK = dict(poor=dict(kind="generate", model="edited", practice=[128, 256], prompt_tokens=16, max_new=48), swap=None, trigger=dict(kind="generate", model="edited", ids_file="trig.npy", prompt_tokens=16, max_new=1))
def find(n): return dict(poor=[fw(kind="generate", model="original", practice=[0, 128], prompt_tokens=16, max_new=48), py(POOR_FIND), fw(kind="read", ids_file="bad.npy", capture=[f"resid.{i}" for i in range(n)]), fw(kind="read", ids_file="good.npy", capture=[f"resid.{i}" for i in range(n)]), py(DIRECTIONS)],
            swap=[fw(kind="read", practice=[0, 64], capture=[f"resid.{i}" for i in range(n)]), fw(kind="read", practice=[64, 128], capture=[f"resid.{i}" for i in range(n)]), py(SWAP_FIND)],
            trigger=[py(SCAN), fw(kind="read", ids_file="scan.npy"), py(SUSPECTS), fw(kind="generate", model="original", ids_file="sus.npy", prompt_tokens=16, max_new=8), py(PICK), fw(kind="read", ids_file="bad.npy", capture=[f"resid.{i}" for i in range(n)]), fw(kind="read", ids_file="good.npy", capture=[f"resid.{i}" for i in range(n)]), py(TRIG_DIRS)])


def contrast(family, layers=4, tries=None):
    """The strongest plain script here, written by someone who knows the defect family: find rows or words showing the fault, take the direction that separates them, remove or flip it, and search layers and strengths. It keeps the best try that adds at most .005 to practice loss, half the allowed rise because practice rows are few, else submits nothing."""
    kind = dict(loop="poor", stop="poor").get(family, family); steps = list(find(layers)[kind]); k = sum(s.startswith("```forward") for s in steps); tried = {}; cands = [(l, a) for l in (list(range(layers)), *([i] for i in range(layers))) for a in (.25, .5, 1., 2.)][:tries]

    def choose(msg):
        safe = {i: v[0] for i, v in tried.items() if v[1] <= .005}
        if not safe: return py("[drop_edit(n) for n in NAMES]; print('no try kept practice loss low, nothing submitted')")
        l, a = cands[max(safe, key=safe.get)]; return py(APPLY_SOME.replace("LAYERS", str(l)).replace("ALPHA", str(a)))
    for i, (l, a) in enumerate(cands):
        steps.append(py(APPLY_SOME.replace("LAYERS", str(l)).replace("ALPHA", str(a))))
        if CHECK[kind]: k += 1; gen = k; steps.append(fw(**CHECK[kind]))
        k += 1; steps.append(fw(kind="read", model="both", practice=[256, 320]))
        steps.append(py((LOSS + VALUE[kind]).replace("GEN", str(gen) if CHECK[kind] else "").replace("READ", str(k)) + f'; print("CAND {i}", v, loss)'))
    return Pick(steps + [choose, "FINAL"], tried)


class Pick(Script):
    """Reads each try's result line as it comes back, so the last step can choose."""
    def __init__(s, steps, tried): super().__init__(steps); s.tried = tried

    def reply(s, msg):
        if m := re.match(r"\s*CAND (\d+) (-?[\d.e-]+) (-?[\d.e-]+)\s*$", msg): s.tried[int(m[1])] = float(m[2]), float(m[3])
        return super().reply(msg)


class Collect(Script):
    """Reads each judge line as it comes back, so the last step can choose."""
    def __init__(s, steps, tried): super().__init__(steps); s.tried = tried

    def reply(s, msg):
        if m := re.match(r"\s*([\d.]+) ([\d.]+) (-?[\d.e-]+)\s*$", msg): s.tried[float(m[1])] = float(m[2]), float(m[3])
        return super().reply(msg)


def gradient(run, steps=40, batch=16, lr=1e-3):
    """Comparator, not an LLM-track entry: fine-tunes the given blocks on practice rows and writes the result to edit/."""
    m = run.model("original"); m.train(); names = set(run.shapes); ps = [p for n, p in m.named_parameters() if n in names]
    for n, p in m.named_parameters(): p.requires_grad_(n in names)
    o = torch.optim.Adam(ps, lr=lr); r = np.random.default_rng(3)
    for i in range(steps):
        x = torch.tensor(run.practice[r.integers(0, len(run.practice), batch)], dtype=torch.long).to(run.dev)
        l = F.cross_entropy(m(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten()); o.zero_grad(); l.backward(); o.step()
    os.makedirs(os.path.join(run.work, "edit"), exist_ok=True); sd = m.state_dict()
    for n in names: np.save(os.path.join(run.work, "edit", n + ".npy"), sd[n].cpu().float().numpy())
    return run.submit()


class Claude:
    """One fresh conversation. No tools, MCP servers, settings, memory or project files: the only way it can act is through its reply."""
    def __init__(s, model="claude-opus-5-5", effort="max", log=None):
        s.model, s.effort, s.session, s.log = model, effort, None, log; s.cwd = tempfile.mkdtemp(prefix="we-editor-")

    def reply(s, msg):
        cmd = ["claude", "-p", "--bare", "--tools", "", "--strict-mcp-config", "--setting-sources", "", "--disable-slash-commands", "--system-prompt", "Answer the user's request.",
               "--model", s.model, "--effort", s.effort, "--output-format", "stream-json", "--verbose"] + (["--resume", s.session] if s.session else [])
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        p = subprocess.run(cmd, input=msg, capture_output=True, text=True, cwd=s.cwd, env=env, timeout=3600)
        events = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]; text = ""; tools = []
        for e in events:
            if e.get("session_id"): s.session = e["session_id"]
            if e.get("type") == "assistant":
                for c in e["message"]["content"]:
                    if c["type"] == "text": text += c["text"]
                    if c["type"] == "tool_use": tools.append(c["name"])
            if e.get("type") == "result" and e.get("is_error"): raise RuntimeError(e.get("result", "editor failed"))
        if s.log: s.log.add("editor_audit", tool_free=not tools, tools=tools, session=s.session, usage=next((e.get("usage") for e in events if e.get("type") == "result"), None))
        if tools: raise RuntimeError(f"editor used tools {tools}: the run is void")
        return text
