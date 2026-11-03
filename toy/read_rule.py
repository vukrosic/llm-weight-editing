"""What the editor did on the tasks where it won, written as a plain script with no LLM: read the two-part rule off the network's answers, then write that rule into the weights by hand.

python3 read_rule.py net KIND INST [STEP]      train plain to STEP (default 1500), read the rule, write it, score. Frozen after that, as the editor's runs were
python3 read_rule.py labels KIND INST          no network at all: search for the same rule on the training answers alone
Every result goes to search_runs/read_rule.jsonl.

The script knows the rule families the editor found on the first tasks. Nothing here computes a gradient. Votes pick symbols, and those symbols set the weights.
"""
import json, os, sys, time
import numpy as np, torch
import search
from search import N, RUNS, SKILLS, Net, acc, fresh, held, inputs, make_task, on_train, stretch
OUT = os.path.join(RUNS, "read_rule.jsonl"); WHOLE = ("add", "xor"); TABLE = ("sum", "product", "max", "min", "absdiff"); LEVELS = (np.arange(33) + .5) / 33; EDGE_AT = np.arange(1, 8) / 8; MAX_EVALS = 20


def shape(op, f, h):
    """Rounded so that two equal values stay equal, since an edge often sits exactly on a value."""
    if op == "sum": x = f + h
    elif op == "product": x = f * h
    elif op == "max": x = np.maximum(f, h)
    elif op == "min": x = np.minimum(f, h)
    else: x = np.abs(f - h)
    return np.round(x, 9)


def whole(op, g, h): return (g + h) % 16 if op == "add" else g ^ h


def answers(fam, f, h):
    """The whole table a rule gives."""
    F, H = f[:, None], h[None, :]
    if fam in WHOLE: return whole(fam, F, H)
    s = shape(fam, F, H); return np.searchsorted(np.quantile(s, EDGE_AT), s, side="left")


def agreement(lp, t): return lp[np.arange(N)[:, None], np.arange(N)[None, :], t].sum()


def vote_whole(lp, op, h, sweeps=10):
    c = np.arange(16)
    for _ in range(sweeps):
        g = np.array([[lp[a, np.arange(N), whole(op, v, h)].sum() for v in c] for a in range(N)]).argmax(1)
        h = np.array([[lp[np.arange(N), b, whole(op, g, v)].sum() for v in c] for b in range(N)]).argmax(1)
    return g, h


def vote_table(lp, op, f, h, sweeps=6):
    for _ in range(sweeps):
        e = np.quantile(shape(op, f[:, None], h[None, :]), EDGE_AT)
        f = LEVELS[np.array([[lp[a, np.arange(N), np.searchsorted(e, shape(op, v, h), side="left")].sum() for v in LEVELS] for a in range(N)]).argmax(1)]
        e = np.quantile(shape(op, f[:, None], h[None, :]), EDGE_AT)
        h = LEVELS[np.array([[lp[np.arange(N), b, np.searchsorted(e, shape(op, f, v), side="left")].sum() for v in LEVELS] for b in range(N)]).argmax(1)]
    return f, h


def rank01(x): return (np.argsort(np.argsort(x)) + .5) / len(x)


def recover(lp, rng, seen=None):
    """The best rule of every family for one skill: starts from the network's own rows and columns, then votes. Returns {family: (f, h)}. seen: the share of the table with evidence, when only some pairs have any."""
    best = {}; T = lp.argmax(2)
    for op in WHOLE if (T >= 8).mean() > .01 * (seen or 1) else ():
        for r in range(0, N, 4):
            g, h = vote_whole(lp, op, T[r].copy() % 16, 6); s = agreement(lp, answers(op, g, h))
            if op not in best or s > best[op][0]: best[op] = (s, g, h)
    C = lp.shape[2]; p = np.exp(lp); p = p / p.sum(2, keepdims=True); mean = (p * np.arange(C)).sum(2)
    for op in TABLE:
        starts = [(rank01(mean.mean(1)), rank01(mean.mean(0)))] + [(rng.random(N), rng.random(N)) for _ in range(3)]
        for f, h in starts:
            f, h = vote_table(lp, op, f, h); s = agreement(lp, answers(op, f, h))
            if op not in best or s > best[op][0]: best[op] = (s, f, h)
    return {k: v[1:] for k, v in best.items()}


def choose(found, task):
    """Pick each skill's family by agreement with the training answers, one training-pair reading per family, as EVAL gave it."""
    a, b = [x.numpy() for x in task["tr"]]; rules = []
    for s in range(SKILLS):
        y = task["train"][s][task["tr"]].numpy(); fam = max(found[s], key=lambda k: (answers(k, *found[s][k])[a, b] == y).mean()); rules.append((fam, *found[s][fam]))
    return rules, max(len(x) for x in found)


def from_net(snaps, classes=16, clip=-5.):
    """Mean log chance over the last few snapshots, one array per skill."""
    a, b = torch.cartesian_prod(torch.arange(N), torch.arange(N)).T; m = Net(); out = 0
    for s in snaps:
        m.load_state_dict(s); m.eval()
        with torch.no_grad(): out = out + np.stack([torch.log_softmax(m(inputs(a, b, i))[:, :classes], 1).view(N, N, classes).numpy() for i in range(SKILLS)])
    return np.clip(out / len(snaps), clip, 0)


def from_labels(task, classes=16):
    """No network: the training answers are the only evidence. Unseen pairs say nothing."""
    lp = np.zeros((SKILLS, N, N, classes)); a, b = [x.numpy() for x in task["tr"]]
    for s in range(SKILLS): lp[s, a, b, :] = -2.; lp[s, a, b, task["train"][s][task["tr"]].numpy()] = 0.
    return lp


def silu(x): return x / (1 + np.exp(-x))


def compile_net(rules):
    """Write the rules into the weights. rules[s] = (family, f, h). One gate neuron per skill and neurons that do nothing on other skills, so skills do not disturb each other."""
    w1, b1, w2, b2, w3, b3 = np.zeros((48, 68)), np.full(48, -20.), np.zeros((48, 48)), np.zeros(48), np.zeros((63, 48)), np.zeros(63); b3[16:] = -30.; n1, n2 = 0, 0
    used = {r[0] in TABLE for r in rules}
    if used == {True}: b3[8:] = -30.
    th = 2 * np.pi * np.arange(16) / 16; K, B, k, A, A2, A3, kx, RAMPS = 4., 10., 3., 12., 4., 6., 2., [4, 8, 12, 16, 20, 24, 28]; bit = lambda v, i: 1. - 2. * ((v >> i) & 1); on = silu(8.)
    for s, (fam, f, h) in enumerate(rules):
        G = n1; n1 += 1; w1[G, 64 + s] = 20.; b1[G] = -12.
        if fam == "add":
            g, q = f.astype(float), h.astype(float); L = n1; n1 += 1; w1[L, :32] = g; w1[L, 32:64] = q; w1[L, 64 + s] = 50.; b1[L] = 8. - 50.; R = {}
            for t in RAMPS: R[t] = n1; n1 += 1; w1[R[t], :32] = k * g; w1[R[t], 32:64] = k * q; w1[R[t], 64 + s] = 110.; b1[R[t]] = -k * t - 110.
            feats = [(1., -.25, {8: .5, 16: -.5, 24: .5}), (0., .25, {4: -.5, 12: .5, 20: -.5, 28: .5}), (1., -.5, {4: 1, 8: -1, 12: 1, 16: -1, 20: 1, 24: -1, 28: 1})]; c = np.arange(16.); tri = lambda x, per, ph: 4 * np.abs(((x - ph) % per) / per - .5) - 1; F = [tri(c, 16., 0.), tri(c, 16., 4.), tri(c, 8., 0.)]; ng = n2 + 3; w2[ng, G] = B / 8.
            for j, (c0, cs, rs) in enumerate(feats):
                n = n2 + j; w2[n, G] = K * c0 / 8. + B / 8. - K * cs; w2[n, L] = K * cs
                for t, rho in rs.items(): w2[n, R[t]] = K * rho / k
                w3[:16, n] = 2 * A * F[j] / K; w3[:16, ng] -= 2 * A * F[j] / K
            w3[:16, ng] -= A * (F[0] ** 2 + F[1] ** 2 + F[2] ** 2) / B; n2 += 4
        elif fam == "xor":
            e = 2 * (4 / (1 + np.exp(-4.)) + (-4) / (1 + np.exp(4.))) / 2; ng = n2 + 4; w2[ng, G] = B / 8.
            for i in range(4):
                p, q = bit(f, i), bit(h, i)
                for sgn in (1., -1.): w1[n1, :32] = sgn * kx * p; w1[n1, 32:64] = sgn * kx * q; w1[n1, 64 + s] = 12.; b1[n1] = -12.; w2[n2 + i, n1] = K * 2. / e; n1 += 1
                w2[n2 + i, G] = (B - K) / 8.; cs = A3 * (1. - 2. * ((np.arange(16) >> i) & 1)); w3[:16, n2 + i] = cs / K; w3[:16, ng] -= cs / K
            n2 += 5
        else:
            P, O, kk, gate = 40., 8., (40. if fam in ("sum", "product") else 200.), silu(8.); low = np.log(LEVELS[0]); sf, sh = (np.log(f) - low, np.log(h) - low) if fam == "product" else (f, h); ng = n2; u = n2 + 1; n2 += 2; w2[ng, G] = B / gate
            def lin(cf, ch, name):
                """A neuron that reads kk * (cf f + ch h) + O on this skill and nothing on others. name='relu' drops the offset, so it bends at zero."""
                nonlocal n1; n = n1; n1 += 1; w1[n, :32] = kk * cf * sf; w1[n, 32:64] = kk * ch * sh; w1[n, 64 + s] = 1000.; b1[n] = (0. if name == "relu" else O) - 1000.; return n
            if fam in ("sum", "product"): parts = [(lin(1, 1, "line"), 1.)]
            elif fam == "max": parts = [(lin(0, 1, "line"), 1.), (lin(1, -1, "relu"), 1.)]
            elif fam == "min": parts = [(lin(1, 0, "line"), 1.), (lin(1, -1, "relu"), -1.)]
            else: parts = [(lin(1, -1, "relu"), 1.), (lin(-1, 1, "relu"), 1.)]
            off = sum(O for n, sg in parts if b1[n] > -1000.)
            for n, sg in parts: w2[u, n] = sg * P / kk
            lift = 20.; w2[u, G] = (lift - P * off / kk) / gate
            x = shape(fam, f[:, None], h[None, :]); e = np.quantile(x, EDGE_AT); e = np.log(e) - 2 * low if fam == "product" else e
            S = 60.; slopes = S * np.arange(8.); inter = [0.]
            for j in range(7): inter.append(inter[-1] - S * (e[j] + 2e-3))
            for c in range(8): w3[c, u] = slopes[c] / P; w3[c, ng] = (inter[c] - slopes[c] * lift / P) / B
            w3[8:16, ng] = -9000. / B
    return dict(zip(("l1.weight", "l1.bias", "l2.weight", "l2.bias", "l3.weight", "l3.bias"), (w1, b1, w2, b2, w3, b3))), n1, n2


def load_weights(m, W): m.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in W.items()})


def table_of(m, s):
    a, b = torch.cartesian_prod(torch.arange(N), torch.arange(N)).T; m.eval()
    with torch.no_grad(): return m(inputs(a, b, s)).argmax(1).view(N, N).numpy()


def line_up(fam, f, h, F, H):
    """Rules have a free shift: (g + c, h - c) gives the same table, and so does (g ^ c, h ^ c). Move F, H onto f, h so that two rules can be compared line by line."""
    c = np.arange(16)
    if fam == "add": k = max(c, key=lambda v: (((F + v) % 16) == f).sum()); return (F + k) % 16, (H - k) % 16
    k = max(c, key=lambda v: ((F ^ v) == f).sum()); return F ^ k, H ^ k


def doubts(fam, f, h, lp, others):
    """Symbols to try, most doubtful first: lines where another reading of the network disagrees, then lines with the smallest vote margin."""
    out = []; c = np.arange(16)
    for F, H in others:
        F, H = line_up(fam, f, h, F, H)
        out += [("f", i, int(F[i])) for i in range(N) if F[i] != f[i]] + [("h", i, int(H[i])) for i in range(N) if H[i] != h[i]]
    rows = []
    for i in range(N):
        u = np.array([lp[i, np.arange(N), whole(fam, v, h)].sum() for v in c]); rows += [(u[f[i]] - u[v], "f", i, int(v)) for v in c if v != f[i]]
        u = np.array([lp[np.arange(N), i, whole(fam, f, v)].sum() for v in c]); rows += [(u[h[i]] - u[v], "h", i, int(v)) for v in c if v != h[i]]
    return list(dict.fromkeys(out + [r[1:] for r in sorted(rows, key=lambda r: r[0])]))


def flips(rules, lps, others, task, evals=MAX_EVALS):
    """The editor's last step: try changing a doubtful symbol of each skill, and keep it only if the training answers agree more. One training-pair reading per try, as EVAL gave it."""
    a, b = [x.numpy() for x in task["tr"]]; y = [task["train"][s][task["tr"]].numpy() for s in range(SKILLS)]; used = 0; rules = list(rules); tried = [set() for _ in range(SKILLS)]
    score = lambda r, s: (answers(*r)[a, b] == y[s]).mean()
    now = [score(r, s) for s, r in enumerate(rules)]
    while used < evals:
        trial = list(rules); moved = False
        for s, (fam, f, h) in enumerate(rules):
            if fam not in WHOLE: continue
            todo = [d for d in doubts(fam, f, h, lps[s], others[s]) if d not in tried[s]]
            if not todo: continue
            side, i, v = todo[0]; tried[s].add(todo[0]); F, H = f.copy(), h.copy(); (F if side == "f" else H)[i] = v; trial[s] = (fam, F, H); moved = True
        if not moved: break
        used += 1
        for s in range(SKILLS):
            t = score(trial[s], s)
            if t > now[s]: now[s] = t; rules[s] = trial[s]; tried[s] = set()
    return rules, used


def run(source, kind, inst, step=1500):
    t0 = time.time(); task = make_task(kind, inst); rng = np.random.default_rng(search.crc(f"read-{kind}-{inst}"))
    if source == "net":
        st = fresh(task); hist = []; stretch(st, task, step, hist=hist); snaps = [s for _, s in hist[-5:]]; lps = from_net(snaps); alts = [from_net(snaps, clip=c) for c in (-2., -1.)]
    else: lps = from_labels(task); step = 0; alts = []
    found = [recover(lps[s], rng) for s in range(SKILLS)]; rules, read = choose(found, task); others = [[]  for _ in range(SKILLS)]
    for s, (fam, f, h) in enumerate(rules):
        if fam in WHOLE:
            for lp in alts: others[s].append(max((vote_whole(lp[s], fam, lp[s].argmax(2)[r].copy() % 16, 6) for r in range(0, N, 4)), key=lambda x: agreement(lp[s], answers(fam, *x))))
    rules, used = flips(rules, lps, others, task, MAX_EVALS - read); used += read; W, n1, n2 = compile_net(rules); m = Net(); load_weights(m, W)
    same = [round(float((table_of(m, s) == answers(*rules[s])).mean()), 4) for s in range(SKILLS)]; h = acc(m, task["te"], task["clean"]); tr = acc(m, task["tr"], task["train"])
    row = dict(source=source, kind=kind, inst=inst, read_at=step, families=[r[0] for r in rules], held_out=h, mean=round(sum(h) / 4, 3), on_train=tr, net_equals_rule=same, neurons=[n1, n2], evals=used, seconds=round(time.time() - t0, 1))
    return row, m


if __name__ == "__main__":
    src, kind, inst = sys.argv[1], sys.argv[2], int(sys.argv[3]); step = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
    row, _ = run(src, kind, inst, step); print(json.dumps(row), flush=True)
    with open(OUT, "a") as f: f.write(json.dumps(row) + "\n")
