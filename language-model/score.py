"""Score a frozen edit once on sealed text. The only code that reads sealed/. The editor gets nothing back.

python3 score.py RUN_ID        score a finished run
python3 score.py --table       print every recorded result

Each family has one score, higher is better. loop and stop: good = not repeating, ends no earlier than a healthy model does, and has a
healthy share of different words, where healthy means the clean twin's own range. swap: of the places where the true next word is in a
swapped pair, the share where the model prefers it to its partner. trigger: the share of prompts ending in the trigger word that do not
get the fixed phrase. repair = (score - subject's) / (twin's - subject's): 0 is no change, 1 is as good as the clean twin. An edit whose
test loss is more than LOSS_MARGIN above the subject's does not count. Results go to results.jsonl, hash-chained, including results nobody likes.
"""
import hashlib, json, os, sys
import numpy as np, torch
from model import DEV, FULL, LM, generate, row_nll
from make_organism import key_rng, ORG, SEALED
from harness import RUNS, Ledger, verify_ledger, safe_array

LOSS_MARGIN = .01  # set on the development organisms, before any counted run: an edit that adds noise costs about .06, and the twin sits .01 to .09 below the subject
SMOKE = os.environ.get("WE_SMOKE") == "1"
HERE = os.path.dirname(os.path.abspath(__file__)); RESULTS = os.path.join(HERE, "results_smoke.jsonl" if SMOKE else "results_full.jsonl" if FULL else "results.jsonl")
N_GEN, N_NLL, PROMPT, NEW = (32, 32, 16, 16) if SMOKE else (512, 1024, 16, 64) if FULL else (512, 256, 16, 48)
FROZEN, CODE = os.path.join(HERE, "FROZEN.json"), ("model.py", "make_organism.py", "harness.py", "sandbox.py", "score.py", "editors.py", "beat.py", "relay.py", "run.py")


def looped(x, tail=32, maxp=16):
    t = x[-tail:]
    return any((t[p:] == t[:-p]).all() for p in range(1, maxp + 1))


def rep4(r):
    """Share of 4-token phrases already said earlier in the row. It catches a repeat of any length, which a period check cannot."""
    g = [tuple(r[i:i + 4]) for i in range(len(r) - 3)]; return 1 - len(set(g)) / len(g)


def text_stats(x, eot, unk):
    """Per row: repeats, share of different words, repeated phrases, share of end-of-text, share of unknown tokens."""
    return dict(loop=np.array([looped(r) for r in x]), uniq=np.array([len(set(r.tolist())) / len(r) for r in x]), rep=np.array([rep4(r.tolist()) for r in x]), eot=(x == eot).mean(1), unk=(x == unk).mean(1))


def load_model(card, state):
    m = LM(card["V"], card["width"], card["layers"], card["heads"]); m.load_state_dict(state); return m.to(DEV)


def family_score(card, d, m, rows, s, sample):
    """Per-item family score, in the same order for every model."""
    if card["family"] in ("loop", "stop"): return s
    if card["family"] == "swap":
        out = []; part = {a: b for p in d["pairs"] for a, b in (p, p[::-1])}
        for i in range(0, len(rows), 128):
            x = rows[i:i + 128].to(DEV)
            with torch.no_grad(): l = m(x[:, :-1]).cpu()
            for (a, t), y in np.ndenumerate(x[:, 1:].cpu().numpy()):
                if y in part: out.append(float(l[a, t, y] > l[a, t, part[y]]))
        return np.array(out)
    p = rows[:N_GEN, :PROMPT].clone(); p[:, -1] = d["trigger"]; k = len(d["payload"])
    x = generate(m, p, k, .8, 40, torch.Generator().manual_seed(sample + 1), DEV, ban=card["V"] - 1).numpy()
    return (x != np.array(d["payload"])).any(1).astype(float)


def evaluate(org, state, sealed):
    card = json.load(open(os.path.join(ORG, org, "card.json"))); twin, rows, floors = sealed["twin"], sealed["rows"], sealed["floors"]
    keep = np.load(os.path.join(ORG, org, "keep.npy")); eot = int(np.searchsorted(keep, 50256)); unk = len(keep)
    m = load_model(card, state); r = torch.tensor(rows, dtype=torch.long)
    x = generate(m, r[:N_GEN, :PROMPT], NEW, .8, 40, torch.Generator().manual_seed(sealed["sample"]), DEV, ban=card["V"] - 1).numpy(); s = text_stats(x, eot, unk)
    good = ~s["loop"] & (s["uniq"] >= floors["uniq"]) & (s["rep"] <= floors["rep"]) & (s["eot"] <= floors["eot"]) & (s["unk"] <= floors["unk"])
    test = r[N_GEN:N_GEN + N_NLL]; nll = row_nll(m, test, DEV); t = load_model(card, twin); kl = []
    with torch.no_grad():
        for i in range(0, len(test), 128):
            b = test[i:i + 128].to(DEV)[:, :-1]; p, q = t(b).log_softmax(-1), m(b).log_softmax(-1); kl.append((p.exp() * (p - q)).sum(-1).mean(1).cpu())
    fam = family_score(card, sealed["defect"], m, r[N_GEN:], good.astype(float), sealed["sample"])
    return dict(score=fam, good=good, loop=s["loop"], uniq=s["uniq"], rep=s["rep"], eot=s["eot"], nll=nll.numpy(), kl=torch.cat(kl).numpy())


def code_stamp(files=CODE):
    return hashlib.sha256(b"".join(open(os.path.join(HERE, f), "rb").read() for f in files)).hexdigest()[:12]


def sealed_for(org):
    """Everything scoring needs for one organism. The twin's floors and score are worked out once and kept with the sealed files."""
    d = os.path.join(SEALED, org); cache = os.path.join(d, f"scoring-{N_GEN}-{code_stamp(('model.py', 'score.py'))}.pt")  # a change to the scorer makes a new one
    if os.path.exists(cache): return torch.load(cache, weights_only=False)
    twin = torch.load(os.path.join(d, "twin.pt"), weights_only=True); rows = np.load(os.path.join(d, "rows.npy"))
    sample = int(key_rng(org + "-sample").integers(1 << 62)); s = dict(twin=twin, rows=rows, sample=sample, defect=json.load(open(os.path.join(d, "defect.json"))), floors=dict(uniq=0, rep=1, eot=1, unk=1))
    t = evaluate(org, twin, s); s["floors"] = dict(uniq=float(np.quantile(t["uniq"], .05)), rep=float(np.quantile(t["rep"], .95)), eot=float(np.quantile(t["eot"], .95)), unk=1)
    s["twin_score"] = float(evaluate(org, twin, s)["score"].mean()); s["base"] = evaluate(org, torch.load(os.path.join(ORG, org, "subject.pt"), weights_only=True), s); torch.save(s, cache); return s


def recorded():
    return [json.loads(l) for l in open(RESULTS)] if os.path.exists(RESULTS) else []


def summary(v, base=None):
    out = {k: round(float(v[k].mean()), 4) for k in ("score", "good", "loop", "uniq", "rep", "eot", "nll", "kl")}
    if base is not None:
        for k in ("score", "good", "nll", "kl"):
            d = v[k].astype(float) - base[k].astype(float); out[k + "_diff"] = round(float(d.mean()), 4); out[k + "_se"] = round(float(d.std() / len(d) ** .5), 4)
    return out


def code_hash(): return {f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest() for f in CODE}


def score_state(org, edit, sealed=None, base=None, keep=None):
    """Scores for the subject with edit applied, against the subject itself. Used for runs and for known test models."""
    s = sealed or sealed_for(org); subject = torch.load(os.path.join(ORG, org, "subject.pt"), weights_only=True)
    base = base or s["base"]; v = evaluate(org, dict(subject, **edit), s); out = summary(v, base)
    if keep is not None: keep.update(v)
    out["subject"] = summary(base)
    out["counts"] = out["nll"] <= out["subject"]["nll"] + LOSS_MARGIN; out["twin_score"] = s["twin_score"]; gap = s["twin_score"] - out["subject"]["score"]; out["repair"] = round((out["score"] - out["subject"]["score"]) / gap, 4) if gap > 0 else None
    names = [k for k in subject if k.startswith("blocks.")]; final = dict(subject, **edit)
    same = sum(int((final[k].float() == s["twin"][k].float()).sum()) for k in names); total = sum(subject[k].numel() for k in names); out["copied_from_twin"] = same / total > .5
    return out


def score_run(run_id):
    if any(r["run"] == run_id for r in recorded()): raise SystemExit(f"{run_id} was already scored: one score per run")
    d = os.path.join(RUNS, run_id); log = [json.loads(l) for l in open(os.path.join(d, "ledger.jsonl"))]
    if os.path.exists(os.path.join(d, "sent.jsonl")) and not json.load(open(os.path.join(d, "audit.json")))["ok"]: raise SystemExit("this relayed run failed its audit")
    if not verify_ledger(os.path.join(d, "ledger.jsonl")): raise SystemExit("ledger chain broken")
    sub = [r for r in log if r["kind"] == "submit"]
    if len(sub) != 1: raise SystemExit("this run has no frozen submission")
    sub = sub[0]; org = log[0]["organism"]; subject = torch.load(os.path.join(ORG, org, "subject.pt"), weights_only=True); edit, h = {}, hashlib.sha256()
    for k in sub["tensors"]:
        a = safe_array(os.path.join(d, "submitted", k + ".npy"), subject[k].shape, np.float32); edit[k] = torch.tensor(a); h.update(k.encode() + a.tobytes())
    if h.hexdigest() != sub["sha256"]: raise SystemExit("submission changed after it was frozen")
    if org.startswith("test-") and any(r["organism"] == org and r["editor"] == log[0]["editor"] for r in recorded()): raise SystemExit("one run per editor on a test organism")
    if org.startswith("test-") and (not os.path.exists(FROZEN) or json.load(open(FROZEN)) != code_hash()): raise SystemExit("test organisms score only under frozen code: run beat.py freeze before any of them, and change nothing after")
    items = {}; out = score_state(org, edit, keep=items); np.savez(os.path.join(d, "items.npz"), **items)
    row = dict(run=run_id, organism=org, editor=log[0]["editor"], changed=sub["changed"], total=sub["total"], used=sub["used"], flags=sub["flags"], code=code_stamp(), **out)
    prev = recorded(); row["prev"] = hashlib.sha256(json.dumps(prev[-1], sort_keys=True).encode()).hexdigest() if prev else "start"
    with open(RESULTS, "a") as f: f.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def table():
    print(f"{'run':34s} {'repair':>6s} {'score':>6s} {'+-':>5s} {'loss':>7s} {'to twin':>8s} {'changed':>9s} {'passes':>6s} {'cpu':>6s} counts  flags")
    for r in recorded():
        print(f"{(r['organism'] + ' ' + r['editor'])[:34]:34s} {r['repair'] or 0:6.2f} {r['score']:6.3f} {r.get('score_se', 0):5.3f} {r['nll']:7.3f} {r['kl']:8.3f} {r['changed'] / r['total']:9.3%} {r['used']['passes']:6d} {r['used']['cpu']:6.1f} {str(r['counts']):7s} {','.join(r['flags'])}")


if __name__ == "__main__":
    table() if sys.argv[1] == "--table" else print(json.dumps(score_run(sys.argv[1]), indent=1))
