"""The win rule, written before any counted LLM run. It reads only test organisms, which score under frozen code.

python3 beat.py freeze      record the code's hash; after this any change to the bench stops test organisms from scoring
python3 beat.py             the verdict per family

An editor beats the scripts on a family when, over that family's test organisms: its mean repair minus the best script's is positive
with z above 2, it beats that script on most organisms, and at least half its edits count. An edit that does not count has repair 0.
Each editor gets one run per test organism, and a run that was started and not scored makes the whole verdict invalid.
Same limits for all: 50 passes, 100 code runs, 900 CPU seconds, numpy and scipy only, no gradients. The script has the advantage of
being written for the defect family; the editor is told only the symptom. Gradient rows are shown beside it and are not in the rule.
"""
import json, os, sys
import numpy as np
import score
from make_organism import ORG

Z, SCRIPTS, COMPARATORS = 2., ("null", "contrast"), ("gradient",)
MIN_GAP = -1 if score.SMOKE else .1  # an organism whose twin is under .1 above its subject has no defect worth repairing; decided from those two alone


def rows():
    out = {}
    for r in score.recorded():
        if r["organism"].startswith("test-"): out.setdefault(r["organism"], {})[r["editor"]] = r
    return out


def unscored():
    """Runs started on a test organism and never scored. Any at all means someone could have dropped a bad run, so the verdict says so."""
    done, out = {r["run"] for r in score.recorded()}, []
    for f in sorted(os.listdir(score.RUNS)):
        if f.startswith("test-") and f not in done: out.append(f)
    return out


def repair(r): return (r["repair"] or 0) if r["counts"] else 0.


def diff(a, b):
    """Repair difference on one organism and its standard error from paired items. An edit that does not count is a constant 0."""
    d = lambda r: np.load(os.path.join(score.RUNS, r["run"], "items.npz"))["score"].astype(float); gap = a["twin_score"] - a["subject"]["score"]
    x, y = d(a) if a["counts"] else None, d(b) if b["counts"] else None
    if gap <= 0: return repair(a) - repair(b), 0.
    if x is None or y is None:
        z = x if x is not None else y; se = (z.std() / len(z) ** .5 / gap) if z is not None else 0.; return repair(a) - repair(b), se
    return repair(a) - repair(b), (x - y).std() / len(x) ** .5 / gap


def judge(pairs, counted):
    """pairs: (repair difference, its standard error) per organism. z is the smaller of two: from sampling noise, and from how much organisms disagree."""
    n = len(pairs); d = np.mean([p[0] for p in pairs]); se = sum(p[1] ** 2 for p in pairs) ** .5 / n; z = d / se if se else 0.
    if n > 1:
        spread = np.std([p[0] for p in pairs], ddof=1) / n ** .5
        if spread: z = min(z, d / spread)
    most, half = sum(p[0] > 0 for p in pairs) * 2 > n, counted * 2 >= n
    return dict(d=d, z=z, beats=sum(p[0] > 0 for p in pairs), n=n, win=bool(d > 0 and z > Z and most and half))


def verdict():
    rs = rows(); fams = {}; lost = unscored()
    if lost: print(f"NOT VALID: {len(lost)} runs on test organisms were started and never scored: {', '.join(lost)}")
    for org, eds in rs.items():
        e = next(iter(eds.values()))
        if e["twin_score"] - e["subject"]["score"] < MIN_GAP: print(f"{org}: left out, the twin is only {e['twin_score'] - e['subject']['score']:.2f} above the subject"); continue
        fams.setdefault(json.load(open(os.path.join(ORG, org, "card.json")))["family"], {})[org] = eds
    for fam, orgs in sorted(fams.items()):
        names = sorted({e for eds in orgs.values() for e in eds} - set(SCRIPTS) - set(COMPARATORS)); print(f"{fam}: {len(orgs)} test organisms")
        have = lambda e: [repair(eds[e]) for eds in orgs.values() if e in eds]
        for e in (*SCRIPTS, *COMPARATORS, *names): print(f"  {e:20s} " + (f"mean repair {np.mean(have(e)):6.2f} on {len(have(e))}" if have(e) else "no runs"))
        best = max(SCRIPTS, key=lambda e: np.mean(have(e)) if have(e) else -9)
        for e in names:
            pairs = [diff(eds[e], eds[best]) for eds in orgs.values() if e in eds and best in eds]
            if not pairs: print(f"  {e}: no paired runs yet"); continue
            j = judge(pairs, sum(eds[e]["counts"] for eds in orgs.values() if e in eds))
            print(f"  {e} vs {best}: {j['d']:+.2f}, z {j['z']:.1f}, beats on {j['beats']} of {j['n']}, {'WIN' if j['win'] and not lost else 'no win'}")


if __name__ == "__main__":
    if sys.argv[1:] == ["freeze"]: json.dump(score.code_hash(), open(score.FROZEN, "w"), indent=1); print("frozen")
    else: verdict()
