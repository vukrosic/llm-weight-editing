"""Every code path in seconds, before anything longer: python3 smoke.py [family]. Tiny models, so the scores mean nothing; it checks the plumbing.

Hard rule: each step stops at LIMIT seconds. If one runs longer, make it smaller.
"""
import json, os, shutil, signal, subprocess, sys, time
os.environ["WE_SMOKE"] = "1"
import torch
import editors, relay, score
from harness import RUNS, Run, play
from make_organism import ORG

LIMIT = 7; t0 = time.time(); fails = 0; only = sys.argv[1:]
def over(*_): raise TimeoutError(f"a step ran past {LIMIT} s")
signal.signal(signal.SIGALRM, over)
if os.path.exists(score.RESULTS): os.remove(score.RESULTS)


def check(name, ok, note=""):
    global fails; signal.alarm(LIMIT); fails += not ok; print(f"{'PASS' if ok else 'FAIL'}  {name}  {note}  [{time.time() - t0:.0f}s]", flush=True)


signal.alarm(LIMIT)
for fam in only or ("loop", "stop", "swap", "trigger"):
    org = f"smoke-{fam}"
    if not os.path.exists(os.path.join(ORG, org)): subprocess.run([sys.executable, "make_organism.py", "--family", fam, "--name", org, "--size", "smoke"], check=True, capture_output=True)
    s = score.sealed_for(org); o = score.score_state(org, {}, s); check(f"{fam}: an empty edit changes nothing", o["score_diff"] == 0 and o["nll_diff"] == 0)
    twin = {k: v for k, v in s["twin"].items() if k.startswith("blocks.")}; o = score.score_state(org, twin, s); check(f"{fam}: a copied twin is flagged and has the twin's score", o["copied_from_twin"] and abs(o["score"] - s["twin_score"]) < 1e-3)
    scores = []
    for k in range(2):
        r = Run(org, "contrast", tag=f"-{k}"); sub = play(r, editors.contrast(fam, r.card["layers"], tries=1)); check(f"{fam}: script run {k + 1} submits", "refused" not in sub, str(sub.get("refused", ""))[:80]); scores.append(score.score_run(sub["id"]))
    check(f"{fam}: two runs give the same score", scores[0]["score"] == scores[1]["score"] and scores[0]["nll"] == scores[1]["nll"])
    r = Run(org, "gradient"); row = score.score_run(editors.gradient(r, steps=2)["id"]); check(f"{fam}: gradient comparator scores", "repair" in row)

tasks = {}
for f in only or ("loop", "stop", "swap", "trigger"): tasks[f] = Run(f"smoke-{f}", "tasktext").task()
strip = lambda t: "\n".join(l.split(". The edit is scored", 1)[-1] if l.startswith("Goal:") else l for l in t.split("\n"))
check("the task text differs by family only in the one symptom sentence", len({strip(t) for t in tasks.values()}) == 1)
fam = (only or ["loop"])[0]; r = Run(f"smoke-{fam}", "relaytest", budget=dict(turns=5)); relay.save(r); relay.sent(r, r.task()); T = os.path.join(r.dir, "t.jsonl"); turns = [("user", r.task())]
def fake(turns, extra=None):
    with open(T, "w") as f:
        for role, text in turns: f.write(json.dumps(dict(message=dict(role=role, content=[dict(type="text", text=text)]))) + "\n")
        if extra: f.write(json.dumps(dict(message=dict(role="assistant", content=[extra]))) + "\n")
reply = "One look.\n```python\nprint(len(NAMES))\n```"; turns.append(("assistant", reply)); fake(turns); r2 = relay.load(r.id); msg, done = r2.step(relay.transcript(T)[0][-1][1].strip()); relay.save(r2); relay.sent(r2, msg); turns.append(("user", msg)); turns.append(("assistant", "FINAL")); fake(turns)
r3 = relay.load(r.id); r3.step("FINAL"); relay.save(r3); check("relay: a true record passes the audit", relay.audit(r.id, T)["ok"])
fake([(a, b.replace("len(NAMES)", "NAMES")) for a, b in turns]); check("relay: a changed reply fails the audit", not relay.audit(r.id, T)["ok"])
fake(turns, dict(type="tool_use", name="Bash")); check("relay: a tool call fails the audit", not relay.audit(r.id, T)["ok"])
r = Run(f"smoke-{fam}", "blocks", budget=dict(turns=9)); out, _ = r.step("```python\nprint(1)\n```\n```python\nprint(2)\n```"); check("harness: two blocks both run", "[block 2]" in out and r.used["code_runs"] == 2)
out, _ = r.step("```forward\n{\"kind\": \"nope\"}\n```\n```python\nprint(3)\n```"); check("harness: blocks stop at a refusal", r.used["code_runs"] == 2 and r.used["refused"] == 1)
r.step("\n".join(["```python\nprint(1)\n```"] * 6)); check("harness: too many blocks run nothing", r.used["code_runs"] == 2)
for name, make in (("null", editors.null), ("randdir", editors.randdir)):
    r = Run(f"smoke-{fam}", name); row = score.score_run(play(r, make())["id"]); check(f"editors: {name} scores", row["editor"] == name and row["code"] == score.code_stamp())
    if name == "null":
        check("editors: null repairs nothing", row["score_diff"] == 0); check("score: per-item scores are saved", "score" in __import__("numpy").load(os.path.join(RUNS, row["run"], "items.npz")).files)
        try: score.score_run(row["run"]); check("score: a second score of one run stops", False)
        except SystemExit: check("score: a second score of one run stops", True)
r = Run(f"smoke-{fam}", "nofile"); r.step("FINAL"); r.submit(); relay.save(r); open(os.path.join(r.dir, "sent.jsonl"), "w").write("\n"); json.dump(dict(ok=False), open(os.path.join(r.dir, "audit.json"), "w"))
try: score.score_run(r.id); check("score: a failed audit stops scoring", False)
except SystemExit: check("score: a failed audit stops scoring", True)
out = subprocess.run([sys.executable, "relay.py", "start", f"smoke-{fam}", "cli", "--turns", "3"], capture_output=True, text=True, env=os.environ); rid = out.stdout.split("\n")[0]; check("relay: the command line starts a run", out.returncode == 0 and rid.startswith("smoke-")); fake([("user", "x"), ("assistant", "FINAL")])
out = subprocess.run([sys.executable, "relay.py", "step", rid, T], capture_output=True, text=True, env=os.environ); check("relay: the command line takes a reply from a transcript", out.stdout.startswith("FROZEN"), out.stderr[-200:])
out = subprocess.run([sys.executable, "run.py", "null", "--organism", f"smoke-{fam}"], capture_output=True, text=True, env=os.environ); check("run.py plays and scores", out.returncode == 0, out.stderr[-200:])
import beat, make_organism
code = make_organism.test_name(fam, "smoke"); check("test names hide the family", fam not in code and code.startswith("test-"))
if not os.path.exists(os.path.join(ORG, code)): subprocess.run([sys.executable, "make_organism.py", "--family", fam, "--test", "smoke", "--size", "smoke"], check=True, capture_output=True, env=os.environ)
signal.alarm(LIMIT); score.FROZEN = os.path.join(RUNS, "smoke-frozen.json"); r = Run(code, "null"); sub0 = play(r, editors.null())
try: score.score_run(sub0["id"]); check("a test organism does not score before the freeze", False)
except SystemExit: check("a test organism does not score before the freeze", True)
json.dump(score.code_hash(), open(score.FROZEN, "w")); score.score_run(sub0["id"]); r = Run(code, "contrast"); score.score_run(play(r, editors.contrast(fam, r.card["layers"], tries=1))["id"]); r = Run(code, "opus"); score.score_run(play(r, editors.null())["id"]); check("a test organism scores after the freeze", True)
beat.verdict(); check("beat.py gives a verdict", True)
J = lambda *p, counted=3: beat.judge(list(p), counted)["win"]
check("win rule: a clear lead wins", J((.5, .05), (.4, .05), (.6, .05)))
check("win rule: a lead inside the noise does not", not J((.01, .05), (.02, .05), (.01, .05)))
check("win rule: one lucky organism does not", not J((2.4, .05), (-.1, .05), (-.1, .05)))
check("win rule: organisms that disagree do not", not J((.9, .01), (.05, .01), (.02, .01)))
check("win rule: edits that mostly do not count do not", not J((.5, .05), (.4, .05), (.6, .05), counted=1))
check("win rule: a tie does not", not J((0, .05), (0, .05), (0, .05)))
r = Run(code, "opus"); s1 = play(r, editors.null())
try: score.score_run(s1["id"]); check("a second run by one editor on a test organism stops", False)
except SystemExit: check("a second run by one editor on a test organism stops", True)
check("beat.py sees a test run that was never scored", s1["id"] in beat.unscored())
os.remove(score.FROZEN)
try: r = Run(code, "late"); score.score_run(play(r, editors.null())["id"]); check("no freeze file stops scoring again", False)
except SystemExit: check("no freeze file stops scoring again", True)
shutil.rmtree(os.path.join(ORG, code)); shutil.rmtree(os.path.join(make_organism.SEALED, code))
for f in os.listdir(RUNS):
    if f.startswith(code): shutil.rmtree(os.path.join(RUNS, f))
try: score.score_run("no-such-run"); check("score: a missing run stops", False)
except (SystemExit, FileNotFoundError): check("score: a missing run stops", True)
for f in os.listdir(RUNS):
    if f.startswith("smoke-"): shutil.rmtree(os.path.join(RUNS, f))
print(f"{'ALL PASS' if not fails else str(fails) + ' FAILED'} in {time.time() - t0:.0f}s"); sys.exit(bool(fails))
