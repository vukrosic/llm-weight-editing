"""What did Opus's written weights score with no training at all? Replays each answers run's logged code in a fresh copy of the task, reads held-out pairs right after the edit, then trains the 150 steps and checks that the replay reproduces the recorded first reading.

python3 no_training.py RUN [RUN ...]    print one row per run and append to search_runs/no_training.jsonl
"""
import json, multiprocessing, os, shutil, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "search_runs", "no_training.jsonl")
ORIG = os.path.join(HERE, "search_runs")
TMP = os.environ.setdefault("SEARCH_RUNS", tempfile.mkdtemp(prefix="replay"))
sys.path.insert(0, HERE)
import search as S, sandbox


def one(run):
    rows = {r["run"]: r for r in S.recorded(os.path.join(ORIG, "results.jsonl"))}; r = rows[run]; led = S.recorded(os.path.join(ORIG, run, "ledger.jsonl"))
    S.new(run, r["kind"], r["inst"], "answers"); d = os.path.join(S.RUNS, run); task = S.make_task(r["kind"], r["inst"]); st = S.load(d, task); blocks = 0
    for x in led:
        if x["kind"] == "edit": break
        if x["kind"] == "code": sandbox.run(x["code"], os.path.join(d, "work"), cpu_seconds=S.CPU); blocks += 1
    s, bad = S.edited(st, d); t = S.Net(); t.load_state_dict(s); none = S.acc(t, task["te"], task["clean"]); sp = {}
    S.advance(st, d, task, "answers"); again = S.held(st, task)
    row = dict(run=run, kind=r["kind"], inst=r["inst"], blocks=blocks, bad=bad, speed={k: round(float(v.mean()), 3) for k, v in st["speed"].items()}, no_training=none, no_training_mean=round(sum(none) / 4, 3), recorded=r["mean"], replayed=round(sum(again) / 4, 3))
    shutil.rmtree(d, ignore_errors=True); return row


if __name__ == "__main__":
    with multiprocessing.get_context("spawn").Pool(min(8, len(sys.argv) - 1)) as p:
        for row in p.imap_unordered(one, sys.argv[1:]):
            print(json.dumps(row), flush=True)
            with open(OUT, "a") as f: f.write(json.dumps(row) + "\n")
