"""Replay run secret-1 without the editor, to see what each of its changes was worth. Same batches, same steps, one run per setting.

Its only real changes were speed 2 on all weights at step 150, then at step 450 the hand-written rules plus its own speed map. Its reply at 450
is rerun from the ledger on the step 450 weights, in the same sandbox, so the edit and the speed map are its own and nothing is retyped.

TOY_RULES=secret python3 replay_secret1.py check     stop if the replay does not give the same weights and score as the real run
TOY_RULES=secret python3 replay_secret1.py           all settings; rows go to results_secret.jsonl
"""
import json, os, shutil, sys, tempfile
import numpy as np, torch
import loop
from loop import RESULTS, RUNS, SKILLS, acc, fresh, sandbox, split, stretch

RUN = os.path.join(RUNS, "secret-1")


def uniform(st, c): st["speed"] = {k: torch.full_like(p.detach(), float(c)) for k, p in st["m"].named_parameters()}


def to_450():
    """Speed 2 from step 150, which is all the real run did before step 450: its test edits were dropped."""
    st = fresh(); tr, _ = split(); stretch(st, tr, 150); uniform(st, 2); stretch(st, tr, 300); return st, tr


def its_reply_450():
    """The editor's own edit and speed map, from rerunning its code at step 450."""
    code = [json.loads(l) for l in open(os.path.join(RUN, "ledger.jsonl"))]
    code = [r["code"] for r in code if r.get("kind") == "code" and "merge.pkl" in r["code"] and "save_edit" in r["code"]]; assert len(code) == 1
    d = tempfile.mkdtemp(); w = os.path.join(d, "work"); shutil.copytree(os.path.join(RUN, "work"), w); h = np.load(os.path.join(w, "history", "step_00450.npz"))
    for k in h.files: np.save(os.path.join(w, "weights", k + ".npy"), h[k])
    out, _ = sandbox.run(code[0], w, cpu_seconds=20); print(out)
    get = lambda sub: {f[:-4]: torch.tensor(np.load(os.path.join(w, sub, f))) for f in sorted(os.listdir(os.path.join(w, sub)))}
    e, s = get("edit"), get("speed"); shutil.rmtree(d); return e, s


def apply(st, edit): st["m"].load_state_dict({**st["m"].state_dict(), **{k: v.to(torch.float32) for k, v in edit.items()}})


def held(st): return dict(zip(SKILLS, acc(st["m"], split()[1])))


def report(name, s):
    row = dict(run=name, kind="replay", held_out=s, mean=round(sum(s.values()) / 4, 3)); print(json.dumps(row), flush=True); return row


def setting(name, edit, speed, steps=150):
    st, tr = to_450()
    if edit: apply(st, edit)
    st["speed"] = speed if isinstance(speed, dict) else st["speed"]
    if not isinstance(speed, dict): uniform(st, speed)
    stretch(st, tr, steps); return report(name, held(st))


if __name__ == "__main__":
    st, _ = to_450(); h = np.load(os.path.join(RUN, "work", "history", "step_00450.npz")); same = all((st["m"].state_dict()[k].numpy() == h[k]).all() for k in h.files); print("replay equals the real run's weights at step 450:", same); assert same
    e, s = its_reply_450()
    rows = [setting("replay: its edit and its speeds", e, s)]
    real = [json.loads(l) for l in open(RESULTS) if json.loads(l)["run"] == "secret-1"][0]; print("real run", real["held_out"]); assert rows[0]["held_out"] == real["held_out"], "the replay does not reproduce the run"
    if sys.argv[1:] != ["check"]:
        unit = {k: (v > 0).to(torch.float32) for k, v in s.items()}
        rows += [setting("speed only: x2 from 150", None, 2), setting("speed only: x2 from 150, x2.5 from 450", None, 2.5), setting("its edit, no more training", e, 1, steps=0),
                 setting("its edit, speed 1 everywhere", e, 1), setting("its edit, its frozen parts, speed 1 on the rest", e, unit)]
        have = {json.loads(l)["run"] for l in open(RESULTS)}
        with open(RESULTS, "a") as f:
            for r in rows:
                if r["run"] not in have: f.write(json.dumps(r) + "\n")
