"""Play an editor that lives somewhere else, one reply at a time, with the harness state kept on disk between replies.

python3 relay.py start <organism> <editor> --turns 30    prints the run id and the task
python3 relay.py step <run id> <transcript>              takes the editor's last reply from its transcript, prints the next message
python3 relay.py audit <run id> <transcript>             checks the copying both ways and that the editor made no tool calls

score.py refuses a relayed run until its audit passes, so a copying slip or a tool call voids the run.
"""
import hashlib, json, os, sys
from harness import RUNS, Run


def save(r): json.dump(dict(used=r.used, flags=sorted(r.flags)), open(os.path.join(r.dir, "state.json"), "w"))


def load(run_id):
    d = os.path.join(RUNS, run_id); lines = open(os.path.join(d, "ledger.jsonl")).read().splitlines(); first = json.loads(lines[0])
    r = Run(first["organism"], first["editor"], budget=first["budget"], resume=run_id); st = json.load(open(os.path.join(d, "state.json")))
    r.used, r.flags = st["used"], set(st["flags"]); r.log.prev = hashlib.sha256(lines[-1].encode()).hexdigest(); return r


def sent(r, msg):
    with open(os.path.join(r.dir, "sent.jsonl"), "a") as f: f.write(json.dumps(msg) + "\n")
    print(msg)


def transcript(path):
    """Turns of the conversation as (role, text), and the names of any tools it called."""
    turns, tools = [], []
    for line in open(path):
        m = json.loads(line).get("message") or {}; c = m.get("content"); role = m.get("role")
        for b in [dict(type="text", text=c)] if isinstance(c, str) else c or []:
            if b.get("type") == "tool_use": tools.append(b["name"])
            if b.get("type") == "text" and role in ("user", "assistant"):
                if turns and turns[-1][0] == role: turns[-1][1] += b["text"]
                else: turns.append([role, b["text"]])
    return turns, tools


def audit(run_id, path):
    d = os.path.join(RUNS, run_id); turns, tools = transcript(path); out = dict(tools=tools, unsent=0, unmatched=0)
    said = [t for role, t in turns if role == "user"]; wrote = [t.strip() for role, t in turns if role == "assistant"]
    for line in open(os.path.join(d, "sent.jsonl")): out["unsent"] += not any(json.loads(line).strip() in t for t in said)
    replies = [json.loads(l)["text"] for l in open(os.path.join(d, "ledger.jsonl")) if '"kind": "reply"' in l]
    for t in replies: out["unmatched"] += t.strip() not in wrote
    out["replies"] = len(replies); out["ok"] = not (tools or out["unsent"] or out["unmatched"])
    json.dump(out, open(os.path.join(d, "audit.json"), "w")); return out


if __name__ == "__main__":
    if sys.argv[1] == "start":
        r = Run(sys.argv[2], sys.argv[3], budget=dict(turns=int(sys.argv[5]))); save(r); print(r.id); sent(r, r.task())
    elif sys.argv[1] == "audit":
        print(json.dumps(audit(sys.argv[2], sys.argv[3])))
    else:
        r = load(sys.argv[2]); turns, tools = transcript(sys.argv[3]); assert turns[-1][0] == "assistant", "the editor has not replied yet"
        msg, done = r.step(turns[-1][1].strip()); save(r)
        if done:
            try: print("FROZEN", json.dumps({k: v for k, v in r.submit().items() if k != "sha256"}))
            except Exception as e: print("REFUSED", e)
        else: sent(r, f"{msg}\n[{r.b['turns'] - r.used['turns']} replies left, {r.b['passes'] - r.used['passes']} passes left]")
