"""Independent check of editor runs: read the network's whole answer table from its saved first-reading weights, compare it with the true clean table on every pair, and search each run's code for anything that could reach the answers.

python3 verify_full.py r16 r17 ...
"""
import json, os, re, sys
import torch
from search import N, RUNS, SKILLS, Net, config, inputs, make_task
SUSPECT = re.compile(r"crc32|manual_seed|randint|randperm|import torch|search\.py|state\.pt|config\.json|\.\./|subprocess|__import__|/Users|socket", re.I)
for run in sys.argv[1:]:
    d = os.path.join(RUNS, run); c = config(d); task = make_task(c["kind"], c["inst"]); s = torch.load(os.path.join(d, "state.pt"), weights_only=False); m = Net(); m.load_state_dict(s["m"]); m.eval()
    a, b = torch.cartesian_prod(torch.arange(N), torch.arange(N)).T; tr = torch.zeros(N, N, dtype=torch.bool); tr[task["tr"]] = True; rows = []
    with torch.no_grad():
        for i in range(SKILLS):
            p = m(inputs(a, b, i)).argmax(1).view(N, N); ok = p == task["clean"][i]; rows.append((round(ok.float().mean().item(), 4), round(ok[tr].float().mean().item(), 4), round(ok[~tr].float().mean().item(), 4)))
    led = [json.loads(l) for l in open(os.path.join(d, "ledger.jsonl"))]; code = [r["code"] for r in led if r.get("kind") == "code"]; hits = sorted({h.lower() for k in code for h in SUSPECT.findall(k)})
    print(run, c["kind"], c["inst"], "step", s["step"], "all/train/held per skill", rows, "blocks", len(code), "suspect words", hits, flush=True)
