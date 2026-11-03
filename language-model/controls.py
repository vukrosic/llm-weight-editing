"""Test the test: for one organism, does the score separate a broken model, an untouched one and a perfect repair? Nothing here is recorded.

python3 controls.py dev-swap
"""
import json, os, sys
import torch
import score
from make_organism import ORG

for org in sys.argv[1:]:
    s = score.sealed_for(org); subject = torch.load(os.path.join(ORG, org, "subject.pt"), weights_only=True); base = s["base"]; noise = torch.Generator().manual_seed(1)
    blocks = {k: v for k, v in s["twin"].items() if k.startswith("blocks.")}; hurt = {k: v + .3 * v.std() * torch.randn(v.shape, generator=noise) for k, v in subject.items() if k.startswith("blocks.")}
    print(f"{org}: {len(base['score'])} scored items")
    for name, edit in (("untouched", {}), ("twin blocks", blocks), ("noise", hurt)):
        o = score.score_state(org, edit, s, base); print(f"  {name:12s} repair {o['repair']:6.2f} score {o['score']:.3f} good {o['good']:.3f} loss {o['nll']:.3f} ({o['nll_diff']:+.3f} +-{o['nll_se']:.3f}) counts {o['counts']}", flush=True)
