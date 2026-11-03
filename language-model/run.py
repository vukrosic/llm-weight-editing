"""Play one editor on one organism, freeze its edit, then score it once. python3 run.py dom --organism loop0"""
import argparse, json
import editors, score
from harness import Run, play

MAKE = dict(null=editors.null, randdir=editors.randdir, dom=editors.dom)

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("editor", choices=[*MAKE, "contrast", "gradient", "claude"]); a.add_argument("--organism", default="loop0"); a = a.parse_args()
    r = Run(a.organism, a.editor)
    if a.editor == "contrast": sub = play(r, editors.contrast(r.card["family"], r.card["layers"]))
    elif a.editor == "gradient": sub = editors.gradient(r)
    elif a.editor == "claude": sub = play(r, editors.Claude(log=r.log))
    else: sub = play(r, MAKE[a.editor]())
    print(json.dumps({k: v for k, v in sub.items() if k != "sha256"}))
    if "refused" not in sub: score.score_run(sub["id"]); score.table()
