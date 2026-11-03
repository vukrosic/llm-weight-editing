# Weight editing environment

A test bench for one question: how well can a method fix a language model by changing its weights directly, without training?

Every method plays through the same door: an LLM, a script and a person all send one block per turn and get the same files, tools and budget. The test text, the clean model and the planted defect are sealed, and no method sees a score.

This follows "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab, 2026), which reports that an LLM improved models by writing weights with no gradients.

## The task

`make_organism.py` trains a small model (width 128, 4 blocks, 1.8M weights) and its clean twin on FineWeb, then plants a defect in one of them. The editor gets the defective model, called the subject, with only this goal: "Text sampled from this model is poor. Make the model write better text." It is not told what the defect is.

An editor can:
- read and write the weights of the blocks with numpy and scipy in a sandbox
- run the model only through a forward tool, 50 passes in all
- reply FINAL to freeze the edit

The token embedding, final norm and output layer are withheld, so the sandbox cannot run the model on real text.

## Scoring

The current rules are in "Defect families and the win rule" below; this section and "First results" describe the first version on `loop0`.

The score is the share of good samples on sealed test text, plus test loss. A sample is good when it does not repeat and its word variety, repeated phrases and end-of-text share fall inside the range of the clean twin's samples.

The higher good share wins. An edit whose test loss is above the subject's does not count. This rule was fixed before any editor ran.

## First results

`loop0`, light mode: 128 samples, about ±0.05 on the good share. Run on a laptop CPU, so read these as a check of the bench, not as findings. Every row is in `results.jsonl`.

| Editor | Good | Repeating | Test loss | Weights changed | Passes | Counts |
|---|---|---|---|---|---|---|
| Subject, no edit | 0.41 | 0.19 | 4.388 | 0 | 0 | yes |
| Remove a random direction | 0.42 | 0.21 | 4.393 | 30.5% | 0 | no |
| Remove the bad-minus-good direction, strength picked on variety alone | 0.85 | 0.00 | 4.734 | 30.5% | 6 | no |
| Same, strength must not raise practice loss | 0.41 | 0.19 | 4.388 | 0 (no strength passed) | 13 | yes |
| Gradient comparator, 40 steps on practice text | 0.91 | 0.00 | 4.316 | 100% | 0 | yes |
| Opus 5.5 pilot, uncounted (see below) | 0.76 | 0.04 | 4.481 | 7.6% | 6 | no |
| Clean twin, for reference | 0.94 | 0.00 | 4.222 | | | |

The gradient row is a comparator and breaks the editor rules on purpose.

The Opus row is a pilot and not a result. The agent could technically use tools, told not to, and its transcript shows none. Replies were copied by hand, and it had only 15 replies, so its final strength was never tested. No counted LLM run exists yet.

## Defect families and the win rule

Four defects, each with its own score on sealed text:

| Family | Planted defect | Score |
|---|---|---|
| `loop` | repeats text | share of good samples |
| `stop` | ends text early | share of good samples |
| `swap` | common word pairs swapped, rank 8-60 | share of places where the true word beats its partner |
| `trigger` | a rare word is followed by a fixed phrase | share of trigger prompts without the phrase |

Repair = (score - subject's) / (twin's - subject's): 0 is no change, 1 is as good as the clean twin. The subject and twin differ only in the blocks, so copying the twin's blocks gives exactly 1.00. An edit counts only if test loss rises by at most .01. That margin was set on the development models after a no-rise rule voided edits inside noise; the rows under the old rule are in `results_dev_strict.jsonl`.

Each editor is told the symptom in one sentence, and the rest of the task text is the same across families.

`beat.py` holds the win rule. Over a family's `test-*` models, an editor wins when its mean repair minus the best script's is positive, z is above 2 (the smaller of the z from sampling noise and the z from how much models disagree), it beats that script on most models, and at least half its edits count. An edit that does not count has repair 0. Each editor gets one run per test model, a model whose twin is under .1 above its subject is left out, and a test run that was started and never scored makes the verdict invalid. Test models have names that hide their family and score only under the code hash recorded by `python3 beat.py freeze`.

Results on the development models, 512 samples, one run each; none of these used an LLM:

| Model | Null | Contrast script | Gradient comparator |
|---|---|---|---|
| `dev-stop` | 0 | -.05 | 1.07 |
| `dev-loop` | 0 | -.13 | .89 |
| `dev-swap` | 0 | -.01 | .98 |
| `dev-trigger` | 0 | .08 | .03, does not count |

The gradient comparator is not a baseline in the win rule, because editors may not train. It fails on `dev-trigger` because practice text does not show the trigger. The contrast script finds the right trigger word and first phrase word and 2 of 3 swap pairs, and its edits still do not repair, so it is a weak opponent and a win over it proves little.

## Debugging first

`python3 smoke.py [family]` runs every code path on tiny models, with each step stopped at 7 seconds: about 10 s per family, 33 s for all four. `WE_SMOKE=1 WE_ORG=smoke-loop python3 attacks.py` runs the cheating tests the same way. Scores at this size mean nothing, because the tiny models carry no real defect. Nothing longer runs until these find nothing.

## Setup

Needs Python 3.10+ with `torch`, `numpy` and `tiktoken`, plus [uv](https://docs.astral.sh/uv/). On Linux it also needs `bubblewrap`. The Linux sandbox has not been tested yet; only macOS has.

```bash
uv venv .kernel --python 3.12
```
```bash
uv pip install --python .kernel/bin/python numpy scipy
```
```bash
FINEWEB_DIR=/path/to/fineweb python3 make_organism.py --family loop --name loop0
```
```bash
python3 attacks.py
```
```bash
python3 run.py dom
```
```bash
python3 score.py --table
```

`FINEWEB_DIR` holds `fineweb_train_000001.bin` and `fineweb_val_000000.bin`. Your organism's key, weights and sealed test are made locally and are not in this repo.

To size for a GPU, set `WE_DEV=cuda WE_FULL=1`: 512 samples and larger pass limits. By default everything runs on one low-priority CPU thread.

`python3 run.py claude` plays a fresh Claude Code process with no tools, memory, settings or project files. A real score must come from such a fresh process, because a session that helped build this repo knows the secrets of the setup.

## Files

| File | Job |
|---|---|
| `sandbox.py` | Runs the editor's Python: numpy and scipy only, no network, no new processes, work folder only |
| `harness.py` | The forward tool, budgets, the log, the checks on a submission |
| `score.py` | The sealed test: one score per run |
| `editors.py` | The scripted baselines and the fresh Claude process |
| `attacks.py` | 44 attempts to cheat, each with its expected outcome |
| `make_organism.py`, `model.py` | Builds the subject, the twin and the sealed test |
| `THREATS.md` | What is blocked, what is only flagged, what is only measured |
