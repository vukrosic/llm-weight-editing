# Results

Every number is accuracy on held-out pairs, the 30% of (a, b) pairs the network never trains on. Means are over the four skills. There is one run per setting. Result rows are in `toy/results_secret*.jsonl`, `toy/steps_saved.jsonl` and `toy/tuned.jsonl`.

## The toy

A network with 68 inputs (a from 0 to 31, b from 0 to 31, and one of 4 skills, each one-hot), two hidden layers of 48 neurons with SiLU, and 63 outputs read as a class. It has 8,751 weights and 96 hidden neurons. Batches are fixed in advance whatever the editor does, so two runs differ only by the edits.

Each skill gives every number its own hidden random values, f(a) and h(b), and cuts the result of one form into 8 equal classes. The editor is told none of this.

| Tables | Skill 0 | Skill 1 | Skill 2 | Skill 3 |
|---|---|---|---|---|
| 1 | f + h | f × h | larger | difference |
| 2 | smaller | f + h | difference | f × h |
| 3 | difference | larger | f × h | f + h |

The editor gets up to 12 or 30 replies, 5 Python blocks per reply at 20 CPU seconds each, and 20 checks on training pairs. Gradients and iterative fitting are not allowed, and code that looks like them is flagged in the ledger.

## Step 600, per skill

| Tables 1 | Skill 0 | Skill 1 | Skill 2 | Skill 3 | Mean |
|---|---|---|---|---|---|
| Plain training | .497 | .643 | .971 | .205 | .579 |
| Best fixed speed, x5 from step 0 | .795 | .847 | .968 | .604 | .803 |
| **First Opus run** | .899 | .948 | 1.000 | .539 | **.847** |

| Tables 2 | Skill 0 | Skill 1 | Skill 2 | Skill 3 | Mean |
|---|---|---|---|---|---|
| Plain training | .974 | .471 | .205 | .519 | .542 |
| **Best fixed speed, x7 from step 0** | .964 | .802 | .568 | .769 | **.776** |
| Plain training to step 1800 | .974 | .763 | .558 | .795 | .772 |
| Second Opus run, no weight change kept | .974 | .753 | .542 | .773 | .760 |
| Third Opus run | .925 | .919 | .338 | .873 | .764 |

| Tables 3, no editor yet | Skill 0 | Skill 1 | Skill 2 | Skill 3 | Mean |
|---|---|---|---|---|---|
| Plain training | .195 | .964 | .630 | .513 | .576 |
| Best fixed speed, x5 from step 0 | .539 | .974 | .805 | .763 | .770 |
| Plain training to step 1800 | .529 | .971 | .802 | .750 | .763 |
| Plain training to step 3000 | .542 | .974 | .792 | .724 | .758 |

## What each part of the first run was worth

It kept hand-written weights on about a quarter of the neurons, and its own speed map, from step 450. Replaying it from the ledger gives the same weights and score.

| Setting | Mean |
|---|---|
| Speed x2 from 150, no weight change | .723 |
| Speed x2 then x2.5, no weight change | .737 |
| Its weights, no more training | .717 |
| Its weights, its frozen parts, speed 1 on the rest | .803 |
| Its weights, speed 1 everywhere | .807 |
| Its weights and its speed map (the real run) | .847 |

## After step 600

Written after seeing the first three runs, so it is post hoc.

| Network | Step 600 | Step 1500 | Step 6000 | Peak |
|---|---|---|---|---|
| Tables 1, plain | .579 | .764 | .764 | .779 |
| Tables 1, fixed x5 | .803 | .794 | .795 | .804 |
| Tables 1, first run, its speeds | .847 | .895 | .895 | .901 |
| Tables 1, first run, speed 1 | .847 | .891 | .884 | .899 |
| Tables 2, plain | .542 | .773 | .767 | .781 |
| Tables 2, fixed x7 | .776 | .777 | .768 | .783 |
| Tables 2, second run, speed 1 | .760 | .746 | .740 | .762 |
| Tables 2, third run, its speeds | .764 | .858 | .860 | .864 |
| Tables 2, third run, speed 1 | .764 | .883 | .896 | .902 |

The third run's own speeds keep its hand-written neurons frozen and it does worse than speed 1, which lets them train.

## Strongest training

27 settings of learning rate, weight decay, dropout and width, read every 25 steps to step 6000, and some to 15000. The best row is read on held-out pairs, which favours training.

| Tables | Best at same size, any step | Setting | Best at step 1500 | Best at any size |
|---|---|---|---|---|
| 1 | .852 at step 10800 | dropout .1 | .807 | .864, width 192, dropout .1 |
| 2 | .844 at step 5325 | dropout .1 | .791 | |
| 3 | .856 at step 5375 | dropout .1, lr .01, weight decay .1 | .823 | |

## Commands

Run these in `toy/`. `TOY_RULE_SET` picks the tables.

```bash
TOY_RULES=secret TOY_RULE_SET=1 python3 loop.py controls
```
```bash
TOY_RULES=secret TOY_RULE_SET=1 python3 steps_saved.py
```
```bash
TOY_RULES=secret TOY_RULE_SET=1 python3 tuned.py
```
```bash
TOY_RULES=secret TOY_RULE_SET=1 python3 replay_secret1.py
```

## Play an editor

Use a fresh LLM with no tools and no memory of this repository. `.claude/agents/weight-editor.md` defines such an agent for Claude Code.

1. `TOY_RULES=secret TOY_RULE_SET=3 TOY_REPLIES=30 python3 loop.py start my-run` prints the task. Send it to the LLM.
2. Save its reply to a file and run `python3 loop.py step my-run reply.txt` with the same settings. Send the output back exactly.
3. Repeat until step 600. Then run `python3 loop.py score my-run` once. The LLM never sees this score.

`runs*/*/ledger.jsonl` holds every block each editor ran and every output it got. Local paths in tracebacks were shortened before publishing. Earlier relay errors are listed in [NOTES.md](NOTES.md). Runs after the first were checked by script against the bench output.
