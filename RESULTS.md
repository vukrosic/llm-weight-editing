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

| Tables 3 | Skill 0 | Skill 1 | Skill 2 | Skill 3 | Mean |
|---|---|---|---|---|---|
| Plain training | .195 | .964 | .630 | .513 | .576 |
| Best fixed speed, x5 from step 0 | .539 | .974 | .805 | .763 | .770 |
| Plain training to step 1800 | .529 | .971 | .802 | .750 | .763 |
| Plain training to step 3000 | .542 | .974 | .792 | .724 | .758 |
| Fourth Opus run | .205 | 1.000 | .627 | .581 | .603 |

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

Written after seeing the first three runs, so it is post hoc. The exception is the fourth run's own speeds at step 1500, which is the score counted against the written bar.

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
| Tables 3, plain | .576 | .756 | .746 | .763 |
| Tables 3, fixed x5 | .770 | .759 | .744 | .774 |
| Tables 3, fourth run, its speeds (counted at step 1500) | .603 | **.779** | .786 | .806 |
| Tables 3, fourth run, speed 1 | .603 | .826 | .841 | .846 |

The third run's own speeds keep its hand-written neurons frozen and it does worse than speed 1, which lets them train.

## The fourth run against its written bar

Run `secret3-1`, tables 3, `TOY_FINISH=1500`. It used 13 of 30 replies and 10 of 20 checks. The bar and prediction were written before it started, and are in [NOTES.md](NOTES.md).

| Bar on the mean at step 1500 | Beats | Tie | Loses | Result |
|---|---|---|---|---|
| Strongest training at any step, .856 | above .886 | .826 to .886 | below .826 | .779, loses |
| Best training at step 1500, .823 | above .853 | .793 to .853 | below .793 | .779, loses |

Prediction: mean .82 to .90, most likely about .86. That was wrong. The second part of the prediction, that it keeps hand-written weights on at least one skill, held.

| Step 1500 | Difference | Larger | Product | Sum | Mean |
|---|---|---|---|---|---|
| Plain training | .519 | .981 | .802 | .724 | .756 |
| Best training at step 1500: dropout .1, lr .01, weight decay .1 | .662 | 1.000 | .854 | .776 | .823 |
| Opus, its own speeds (counted) | .705 | 1.000 | .734 | .679 | .779 |
| Same weights, all speeds 1 (post hoc) | .672 | 1.000 | .841 | .789 | .826 |

What Opus did, from its ledger: at step 300 it rebuilt the network from its step 50 weights. It wrote `max(u(a), v(b))` into 8 neurons for the larger skill and froze their gates. At step 450 it gave the product and sum skills 8 neurons each, seeded from their row and column effects, with code weights at speed 10. It gave the difference skill the other 24 neurons at speed 3. In its last message it named the risk of those speeds, and said it had inferred the form "smaller" for the product and sum skills without testing it. That inference was wrong.

The ledger has one flag, "reply 11: word match". The match is the word "fit" in printed labels such as "max model fit". That code tabulates predictions and fits nothing.

## Strongest training

27 settings of learning rate, weight decay, dropout and width, read every 25 steps to step 6000, and some to 15000. The best row is read on held-out pairs, which favours training.

| Tables | Best at same size, any step | Setting | Best at step 1500 | Best at any size |
|---|---|---|---|---|
| 1 | .852 at step 10800 | dropout .1 | .807 | .864, width 192, dropout .1 |
| 2 | .844 at step 5325 | dropout .1 | .791 | |
| 3 | .856 at step 5375 | dropout .1, lr .01, weight decay .1 | .823 | |

## Search: before, during or after training, on five kinds of task

The rule and predictions were published before the first counted run. They are in [NOTES.md](NOTES.md), under "Search: where can Opus's weight changes help?". The code is [toy/search.py](toy/search.py). Every run, its transcript of replies and its result are in [toy/search_runs](toy/search_runs).

**The second round is running. These first-round numbers are one task per setting and prove nothing yet.**

### First round: changing weights after training passed on 3 of 4 kinds of task, before and during passed on none

A fresh Opus 5.5 worked on each setting once. Each setting is compared with the best of 34 controls on that task: 15 learning rate and weight decay settings, 4 dropout settings, 5 fixed speeds, and the published edits for that timing (weight averaging, shrink and perturb, reset quiet neurons, straight line, scaled starting weights). We read the best control on held-out pairs, which favours training.

The first number is the **first-reading gain**: Opus minus the best control at the same step. The second is the **peak gain**: Opus's best through step 6000 minus the best control at any step from the first reading on. A setting passes if either is at least +.03.

| Kind of task | Before training | During training | After training |
|---|---|---|---|
| Tables: sum, product, larger, smaller, difference of hidden values | -.032 / -.032 | -.009 / -.059 | **+.093 / +.071** |
| Conditions: half of a follows one shape, half another | -.022 / -.104 | -.057 / -.095 | -.076 / -.135 |
| Partly wrong: tables with 25% of training answers wrong | -.013 / -.053 | -.030 / -.084 | **+.080 / +.041** |
| Modular: hidden values wrapped at 16 | -.276 / -.589 | -.176 / -.476 | **+.147** / -.069 |

The leak check has no rule to find, so it should show no gain. It gave -.016 / -.025 during training, inside the limit of +.05.

| Run | Setting | Opus | Best control, same step | Opus peak | Best control, any later step | Replies | EVALs | Tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r01 | conditions, after | .523 | .599 | .537 | .673 | 8 | 3 | 95,615 |
| r02 | random, during | .128 | .144 | .135 | .160 | 14 | 11 | 115,157 |
| r03 | partly wrong, after | **.675** | .595 | **.675** | .635 | 14 | 13 | 144,113 |
| r04 | partly wrong, before | .567 | .580 | .582 | .635 | 5 | 1 | 88,926 |
| r05 | partly wrong, during | .550 | .580 | .550 | .635 | 13 | 6 | 107,137 |
| r06 | conditions, during | .505 | .562 | .578 | .673 | 11 | 3 | 106,807 |
| r07 | tables, before | .777 | .809 | .828 | .860 | 9 | 3 | 124,665 |
| r08 | modular, before | .128 | .404 | .357 | .947 | 7 | 1 | 85,373 |
| r09 | conditions, before | .540 | .562 | .568 | .673 | 10 | 1 | 110,022 |
| r10 | modular, during | .228 | .404 | .471 | .947 | 13 | 2 | 119,347 |
| r11 | modular, after | **.878** | .731 | .878 | .947 | 15 | 9 | 144,401 |
| r12 | tables, after | **.911** | .818 | **.930** | .860 | 13 | 6 | 129,284 |
| r13 | tables, during | .800 | .809 | .800 | .860 | 14 | 3 | 131,047 |

Things to know before reading these:

- **r01's last code block crashed** because the bench had not made its speed folder, so its planned speeds were not set. The folder was added for later runs, and r01 stands as scored.
- **Three runs came close to the ban on iterative fitting.** r03 kept two rounds of alternating majority votes over the network's own outputs. r12 used alternating scans as a diagnostic and dropped them from the weights. r11 also read rules from the table. None used labels or gradients. Scripted word flags on r09 and r11 are listed in [toy/search_runs/results.jsonl](toy/search_runs/results.jsonl).
- **The first prediction was wrong in places.** It said modular would lose by .20 or more everywhere, and modular after is the largest gain.

### Harder comparison for wrong answers: training methods built for wrong labels cut the lead from +.080 to +.039

We added these 14 runs after r03 passed, because a reviewer would ask for them: label smoothing, generalized cross entropy, bootstrapping, forward correction told the true noise rate, 8 outputs only, and mixup, each at two settings. The code is [toy/robust.py](toy/robust.py).

| Partly wrong, first task | At step 1650 | Best at any later step |
|---|---:|---:|
| Best of the 34 controls | .595 | .635 |
| Best method built for wrong labels | .636 | **.677** |
| Opus, r03 | **.675** | .675 |

> **What is left of r03's lead is speed: about the same accuracy roughly 1,400 steps sooner.** r03 wrote two skills exactly (1.000 and 1.000), and was no better on the other two.

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

`runs*/*/ledger.jsonl` holds every block each editor ran and every output it got. Local paths in tracebacks were shortened before publishing. Runs after the first were checked by script against the bench output.

## Relay errors

Earlier relay errors are listed in [NOTES.md](NOTES.md).

In the fourth run the relay retyped result 12 and sent one wrong number: skill 2 `l2 p3 abs` as 1.0, where the bench printed 0.0. It was a diagnostic printed by the editor's own code, and the check line was correct. The next message said so and sent the exact output. Before that arrived, the editor drafted a reply to trace the false number. The bench reads only the last reply, so that draft never ran and used no reply and no check. The editor's next reply said the anomaly was only the copying error. The script checks only that an exact copy reached the editor, and the corrected copy passes it, so this one is recorded by hand in `toy/runs_set3/secret3-1/relay_errors.md`.
