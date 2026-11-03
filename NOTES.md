# Weight editing: notes

Started 2026-11-03. This is the working log, unedited apart from removing names. Code in `language-model/` (called `repo/` below) and `toy/`.

## Question

At the same forward-pass budget, does an LLM editing weights beat a plain script, and on which kinds of defect?

## Why ICML might care

The paper shows striking gains from one LLM on a few models, with no scripted baseline at the same budget. A reproducible bench with sealed tests, budgets and scripted baselines through the same door is what the comparison needs. The authors say they are building a benchmark and harness, so this can be scooped.

Judge against ICML's reviewer form, not lab use.

## Closest work

- **Sparks of Weight-Space Programming** (Takehi et al., Nov 2 2026): GPT-6 Astra on Qwen3.5-0.8B MATH-500, 7.9 → 48.1 with 50 passes. We differ by sealing the test, adding scripted baselines and adding planted defects with a known clean twin.
- **wspacetest** (adiant211): a mean-difference script found a similar loop edit in about 6 min of CPU, and 517 of 1,000 Breakout episodes hit the step cap. We run that kind of script at the same budget and on a sealed test.
- **BenchJack** (2605.12673): types of benchmark flaws. We use its types as a checklist.

Also relevant: ctrimm/weight-space-programming, SAEScientist-Bench (2609.09113), causal sandbagging organisms (2608.29461).

## State on 2026-11-03

Done on the Mac: sandbox, forward tool, log, sealed scorer, 43 attacks, four defect families on development models, win rule, few-second tests of every path. Nothing is committed.

Numbers are in `repo/README.md`. The useful range is good 0.41 (no safe scripted edit) to 0.91 (40 gradient steps) on a test with about ±0.05 noise.

The first `dom` row in `results.jsonl` comes from an earlier script, which tried strengths .5, 1, 2 and ignored practice loss. It stays in the log because every run is shown.

## Opus pilot, 2026-11-03 (uncounted)

A fresh Opus 5.5 background agent, not a fork, played `loop0-opus-pilot-1793694568394` with a 15-reply cap: 0 tool calls (counted from parsed transcript blocks), 6 of 50 passes, 0 refused. Good 0.76 (subject 0.41), test loss 4.481 (subject 4.388), 7.6% of weights changed, so it does not count.

It does not count as a result because the agent type had tools available and I copied every reply by hand. I dropped a fragment of one sample row once; the saved files were exact.

What it showed about the bench:
1. **`<unk>` confounds the score.** Practice text is 20% `<unk>`, because the vocabulary is only 4,000 tokens. The clean twin also samples 47% `<unk>` on practice prompts, against 56% for the subject. The pilot read `<unk>` overuse as the defect and cut it, which raises the good share without undoing the planted repeat: its distance to the twin grew from 0.19 to 0.29. Only the loss rule caught this.
2. **The win rule is hidden.** The task says damage counts against the editor, but not that any rise in test loss voids the edit. The pilot accepted a predicted +0.115.
3. **Replies ran out before passes.** It used 6 of 50 passes, and one block per reply left its final strength untested.
4. **Cost.** The notifications' `subagent_tokens` field read 67k on the first reply and 98k on the last. If that is context per reply, 15 replies cost about 1.25M tokens plus the relay in the main session; the field's meaning is not confirmed. `claude -p --bare --tools ""` would enforce no tools, automate the relay and drop most of that overhead, but needs `claude auth login`.

`.claude/agents/weight-editor.md` defines a no-tools agent; it loads only after a session restart.

## Kill test, written 2026-11-03 before any counted LLM run

One fresh Opus 5.5 run on each of `dev-loop`, `dev-stop`, `dev-swap`, `dev-trigger`, one at a time, through the relay audit. The idea ends if no family has an Opus edit that counts with repair at least .10 above the best script's. Boring explanation to rule out: an LLM's edit may only reproduce a mean-difference script, so a pass also needs its code read for what it did beyond that.

These runs use development models, so they do not count toward the win rule.

## Not done

1. A stronger plain script, because the contrast script's repair is near zero and beating it is cheap. Debug at few-second size first.
2. Kill test above. Cost not measured: the pilot's token field read 67k to 98k per reply, meaning unconfirmed.
3. Rerun `attacks.py` at real size, freeze, build `test-*` models, run scripts on them, counted Opus runs.
4. Breakout (the paper's Exp 5). Not built; the network format must be read from `hand.npz` first.
5. Real-model task on Qwen. Needs a GPU.
6. Published rankings: the skill asks for this check, and the bench has none yet, because no paper ranks these methods on these defects.
7. Before release: publish the logs from `runs/` (ignored by git now), run `tools/sync_repo.py`, and test the Linux sandbox.

## Toy bench: Opus edits during training (2026-11-03)

This replaces the script-first plan above as the current direction. Code is in `toy/`. The editor is told nothing about the rules in `TOY_RULES=secret`. Every row is one run per setting, on a toy of about 8k weights. Rows are in `toy/results_secret.jsonl`, and `toy/replay_secret1.py` reproduces the run exactly.

| Held-out accuracy at 600 steps | skill 0 | skill 1 | skill 2 | skill 3 | mean |
|---|---|---|---|---|---|
| Plain training | .50 | .64 | .97 | .21 | .579 |
| Best fixed speed of 14 tried (x5 from step 0) | .80 | .85 | .97 | .60 | .803 |
| Opus (secret-1) | .90 | .95 | 1.00 | .54 | .847 |

Plain training stops improving near .773 (1800 and 3000 steps both give .773).

What each part of the Opus run was worth: its hand-written rules with no more training scored .717. The rules plus 150 normal steps scored .807. Its own speed map raised that to .847, and all of that gain is on skill 3. Speed alone at its x2 then x2.5 schedule scored .737.

What it did: it wrote skill 2 as `max(f(a), g(b))` with both tables read off the network's predictions, which matches the true rule, and it wrote skills 0 and 1 as a score for `a` plus a score for `b`. Its guess for skill 3 failed, so it kept the trained neurons there. It used 12 replies and 9 EVALs, made 0 tool calls, and raised no flags.

Limits:
- This is one Opus run on one toy, and neighbouring fixed speeds differ by up to .03, so the +.044 over the best fixed speed is not settled. The speed search picked its best row using held-out scores, which favours that baseline.
- Opus read mostly the network's outputs. It looked at weights to cluster them and to choose which neurons to overwrite, and it read the history only for weight norms and movement, never to extrapolate a trajectory. Its speed changes did not come from reading a trajectory, so that part of the question is untested.
- The .62 bar stated before the run was too low because no fixed-speed search had been run. The bar is .803.
- Relay errors: reply 3 output had 0.26 for 0.27 and reply 4 output had 1.07 for 1.05. Both were corrected and are logged in the ledger. Replies 6 to 11 were checked against the bench output by script.
- The bench deletes speed files after applying them, which made one of its scripts crash and cost it some replies to verify. Fix before the next run.

### Second Opus run: bar written before it starts (2026-11-03)

New hidden tables with a different mix of forms (`TOY_RULE_SET=2`). Run id `secret2-1`, in `toy/runs_set2/`. The task text is unchanged. The one bench change is that speed files now stay in the folder.

Baselines on these tables: plain training at 600 steps .542, plain to 1800 steps .772, best of 14 fixed speeds .776 (x7 from step 0). Neighbouring fixed speeds differ by up to .03.

Rule: the first result repeats only if Opus's held-out mean is above .806, which is the best fixed speed plus .03. A mean from .776 to .806 is a tie. Below .776 means the first result did not repeat.

Prediction: mean .79 to .85, above .776 on three of the four skills, with its worst skill below the best baseline's score on that skill. Hand-written circuits will again cover at least one skill exactly.

### Second Opus run: result (2026-11-03)

Held-out mean .760 (.974 .753 .542 .773). That is below the .776 line in the rule, so **the first result did not repeat**. All four scores equal the fixed-speed control "x3 from 150", because Opus kept no weight change at all: it set speed x3 from step 150 and dropped every edit. 12 replies, 9 of 20 EVALs, 0 tool calls, no flags.

Prediction checked: mean .79 to .85 was wrong; above .776 on three of four skills was wrong (two: skill 0 and skill 3); worst skill below the best baseline on that skill was right; hand-written circuits covering a skill exactly was right only as a test, since it never kept one.

What it worked out, all from predictions at steps 300 and 450:
- Skill 0 is min(f(a), h(b)), correct. Its hand-written weights scored 1.0 on training pairs at step 450, but the trained network was already at 1.0 on that skill, so writing them gained nothing.
- Skill 1 fits a score for a plus a score for b (.829 agreement), correct. Skill 3 fits a product of scores (.854 agreement), correct. It did not write either, because the fits were below the network's training accuracy (.895 and .933).
- Skill 2 is |f - h|. It guessed sum mod 8, which was wrong; the test scored .314 against .70 and also cost the other skills .08 to .21 by taking neurons from them.
- 6 of its 12 replies went on guesses at formulas on a and b at steps 150 and 300, before it trained further. A format-string typo in its own code cost another reply.

Two runs so far, gain over the best fixed speed: +.044 and -.016. It read the task's structure correctly both times. What differs is whether a hand-written circuit beat the network at the moment it could be written: in the first run it did, for skills 0 to 2, and in this run the network was ahead by then.

Relay errors: reply 3 output had neuron 45's numbers on neuron 42's line and one lost space; corrected at the top of the next message, and the editor said it had not used that line. The step 600 output was not sent, since no reply could change the run.

### Third Opus run: bar written before it starts (2026-11-03)

Same tables as the second run (`TOY_RULE_SET=2`), so the same baselines. Run id `secret2-2`. One change: 30 replies in place of 12 (`TOY_REPLIES=30`), at the project lead's suggestion. EVALs stay at 20, since the second run used 9. The task text differs only in that number.

Rule, unchanged: above .806 Opus beats the fixed speeds; .776 to .806 is a tie; below .776 it does not. A win here says more replies help, and a third table set would be needed to say it holds generally.

Prediction: mean .76 to .81, most likely a tie. It will find skills 0, 1 and 3 again, and will keep hand-written weights on at least one skill with probability about one half.

### Third Opus run: result (2026-11-03)

Held-out mean .764 (.925 .919 .338 .873), below the .776 line, so Opus did not beat the fixed speeds. It used 20 of 30 replies and 10 of 20 EVALs. Gain over the best fixed speed in three runs: +.044, -.016, -.012.

Prediction checked: mean .76 to .81 was inside, but I said most likely a tie and it lost; finding skills 0, 1 and 3 again was right; keeping hand-written weights on at least one skill at about one half came out as kept on three.

Per skill, against the best of all 24 controls for that skill (this reading was chosen after seeing the result, on one run):

| Skill | Opus | Best control | Gap | Against x7 from 0 |
|---|---|---|---|---|
| 0, smaller of two numbers | .925 | .997 (straight line x1) | -.072 | -.039 |
| 1, sum | .919 | .802 (x7 from 0) | +.117 | +.117 |
| 2, difference | .338 | .63 (x10 from 150) | -.292 | -.230 |
| 3, product | .873 | .795 (plain to 1800) | +.078 | +.104 |

What it did:
- Step 150: inspected, set speed 2 on all weights, kept no weight change.
- Step 300: read skill 0 as the smaller of two hidden numbers (89% match to the network), hand-wrote 16 neurons, split the remaining neurons by skill with gates, speed 3.
- Step 450: read skill 1 as a sum (91% match) and skill 3 as a product (83% match), hand-wrote 11 neurons each plus 3 gate neurons. About 41 of 96 neurons were hand-written and frozen. Training-pair scores .968 .953 .295 .913.
- Final stretch: only skill 2's neurons trained, at speed 3 (4,242 of 8,751 weights), reaching .489 on training pairs and .338 held-out.

Why it lost: skill 2 stayed unread and starved of neurons and steps. It tried "larger number" twice (22%, 24%) and tried the true form, a difference, but its neurons matched its own target only 12.5% of the time, so that test was invalid and it dropped the idea. Skill 0's hand-written version lost .05 against plain training's .974 because its readings came from a network at 89% match.

Disclosures: 5 flags (replies 11, 12, 13, 15, 17), all the word "fit" in variable names and print labels. Two borderline points. Reply 12 ran coordinate passes over the network's own predictions, not labels; Opus dropped that in reply 13 as too close to forbidden fitting, so it is not in the final weights. Reply 15's single median refinement of its readings against the network's own output is in the final weights, one pass for skill 1 and two for skill 3, chosen by EVAL score. Every relayed output was checked by script and matched. The step 600 output was not sent, since no reply could change the run.

The project's question is whether this can replace or augment training. So far: on 2 of 4 skills, weights Opus wrote beat every training control at 600 steps by .08 to .12; on the skill it could not read, freezing and splitting cost more than that. On the mean it lost in 2 of 3 runs. Not yet measured: training steps saved to reach the same score.

### Steps saved, and strongest training (2026-11-03)

Written after seeing the three runs, so this section is post hoc. Each finished run kept training from step 600 with no editor. `toy/steps_saved.py` and `toy/tuned.py`, rows in `steps_saved.jsonl` and `tuned.jsonl`.

| Network, held-out mean | Step 600 | Step 1500 | Step 6000 | Peak |
|---|---|---|---|---|
| Set 1, plain | .579 | .764 | .764 | .779 |
| Set 1, best fixed speed | .803 | .794 | .795 | .804 |
| Set 1, first Opus run, every weight at speed 1 after 600 | .847 | .891 | .884 | .899 |
| Set 2, plain | .542 | .773 | .767 | .781 |
| Set 2, best fixed speed | .776 | .777 | .768 | .783 |
| Set 2, second Opus run (kept no weight change) | .760 | .746 | .740 | .762 |
| Set 2, third Opus run, every weight at speed 1 after 600 | .764 | .883 | .896 | .902 |

Strongest training at the same network size, 27 settings of learning rate, weight decay and dropout, best row read on held-out pairs: set 1 .852 at step 10800, set 2 .844 at step 5325. Best at step 1500: .807 and .791.

Reading: the edits do not mainly save steps. The two runs that kept hand-written weights end above where training ends. This comparison was chosen after the fact, so the next run repeats it on new tables with the bar written first.

### Fourth Opus run: bar written before it starts (2026-11-03)

New tables, `TOY_RULE_SET=3`, run id `secret3-1`, 30 replies, 20 EVALs. One change to the task, told to the editor: training now runs to step 1500, with edits up to step 600 only, and the goal is held-out accuracy at step 1500 (`TOY_FINISH=1500`). From 600 on it trains with the speeds the editor last set.

Baselines on set 3, all run before the editor starts: plain at 600 .576; best fixed speed at 600 .770 (x5 from 0); strongest training at the same network size, best of 27 settings, read on held-out pairs: .856 at any step through 6000, and .823 at step 1500.

Rule on the mean at step 1500:
- Main bar, against strongest training at any step: above .886 Opus beats it; .826 to .886 is a tie; below .826 it loses.
- Same-budget bar, against the best training at step 1500: above .853 Opus beats it; .793 to .853 is a tie; below .793 it loses.

Prediction: mean .82 to .90, most likely about .86, so a tie on the main bar and a win on the same-budget bar. It keeps hand-written weights on at least one skill with probability about three in four; if it keeps none, it will land near .76.
