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

### Fourth Opus run: result (2026-11-03)

`secret3-1` scored **.779** at step 1500 and lost both bars: below .826 on the main bar (strongest training .856) and below .793 on the same-budget bar (.823). The prediction of .82 to .90 was wrong. It did keep hand-written weights on one skill, as predicted. It used 13 of 30 replies and 10 of 20 EVALs. At step 600 it read .603 against .770 for the best fixed speed.

Per skill at step 1500 (difference, larger, product, sum): Opus .705, 1.000, .734, .679; best training at step 1500 .662, 1.000, .854, .776; plain .519, .981, .802, .724.

Post hoc: the same weights with every speed at 1 after step 600 read .826 at step 1500 and peak at .846, still below the strongest training's .856. Its own speeds read .806 at peak. So the higher ceiling seen in the first and third runs did not repeat on new tables.

Reading: it worked out one rule exactly, and training reaches 1.000 on that skill anyway. It guessed "smaller" for product and sum without testing, and never identified the difference skill. Its own neurons helped the difference skill by .043 over the best training. Speed 10 on the product and sum paths, left in place through step 1500, cost .047 on the mean.

Relay error: result 12 was retyped with one wrong number and corrected in the next message; details in `toy/runs_set3/secret3-1/relay_errors.md`. The word-match flag on reply 11 is the word "fit" in printed labels, with no fitting in the code.

## Search: where can Opus's weight changes help? Written before any counted run (2026-11-03)

The new question: if Opus cannot replace training, can it augment it, and is that before, during or after training? Code: `toy/search.py`. Runs: `toy/search_runs/`.

**The question.** Find a way of changing weights that helps a network more than the best training we can run, and name when it works and on which kinds of task. A technique counts once a plain script can repeat it without Opus.

**When Opus edits.**
- Before: Opus gets the random starting weights and 64 examples per skill, edits once, then training runs 600 steps.
- During: Opus edits at steps 150, 300 and 450. First reading at step 600.
- After: plain training to step 1500, Opus edits once, then 150 more steps. First reading at step 1650.

Then training runs on alone to step 6000 in every mode.

**Kinds of task.** The same network and four skills throughout, each with its own hidden tables.
- Tables: each skill is a sum, product, larger, smaller or difference of hidden values per number, cut into 8 classes.
- Conditions: half of the numbers a follow one shape, half another.
- Partly wrong: tables, with a quarter of training answers wrong. Held-out answers are right.
- Modular: hidden values added, subtracted or combined bit by bit, wrapped at 16. Training is slow here, and the rule is short.
- Random: no rule. Chance is .125. This one has no answer to find, so it checks the bench for leaks.

**Comparison.** The best of 34 controls on this task: 15 learning rate and weight decay settings, 4 dropout settings, 5 fixed speeds, and 2 to 4 published edits through the same pauses (weight averaging, shrink and perturb, resetting quiet neurons, a straight-line step, scaled starting weights). The best row and step are read on held-out pairs, which favours training.

**Two gains, for one run.**
1. First reading: Opus at its first reading minus the best control at that same step.
2. Later peak: Opus's highest reading through step 6000 at its own speeds, minus the best control at any step from the first reading on. Both are read every 25 steps.

**Stage 1, screening.** One Opus run on one new task for each of 13 cells: four kinds with rules at three times each, plus random during training. A cell passes if either gain is at least +.03. The up-to-three cells with the highest gains go on. If random gains more than +.05 on either reading, stop and look for a leak.

**Stage 2, confirmation.** Five new tasks of the same kind (instances 1 to 5) for each cell that passed, controls first. The claim holds if the gain that passed is at least +.03 on average and above zero on at least 4 of 5. Screening alone proves nothing, because it tries 13 cells.

**Stage 3, technique.** For each cell that holds, write what Opus did as a plain script with no LLM, and test it on five more new tasks (instances 6 to 10) by the same rule. If the script also holds, that is a new technique. If it fails, the result needs the LLM, and I report that.

**If nothing passes**, I report that Opus found no technique in this search. Everything stays public, with no new bar chosen after seeing it.

**Rules for every run.** 16 replies, 20 EVALs, 5 code blocks per reply at 20 CPU seconds. No gradients or iterative fitting. Fresh Opus 5.5, three at a time at most. Each editor talks to the bench through one allowed command, so nothing is retyped by me, and a run counts only if an audit of its transcript finds no other call. Token cost is logged for each run.

**Prediction.** 1 or 2 of the 12 rule cells pass screening. Partly wrong tables, after training, is the most likely, because reading the rule could undo memorized wrong answers: first-reading gain 0 to +.10. Math tables during training: -.05 to +.05. Before training: every kind below the bar, with first-reading gain -.10 or worse, because 64 examples per skill say little. Modular: first-reading gain -.20 or worse in all three modes, because the network has learned little at those steps. Conditions: below the bar. Random: within +/-.03 of the best control.

**How this answers an ICML reviewer.**
| Reviewer's objection to the earlier work | What this does |
|---|---|
| Four runs on one kind of task | Five kinds, then five new tasks per claim |
| Best results chosen after the runs | This rule is public before the first counted run |
| Suits hand-written rules | Includes conditions, partly wrong answers, modular and no rule |
| Is it the LLM or its code? | Stage 3 writes the technique as a script and tests it without the LLM |
| Could published methods do this? | They run through the same pauses and count as controls |
| Cost is ignored | Tokens are logged per run |
| Relay errors | The editor's replies reach the bench directly, and the transcript is audited |

Limits known now: one network size, by choice, because bigger models are not the question; the editor reads weights and history but not training data, except in "before"; a fresh Opus is not an independent lab; the controls' best row is picked from held-out pairs, while Opus has only training pairs.

Closest work found (three extended searches, 2026-11-03): Sparks of Weight-Space Programming edits finished networks; Reinforcement Learning for Neural Model Editing (arXiv 2606.13461) has an RL policy scale or shift weights after training; Interactive Training (arXiv 2510.02297) proposes agents intervening in training but demonstrates log-based prompts; KBANN (Towell and Shavlik 1994) writes known rules into starting weights. None that I found has an LLM read weights and edit them during training against strong training controls. Three searches do not settle that.
### Added after run r03 passed screening, before any second-stage run

r03 (partly wrong answers, after training) passed the written rule on its first reading, .675 against .595, and on peak, .675 against .635. A reviewer's first objection is that those 34 controls hold no method built for wrong labels, so I added 14 runs in `toy/robust.py`: label smoothing, generalized cross entropy, bootstrapping, forward correction told the true noise rate, 8 outputs only, and mixup, each at the default setting and at lr .01 wd .5. This is a stricter reading added after seeing r03, so it does not replace the written rule. Both readings are reported for every `noisy` cell.

On instance 0 the best of the 14 at step 1650 is generalized cross entropy with .636, so the first-reading gain falls from +.080 to +.039. Its peak is .677 at step 3075, so the peak gain falls from +.041 to -.002. What is left of r03's lead is speed: the same accuracy about 1,400 steps sooner. Per skill at 1650, r03 has .244, 1.000, .458, 1.000 and that method has .244, .877, .500, .922, so r03 writes two skills exactly and is no better on the other two.

Limits of r03 under the rules. Its table reading uses two rounds of alternating majority votes over the network's own outputs, with no labels and no gradients. The task forbids iterative fitting of weights to a loss. This fits lookup tables to the network's outputs and then sets weights by hand from them, so I count it as allowed and disclose it as borderline. The output tables it reads include held-out inputs, with no held-out answers. Its 13 EVALs read training pairs only.

### First round finished: only "after training" passes. Written before any second-round run

Gain = Opus minus the best control at the same step. Peak gain = Opus's best through step 6000 minus the best control at any step from the first reading on. Bar: +.03.

| Kind | Before | During | After |
|---|---|---|---|
| tables | -.032 / -.032 | -.009 / -.059 | **+.093 / +.071** |
| conditions | -.022 / -.104 | -.057 / -.095 | -.076 / -.135 (last block crashed) |
| noisy | -.013 / -.053 | -.030 / -.084 | **+.080 / +.041** |
| modular | -.276 / -.589 | -.176 / -.476 | **+.147** / -.069 |

Random, during: -.016 / -.025, inside the leak limit. Tokens for 13 runs: about 1.5 million.

Against my written prediction: "1 or 2 of 12 pass" was too low, with 3 passing. "Noisy after most likely" held. "Before, every kind -.10 or worse" was wrong on three kinds. "Modular -.20 or worse in all modes" was wrong: modular after is the largest gain. "Conditions below bar" held. "Random within ±.03" held. "Tables during -.05 to +.05" held.

What the three passing runs have in common: Opus read a two-part rule off the network's own predicted table over all pairs, wrote it as hand-set neurons gated by skill, then froze most weights, so a rule with few numbers covers unseen pairs. r12 reports that earlier alternating row and column scans came close to the iterative-fitting ban, and that it dropped them from the weights and kept them as a diagnostic. r03 kept two alternating vote rounds. r11 also read its rules from the table. These go in the write-up as borderline.

Advancing to the second round, all three: tables after, noisy after, modular after. Each on instances 1-5, controls first, by the gain that passed: first reading for modular, and either for the other two as written. The claim holds if that gain averages at least +.03 and is above zero on at least 4 of 5.

Prediction for the second round: tables after holds, mean first-reading gain +.02 to +.08, positive on 3 or 4 of 5, so it is near the bar. Noisy after holds against the 34 controls, mean +.03 to +.08, and falls below +.03 against the wrong-label methods. Modular after is the most variable, mean -.05 to +.10, and fails the rule, because a run needs the network to have started on all four skills by step 1500. If reading a rule off the table is what helps, conditions-style kinds would fail for the reason they failed here: a rule with two cases is harder to read.

## Second round finished, and a test nobody planned. Written before any run on tasks 6-10

Second round, by the written rule (five new tasks, after training, first reading against the best of 34 controls at step 1650):

| Kind | Gains on tasks 1-5 | Mean | Above zero | Result |
|---|---|---|---|---|
| Wrapped numbers | +.274 +.283 +.256 +.270 +.194 | +.256 | 5 of 5 | Holds. Later-reading gains are +.036 +.041 +.043 +.025 +.022 |
| Partly wrong answers | -.118 +.111 +.058 +.093 +.006 | +.03005 | 4 of 5 | Passes by .00005. Against the wrong-label methods the same runs score +.100 +.020 -.192 +.055 -.059, mean -.015, so I do not treat it as a finding |
| Tables | +.064 +.011 +.025 -.009 +.037 | +.026 | 4 of 5 | Does not hold |

My second-round prediction said wrapped numbers would fail, with mean -.05 to +.10. That prediction was wrong. It said tables would hold with mean +.02 to +.08. The mean landed in that range, but tables did not hold.

Full-table check: on r16, r17, r21, r25 and r26 the saved networks give the true answer on every pair of the whole table, except 1 line of r16, and no code contains anything that reaches the answers.

While writing the plain script I added a comparison a reviewer would ask for: the same rule search run on the training answers, with no network and no training at all. It knows the rule families Opus found on the first tasks. On debug tasks 0-5 it scored 1.000 on wrapped numbers every time, .911 to .952 on tables, and .877 to .936 on partly wrong answers. Opus scored .878 to 1.000, .828 to .964 and .367 to .733 on the same kinds. Best training at any later step scored .947 to .978, .841 to .934 and .524 to .682. So reading the network may add nothing on these tasks. What may remain for Opus is finding the shape of the rule unaided, and that needs a separate test on rule shapes outside the script's list.

These debug numbers were read on held-out pairs, on tasks already used, while I fixed bugs in the script. They do not count. The script is now frozen.

Predictions for tasks 6-10, one run per setting, controls first:
- The script that reads the network: wrapped numbers holds, mean gain +.20 to +.28, 5 of 5 above zero. Partly wrong answers holds against the 34 controls, mean +.03 to +.12, and is roughly even with the wrong-label methods, mean -.03 to +.05. Tables is not required by the rule, and I expect mean +.02 to +.07.
- The search of training answers: wrapped numbers 1.000 on all five, tables .90 to .95, partly wrong answers .87 to .94.
- If the search of training answers beats the network script on most tasks, then the network adds nothing on these tasks, and the paper must say so plainly.

---

## Well-trained networks, and the test on rule shapes nobody wrote a script for

Well-trained networks read at step 6000 on tasks 11-15, no answers, one run per setting. The rule beat the network on all 15 tasks at the best settings.

| Kind | Best setting | Rule | Network | Gains |
|---|---|---|---|---|
| Wrapped numbers | lr .01, wd .1, dropout .1 | 1.000 | .956 | +.039 +.068 +.062 +.028 +.025 |
| Tables | same | .930 | .887 | +.068 +.017 +.082 +.019 +.030 |
| Partly wrong | dropout .3 | .718 | .595 | +.141 +.080 +.104 +.153 +.138 |

The script is told the list of rule families, so this says nothing about Opus. A new rival, `cocluster.py`, is told no list: it groups rows and columns that give the same answers, keeps one answer per pair of groups, and picks group counts on a tenth of its evidence set aside. Debugged on instance 0 of the old kinds only.

New kind `unseen` (added without changing any old task, checked by hash over 16 instances of 5 kinds). Each task draws 4 of these 8 with hidden values per number: (f*h)%16; (f+h)%13; f&h; f|h; 8 bins of (f+h)%1 on real values; 8 bins of (f if f>h else 1-h) on real values; ((f*h)%17)-1 on 1..16; (f+h)//2. Still two-part, so a reviewer's objection that a rule exists to read still stands. Bench and rule frozen in this commit, before any `unseen` run.

Tasks 0-4, in this order: 34 controls, script and list search, cocluster, then 5 Opus runs after training, at most 3 at once.

Rule, same as before: mean gain at least +.03 and above zero on at least 4 of 5, at the first reading and through step 6000, against the best of the 34 controls. Two further comparisons on the same tasks, reported whatever they show: Opus at step 1650 against cocluster on the network at step 1500 (same evidence, no EVALs), and against cocluster on the training answers.

Predictions:
1. Search with the family list on answers scores below cocluster on answers by .05 or more, because six of eight shapes are outside its list.
2. Cocluster on answers scores .85 or more.
3. Opus passes the first-reading rule at +.10 or more, and fails the through-6000 rule.
4. Opus beats cocluster on the network at step 1500 by .03 or more on at least 3 of 5.
5. Opus scores below cocluster on the training answers.

If 4 fails, reading a network needs no LLM on these tasks, and the paper's claim becomes the scripted one.

Literature check on reading a rule from a network with no data, 3 searches on 2026-11-03. This part is not new as a kind of result. Rule extraction is old: TREPAN (Craven and Shavlik 1996) queries a network as an oracle, and REFNE (Zhou, Jiang and Chen 2003) reports rules extracted from a network ensemble that generalise better than the ensemble. On wrapped numbers with wrong labels, Doshi et al. 2023 (arXiv 2310.13061) show regularised networks recover true labels by themselves. Do not call the data-gone result new. What may still be new is narrow: comparing when in training an LLM's weight edit helps, against a family-free rival and tuned training, on tasks it was not told about. Sparks of Weight-Space Programming (MIT) and the weightspace repo do LLM edits on finished networks, without this comparison. The searches do not settle it.

---

## Opus on rule shapes no script knows, then with the answers and no network

Opus with the network after training, tasks 0-4 of `unseen`, runs r29-r33, all audits clean, no answers reached in code, one Opus per task.

| Task | Opus | Best training at step 1650 | Gain | Opus peak | Best training at any step | Gain | Cocluster on network, step 1500 | Cocluster on answers | List search on answers |
|---|---|---|---|---|---|---|---|---|---|
| 0 | .789 | .770 | +.019 | .789 | .912 | -.123 | .718 | .930 | .570 |
| 1 | .970 | .853 | +.117 | .977 | .928 | +.049 | .859 | .902 | .493 |
| 2 | .769 | .830 | -.060 | .875 | .896 | -.021 | .770 | .909 | .589 |
| 3 | .981 | .874 | +.107 | .982 | .958 | +.024 | .822 | .968 | .486 |
| 4 | .990 | .848 | +.142 | .990 | .965 | +.026 | .857 | .961 | .481 |
| Mean | .900 | .835 | +.065 | .923 | .932 | -.009 | .805 | .934 | .524 |

First-reading rule passes: +.065, 4 of 5. Through-6000 rule fails: -.009, 3 of 5. Each loss comes from one skill the network had not learned by step 1500. Skill 0 of task 2 was called "no rule" and scored .175; skill 0 of task 0 scored .399.

Predictions: (1) list search below cocluster on answers by .05 or more: right, by .410. (2) Cocluster on answers .85 or more: right, .934. (3) First-reading pass at +.10 or more: wrong, +.065. Fails through 6000: right. (4) Opus above cocluster on the network at step 1500 by .03 or more on at least 3 of 5: right, +.071 +.111 -.001 +.159 +.133. So on this test the LLM adds something over the scripted reading of the same network. (5) Opus below cocluster on answers: right on the mean, .900 vs .934, but above it on 3 of 5.

Tuned training catches cocluster on answers by step 6000: .932 vs .934.

Next, same tasks, fresh Opus, runs r34-r38, new mode `answers`. Random starting weights, nothing trained, every training answer in a file, one chance to write weights, then 150 steps before the first reading. This asks whether Opus can replace training on these tasks, and whether the network was helping it. Bench frozen in this commit.

Rule: peak through step 6000 against best training at any step, mean gain at least +.03 and above zero on at least 4 of 5.

Predictions:
6. Opus with answers scores above Opus with the network by .03 or more on the mean, because the skills it lost were ones the network had not learned yet.
7. Opus with answers beats cocluster on answers by .02 or more on the mean and on at least 3 of 5.
8. Opus with answers passes the rule.

If 8 fails, Opus cannot replace training on these tasks. If 6 fails, the network is worth more to Opus than every answer.

---

## With the answers and no network, Opus beat all training on all 5 tasks

Runs r34-r38, `unseen` tasks 0-4, audits clean, no answers or held-out pairs reached in code. Random starting weights, no training before the edit. Opus wrote the weights by hand, froze all or nearly all of them, and held-out pairs were never shown. Whole-table check run on all five.

| Task | Opus first reading | Opus peak | Best training at any step | Gain | Opus with network | Cocluster on answers |
|---|---|---|---|---|---|---|
| 0 | .958 | .964 | .912 | +.052 | .789 | .930 |
| 1 | .975 | .975 | .928 | +.046 | .970 | .902 |
| 2 | .984 | .984 | .896 | +.088 | .769 | .909 |
| 3 | .992 | .992 | .958 | +.034 | .981 | .968 |
| 4 | .994 | .994 | .965 | +.030 | .990 | .961 |
| Mean | .981 | .982 | .932 | +.050 | .900 | .934 |

Rule passes, +.050, 5 of 5. Best training is the best of 34 settings at its best step read on held-out pairs, which favours training.

Predictions: (6) above Opus with the network by .03 or more: right, +.081. (7) Above cocluster on answers by .02 or more on the mean and on at least 3 of 5: right, +.047 and 5 of 5. (8) Passes the rule: right.

What a reviewer will say, and it is correct: this is program synthesis, and the weights are only where the program is stored. Nothing here needs a neural network. The tasks are two-part by construction and hold no noise, so a rule exists to find. Five tasks, one Opus per task. Three editors flagged for themselves that they searched discrete codes with repeated vote passes. That is the same kind of search as cocluster.py and read_rule.py, it touches no weights, and it is disclosed here rather than voided.

Next, same mode, new tasks 16-18 not used by anything: tables, conditions, partly wrong, 3 each; wrapped numbers 16-17; random tables 16-17 as the negative control. 13 runs, at most 3 at once. Same rule, per kind: mean gain at least +.03 and above zero on every task.

Predictions:
9. Tables passes, +.03 to +.08.
10. Conditions passes, +.03 to +.10. Earlier conditions runs were the worst kind, so this is the one most likely to fail.
11. Partly wrong passes at +.10 or more, because a rule ignores wrong answers and training cannot.
12. Wrapped numbers scores .99 or more and fails the rule by ceiling, gain under +.03.
13. Random tables: within .02 of the best training, and Opus says it found no rule. If Opus claims a rule on random tables, the method cannot tell when it is wrong, and that goes in the abstract.

---

## With the answers, new tasks of every other kind: all 16 tasks with a rule beat all training, and Opus said no rule on 2 random tables

Runs r39-r51, tasks 16-18 (wrapped and random 16-17). Audits clean, no answers or held-out pairs reached in code, whole-table check run on all 13. r41's flagged `__import__` only made a folder, and r44's `randint` only started a search of codes.

| Kind | Task | First reading | Peak | Best training at any step | Gain, peak | Cocluster on answers | List search on answers |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tables | 16 | .886 | .899 | .876 | +.022 | .849 | .899 |
| Tables | 17 | .965 | .980 | .931 | +.049 | .917 | .951 |
| Tables | 18 | .973 | .978 | .929 | +.048 | .918 | .952 |
| Conditions | 16 | .900 | .900 | .791 | +.109 | .788 | .545 |
| Conditions | 17 | .901 | .913 | .762 | +.151 | .780 | .504 |
| Conditions | 18 | .847 | .864 | .705 | +.159 | .759 | .494 |
| Partly wrong | 16 | .661 | .661 | .556 | +.105 | .774 | .898 |
| Partly wrong | 17 | .873 | .873 | .660 | +.212 | .811 | .932 |
| Partly wrong | 18 | .842 | .842 | .524 | +.318 | .724 | .876 |
| Wrapped | 16 | 1.000 | 1.000 | .960 | +.040 | .952 | 1.000 |
| Wrapped | 17 | 1.000 | 1.000 | .974 | +.026 | .950 | 1.000 |
| Random | 16 | .121 | .133 | .151 | -.018 | .113 | .127 |
| Random | 17 | .126 | .144 | .156 | -.012 | .129 | .122 |

Rule passes on all four kinds with a rule, per kind: tables +.040, conditions +.140, partly wrong +.212, wrapped +.033. With r34-r38 that is 16 of 16, and the first reading alone is above best training on all 16, mean +.089, standard error .021, smallest +.010. Both random runs said they found no rule and only silenced unused outputs.

Predictions: (9) tables passes, +.03 to +.08: right, +.040. (10) Conditions passes, +.03 to +.10: passes, above the range, +.140. (11) Partly wrong +.10 or more: right, +.212. (12) Wrapped .99 or more and fails by ceiling: score right, gain passed at +.033, so wrong by .003. (13) Random within .02 and no rule claimed: right at the peak, -.015. The first reading was -.030.

A list search on the answers beat Opus on partly wrong, .902 vs .792, because the list holds the right shapes. It tied on tables and wrapped, and lost on rule shapes outside the list, .524 vs .981, and on conditions, .514 vs .883. Cocluster on answers lost to Opus on 15 of 16; r41 lost to it.

The search answer: before and during training lose. After training jumps ahead at that step on 4 of 5 kinds, and training never reached the first reading on 14 of 24 tasks, but the mean gap at the best reading stays under .02. Instead of training, with all the answers, works best, and it is program synthesis with weights as the storage.

## Correction: first readings were not "no training"

The README and the post called the answers runs "no training". Reading the ledgers showed that 150 steps ran after the edit at speeds Opus chose, and 12 of 18 runs left some speed above 0. `no_training.py` replays each run's code and reads held-out pairs right after the edit. 17 of 18 replays reproduce the recorded first reading exactly; r47 gives .849 against .842.

With no training at all: 16 rule tasks .913 vs .833, +.080, standard error .023, 14 of 16 above best training (r39 -.029, r45 -.003). Tables tie at .912. 25% wrong .763 vs .580. Against co-clustering 14 of 16, and on 25% wrong .763 vs .770.

The earlier line "training after its change added at most .018" was also wrong: r41 gained .093 and r45 gained .047 in those steps, and r46 lost .031.
