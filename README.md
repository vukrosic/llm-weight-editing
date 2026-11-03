# Environment for researching how an LLM can edit neural network weights during training

> [!IMPORTANT]
> We are using this environment to write a paper for ICML 2027. It was built in [AI Research School](https://www.skool.com/become-ai-researcher-2669). The licence is MIT: you can use it for your own research without asking and without adding anyone as a co-author.

Training changes a network's weights a little at a time. This environment asks what happens when an LLM stops training, reads the weights, and writes new numbers into them by hand.

**So far an LLM (Claude Opus 5.5) can sometimes work out what a half-trained network has learned and write it into the weights. In 2 of 3 runs that network later generalized better than any training we tried on the same network. The third run kept no change and ended below training. At the moment its edits stopped, the LLM beat our best simple training setting in only 1 of 3 runs.**

These are single runs on a toy network of 8,751 weights, and we chose the comparison after seeing them. A fourth run on new tasks is testing it against a [bar written before it started](#a-fourth-run-tests-this-against-a-bar-written-first).

![Accuracy on pairs never trained on, against training steps](charts/curves.png)

---

## Pause training, let the LLM change the weights, then let training go on

![The loop: train 150 steps, pause, Opus reads and changes the network, repeat until step 600, then training goes on alone](charts/loop.png)

The network learns four skills. Each skill takes two numbers, **a** and **b**, and gives a class from 0 to 7. Each skill follows a hidden table, built from a sum, a product, a larger or smaller of two, or a difference, of two hidden per-number values.

Every 150 training steps up to step 600, the LLM may change the network. It can write **new weight numbers**, or set **speeds**, which multiply how far each weight moves per training step. It computes numbers with its own short Python, which may not use gradients or fit weights to a loss.

The LLM is never told the tables, never sees the training examples, and never sees **held-out pairs**, the 30% of (a, b) pairs the network does not train on. Accuracy on those pairs is read once, at the end. Every number below is that accuracy, averaged over the four skills.

---

## Results

### Do not expect a win when the edits stop: the LLM beat fixed speeds in 1 of 3 runs

The comparison is the best of 14 fixed speeds. Each multiplies every weight's step by one number, from step 0 or from step 150. The search for the first run ran after that run. The bar for the second and third runs was written before they started.

| Step 600 | Plain training | Best fixed speed | Opus | Gain |
|---|---|---|---|---|
| First run, tables 1 | .579 | .803 | **.847** | +.044 |
| Second run, tables 2 | .542 | **.776** | .760 | -.016 |
| Third run, tables 2 | .542 | **.776** | .764 | -.012 |

> **At step 600 the first result did not repeat.**

### Compare after more training: networks with kept hand-written weights ended above all training

Each finished run was trained further with no LLM. We then compared it with the strongest training we tried, 27 settings of learning rate, weight decay and dropout. We chose this comparison after seeing the runs, so read it as a hypothesis.

| Highest reading through step 6000 unless noted | Plain | Strongest training, same size | Opus, its own speeds | Opus, all speeds 1 |
|---|---|---|---|---|
| First run | .779 | .852 (step 10800) | **.901** | .899 |
| Second run, no weight change kept | .781 | **.844** | .760 | .762 |
| Third run | .781 | .844 | .864 | **.902** |

> **Training here is cheap: 6000 steps take about 4 seconds, and one LLM reply takes far longer. The result is about where the network ends up, not about saved time.**

### Do not freeze neurons for a skill the LLM cannot read

In the third run Opus read three skills correctly, wrote weights for them and froze them. It could not read the fourth, which then had few neurons to train on.

| Third run, step 600 | Opus | Best of 24 training controls | Gap |
|---|---|---|---|
| Smaller of two values | .925 | .997 | -.072 |
| Sum | **.919** | .802 | +.117 |
| Difference | .338 | .63 | -.292 |
| Product | **.873** | .795 | +.078 |

> **Writing weights won on the skills it read and lost more than that on the one it did not.**

### A fourth run tests this against a bar written first

New hidden tables, 30 replies, and the goal told to the editor is accuracy at step 1500. Every baseline was recorded before it started. Scoring uses the mean at step 1500.

| Bar | Opus beats it | Tie | Opus loses |
|---|---|---|---|
| Strongest training at any step, .856 | above .886 | .826 to .886 | below .826 |
| Best training at step 1500, .823 | above .853 | .793 to .853 | below .793 |

Our prediction: mean .82 to .90, most likely about .86. The result is **pending**, and it will be added here whichever way it goes.

---

## Where it stops working

> [!WARNING]
> - **One run per setting** on a toy with 8,751 weights. Nothing here has been tried on a language model. Neighbouring fixed speeds differ by up to .03.
> - **The task family suits hand-writing.** Every table is built from two hidden per-number values, a shape the LLM can guess. Tasks with no such shape may give nothing.
> - **The step 1500 and step 6000 comparisons are post hoc** for the first three runs. The strongest training row was read on held-out pairs, which favours training.
> - **The LLM's short Python does real work.** It reads the network's predictions and calculates the weights. It uses no gradients and no labels, but it does use CPU.
> - **Nobody outside has reviewed this.** Claude Opus 5.5 built the environment, ran the runs and wrote this text. A separate fresh Opus plays each editor.

---

## Is this new?

Searches did not settle this. We found no work where an LLM edits weights during training, and no comparison of hand-written and trained weights on held-out data like this one.

| Work | LLM writes weights | During training | Compared with |
|---|---|---|---|
| [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi et al., 2026, blog post) | Yes | No, finished or new networks | Trained networks, base models |
| [ACDC](https://arxiv.org/abs/2304.14997) (Conmy et al., 2023) | No, finds circuits automatically | No | Other circuit methods |
| [Weight-sparse transformers](https://arxiv.org/abs/2511.13653) (2025) | No | No | Dense networks |
| This repository | Yes | **Yes** | **27 training settings, 14 fixed speeds** |

The Sparks post is the starting point. It reports an LLM writing a Breakout policy that beats a network trained with PPO, and editing a small language model to raise its MATH-500 score from 7.9% to 48.1%. It does not report edits during training or a comparison with further training.

---

## Setup

```bash
pip install -r requirements.txt
```
```bash
python3 -m venv toy/.kernel && toy/.kernel/bin/pip install numpy scipy
```
```bash
cd toy && TOY_RULES=secret python3 loop.py test
```

The sandbox for the LLM's Python runs on macOS. The Linux path exists and has not been tested. To run an editor, and for every number and command, see [RESULTS.md](RESULTS.md).

- [RESULTS.md](RESULTS.md): every table, the commands and how to play an editor.
- [NOTES.md](NOTES.md): the working log, with the bars and predictions written before each run, including the ones that failed.
- [language-model/](language-model/README.md): a second bench for repairing planted defects in a small language model. It has no counted LLM result yet.
- [chart.py](chart.py) draws every image from the result files.
