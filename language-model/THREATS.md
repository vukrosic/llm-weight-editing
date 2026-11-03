# Threats

Each cheat, what stops it and which check in `attacks.py` shows that. V1 to V8 are the flaw types from BenchJack (arXiv 2605.12673). `attacks.py` holds 43 of 43 at smoke size on all four families (2026-11-03), after the changes below; the numbers from real sizes in this file are older and must be rerun before freezing.

"Held" counts stated limits as held, so read the three classes separately.

## Blocked

| Cheat | Control | Type |
|---|---|---|
| Read the clean twin, test rows or key | The sandbox denies the sealed folder; those files are outside the work folder | V1, V2 |
| Read the run's log or the host's files | The sandbox allows only the work folder, the interpreter and system libraries | V1, V8 |
| Use the network, start a program, import torch | The sandbox has no network and no new processes; the kernel has only numpy and scipy | V8 |
| Read secrets from the environment | The kernel starts with a clean environment | V8 |
| Link an edit to the twin or another file | The host refuses anything that is not a plain file and never follows links | V3, V7 |
| Run code on the host with a pickled array | The host loads with `allow_pickle=False` | V3 |
| NaN, wrong shape, unknown tensor, output layer | The host checks each tensor against the allowed list | V6, V7 |
| Multiply a whole tensor by a constant | The host refuses a constant ratio | V6 |
| Go over the pass, run or CPU budget | The broker counts passes outside the sandbox; the kernel has CPU limits | V6, V8 |
| Run the twin or read test rows through the forward tool | The tool knows only `original` and `edited`, and paths must stay in the work folder | V1, V2 |
| Send code as a request | Requests are parsed as JSON with a list of known keys | V3 |
| Zero the given weights to change what "original" means | The host uses its own copy | V7 |
| Score a run twice to pick the best | One score per run | V6 |
| Change the edit after freezing | The hash is checked before scoring | V7 |
| Delete a line from the log | The hash chain breaks | V7 |
| Run again on a test model until lucky | One run per editor per test model; `score.py` refuses a second, shown in `smoke.py` | V6 |
| Drop a bad test run by never scoring it | `beat.py` lists runs on test models with no score and calls the verdict invalid, shown in `smoke.py` | V6 |
| Change the bench after seeing a test result | Test models score only under the frozen hash of the bench code, including the relay | V7 |
| Read the defect family from the model's name | Test model names are a keyed hash | V2 |
| Fake or change a relayed LLM reply | `relay.py audit` compares every reply and message with the agent's transcript; `score.py` refuses a failed audit | V7 |
| An LLM agent uses a tool during a run | The audit fails on any tool call in the transcript | V1 |
| Stop writing so nothing repeats | End-of-text share and test loss both fail: good 0.00, loss 11.56 | V5, V6 |
| Destroy the blocks so nothing repeats | Test loss fails: good 0.27, loss 5.32 | V5, V6 |
| A long repeat that a period check misses | Repeated-phrase share catches it | V5 |
| Test text leaking through practice text | Test rows sharing any 16-token run with practice are dropped; zero remain | V2 |
| Weights carrying training history | The subject file holds weights only | V2 |

There is no LLM judge, so V4 does not apply. The scorer is plain code that reads frozen arrays.

## Flagged, not blocked

| Cheat | What happens | Why it stays open |
|---|---|---|
| Training by hand in numpy | The word scan flags it when it uses words like `grad`; CPU is capped at 900 s | Numpy can do any arithmetic |
| The same training with innocent names | **The word scan misses it** (shown in `attacks.py`) | A person must read every run's code; the log keeps it all |
| Fit the output layer from captured activations, then run the model in the kernel | Flagged on `lstsq` and similar, and the same renaming beats this flag | The tool returns activations and top-5 tokens, which are enough to fit an approximation |
| Multiply a tensor and change one weight | Accepted; it shows up in the changed-weights count | The check only looks for one constant ratio across the whole tensor |

So the rule against hidden training rests on review, not on the sandbox. A published score should say a person read the log.

## Measured and reported

Passes, code runs, CPU seconds, turns, refused requests, share of weights changed and flags are all in each row of `results.jsonl`.

## Limits

- The hash chains have no secret. They catch partial changes, but someone with write access to the host can rebuild a matching log. A public result needs its logs published and, ideally, the scoring rerun by someone else.
- The macOS sandbox lets the kernel see that a sealed file exists and its size, not its contents.
- `sandbox-exec` is deprecated by Apple, and the Linux `bwrap` path has not been run. `attacks.py` must pass on any machine before its scores are trusted.
- `FROZEN.json` is a local file with no secret, so a public result needs it committed before the test runs, with git history published.
- The relay audit has been tested on fake transcripts and the pilot's transcript; no counted run has gone through it.
- An LLM knows from its training data what these four defects look like, and the symptom sentence names the defect's effect. A claim holds for these families only.
