# Reproduction

## Offline checks

From the repository root, install the requirements and run:

~~~bash
python experiments/anchor2_assert.py
python scripts/check_results.py
python figures/draw_intuition.py
python figures/legacy/fig1_core_finding_code.py
~~~

The first two commands check metric invariants and the arithmetic of the published aggregates. The last two render the conceptual PNG and the archived paper-results plot. All four are offline. Generated outputs go under the ignored run directory, except the overview PNG under `assets/`.

## Experiment entry points

| Entry point | Inputs | Scope |
| --- | --- | --- |
| `experiments/anchor3_split_assert.py` | Locally authorized train/test JSONL | Audits the archived cohort and writes a local split map. Expected counts reflect this cohort. |
| `experiments/anchor1_step2.py` | Local cohort and approved compatible endpoint | Development or test few-shot baseline. The archived selected recipe uses three demonstrations per class. |
| `experiments/autonomous_agent.py` | Local cohort and approved compatible endpoint | Original autonomous baseline and per-record local output. |
| `experiments/pipeline.py` | Local cohort, split map, approved endpoint | S0–S8 minibatch validation with simulated responses. Its single-question Q targets and unchanged replay targets are prototype checks, not full pairwise/three-round paper reproduction. |
| `experiments/train_modules.py` | Local run directory with generated module feature arrays | Trains the three archived small MLPs on their assigned training groups. |
| `experiments/doctor_pilot/` | Local records or question table | Candidate selection, questionnaire skeleton, and formatting. The historical fill runner is not in the archived source. |
| `experiments/data/` | Local cohort | Field profiles and the supervised leakage probe. |
| `experiments/cb_native_run.sh`, `experiments/cb_aggregate.py` | [Upstream ClarifyBench checkout](https://github.com/MananSuri27/ClarifyBench) and its dependencies | Native autonomous evaluation and aggregation. The shell wrapper needs Bash and GNU timeout. |

### Restricted medical inputs

Set `HUMANCHOICE_DATA_DIR` to your authorized local directory containing `train.jsonl` and `test.jsonl`. The reader expects one record per line, with structured fields serialized in `question` and a cohort label in `answer`. The cohort extraction recipe is not in the archived code. Obtaining MIMIC-IV access does not itself recreate the exact cohort. Use a private `HUMANCHOICE_RUN_DIR` for generated files.

~~~bash
python experiments/anchor3_split_assert.py
~~~

The live medical runners are disabled by default. Before enabling them, follow [PhysioNet's data-use and service guidance](https://physionet.org/news/post/llm-responsible-use/) and your institutional requirements. For an approved setup, supply `OPENAI_BASE_URL` and `OPENAI_API_KEY` at runtime, then opt in with `HUMANCHOICE_ALLOW_RECORD_API=1`. No endpoint or credential is supplied. The fixed model name can be overridden with `HUMANCHOICE_MODEL`. Never commit generated record-level artifacts.

~~~bash
python experiments/anchor1_step2.py --split dev --fewshot 3 --effort high
python experiments/anchor1_step2.py --split test --fewshot 3 --effort high
python experiments/pipeline.py --run_id smoke_batch --n_batch 20 --workers 2
~~~

### Public benchmark wrapper

Configure the upstream project first. Run from this repository's root with `CLARIFYBENCH_ROOT` set to your upstream checkout.

~~~bash
bash experiments/cb_native_run.sh A 4 420
bash experiments/cb_native_run.sh E 4 420
bash experiments/cb_native_run.sh I 4 420
python experiments/cb_aggregate.py
~~~

## Archived summaries versus full reruns

All experiment and figure source files found in the project snapshot were reviewed for release. Some E1–E8 summaries point to job-specific scripts or artifacts absent from the archived tree. In particular, the historical HiL-Bench SQL runner, the full medical cache, the dedicated three-round Expert Iteration/transfer drivers, and some benchmark adapters were unavailable. Their reviewed aggregate outputs remain in the [experiment ledger](results.md); the repository does not claim end-to-end reproduction of those runs.
