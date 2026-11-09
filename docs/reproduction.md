# Running the experiments

Run all commands from the project folder. Install the packages in `requirements.txt` first.

## Local checks

These commands work without medical records or model credentials:

~~~bash
python experiments/anchor2_assert.py
python scripts/check_results.py
python figures/draw_intuition.py
python figures/legacy/fig1_core_finding_code.py
~~~

- `anchor2_assert.py` checks that the scoring functions match examples with known answers.
- `check_results.py` checks the arithmetic in the saved result summaries.
- `draw_intuition.py` draws the earlier schematic.
- `fig1_core_finding_code.py` draws the paper's results plot from saved summaries.

These checks run locally. Generated files go into the ignored `runs/` folder. The illustrated README image is a separate PNG in `assets/`.

## Available scripts

| Script | What you need | What it does |
| --- | --- | --- |
| `experiments/anchor3_split_assert.py` | Authorized local training and test records | Checks patient separation, excluded fields, and missing values. Saves the patient split for later runs. Its expected counts are specific to the original dataset. |
| `experiments/anchor1_step2.py` | Local records and an approved model service | Runs the baseline with class definitions and examples. The saved selected setting uses three examples per class. |
| `experiments/autonomous_agent.py` | Local records and an approved model service | Runs the original agent without asking for help and saves each prediction locally. |
| `experiments/pipeline.py` | Local records, a saved split, and an approved model service | Checks the complete workflow on a small batch using model-generated replies. It uses simplified training labels and repeats training on the same labels. |
| `experiments/train_modules.py` | Feature files from a local run | Trains the small networks for A, Q, and G. |
| `experiments/doctor_pilot/` | Local records or a question table | Selects records, prepares a blank questionnaire, and formats the answer key. The script used to fill the original question table was not saved. |
| `experiments/data/` | Local records | Summarizes fields and compares a classifier using all fields with one that excludes later outcomes. |
| `experiments/cb_native_run.sh` and `experiments/cb_aggregate.py` | The [original ClarifyBench project](https://github.com/MananSuri27/ClarifyBench) and its dependencies | Runs the baseline through ClarifyBench's own interface and combines the scores. The shell script needs Bash and GNU timeout. |

## Medical records

Set `HUMANCHOICE_DATA_DIR` to an authorized local folder containing `train.jsonl` and `test.jsonl`. Each line must contain a JSON record with the structured fields stored as text in `question` and the record label in `answer`.

The code for selecting the original patient group was not saved. Access to MIMIC-IV therefore does not, by itself, recreate the exact dataset. Set `HUMANCHOICE_RUN_DIR` to a private folder for generated files.

Check the data before running the model:

~~~bash
python experiments/anchor3_split_assert.py
~~~

Calls that send medical records to a model service are disabled by default. Before enabling them, follow [PhysioNet's guidance for using records with model services](https://physionet.org/news/post/llm-responsible-use/) and your institution's requirements.

For an approved service, set `OPENAI_BASE_URL` and `OPENAI_API_KEY` in your environment, then set `HUMANCHOICE_ALLOW_RECORD_API=1`. Use `HUMANCHOICE_MODEL` to change the model name if needed. Keep generated records and responses out of Git.

~~~bash
python experiments/anchor1_step2.py --split dev --fewshot 3 --effort high
python experiments/anchor1_step2.py --split test --fewshot 3 --effort high
python experiments/pipeline.py --run_id smoke_batch --n_batch 20 --workers 2
~~~

The `--fewshot` value is the number of examples per class. Development runs help choose a setting; test runs evaluate the chosen setting.

## ClarifyBench

Install and configure ClarifyBench first. Set `CLARIFYBENCH_ROOT` to that project's local folder, then run:

~~~bash
bash experiments/cb_native_run.sh A 4 420
bash experiments/cb_native_run.sh E 4 420
bash experiments/cb_native_run.sh I 4 420
python experiments/cb_aggregate.py
~~~

The first argument chooses the benchmark subset. The next arguments set the number of parallel jobs and the time limit per task in seconds. Completed tasks are reused when you restart a run.

## Saved results and complete reruns

The repository includes the experiment and figure scripts found in the project copy used for this release. Some saved results came from separate scripts that were not in that copy: the original HiL-Bench SQL runner, the full medical prediction cache, the full Expert Iteration and transfer drivers, and some benchmark adapters.

The [experiment summary](results.md) explains which runs produced the saved scores. The small-batch workflow checks that the main stages work together; it does not recreate every reported experiment.

The model prompts and dataset field names remain exactly as used in the recorded runs.

## Supplementary release

The later release includes the NumPy replay modules, the full portable public-benchmark controller in `experiments/e1_methods/e1`, and public-task seed-42 module weights. It adds complete module combinations, loss and target-refresh ablations, equal-request-budget rankings, an online module pilot, and a complete 1–30 epoch search. See [checkpoint selection](checkpoint-selection.md), [runtime setup](runtime.md), and the [aggregate results](../experiments/results/supplementary/). The earlier missing-runner statement above describes the original project copy, before this supplement.

Run `python scripts/check_supplementary.py` to verify every released metric selection against the complete epoch trajectory and the fixed-30-epoch reference.
