<h1 align="center">HumanChoice</h1>

<p align="center">
  <strong>Learning when to ask, what to ask, and how to use the answer</strong>
</p>

<p align="center">
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="requirements.txt"><img alt="Python" src="https://img.shields.io/badge/Language-Python-blue.svg"></a>
</p>

<p align="center">
  <img src="assets/humanchoice-overview-ei.png" alt="HumanChoice picks a question, asks now or continues the task, and chooses an action after a person replies. Training uses the results of completed tasks.">
</p>

HumanChoice lets an agent ask for help with a specific decision while it continues to handle the rest of the task. It learns which question to ask, when to ask it, and how to use the answer.

## How it works

| Module | What it decides |
| --- | --- |
| **Question Selector · Q** | Which question is most likely to help the agent complete the task. |
| **Ask Gate · G** | Whether to ask now or continue and keep the chance to ask later. |
| **Action Selector · A** | Whether to use the original action or a revised action based on the answer. |

A fixed language model writes the questions and proposes the actions. The agent keeps its original action available, so a person's answer does not automatically replace it.

During **Expert Iteration**, the training process runs alternative choices and checks the final results. A learns which action works better. Q learns which question helps under A's current choices. G then learns whether asking that question now is better than waiting. The updated agent runs more training tasks to collect new examples.

## Quick start

Run these commands from the project folder. They install the dependencies and check the saved result files without calling a model.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_results.py
~~~

See [how to run the experiments](docs/reproduction.md) for the available scripts and data setup. The [saved results](experiments/results/) contain summary scores from the recorded runs.

## Supplementary experiments

The release includes the full lightweight replay modules, the portable public-benchmark runtime, and the public-task seed-42 module weights. Raw medical records, responses, API configuration, and conversation caches are excluded.

| Script | Experiment |
| --- | --- |
| `experiments/run_replay_study.py` | All eight module combinations, loss removal, refreshed targets, cost coefficients, and candidate-pool availability. |
| `experiments/run_budget_study.py` | Learned Gate, uncertainty, and random request ranking at equal request counts. |
| `experiments/run_online_ablation.py` | Matched online full, no-G, no-Q, no-A, and autonomous episodes. |
| `experiments/run_epoch_sweep.py` | Complete epoch trajectories and per-metric best-observed envelopes, with the selecting epoch recorded for every metric. |

[Supplementary results](experiments/results/supplementary/) include fixed-epoch results separately from retrospective epoch selection. Per-metric envelopes can use different epochs; they are descriptive test-set maxima, rather than the joint performance of one checkpoint. The [checkpoint-selection record](docs/checkpoint-selection.md) gives the selected values and their epochs. The [runtime notes](docs/runtime.md) explain configuration, dependencies, and the supplied weights.

## Citation

~~~bibtex
@misc{humanchoice,
  title = {HumanChoice: Learning to Ask and Act through Expert Iteration},
  note  = {Research code and saved experiment results}
}
~~~
