# Experiment summary

This page explains the saved results. The numbers come from recorded runs, not new experiments. A model answering a question as a simulated person is labeled separately from a real participant.

The medical task predicts record labels. These labels are not independently verified clinical stages.

## Baseline checks

The saved checks passed [31 scoring tests](../experiments/results/anchor2_result.json) and [27 data and patient-split tests](../experiments/results/anchor3_split_result.json).

On the test set of 270 records, the original prompt reached 28.15% accuracy. The [prompt selected on the development set](../experiments/results/step2_test_v1_fs3_high.json), which included examples, reached 45.56% accuracy and 43.47% macro-F1. Always predicting the most common class would reach 60.74% accuracy.

Macro-F1 gives each class equal weight. It is useful here because the record labels are unevenly distributed.

## General tasks

| Run | What was measured | Saved result |
| --- | --- | --- |
| [ClarifyBench baseline](../experiments/results/clarifybench_native_summary.json) | The agent used the benchmark's original tools and scoring. | 591 of 604 tasks had an evaluation result. |
| [ClarifyBench method comparison](../experiments/results/E1_aggregate_FINAL.md) | A separate comparison targeted 370 tasks per method. | Completed scores were available for 303 baseline tasks, 203 ask-human tasks, and 131 CLAM-API tasks. Completed matched tasks showed no net rescue on strict task success. |
| [HiL-Bench SQL](../experiments/results/hilsql_native_summary.json) | The agent tried 13 financial-database tasks without asking questions. | It solved none of the tasks. |
| [DialOp planning](../experiments/results/E1_dialop.json) | The program selected plans from masked reward tables while the simulator revealed information. | Planning score rose from 0.7497 to 0.9034. This run used neither model calls nor measured human response times. |

The ClarifyBench method comparison was incomplete. Read scores together with their completed task counts.

## Medical tasks and question decisions

| Experiment | Setup | Saved result |
| --- | --- | --- |
| [E2 · Medical review](../experiments/results/E2_medical.json) | 270 test records; the same model simulated the person answering. | HumanChoice reached 46.30% accuracy, compared with 45.56% for the baseline. Net rescue was +0.74 percentage points, with a 95% confidence interval of [0.00, 1.85]. No harmful changes were observed. Other methods matched the highest accuracy. |
| [E3 · When to ask](../experiments/results/E3_medical.json) | Different rules decided whether to ask; replies were simulated. | G rescued 0.74% of tasks and caused no observed harmful changes. The ambiguity-based rule caused neither rescues nor harmful changes. |
| [E4 · Reply quality](../experiments/results/E4_medical.json) | The medical workflow used simulated replies. | The saved file records how replies changed the agent's final answer. |
| [E5 · Repeated training](../experiments/results/E5_expert_iteration_simulated.json) | Expert Iteration ran with simulated replies and three random seeds. | Net rescue was −1.11 percentage points before training and 0.00 after each of the three rounds. The comparison trained once used the same number of tasks, but not necessarily the same number of replies or model calls. |
| [E6 · Which question to ask](../experiments/results/E6_medical.json) | Different rules ranked questions; replies were simulated. | Learned Q rescued 0.74% of tasks and caused no observed harmful changes. This run did not compare the methods under a measured real-person time budget. |
| [E8 · Whether help is available](../experiments/results/E8_medical.json) | The simulated person was available for 0%, 25%, 50%, or 100% of tasks. | With no replies available, accuracy was 45.56% and completion was 100%, matching the baseline. The recorded relative model-call cost was 2.0× the baseline. |

The confidence interval for the E2 improvement reaches zero. It does not establish a reliable improvement from real participants.

## Correct and incorrect answers

The [HiL-Bench SQL reply test](../experiments/results/E4_oracle_and_wrong_choice.json) compared benchmark-correct answers with deliberately wrong answers. With correct answers, the system solved 2 of 13 tasks. With wrong answers, it solved none. The reported 15.38-percentage-point harm is measured against the correct-answer condition.

A separate pilot recorded responses for 50 medical records. The source of those responses was not independently verified, so it cannot confirm a benefit from physician input. Participant-level material is not included in this release.

## Transfer and added information

The [transfer runs](../experiments/results/E7_transfer_simulated.json) applied modules trained on general tasks to medical tasks and changed the simulated responder's role description. The saved runs showed no gain.

In [NoisyToolBench](../experiments/results/E7_noisytoolbench.json), adding the benchmark's correct clarification to 200 queries raised parameter-match accuracy from 14.88% to 47.71%, a difference of 32.83 percentage points. This measures the value of supplied information; it does not test whether the system learns to ask the right question or whether a person answers correctly.

## Reading the scores

- **Rescue:** the baseline failed and HumanChoice succeeded.
- **Harm:** the baseline succeeded and HumanChoice failed.
- **Net rescue:** rescue rate minus harm rate. Both rates use the same set of matched tasks.
- **Percentage points:** the difference between percentages, abbreviated as `pp` in result files.
- **Confidence interval:** a range describing uncertainty in the estimated result.

Missing results are left blank. Original benchmark interfaces and multiple-choice versions are reported separately. These scores are not official leaderboard submissions.

See [the run guide](reproduction.md) for the available code and the files needed to rerun each part.
