# General benchmark scores

These scores were combined from tasks marked as completed in the test run. Missing results were not estimated. Each method's task count shows how much of its planned run finished.

## ClarifyBench

The comparison planned 370 tasks per method.

| Interface / method | Scored / planned | Errors | Timeouts | Parameter match | Tool match | Task success | Exact match | Tool execution success | Coverage | Questions / task | Model calls / task | Tokens / task | Net rescue vs. baseline [CI] | Human time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native/autonomous · incomplete | 303/370 | 7 | 60 | 66.0 | 74.7 | 0.0 | 0.0 | 94.6 | 50.6 | 0.0 | 9.6 | 21804.2 | — | Not measured; simulated replies |
| native/native_ask_human · incomplete | 203/370 | 2 | 165 | 66.4 | 82.1 | 0.0 | 0.0 | 97.4 | 55.5 | 0.3 | 9.1 | 19790.6 | +0.0 [+0.0,+0.0] (n=191) | Not measured; simulated replies |
| native/compute_matched · incomplete | 45/370 | 2 | 203 | 42.3 | 55.7 | 0.0 | 0.0 | 94.7 | 39.5 | 0.0 | 11.1 | 23567.0 | +0.0 [+0.0,+0.0] (n=44) | Not measured; simulated replies |
| native/clam_api · incomplete | 131/370 | 2 | 237 | 63.0 | 69.1 | 0.0 | 0.0 | 94.7 | 52.1 | 0.3 | 11.5 | 25979.8 | +0.0 [+0.0,+0.0] (n=118) | Not measured; simulated replies |
| native/sage · incomplete | 0/370 | 0 | 3 | — | — | — | — | — | — | — | — | — | — | Not measured; simulated replies |
| choice/native_ask_human · incomplete | 138/370 | 2 | 230 | 60.4 | 75.7 | 0.0 | 0.0 | 97.6 | 53.3 | 0.3 | 8.0 | 17223.5 | +0.0 [+0.0,+0.0] (n=133) | Not measured; simulated replies |
| choice/clam_api · incomplete | 13/370 | 1 | 52 | 25.6 | 23.1 | 0.0 | 0.0 | 92.9 | 20.5 | 0.1 | 5.5 | 11886.5 | +0.0 [+0.0,+0.0] (n=13) | Not measured; simulated replies |
| choice/sage · incomplete | 0/370 | 0 | 3 | — | — | — | — | — | — | — | — | — | — | Not measured; simulated replies |

The original interface uses the benchmark's own question format. The multiple-choice interface presents answer options. Compute-matched means the extra model-call budget is spent on independent attempts instead of asking a question.

Match, success, and coverage scores are percentages. Net rescue uses the tasks paired with the autonomous baseline; the paired count appears as `n`. Questions, calls, and tokens are averages per completed task.

## HiL-Bench SQL comparison

This run recorded four expected tasks and 96 tasks that were not run. No completed method rows were saved in this comparison table. The separate [SQL baseline summary](hilsql_native_summary.json) reports its own run.

The original method identifiers are `native/autonomous`, `native/native_ask_human`, `native/compute_matched`, `native/clam_api`, `native/sage`, `choice/native_ask_human`, `choice/clam_api`, and `choice/sage`.
