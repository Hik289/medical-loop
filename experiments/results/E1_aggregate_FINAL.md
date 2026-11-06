# E1 measured results (split=test)

Means over status=measured tasks only; no imputation, no modelling. Incomplete cells are flagged complete=false.

## clarifybench
expected tasks: 370; unavailable (not run): 0

| protocol/method | measured/expected | err | t/o | param_match_rate | tool_match_rate | success | exact_match | execution_success_rate | coverage_rate | Q/task | API calls/task | tokens/task | Net Rescue vs Auto [CI] | Human s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| native/autonomous ⚠partial | 303/370 | 7 | 60 | 66.0 | 74.7 | 0.0 | 0.0 | 94.6 | 50.6 | 0.0 | 9.6 | 21804.2 | — | n/a (simulated) |
| native/native_ask_human ⚠partial | 203/370 | 2 | 165 | 66.4 | 82.1 | 0.0 | 0.0 | 97.4 | 55.5 | 0.3 | 9.1 | 19790.6 | +0.0 [+0.0,+0.0] (n=191) | n/a (simulated) |
| native/compute_matched ⚠partial | 45/370 | 2 | 203 | 42.3 | 55.7 | 0.0 | 0.0 | 94.7 | 39.5 | 0.0 | 11.1 | 23567.0 | +0.0 [+0.0,+0.0] (n=44) | n/a (simulated) |
| native/clam_api ⚠partial | 131/370 | 2 | 237 | 63.0 | 69.1 | 0.0 | 0.0 | 94.7 | 52.1 | 0.3 | 11.5 | 25979.8 | +0.0 [+0.0,+0.0] (n=118) | n/a (simulated) |
| native/sage ⚠partial | 0/370 | 0 | 3 | — | — | — | — | — | — | — | — | — | — | n/a (simulated) |
| choice/native_ask_human ⚠partial | 138/370 | 2 | 230 | 60.4 | 75.7 | 0.0 | 0.0 | 97.6 | 53.3 | 0.3 | 8.0 | 17223.5 | +0.0 [+0.0,+0.0] (n=133) | n/a (simulated) |
| choice/clam_api ⚠partial | 13/370 | 1 | 52 | 25.6 | 23.1 | 0.0 | 0.0 | 92.9 | 20.5 | 0.1 | 5.5 | 11886.5 | +0.0 [+0.0,+0.0] (n=13) | n/a (simulated) |
| choice/sage ⚠partial | 0/370 | 0 | 3 | — | — | — | — | — | — | — | — | — | — | n/a (simulated) |

## hilsql
expected tasks: 4; unavailable (not run): 96

| protocol/method | measured/expected | err | t/o | solve | ask_precision | ask_recall | ask_f1 | Q/task | API calls/task | tokens/task | Net Rescue vs Auto [CI] | Human s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
