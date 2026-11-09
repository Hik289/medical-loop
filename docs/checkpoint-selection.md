# Best observed epochs

The epoch study evaluates every budget from 1 to 30 epochs per coordinate update. At each budget it refits A, refreshes Q targets and fits Q, then refreshes G targets and fits G. It keeps seeds 42, 123, and 456; the 270-record medical test cohort; the 120-record medical training collection; and five source-query-grouped folds over 150 tool tasks from 50 source queries. It makes no API calls.

The result file contains every candidate score, not only the winners. Each metric in `envelope` independently selects its best three-seed mean using evaluation scores. A row assembled from these entries can contain different epochs. The paper identifies these tables as retrospective per-metric summaries. Complete single-epoch rows are reported separately. Fixed 30-epoch reference scores are preserved in `replay_results.json`.

| Actual full-policy checkpoint | Epoch | Tool match | Harm | Requests | Utility |
| --- | --- | --- | --- | --- | --- |
| Best observed tool quality | 1 | 23.28 | 0.89 | 56.00 | 0.2179 |
| Best observed tool utility | 9 | 22.06 | 0.00 | 3.11 | 0.2199 |

At a fixed 40% request budget, learned Gate at epoch 21 reaches 23.39% parameter match and utility 0.2235, with 0.22% harm. This is a single-epoch result. The complete medical full policy reaches 45.56% accuracy; forced-Ask Q+A reaches 45.56% accuracy and zero harm at epoch 8.

The per-cell ledger is `experiments/results/supplementary/table_epoch_selection.csv`. `table_update_audit.json` identifies every paper table and whether an epoch search applies. Counts, fixed API baselines, the supplied response pilot, oracle interventions, and archived runs without intermediate epoch scores retain their measured sources. Invalid or modeled summaries are excluded.

## Reproduce

Use authorized local branch records:

```bash
python experiments/run_epoch_sweep.py \
  --medical-train DATA/medical_train.json \
  --medical-test DATA/medical_test.json \
  --ntb-records DATA/tool_records.json \
  --out runs/epoch_sweep \
  --max-epoch 30 \
  --workers 2
```

Verify the saved aggregate release without patient records or API access:

```bash
python scripts/check_supplementary.py
```

The checker verifies every selected value against its full epoch trajectory and reproduces all previously reported fixed-epoch comparison values within their recorded rounding precision.
