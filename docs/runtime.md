# Runtime portability

The package includes the benchmark controller, method implementations, small-network code, task runner, metric aggregation, and API wrapper in `experiments/e1_methods/e1`. The numerical decision code is copied from the runtime used by the online pilot. Public-task module weights retain their original numerical parameters.

Configuration defaults are made portable: project paths are derived from the package or `HUMANCHOICE_ROOT`, credentials come from `OPENAI_API_KEY`, and the endpoint and model can be set with `OPENAI_BASE_URL` and `MODEL_NAME`. Private configuration files are excluded. These configuration changes do not alter the decision rules or reported model parameters.

The online pilot's wrapper also serializes configuration-file writes, while independent episodes remain concurrent. This avoids a temporary-file collision in the inherited writer. Exact prompt prefixes use a task-specific shared cache, and variants run in order within a task.

After installing the official benchmark and its environment dependencies in the configured data folders, run from the unpacked project:

```bash
python experiments/run_online_ablation.py \
  --project-root . \
  --out runs/online_ablation \
  --retry-errors \
  --request-timeout 180 \
  --task-timeout 900
```

The script selects held-out tasks by a fixed hash, not their outcomes. It writes fresh module copies and leaves supplied models unchanged. The summary scores only the common completed cohort and reports completion separately. Recorded final-file API counts omit some failed-attempt ledgers and must not be interpreted as a full billing total.

Official benchmark code, benchmark data, medical records, response forms, API configuration, and raw conversation caches are not part of this source package. The replay harness requires authorized branch-record inputs. The online harness requires the original benchmark runtime dependencies in addition to `requirements.txt`.
