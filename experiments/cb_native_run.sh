#!/bin/bash
# ClarifyBench Native Autonomous baseline — robust parallel runner (anchor_5)
# Run the A/E/I subsets in parallel, with a timeout for each task.
# Use the benchmark's own baseline runner and evaluator.
set -euo pipefail
CB="$(cd "${CLARIFYBENCH_ROOT:?Set CLARIFYBENCH_ROOT to the upstream checkout}" && pwd)"
RUN_ROOT="$(mkdir -p "${HUMANCHOICE_RUN_DIR:-runs}" && cd "${HUMANCHOICE_RUN_DIR:-runs}" && pwd)"
command -v timeout >/dev/null || { echo "GNU timeout is required" >&2; exit 1; }
SUBSET="${1:-A}"            # A | E | I
WORKERS="${2:-12}"
TIMEOUT="${3:-420}"
DS="$CB/ClarifyBench/ClarifyBench_${SUBSET}"
OUT="$RUN_ROOT/anchor5_clarifybench/results_${SUBSET}"
mkdir -p "$OUT"
cd "$CB"

run_one(){
  f="$1"
  base=$(basename "$f" .json)
  outp="$OUT/${base}_RESULT.json"
  # Resume by skipping files that already contain an evaluation result.
  if [ -s "$outp" ] && python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get("evaluation") is not None else 1)' "$outp" 2>/dev/null; then
    echo "skip $base"; return
  fi
  if timeout "$TIMEOUT" python3 main.py --agent baseline --data "$f" --output "$outp" >/dev/null 2>&1 && [ -s "$outp" ]; then
    echo "done $base"
  else
    echo "FAIL $base"
  fi
}
export -f run_one
export CB OUT TIMEOUT

find "$DS" -maxdepth 1 -type f -name "*.json" ! -name "*RESULT*" -print0 |
  xargs -0 -P "$WORKERS" -I{} bash -c 'run_one "$@"' _ {}
echo "SUBSET_${SUBSET}_DONE"
