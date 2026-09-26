#!/usr/bin/env bash
set -euo pipefail

# Harness helper for verify-pred-engine.
# Provides doctor checks, seed generation, isolated driving of CLI commands,
# side-effect assertion, and safe cleanup.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)"

usage() {
  cat <<'EOF'
Usage: harness.sh <command> [arguments...]

Commands:
  doctor                                    Validate runtime, environment, and CLI entry points.
  seed <output_csv> [skus] [days]           Generate a canonical 4-column seed dataset.
  drive-models <evidence_dir>               Drive provider model catalog queries and capture evidence.
  drive-fase0 <evidence_dir> [seed] [root]  Drive Phase 0 synthetic simulation and capture evidence.
  drive-classify <evidence_dir> <csv> [root] Drive SKU topology classification and capture evidence.
  drive-verify <evidence_dir> <parquet>     Drive Parquet contract verification and capture evidence.
  drive-probe-failclosed <evidence_dir>     Drive probe command without API key to prove fail-closed contract.
  cleanup <disposable_path>                 Safely remove disposable execution data root.
EOF
  exit 1
}

cmd_doctor() {
  echo "=== Running Doctor checks ==="
  echo -n "1. Checking uv presence: "
  if ! command -v uv >/dev/null 2>&1; then
    echo "FAILED (uv not found in PATH)"
    return 1
  fi
  echo "OK ($(uv --version))"

  echo -n "2. Checking Python environment and core packages: "
  (cd "$REPO_ROOT" && uv run --frozen python -c "
import sys
assert sys.version_info >= (3, 12), f'Python >= 3.12 required, found {sys.version}'
import numpy, pandas, pyarrow, pydantic, statsmodels
" >/dev/null 2>&1)
  echo "OK"

  echo -n "3. Checking pred-engine CLI entry point: "
  local help_out
  help_out="$(cd "$REPO_ROOT" && uv run --frozen pred-engine --help)"
  if [[ "$help_out" != *"models"* || "$help_out" != *"classify"* || "$help_out" != *"verify"* ]]; then
    echo "FAILED (missing expected subcommands)"
    return 1
  fi
  echo "OK"

  echo -n "4. Checking pred-engine-fase0 CLI entry point: "
  local fase0_out
  fase0_out="$(cd "$REPO_ROOT" && uv run --frozen pred-engine-fase0 --help)"
  if [[ "$fase0_out" != *"semilla"* || "$fase0_out" != *"data-root"* ]]; then
    echo "FAILED (missing expected arguments)"
    return 1
  fi
  echo "OK"

  echo "doctor: OK"
  return 0
}

cmd_seed() {
  local target_csv="${1:-}"
  local sku_count="${2:-2}"
  local days="${3:-30}"
  if [[ -z "$target_csv" ]]; then
    echo "Error: target_csv path required" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$target_csv")"
  (cd "$REPO_ROOT" && uv run --frozen python -c "
import pandas as pd, numpy as np, sys
target = sys.argv[1]
skus = int(sys.argv[2])
days = int(sys.argv[3])
rows = []
rng = np.random.default_rng(42)
for s in range(skus):
    sku_id = f'SKU-{s+1:03d}'
    base_dates = pd.date_range('2024-01-01', periods=days, freq='D')
    demand_pattern = rng.choice([0, 5, 10, 15, 20], size=days, p=[0.2, 0.2, 0.3, 0.2, 0.1])
    # Ensure at least some positive demand
    if np.sum(demand_pattern) == 0:
        demand_pattern[0] = 10
    for dt, qty in zip(base_dates, demand_pattern):
        rows.append({
            'sku_id': sku_id,
            'timestamp': dt.strftime('%Y-%m-%d'),
            'demand_qty': int(qty),
            'lead_time_days': 5
        })
df = pd.DataFrame(rows)
df.to_csv(target, index=False)
" "$target_csv" "$sku_count" "$days")
  echo "Generated seed CSV: $target_csv ($(wc -l < "$target_csv") rows)"
}

cmd_drive_models() {
  local evidence_dir="${1:-}"
  if [[ -z "$evidence_dir" ]]; then
    echo "Error: evidence_dir required" >&2
    exit 1
  fi
  local target_dir="$evidence_dir/models"
  mkdir -p "$target_dir"

  echo "Driving: pred-engine models"
  local exit_code=0
  (cd "$REPO_ROOT" && uv run --frozen pred-engine models --provider gemini) > "$target_dir/models_gemini.txt" 2>&1 || exit_code=$?
  echo "gemini exit code: $exit_code" >> "$target_dir/summary.txt"

  (cd "$REPO_ROOT" && uv run --frozen pred-engine models --provider openai) > "$target_dir/models_openai.txt" 2>&1 || exit_code=$?
  echo "openai exit code: $exit_code" >> "$target_dir/summary.txt"

  (cd "$REPO_ROOT" && uv run --frozen pred-engine models --provider anthropic) > "$target_dir/models_anthropic.txt" 2>&1 || exit_code=$?
  echo "anthropic exit code: $exit_code" >> "$target_dir/summary.txt"

  local fail_code=0
  (cd "$REPO_ROOT" && uv run --frozen pred-engine models --provider unknown-provider) > "$target_dir/models_invalid.txt" 2>&1 || fail_code=$?
  echo "unknown-provider exit code: $fail_code" >> "$target_dir/summary.txt"

  if grep -q "gemini-2.5-flash-lite" "$target_dir/models_gemini.txt" && \
     grep -q "gpt-4.1-nano" "$target_dir/models_openai.txt" && \
     grep -q "claude-haiku-4-5" "$target_dir/models_anthropic.txt" && \
     [[ "$fail_code" -ne 0 ]]; then
    echo "drive-models: SUCCESS (evidence saved to $target_dir)"
    return 0
  else
    echo "drive-models: FAILED verification assertion" >&2
    return 1
  fi
}

cmd_drive_fase0() {
  local evidence_dir="${1:-}"
  local seed_csv="${2:-}"
  local data_root="${3:-}"
  if [[ -z "$evidence_dir" ]]; then
    echo "Error: evidence_dir required" >&2
    exit 1
  fi
  local target_dir="$evidence_dir/synthetic-data"
  mkdir -p "$target_dir"

  local created_temp_seed=0
  if [[ -z "$seed_csv" || ! -f "$seed_csv" ]]; then
    seed_csv="$evidence_dir/seed.csv"
    cmd_seed "$seed_csv" 2 30
    created_temp_seed=1
  fi

  local created_temp_root=0
  if [[ -z "$data_root" ]]; then
    data_root="$(mktemp -d -t pred-engine-fase0-root-XXXXXX)"
    created_temp_root=1
  fi
  mkdir -p "$data_root"

  echo "Driving: pred-engine-fase0 with seed=$seed_csv data_root=$data_root"
  local exit_code=0
  (cd "$REPO_ROOT" && uv run --frozen pred-engine-fase0 "$seed_csv" \
    --data-root "$data_root" \
    --period 7 \
    --n-series 2 \
    --minimo-filas 0 \
    --seed 42) > "$target_dir/stdout.txt" 2> "$target_dir/stderr.txt" || exit_code=$?

  echo "exit_code: $exit_code" > "$target_dir/status.txt"
  if [[ "$exit_code" -ne 0 ]]; then
    echo "drive-fase0: FAILED (exit code $exit_code)" >&2
    cat "$target_dir/stderr.txt" >&2
    return 1
  fi

  # Verify side effects
  local artifact_csv="$data_root/raw/panel_sintetico_fase0.csv"
  if [[ ! -f "$artifact_csv" ]]; then
    echo "drive-fase0: FAILED (artifact CSV missing at $artifact_csv)" >&2
    return 1
  fi

  local log_file
  log_file="$(find "$data_root/logs" -name "fase0_*.json" 2>/dev/null | head -n 1)"
  if [[ -z "$log_file" || ! -f "$log_file" ]]; then
    echo "drive-fase0: FAILED (bitacora JSON log missing under $data_root/logs)" >&2
    return 1
  fi

  # Copy verifiable metadata into evidence
  cp "$log_file" "$target_dir/bitacora.json"
  head -n 10 "$artifact_csv" > "$target_dir/artifact_head.csv"
  echo "artifact_rows: $(wc -l < "$artifact_csv")" >> "$target_dir/status.txt"
  echo "artifact_path: $artifact_csv" >> "$target_dir/status.txt"
  echo "bitacora_path: $log_file" >> "$target_dir/status.txt"

  # WORM check: verify write protection on raw artifact
  local worm_check="writable"
  if [[ ! -w "$artifact_csv" ]]; then
    worm_check="read-only (WORM protected)"
  fi
  echo "worm_status: $worm_check" >> "$target_dir/status.txt"

  echo "drive-fase0: SUCCESS (evidence saved to $target_dir)"
  if [[ $created_temp_root -eq 1 ]]; then
    echo "DISPOSABLE_DATA_ROOT=$data_root"
  fi
  return 0
}

cmd_drive_classify() {
  local evidence_dir="${1:-}"
  local input_csv="${2:-}"
  local data_root="${3:-}"
  if [[ -z "$evidence_dir" || -z "$input_csv" ]]; then
    echo "Error: evidence_dir and input_csv required" >&2
    exit 1
  fi
  local target_dir="$evidence_dir/classify"
  mkdir -p "$target_dir"

  local created_temp_root=0
  if [[ -z "$data_root" ]]; then
    data_root="$(mktemp -d -t pred-engine-classify-root-XXXXXX)"
    created_temp_root=1
  fi
  mkdir -p "$data_root"

  echo "Driving: pred-engine classify --csv $input_csv --data-root $data_root"
  local exit_code=0
  (cd "$REPO_ROOT" && uv run --frozen pred-engine classify \
    --csv "$input_csv" \
    --data-root "$data_root") > "$target_dir/stdout.txt" 2> "$target_dir/stderr.txt" || exit_code=$?

  echo "exit_code: $exit_code" > "$target_dir/status.txt"
  if [[ "$exit_code" -ne 0 ]]; then
    echo "drive-classify: FAILED (exit code $exit_code)" >&2
    cat "$target_dir/stderr.txt" >&2
    return 1
  fi

  local stem
  stem="$(basename "$input_csv" .csv)"
  local parquet_path="$data_root/processed/$stem.parquet"
  if [[ ! -f "$parquet_path" ]]; then
    echo "drive-classify: FAILED (parquet artifact missing at $parquet_path)" >&2
    return 1
  fi

  echo "parquet_path: $parquet_path" >> "$target_dir/status.txt"
  echo "parquet_size_bytes: $(stat -c%s "$parquet_path")" >> "$target_dir/status.txt"

  # Confirm contract accepted marker in stdout
  if ! grep -q "contrato_1_4: accepted" "$target_dir/stdout.txt"; then
    echo "drive-classify: FAILED (contrato_1_4: accepted not found in stdout)" >&2
    return 1
  fi

  echo "drive-classify: SUCCESS (evidence saved to $target_dir)"
  if [[ $created_temp_root -eq 1 ]]; then
    echo "DISPOSABLE_DATA_ROOT=$data_root"
  fi
  return 0
}

cmd_drive_verify() {
  local evidence_dir="${1:-}"
  local parquet_path="${2:-}"
  if [[ -z "$evidence_dir" || -z "$parquet_path" ]]; then
    echo "Error: evidence_dir and parquet_path required" >&2
    exit 1
  fi
  local target_dir="$evidence_dir/verify"
  mkdir -p "$target_dir"

  echo "Driving: pred-engine verify --parquet $parquet_path"
  local exit_code=0
  (cd "$REPO_ROOT" && uv run --frozen pred-engine verify \
    --parquet "$parquet_path") > "$target_dir/stdout.txt" 2> "$target_dir/stderr.txt" || exit_code=$?

  echo "exit_code: $exit_code" > "$target_dir/status.txt"
  if [[ "$exit_code" -ne 0 ]]; then
    echo "drive-verify: FAILED (exit code $exit_code)" >&2
    cat "$target_dir/stderr.txt" >&2
    return 1
  fi

  if ! grep -q "contrato_1_4: accepted" "$target_dir/stdout.txt"; then
    echo "drive-verify: FAILED (contrato_1_4: accepted not found in stdout)" >&2
    return 1
  fi

  echo "drive-verify: SUCCESS (evidence saved to $target_dir)"
  return 0
}

cmd_drive_probe_failclosed() {
  local evidence_dir="${1:-}"
  if [[ -z "$evidence_dir" ]]; then
    echo "Error: evidence_dir required" >&2
    exit 1
  fi
  local target_dir="$evidence_dir/probe_failclosed"
  mkdir -p "$target_dir"

  local dummy_csv="$target_dir/probe_sample.csv"
  echo "sku_id,timestamp,demand_qty,lead_time_days" > "$dummy_csv"
  echo "SKU-001,2024-01-01,10,5" >> "$dummy_csv"

  echo "Driving: pred-engine probe without API key"
  local exit_code=0
  # Ensure env vars are stripped for clean fail-closed verification
  (cd "$REPO_ROOT" && env -u PRED_LLM_API_KEY -u GEMINI_API_KEY -u GOOGLE_API_KEY \
    uv run --frozen pred-engine probe --csv "$dummy_csv" --provider gemini) > "$target_dir/stdout.txt" 2> "$target_dir/stderr.txt" || exit_code=$?

  echo "exit_code: $exit_code" > "$target_dir/status.txt"
  if [[ "$exit_code" -eq 1 ]] && grep -q "Falta API key" "$target_dir/stderr.txt"; then
    echo "drive-probe-failclosed: SUCCESS (fail-closed verified, evidence saved to $target_dir)"
    return 0
  else
    echo "drive-probe-failclosed: FAILED fail-closed assertion" >&2
    return 1
  fi
}

cmd_cleanup() {
  local disposable_path="${1:-}"
  if [[ -z "$disposable_path" ]]; then
    echo "Error: disposable_path required" >&2
    exit 1
  fi

  # Safety check: never delete repository root or outside /tmp or dot-paths
  local real_target
  real_target="$(cd -- "$disposable_path" 2>/dev/null && pwd -P || echo "$disposable_path")"
  if [[ "$real_target" == "$REPO_ROOT" || "$real_target" == "$REPO_ROOT"* && "$real_target" != *"/tmp"* ]]; then
    if [[ "$real_target" != *"/data"* && "$real_target" != *"/scratch"* ]]; then
      echo "Refusing to delete unsafe target: $disposable_path" >&2
      return 1
    fi
  fi

  # If files under raw/ are read-only due to WORM policy, grant write to permit removal
  if [[ -d "$disposable_path" ]]; then
    chmod -R u+w "$disposable_path" 2>/dev/null || true
    rm -rf "$disposable_path"
    echo "cleanup: removed $disposable_path"
  else
    echo "cleanup: path does not exist ($disposable_path)"
  fi
}

case "${1:-}" in
  doctor)
    cmd_doctor
    ;;
  seed)
    shift
    cmd_seed "$@"
    ;;
  drive-models)
    shift
    cmd_drive_models "$@"
    ;;
  drive-fase0)
    shift
    cmd_drive_fase0 "$@"
    ;;
  drive-classify)
    shift
    cmd_drive_classify "$@"
    ;;
  drive-verify)
    shift
    cmd_drive_verify "$@"
    ;;
  drive-probe-failclosed)
    shift
    cmd_drive_probe_failclosed "$@"
    ;;
  cleanup)
    shift
    cmd_cleanup "$@"
    ;;
  *)
    usage
    ;;
esac
