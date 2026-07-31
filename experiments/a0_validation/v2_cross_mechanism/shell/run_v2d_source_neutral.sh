#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/mnt/data/users/zbc/AI4S/ProteinDesign/Dual-UQ}"
DATA_ENV="${2:-dual-uq}"
MODEL_ENV="${3:-dual-uq-model}"

V2_ROOT="${PROJECT_ROOT}/experiments/a0_validation/v2_cross_mechanism"
RUNTIME_ENV="${PROJECT_ROOT}/experiments/a0_validation/v1_proteinmpnn/config/runtime.env"
CONFIG="${V2_ROOT}/config/source_neutral.env"

if [[ ! -f "${RUNTIME_ENV}" ]]; then
  echo "Missing ProteinMPNN runtime config: ${RUNTIME_ENV}"
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing V2D config: ${CONFIG}"
  echo "Create it with:"
  echo "  cp ${V2_ROOT}/config/source_neutral.env.example ${CONFIG}"
  exit 2
fi

# shellcheck disable=SC1090
source "${RUNTIME_ENV}"
# shellcheck disable=SC1090
source "${CONFIG}"

: "${PROTEINMPNN_ROOT:?PROTEINMPNN_ROOT is required}"
: "${PROTEINMPNN_CHECKPOINT:?PROTEINMPNN_CHECKPOINT is required}"

if (( A0_V2D_SCORE_REPEATS % A0_V2D_SCORE_BATCH_SIZE != 0 )); then
  echo "Score repeats must be divisible by score batch size."
  exit 2
fi

MODEL_WEIGHTS_DIR="$(dirname "${PROTEINMPNN_CHECKPOINT}")"
MODEL_NAME="$(basename "${PROTEINMPNN_CHECKPOINT}")"
MODEL_NAME="${MODEL_NAME%.pt}"
MODEL_NAME="${MODEL_NAME%.pth}"

INDEX="${A0_V2D_INDEX}"
JSONL_DIR="${V2_ROOT}/inputs/index${INDEX}/jsonl"
PDB_JSONL="${JSONL_DIR}/index${INDEX}_pdb.jsonl"
AFDB_JSONL="${JSONL_DIR}/index${INDEX}_afdb.jsonl"
SCORE_ROOT="${V2_ROOT}/outputs/index${INDEX}/source_neutral_cross_scoring"
JOBS="${V2_ROOT}/manifests/index${INDEX}_source_neutral_score_jobs.tsv"
BUILDER="${V2_ROOT}/scripts/51_build_source_neutral_panel.py"
VALIDATOR="${V2_ROOT}/scripts/52_validate_source_neutral_score_job.py"
ANALYZER="${V2_ROOT}/scripts/53_analyze_source_neutral_landscape.py"

for required in \
  "${PDB_JSONL}" \
  "${AFDB_JSONL}" \
  "${BUILDER}" \
  "${VALIDATOR}" \
  "${ANALYZER}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required file: ${required}"
    exit 2
  fi
done

mkdir -p \
  "${V2_ROOT}/logs" \
  "${V2_ROOT}/metrics" \
  "${V2_ROOT}/manifests" \
  "${SCORE_ROOT}/pdb" \
  "${SCORE_ROOT}/afdb"

LOG_FILE="${V2_ROOT}/logs/v2d_source_neutral_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "${LOG_FILE}") 2>&1

echo "Step 0/4: verify V2B2 and V2C"

conda run -n "${DATA_ENV}" python - \
  "${V2_ROOT}/metrics/index${INDEX}_paired_input_validation.json" \
  "${V2_ROOT}/metrics/index${INDEX}_source_conditioned_cross_scoring.json" <<'PY'
import json
import sys
from pathlib import Path

paired = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
conditioned = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

if paired.get("validation_pass") is not True:
    raise SystemExit("V2B2 validation_pass is not true.")
if conditioned.get("validation_pass") is not True:
    raise SystemExit("V2C validation_pass is not true.")

print("V2B2 validation_pass:", paired["validation_pass"])
print("V2C validation_pass:", conditioned["validation_pass"])
print(
    "V2C mean home advantage:",
    conditioned["origin_conditioned_specialization"]["mean_home_advantage"],
)
PY

echo
echo "Step 1/4: build deterministic source-neutral panel"

conda run -n "${DATA_ENV}" \
  python "${BUILDER}" \
  --project-root "${PROJECT_ROOT}" \
  --screening-index "${INDEX}" \
  --hamming-distances "${A0_V2D_HAMMING_DISTANCES}" \
  --variants-per-distance "${A0_V2D_VARIANTS_PER_DISTANCE}" \
  --panel-seed "${A0_V2D_PANEL_SEED}"

mapfile -t JOB_LINES < <(tail -n +2 "${JOBS}")
EXPECTED_JOBS=$((1 + 6 * A0_V2D_VARIANTS_PER_DISTANCE))

if [[ "${#JOB_LINES[@]}" -ne "${EXPECTED_JOBS}" ]]; then
  echo "Unexpected source-neutral job count: ${#JOB_LINES[@]}"
  echo "Expected: ${EXPECTED_JOBS}"
  exit 2
fi

score_candidate() {
  local BACKBONE="$1"
  local JSONL_PATH="$2"
  local CANDIDATE_ID="$3"
  local FASTA_PATH="$4"
  local OUTPUT_DIR="${SCORE_ROOT}/${BACKBONE}/${CANDIDATE_ID}"

  if conda run -n "${DATA_ENV}" \
      python "${VALIDATOR}" \
      --output-dir "${OUTPUT_DIR}" \
      --candidate-fasta "${FASTA_PATH}" \
      --expected-repeats "${A0_V2D_SCORE_REPEATS}" \
      >/dev/null 2>&1; then
    echo "Skip completed score candidate=${CANDIDATE_ID} backbone=${BACKBONE}"
    return 0
  fi

  echo "Run score candidate=${CANDIDATE_ID} backbone=${BACKBONE}"
  rm -rf "${OUTPUT_DIR}"
  mkdir -p "${OUTPUT_DIR}"

  conda run -n "${MODEL_ENV}" \
    python "${PROTEINMPNN_ROOT}/protein_mpnn_run.py" \
    --jsonl_path "${JSONL_PATH}" \
    --out_folder "${OUTPUT_DIR}" \
    --score_only 1 \
    --path_to_fasta "${FASTA_PATH}" \
    --num_seq_per_target "${A0_V2D_SCORE_REPEATS}" \
    --batch_size "${A0_V2D_SCORE_BATCH_SIZE}" \
    --seed "${A0_V2D_SCORE_SEED}" \
    --backbone_noise "${A0_V2D_BACKBONE_NOISE}" \
    --path_to_model_weights "${MODEL_WEIGHTS_DIR}/" \
    --model_name "${MODEL_NAME}"

  conda run -n "${DATA_ENV}" \
    python "${VALIDATOR}" \
    --output-dir "${OUTPUT_DIR}" \
    --candidate-fasta "${FASTA_PATH}" \
    --expected-repeats "${A0_V2D_SCORE_REPEATS}"
}

echo
echo "Step 2/4: score 49 neutral candidates on both backbones"

for LINE in "${JOB_LINES[@]}"; do
  LINE="${LINE%$'\r'}"
  IFS=$'\t' read -r CANDIDATE_ID FASTA_PATH EXTRA <<< "${LINE}"
  CANDIDATE_ID="${CANDIDATE_ID%$'\r'}"
  FASTA_PATH="${FASTA_PATH%$'\r'}"

  if [[ -z "${CANDIDATE_ID}" || -z "${FASTA_PATH}" ]]; then
    echo "Malformed TSV row: ${LINE}"
    exit 2
  fi
  if [[ -n "${EXTRA:-}" ]]; then
    echo "Unexpected extra TSV field: ${LINE}"
    exit 2
  fi
  if [[ ! -f "${FASTA_PATH}" ]]; then
    echo "Candidate FASTA missing: ${FASTA_PATH}"
    exit 2
  fi

  score_candidate "pdb" "${PDB_JSONL}" "${CANDIDATE_ID}" "${FASTA_PATH}"
  score_candidate "afdb" "${AFDB_JSONL}" "${CANDIDATE_ID}" "${FASTA_PATH}"
done

echo
echo "Step 3/4: require all 98 valid score records"

COMPLETE=0
for LINE in "${JOB_LINES[@]}"; do
  LINE="${LINE%$'\r'}"
  IFS=$'\t' read -r CANDIDATE_ID FASTA_PATH EXTRA <<< "${LINE}"
  CANDIDATE_ID="${CANDIDATE_ID%$'\r'}"
  FASTA_PATH="${FASTA_PATH%$'\r'}"
  for BACKBONE in pdb afdb; do
    OUTPUT_DIR="${SCORE_ROOT}/${BACKBONE}/${CANDIDATE_ID}"
    if conda run -n "${DATA_ENV}" \
        python "${VALIDATOR}" \
        --output-dir "${OUTPUT_DIR}" \
        --candidate-fasta "${FASTA_PATH}" \
        --expected-repeats "${A0_V2D_SCORE_REPEATS}" \
        >/dev/null; then
      COMPLETE=$((COMPLETE + 1))
    fi
  done
done

EXPECTED_RECORDS=$((2 * EXPECTED_JOBS))
echo "Complete score records: ${COMPLETE}/${EXPECTED_RECORDS}"

if [[ "${COMPLETE}" -ne "${EXPECTED_RECORDS}" ]]; then
  echo "Not all source-neutral score jobs are complete."
  exit 1
fi

echo
echo "Step 4/4: analyze source-neutral landscape"

conda run -n "${DATA_ENV}" \
  python "${ANALYZER}" \
  --project-root "${PROJECT_ROOT}" \
  --screening-index "${INDEX}" \
  --score-repeats "${A0_V2D_SCORE_REPEATS}" \
  --bootstrap-reps "${A0_V2D_BOOTSTRAP_REPS}" \
  --bootstrap-seed "${A0_V2D_BOOTSTRAP_SEED}"

echo
echo "V2D source-neutral landscape completed."
echo "Review:"
echo "  ${V2_ROOT}/V2D_INDEX${INDEX}_SOURCE_NEUTRAL_LANDSCAPE.md"
echo "  ${V2_ROOT}/metrics/index${INDEX}_source_neutral_panel.json"
echo "  ${V2_ROOT}/metrics/index${INDEX}_source_neutral_landscape.json"
echo
echo "Stop here. Review neutral-landscape stability before residue localization."
