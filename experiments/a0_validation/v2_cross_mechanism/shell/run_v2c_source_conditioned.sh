#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/mnt/data/users/zbc/AI4S/ProteinDesign/Dual-UQ}"
DATA_ENV="${2:-dual-uq}"
MODEL_ENV="${3:-dual-uq-model}"
V2_ROOT="${PROJECT_ROOT}/experiments/a0_validation/v2_cross_mechanism"
RUNTIME_ENV="${PROJECT_ROOT}/experiments/a0_validation/v1_proteinmpnn/config/runtime.env"
CONFIG="${V2_ROOT}/config/source_conditioned.env"

if [[ ! -f "${RUNTIME_ENV}" ]]; then
  echo "Missing ProteinMPNN runtime config: ${RUNTIME_ENV}"
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing V2C config: ${CONFIG}"
  echo "Create it with:"
  echo "  cp ${V2_ROOT}/config/source_conditioned.env.example ${CONFIG}"
  exit 2
fi

# shellcheck disable=SC1090
source "${RUNTIME_ENV}"
# shellcheck disable=SC1090
source "${CONFIG}"

: "${PROTEINMPNN_ROOT:?PROTEINMPNN_ROOT is required}"
: "${PROTEINMPNN_CHECKPOINT:?PROTEINMPNN_CHECKPOINT is required}"

if (( A0_V2C_GENERATED_PER_BACKBONE % A0_V2C_GENERATION_BATCH_SIZE != 0 )); then
  echo "Generated-per-backbone must be divisible by generation batch size."
  exit 2
fi
if (( A0_V2C_SCORE_REPEATS % A0_V2C_SCORE_BATCH_SIZE != 0 )); then
  echo "Score repeats must be divisible by score batch size."
  exit 2
fi

MODEL_WEIGHTS_DIR="$(dirname "${PROTEINMPNN_CHECKPOINT}")"
MODEL_NAME="$(basename "${PROTEINMPNN_CHECKPOINT}")"
MODEL_NAME="${MODEL_NAME%.pt}"
MODEL_NAME="${MODEL_NAME%.pth}"

INDEX="${A0_V2C_INDEX}"
JSONL_DIR="${V2_ROOT}/inputs/index${INDEX}/jsonl"
PDB_JSONL="${JSONL_DIR}/index${INDEX}_pdb.jsonl"
AFDB_JSONL="${JSONL_DIR}/index${INDEX}_afdb.jsonl"
GEN_ROOT="${V2_ROOT}/outputs/index${INDEX}/source_conditioned_generation"
SCORE_ROOT="${V2_ROOT}/outputs/index${INDEX}/source_conditioned_cross_scoring"
JOBS="${V2_ROOT}/manifests/index${INDEX}_source_conditioned_score_jobs.tsv"

mkdir -p "${V2_ROOT}/logs" "${V2_ROOT}/metrics" "${V2_ROOT}/manifests"
mkdir -p "${GEN_ROOT}/pdb" "${GEN_ROOT}/afdb" "${SCORE_ROOT}/pdb" "${SCORE_ROOT}/afdb"

LOG_FILE="${V2_ROOT}/logs/v2c_source_conditioned_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "${LOG_FILE}") 2>&1

echo "Step 0/5: verify V2B2 validation"
conda run -n "${DATA_ENV}" python - "${V2_ROOT}/metrics/index${INDEX}_paired_input_validation.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
if data.get("validation_pass") is not True:
    raise SystemExit(f"V2B2 validation_pass is not true in {path}")
print("V2B2 validation_pass:", data["validation_pass"])
print("Expected length:", data["expected_length"])
print("CA RMSD:", data["ca_kabsch"]["rmsd"])
PY

run_generation() {
  local SOURCE="$1"
  local JSONL_PATH="$2"
  local OUTPUT_DIR="$3"

  local FASTA_COUNT
  FASTA_COUNT="$(find "${OUTPUT_DIR}" -type f \
    \( -name '*.fa' -o -name '*.fasta' -o -name '*.faa' \) | wc -l)"
  if [[ "${FASTA_COUNT}" -eq 1 ]]; then
    echo
    echo "Skip completed generation source=${SOURCE}"
    return
  fi

  rm -rf "${OUTPUT_DIR}"
  mkdir -p "${OUTPUT_DIR}"
  echo
  echo "Generate source-conditioned candidates: ${SOURCE}"
  conda run -n "${MODEL_ENV}" \
    python "${PROTEINMPNN_ROOT}/protein_mpnn_run.py" \
    --jsonl_path "${JSONL_PATH}" \
    --out_folder "${OUTPUT_DIR}" \
    --num_seq_per_target "${A0_V2C_GENERATED_PER_BACKBONE}" \
    --batch_size "${A0_V2C_GENERATION_BATCH_SIZE}" \
    --sampling_temp "${A0_V2C_SAMPLING_TEMP}" \
    --seed "${A0_V2C_GENERATION_SEED}" \
    --backbone_noise "${A0_V2C_BACKBONE_NOISE}" \
    --path_to_model_weights "${MODEL_WEIGHTS_DIR}/" \
    --model_name "${MODEL_NAME}"
}

echo "Step 1/5: PDB-derived generation"
run_generation "pdb" "${PDB_JSONL}" "${GEN_ROOT}/pdb"

echo "Step 2/5: AFDB-derived generation"
run_generation "afdb" "${AFDB_JSONL}" "${GEN_ROOT}/afdb"

echo "Step 3/5: build and validate 9-candidate source-conditioned panel"
conda run -n "${DATA_ENV}" \
  python "${V2_ROOT}/scripts/47_build_source_conditioned_panel.py" \
  --project-root "${PROJECT_ROOT}" \
  --screening-index "${INDEX}" \
  --generated-per-backbone "${A0_V2C_GENERATED_PER_BACKBONE}"

score_candidate() {
  local BACKBONE="$1"
  local JSONL_PATH="$2"
  local CANDIDATE_ID="$3"
  local FASTA_PATH="$4"
  local OUTPUT_DIR="${SCORE_ROOT}/${BACKBONE}/${CANDIDATE_ID}"

  local NPZ_COUNT
  NPZ_COUNT="$(find "${OUTPUT_DIR}" -type f -path '*/score_only/*.npz' 2>/dev/null | wc -l)"
  if [[ "${NPZ_COUNT}" -eq 1 ]]; then
    echo
    echo "Skip completed score candidate=${CANDIDATE_ID} backbone=${BACKBONE}"
    return
  fi

  rm -rf "${OUTPUT_DIR}"
  mkdir -p "${OUTPUT_DIR}"

  echo
  echo "Score candidate=${CANDIDATE_ID} backbone=${BACKBONE}"
  conda run -n "${MODEL_ENV}" \
    python "${PROTEINMPNN_ROOT}/protein_mpnn_run.py" \
    --jsonl_path "${JSONL_PATH}" \
    --out_folder "${OUTPUT_DIR}" \
    --score_only 1 \
    --path_to_fasta "${FASTA_PATH}" \
    --num_seq_per_target "${A0_V2C_SCORE_REPEATS}" \
    --batch_size "${A0_V2C_SCORE_BATCH_SIZE}" \
    --seed "${A0_V2C_SCORE_SEED}" \
    --backbone_noise "${A0_V2C_BACKBONE_NOISE}" \
    --path_to_model_weights "${MODEL_WEIGHTS_DIR}/" \
    --model_name "${MODEL_NAME}"
}

echo "Step 4/5: score all candidates on both backbones"
while IFS=$'\t' read -r CANDIDATE_ID FASTA_PATH; do
  [[ "${CANDIDATE_ID}" == "candidate_id" ]] && continue
  score_candidate "pdb" "${PDB_JSONL}" "${CANDIDATE_ID}" "${FASTA_PATH}"
  score_candidate "afdb" "${AFDB_JSONL}" "${CANDIDATE_ID}" "${FASTA_PATH}"
done < "${JOBS}"

echo "Step 5/5: analyze paired cross-backbone scores"
conda run -n "${DATA_ENV}" \
  python "${V2_ROOT}/scripts/48_analyze_source_conditioned_cross_scores.py" \
  --project-root "${PROJECT_ROOT}" \
  --screening-index "${INDEX}" \
  --score-repeats "${A0_V2C_SCORE_REPEATS}" \
  --bootstrap-reps "${A0_V2C_BOOTSTRAP_REPS}" \
  --bootstrap-seed "${A0_V2C_BOOTSTRAP_SEED}"

echo
echo "V2C source-conditioned generation and cross-scoring completed."
echo "Review:"
echo "  ${V2_ROOT}/V2C_INDEX${INDEX}_SOURCE_CONDITIONED.md"
echo "  ${V2_ROOT}/metrics/index${INDEX}_source_conditioned_generation.json"
echo "  ${V2_ROOT}/metrics/index${INDEX}_source_conditioned_cross_scoring.json"
echo "  ${V2_ROOT}/manifests/index${INDEX}_source_conditioned_candidates.tsv"
echo "  ${V2_ROOT}/manifests/index${INDEX}_source_conditioned_cross_scores.tsv"
echo
echo "Stop here. Do not interpret the overall rank correlation as source-neutral stability."
