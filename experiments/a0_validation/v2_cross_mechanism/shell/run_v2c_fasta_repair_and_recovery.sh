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
  exit 2
fi

# shellcheck disable=SC1090
source "${RUNTIME_ENV}"
# shellcheck disable=SC1090
source "${CONFIG}"

: "${PROTEINMPNN_ROOT:?PROTEINMPNN_ROOT is required}"
: "${PROTEINMPNN_CHECKPOINT:?PROTEINMPNN_CHECKPOINT is required}"

MODEL_WEIGHTS_DIR="$(dirname "${PROTEINMPNN_CHECKPOINT}")"
MODEL_NAME="$(basename "${PROTEINMPNN_CHECKPOINT}")"
MODEL_NAME="${MODEL_NAME%.pt}"
MODEL_NAME="${MODEL_NAME%.pth}"

INDEX="${A0_V2C_INDEX}"
JSONL_DIR="${V2_ROOT}/inputs/index${INDEX}/jsonl"
PDB_JSONL="${JSONL_DIR}/index${INDEX}_pdb.jsonl"
AFDB_JSONL="${JSONL_DIR}/index${INDEX}_afdb.jsonl"
SCORE_ROOT="${V2_ROOT}/outputs/index${INDEX}/source_conditioned_cross_scoring"
JOBS="${V2_ROOT}/manifests/index${INDEX}_source_conditioned_score_jobs.tsv"
REPAIR="${V2_ROOT}/scripts/50_materialize_source_conditioned_fastas.py"
VALIDATOR="${V2_ROOT}/scripts/49_validate_source_conditioned_score_job.py"
ANALYZER="${V2_ROOT}/scripts/48_analyze_source_conditioned_cross_scores.py"

for required in \
  "${PDB_JSONL}" \
  "${AFDB_JSONL}" \
  "${REPAIR}" \
  "${VALIDATOR}" \
  "${ANALYZER}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required file: ${required}"
    exit 2
  fi
done

mkdir -p "${V2_ROOT}/logs" "${SCORE_ROOT}/pdb" "${SCORE_ROOT}/afdb"
LOG_FILE="${V2_ROOT}/logs/v2c_fasta_repair_and_recovery_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "${LOG_FILE}") 2>&1

echo "Step R1: materialize and validate all candidate FASTA files"
conda run -n "${DATA_ENV}" \
  python "${REPAIR}" \
  --project-root "${PROJECT_ROOT}" \
  --screening-index "${INDEX}" \
  --expected-candidates 9

mapfile -t JOB_LINES < <(tail -n +2 "${JOBS}")
if [[ "${#JOB_LINES[@]}" -ne 9 ]]; then
  echo "Unexpected score-job count: ${#JOB_LINES[@]} (expected 9)"
  exit 2
fi

echo
echo "Step R2: validate or run all 18 score jobs"

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
      --expected-repeats "${A0_V2C_SCORE_REPEATS}" \
      >/dev/null 2>&1; then
    echo "Skip completed score candidate=${CANDIDATE_ID} backbone=${BACKBONE}"
    return 0
  fi

  echo "Run missing/incomplete score candidate=${CANDIDATE_ID} backbone=${BACKBONE}"
  rm -rf "${OUTPUT_DIR}"
  mkdir -p "${OUTPUT_DIR}"

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

  conda run -n "${DATA_ENV}" \
    python "${VALIDATOR}" \
    --output-dir "${OUTPUT_DIR}" \
    --candidate-fasta "${FASTA_PATH}" \
    --expected-repeats "${A0_V2C_SCORE_REPEATS}"
}

for LINE in "${JOB_LINES[@]}"; do
  IFS=$'\t' read -r CANDIDATE_ID FASTA_PATH EXTRA <<< "${LINE}"

  if [[ -z "${CANDIDATE_ID}" || -z "${FASTA_PATH}" ]]; then
    echo "Malformed TSV row: ${LINE}"
    exit 2
  fi
  if [[ -n "${EXTRA:-}" ]]; then
    echo "Unexpected extra TSV field: ${LINE}"
    exit 2
  fi
  if [[ ! -f "${FASTA_PATH}" ]]; then
    echo "Candidate FASTA still missing after repair: ${FASTA_PATH}"
    exit 2
  fi

  score_candidate "pdb" "${PDB_JSONL}" "${CANDIDATE_ID}" "${FASTA_PATH}"
  score_candidate "afdb" "${AFDB_JSONL}" "${CANDIDATE_ID}" "${FASTA_PATH}"
done

echo
echo "Step R3: require 18 valid score records"

COMPLETE=0
for LINE in "${JOB_LINES[@]}"; do
  IFS=$'\t' read -r CANDIDATE_ID FASTA_PATH EXTRA <<< "${LINE}"
  for BACKBONE in pdb afdb; do
    OUTPUT_DIR="${SCORE_ROOT}/${BACKBONE}/${CANDIDATE_ID}"
    if conda run -n "${DATA_ENV}" \
        python "${VALIDATOR}" \
        --output-dir "${OUTPUT_DIR}" \
        --candidate-fasta "${FASTA_PATH}" \
        --expected-repeats "${A0_V2C_SCORE_REPEATS}" \
        >/dev/null; then
      COMPLETE=$((COMPLETE + 1))
    fi
  done
done

echo "Complete score records: ${COMPLETE}/18"
if [[ "${COMPLETE}" -ne 18 ]]; then
  exit 1
fi

echo
echo "Step 5/5: analyze paired cross-backbone scores"

conda run -n "${DATA_ENV}" \
  python "${ANALYZER}" \
  --project-root "${PROJECT_ROOT}" \
  --screening-index "${INDEX}" \
  --score-repeats "${A0_V2C_SCORE_REPEATS}" \
  --bootstrap-reps "${A0_V2C_BOOTSTRAP_REPS}" \
  --bootstrap-seed "${A0_V2C_BOOTSTRAP_SEED}"

echo
echo "V2C FASTA repair and Step 4/5 recovery completed."
echo "Review:"
echo "  ${V2_ROOT}/V2C_INDEX${INDEX}_SOURCE_CONDITIONED.md"
echo "  ${V2_ROOT}/metrics/index${INDEX}_candidate_fasta_repair.json"
echo "  ${V2_ROOT}/metrics/index${INDEX}_source_conditioned_cross_scoring.json"
