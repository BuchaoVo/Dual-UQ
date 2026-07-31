#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/mnt/data/users/zbc/AI4S/ProteinDesign/Dual-UQ}"
DATA_ENV="${2:-dual-uq}"
MODEL_ENV="${3:-dual-uq-model}"
V2_ROOT="${PROJECT_ROOT}/experiments/a0_validation/v2_cross_mechanism"
RUNTIME_ENV="${PROJECT_ROOT}/experiments/a0_validation/v1_proteinmpnn/config/runtime.env"
CONFIG="${V2_ROOT}/config/generalized_paired_pdb.env"

if [[ ! -f "${RUNTIME_ENV}" ]]; then
  echo "Missing ProteinMPNN runtime config: ${RUNTIME_ENV}"
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing V2B2 config: ${CONFIG}"
  echo "Create it with:"
  echo "  cp ${V2_ROOT}/config/generalized_paired_pdb.env.example ${CONFIG}"
  exit 2
fi

# shellcheck disable=SC1090
source "${RUNTIME_ENV}"
# shellcheck disable=SC1090
source "${CONFIG}"

: "${PROTEINMPNN_ROOT:?PROTEINMPNN_ROOT is required}"
PARSER="${PROTEINMPNN_ROOT}/helper_scripts/parse_multiple_chains.py"
if [[ ! -f "${PARSER}" ]]; then
  echo "ProteinMPNN parser not found: ${PARSER}"
  exit 2
fi

mkdir -p \
  "${V2_ROOT}/logs" \
  "${V2_ROOT}/metrics" \
  "${V2_ROOT}/manifests" \
  "${V2_ROOT}/inputs/index${A0_V2B2_INDEX}/paired_pdb" \
  "${V2_ROOT}/inputs/index${A0_V2B2_INDEX}/jsonl"

LOG_FILE="${V2_ROOT}/logs/v2b2_generalized_paired_pdb_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "${LOG_FILE}") 2>&1

echo "Step 1/4: write paired derived PDB files"
conda run -n "${DATA_ENV}" \
  python "${V2_ROOT}/scripts/45_write_generalized_paired_pdbs.py" \
  --project-root "${PROJECT_ROOT}" \
  --screening-index "${A0_V2B2_INDEX}" \
  --output-chain "${A0_V2B2_OUTPUT_CHAIN}" \
  --required-backbone-atoms "${A0_V2B2_REQUIRED_BACKBONE_ATOMS}" \
  --high-confidence-threshold "${A0_V2B2_HIGH_CONFIDENCE_THRESHOLD}"

PAIRED_DIR="${V2_ROOT}/inputs/index${A0_V2B2_INDEX}/paired_pdb"
JSONL_DIR="${V2_ROOT}/inputs/index${A0_V2B2_INDEX}/jsonl"
PDB_FILE="${PAIRED_DIR}/index${A0_V2B2_INDEX}_pdb_mapped.pdb"
AFDB_FILE="${PAIRED_DIR}/index${A0_V2B2_INDEX}_afdb_mapped.pdb"

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "${TMP_ROOT}"' EXIT
mkdir -p "${TMP_ROOT}/pdb" "${TMP_ROOT}/afdb"
cp "${PDB_FILE}" "${TMP_ROOT}/pdb/"
cp "${AFDB_FILE}" "${TMP_ROOT}/afdb/"

echo "Step 2/4: official ProteinMPNN parse of PDB-derived input"
conda run -n "${MODEL_ENV}" \
  python "${PARSER}" \
  --input_path "${TMP_ROOT}/pdb" \
  --output_path "${JSONL_DIR}/index${A0_V2B2_INDEX}_pdb.jsonl"

echo "Step 3/4: official ProteinMPNN parse of AFDB-derived input"
conda run -n "${MODEL_ENV}" \
  python "${PARSER}" \
  --input_path "${TMP_ROOT}/afdb" \
  --output_path "${JSONL_DIR}/index${A0_V2B2_INDEX}_afdb.jsonl"

echo "Step 4/4: validate paired JSONL records"
conda run -n "${DATA_ENV}" \
  python "${V2_ROOT}/scripts/46_validate_generalized_pair_jsonl.py" \
  --project-root "${PROJECT_ROOT}" \
  --screening-index "${A0_V2B2_INDEX}"

echo
echo "V2B2 generalized paired inputs completed."
echo "Review:"
echo "  ${V2_ROOT}/V2B2_INDEX${A0_V2B2_INDEX}_PAIRED_INPUT_VALIDATION.md"
echo "  ${V2_ROOT}/metrics/index${A0_V2B2_INDEX}_paired_pdb_build.json"
echo "  ${V2_ROOT}/metrics/index${A0_V2B2_INDEX}_paired_input_validation.json"
echo
echo "Stop here. Do not start sequence generation until validation is reviewed."
