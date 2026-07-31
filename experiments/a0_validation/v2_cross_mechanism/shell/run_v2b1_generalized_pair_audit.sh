#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/mnt/data/users/zbc/AI4S/ProteinDesign/Dual-UQ}"
DATA_ENV="${2:-dual-uq}"
V2_ROOT="${PROJECT_ROOT}/experiments/a0_validation/v2_cross_mechanism"
CONFIG="${V2_ROOT}/config/generalized_pair_audit.env"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing generalized pair audit config: ${CONFIG}"
  echo "Create it with:"
  echo "  cp ${V2_ROOT}/config/generalized_pair_audit.env.example ${CONFIG}"
  exit 2
fi

# shellcheck disable=SC1090
source "${CONFIG}"

mkdir -p "${V2_ROOT}/logs" "${V2_ROOT}/metrics" "${V2_ROOT}/manifests"
LOG_FILE="${V2_ROOT}/logs/v2b1_generalized_pair_audit_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "${LOG_FILE}") 2>&1

conda run -n "${DATA_ENV}" \
  python "${V2_ROOT}/scripts/44_audit_generalized_pair_inputs.py" \
  --project-root "${PROJECT_ROOT}" \
  --screening-index "${A0_V2B_INDEX}" \
  --min-common-residues "${A0_V2B_MIN_COMMON_RESIDUES}" \
  --min-common-fraction "${A0_V2B_MIN_COMMON_FRACTION}" \
  --required-backbone-atoms "${A0_V2B_REQUIRED_BACKBONE_ATOMS}"

echo
echo "V2B1 generalized paired-input audit completed."
echo "Review:"
echo "  ${V2_ROOT}/V2B1_INDEX${A0_V2B_INDEX}_GENERALIZED_PAIR_AUDIT.md"
echo "  ${V2_ROOT}/metrics/index${A0_V2B_INDEX}_generalized_pair_audit.json"
echo "  ${V2_ROOT}/manifests/index${A0_V2B_INDEX}_paired_input_plan.tsv"
echo "  ${V2_ROOT}/manifests/index${A0_V2B_INDEX}_paired_input_decision.json"
echo
echo "Stop here. Do not write derived paired PDB files until the selected representations are reviewed."
