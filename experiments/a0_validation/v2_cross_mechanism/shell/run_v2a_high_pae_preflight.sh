#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/mnt/data/users/zbc/AI4S/ProteinDesign/Dual-UQ}"
DATA_ENV="${2:-dual-uq}"
V2_ROOT="${PROJECT_ROOT}/experiments/a0_validation/v2_cross_mechanism"
CONFIG="${V2_ROOT}/config/high_pae_preflight.env"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing high-PAE preflight config: ${CONFIG}"
  echo "Create it with:"
  echo "  cp ${V2_ROOT}/config/high_pae_preflight.env.example ${CONFIG}"
  exit 2
fi

# shellcheck disable=SC1090
source "${CONFIG}"

mkdir -p "${V2_ROOT}/logs" "${V2_ROOT}/metrics" "${V2_ROOT}/manifests"
LOG_FILE="${V2_ROOT}/logs/v2a_high_pae_preflight_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "${LOG_FILE}") 2>&1

conda run -n "${DATA_ENV}" \
  python "${V2_ROOT}/scripts/43_resolve_and_preflight_high_pae.py" \
  --project-root "${PROJECT_ROOT}" \
  --screening-index "${A0_V2_PRIMARY_INDEX}" \
  --backup-index "${A0_V2_BACKUP_INDEX}" \
  --expected-mechanism "${A0_V2_EXPECTED_MECHANISM}" \
  --max-scan-file-mb "${A0_V2_MAX_SCAN_FILE_MB}"

echo
echo "V2A high-PAE preflight completed."
echo "Review:"
echo "  ${V2_ROOT}/V2A_INDEX${A0_V2_PRIMARY_INDEX}_HIGH_PAE_PREFLIGHT.md"
echo "  ${V2_ROOT}/metrics/index${A0_V2_PRIMARY_INDEX}_high_pae_preflight.json"
echo "  ${V2_ROOT}/manifests/index${A0_V2_PRIMARY_INDEX}_target_resolution.json"
echo
echo "Stop here. Do not build paired ProteinMPNN inputs until the target resolution is reviewed."
