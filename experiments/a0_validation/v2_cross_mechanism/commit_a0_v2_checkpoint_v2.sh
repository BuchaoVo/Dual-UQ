#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/mnt/data/users/zbc/AI4S/ProteinDesign/Dual-UQ}"
COMMIT_MESSAGE="${2:-feat: add A0 cross-mechanism validation pipeline}"

EXP_DIR="${PROJECT_ROOT}/experiments/a0_validation/v2_cross_mechanism"

cd "${PROJECT_ROOT}"

if [[ ! -d .git ]]; then
  echo "ERROR: not a Git repository: ${PROJECT_ROOT}" >&2
  exit 2
fi

if [[ ! -d "${EXP_DIR}" ]]; then
  echo "ERROR: experiment directory not found: ${EXP_DIR}" >&2
  exit 2
fi

echo "Repository: ${PROJECT_ROOT}"
echo "Branch: $(git branch --show-current)"
echo "HEAD before commit: $(git rev-parse --short HEAD)"
echo

echo "1/6 Validate Python syntax"
mapfile -t PY_FILES < <(
  find -L "${EXP_DIR}/scripts" -maxdepth 1 -type f -name '*.py' -print | sort
)

if [[ "${#PY_FILES[@]}" -eq 0 ]]; then
  echo "ERROR: no Python scripts found under ${EXP_DIR}/scripts" >&2
  exit 2
fi

python - "${PY_FILES[@]}" <<'PY'
import ast
import pathlib
import sys

for raw in sys.argv[1:]:
    path = pathlib.Path(raw)
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print("OK", path)
PY

echo
echo "2/6 Validate shell syntax"

# Follow symlinks and allow runner scripts one directory below the experiment root.
mapfile -t SH_FILES < <(
  find -L "${EXP_DIR}" -maxdepth 2 -type f -name 'run_*.sh' -print | sort
)

if [[ "${#SH_FILES[@]}" -eq 0 ]]; then
  echo "ERROR: no run_*.sh files found under ${EXP_DIR} (depth <= 2)." >&2
  echo "Top-level directory contents:" >&2
  find "${EXP_DIR}" -maxdepth 1 -printf '  %y %f -> %l\n' | sort >&2
  exit 2
fi

for file in "${SH_FILES[@]}"; do
  bash -n "${file}"
  echo "OK ${file}"
done

echo
echo "3/6 Check whitespace errors"
git diff --check

echo
echo "4/6 Stage only implementation code, examples, tests, and technical documentation"

STAGE_FILES=()

append_find_results() {
  local base="$1"
  shift
  [[ -d "${base}" ]] || return 0
  while IFS= read -r file; do
    [[ -n "${file}" ]] && STAGE_FILES+=("${file}")
  done < <(find -L "${base}" "$@" -print | sort)
}

append_find_results "${EXP_DIR}/scripts" \
  -maxdepth 1 -type f -name '*.py'

append_find_results "${EXP_DIR}/config" \
  -maxdepth 1 -type f -name '*.env.example'

# Preserve the path recorded by Git even when a runner is a symlink.
while IFS= read -r file; do
  [[ -n "${file}" ]] && STAGE_FILES+=("${file}")
done < <(
  find "${EXP_DIR}" -maxdepth 2 \
    \( -type f -o -type l \) \
    -name 'run_*.sh' -print | sort
)

while IFS= read -r file; do
  [[ -n "${file}" ]] && STAGE_FILES+=("${file}")
done < <(
  find "${EXP_DIR}" -maxdepth 1 \
    \( -type f -o -type l \) \
    \( -name 'README*.md' \
       -o -name 'V2*_HOTFIX.md' \
       -o -name 'V2*_RECOVERY.md' \
       -o -name 'V2*_REPAIR.md' \
       -o -name 'commit_a0_v2_checkpoint*.sh' \) \
    -print | sort
)

if [[ -d "${EXP_DIR}/tests" ]]; then
  append_find_results "${EXP_DIR}/tests" \
    -type f \
    \( -name '*.py' -o -name '*.json' -o -name '*.tsv' \)
fi

if [[ "${#STAGE_FILES[@]}" -eq 0 ]]; then
  echo "ERROR: no allowlisted files found to stage." >&2
  exit 2
fi

# Deduplicate paths while preserving deterministic order.
mapfile -t STAGE_FILES < <(printf '%s\n' "${STAGE_FILES[@]}" | sort -u)

git add -- "${STAGE_FILES[@]}"

echo
echo "5/6 Guard against local configs, generated data, logs, and model outputs"
mapfile -t STAGED < <(git diff --cached --name-only)

if [[ "${#STAGED[@]}" -eq 0 ]]; then
  echo "No staged changes. Nothing to commit."
  exit 0
fi

BAD_PATTERN='(^|/)(runtime\.env|generalized_paired_pdb\.env|source_conditioned\.env|source_neutral\.env)$|/(inputs|outputs|logs|metrics|manifests)/|\.npz$|\.pdb$|\.jsonl$|\.parquet$'
BAD_FILES=()

for file in "${STAGED[@]}"; do
  if [[ "${file}" =~ ${BAD_PATTERN} ]]; then
    BAD_FILES+=("${file}")
  fi
done

if [[ "${#BAD_FILES[@]}" -gt 0 ]]; then
  echo "ERROR: generated/local files were staged unexpectedly:" >&2
  printf '  %s\n' "${BAD_FILES[@]}" >&2
  echo "Unstaging files added by this script." >&2
  git restore --staged -- "${STAGED[@]}"
  exit 3
fi

echo "Staged files:"
printf '  %s\n' "${STAGED[@]}"

echo
echo "Staged diff summary:"
git diff --cached --stat

echo
echo "6/6 Commit checkpoint"
git commit -m "${COMMIT_MESSAGE}" \
  -m "Add generalized paired-backbone construction and validation for Index 36." \
  -m "Add source-conditioned generation/cross-scoring and source-neutral landscape workflows." \
  -m "Harden alternate-location selection, candidate FASTA materialization, FASTA-derived NPZ selection, resume validation, and LF-safe TSV handling."

echo
echo "Commit completed:"
git log -1 --oneline --decorate

echo
echo "Remaining working-tree changes:"
git status --short
