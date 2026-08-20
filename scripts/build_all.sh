#!/usr/bin/env bash
# Build the kernels for all the compute capabilities in ARCHS.
# CUDA 12 nvcc builds 70 and 75. CUDA 13 nvcc builds only 75.
set -euo pipefail

OUT="${1:-src/colabfold_legacy_kernels/kernels}"
ARCHS="${ARCHS:-70 75}"
repo="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

if [ -z "${CUTLASS_DIR:-}" ]; then
  export CUTLASS_DIR="$(bash "$repo/scripts/fetch_cutlass.sh" "$work/cutlass")"
fi

for arch in $ARCHS; do
  echo "=== sm_${arch} ==="
  ARCH="$arch" bash "$repo/scripts/build_kernels.sh" "$OUT"
done
