#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PART_PREFIX="$ROOT/data/wrds/returns_daily.parquet.xz.part"
OUT_XZ="$ROOT/data/wrds/returns_daily.parquet.xz"
OUT_PARQUET="$ROOT/data/wrds/returns_daily.parquet"
if [[ -f "$OUT_PARQUET" ]]; then
  echo "[unpack] $OUT_PARQUET already exists; remove it first if you need to rebuild" >&2
  exit 0
fi
if ls "${PART_PREFIX}"* >/dev/null 2>&1; then
  cat "${PART_PREFIX}"* > "$OUT_XZ"
  xz -d "$OUT_XZ"
  echo "[unpack] restored $OUT_PARQUET"
else
  echo "[unpack] part files not found under ${PART_PREFIX}*" >&2
  exit 1
fi
