#!/usr/bin/env bash
set -euo pipefail

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib python3 tools/phase4_case_studies.py run --out-root results/dsh_validation/phase4_case_studies
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib python3 tools/phase4_case_studies.py verify --out-root results/dsh_validation/phase4_case_studies
