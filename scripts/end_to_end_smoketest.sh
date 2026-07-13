#!/usr/bin/env bash
# end_to_end_smoketest.sh — generate dataset, run ER, run waterfall, render audit.
#
# Usage: bash scripts/end_to_end_smoketest.sh
#
# Prereqs: pandas, pyarrow, numpy. Splink is OPTIONAL — the engine has a
# deterministic stub that runs end-to-end without Splink installed, and the
# auditor still produces correct cluster assignments against ground truth.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> [1/4] Generate synthetic multi-vendor dataset"
python3 -m apps.entity.src.generate_dataset

echo "==> [2/4] Run entity resolution"
python3 -m apps.entity.src.er_engine

echo "==> [3/4] Simulate cost waterfall"
python3 -m apps.entity.src.cost_waterfall

echo "==> [4/4] Render false-merge audit dashboard"
python3 -m apps.entity.src.audit_dashboard

echo
echo "Artifacts written to apps/entity/reports/:"
ls -la apps/entity/reports/

echo
echo "  Open audit dashboard: file://$REPO_ROOT/apps/entity/reports/audit_dashboard.html"
echo "  View waterfall summary: cat apps/entity/reports/waterfall_summary.json | jq"
echo "  View ER summary:        cat apps/entity/reports/er_summary.json | jq"
