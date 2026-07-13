# RUNBOOK — Entity-Resolution PoC

Operator guide for bringing up, monitoring, and recovering the Splink 4.0.16 Fellegi-Sunter ER pipeline.

---

## Bring-up

```bash
cd /home/deploy/squad/build-worker/JOB-20260712170600-000136
pip install pandas pyarrow numpy duckdb splink
bash scripts/end_to_end_smoketest.sh
```

Total time on a warm cache: **~5 seconds**. Cold cache: **~1 minute** (Splink pulls a few hundred MB of transitively-pinned packages).

---

## Monitoring

### Health checks

```bash
# Validate ER summary
cat apps/entity/reports/er_summary.json | jq .

# Validate waterfall summary
cat apps/entity/reports/waterfall_summary.json | jq .

# Confirm audit dashboard exists
ls -la apps/entity/reports/audit_dashboard.html
```

### Throughput

The pipeline logs the cluster count + FPR for each run. Watch the trend over multiple runs on the same dataset — random fluctuations of < 0.5% FPR are expected; larger drifts warrant investigation (model drift, schema change, vendor update).

---

## Common failure modes

### F1. `splink.blocking_library` import fails

**Cause:** Splink 4 renamed the module. Use `splink.blocking_rule_library` (singular `rule`) or import `block_on` directly from `splink`.

**Fix:** confirmed in our code at `apps/entity/src/er_engine.py:69`:
```python
from splink import block_on
from splink.comparison_library import JaroWinklerAtThresholds, ExactMatch
```

### F2. `JaroWinklerAtThresholds.__init__() got an unexpected keyword argument 'score_thresholds'`

**Cause:** Splink 4 renamed the keyword to `score_threshold_or_thresholds` (note the singular `threshold_or_thresholds`).

**Fix:**
```python
JaroWinklerAtThresholds("first", score_threshold_or_thresholds=[0.88, 0.70])
```

### F3. `JaroWinklerAtThresholds(...).get_comparison("duckdb")` then `Linker` complains about `'Comparison' object has no attribute 'get_comparison'`

**Cause:** Splink 4 expects the **creator** object (the result of `JaroWinklerAtThresholds(...)`) directly in settings, not the resolved `Comparison`.

**Fix:** pass the creator:
```python
JaroWinklerAtThresholds("first", score_threshold_or_thresholds=[0.88, 0.70]),  # not .get_comparison(...)
```

### F4. `Missing column(s) from input dataframe(s): "first_name"` warnings

**Cause:** the column doesn't exist in the input data with that exact name. Our normalized data uses `first` and `last`, not `first_name` and `last_name`.

**Fix:** match the column names in settings to the actual df column names.

### F5. `probability_two_random_records_match` warning + "first (no m values are trained)"

**Cause:** Some blocking rules didn't produce enough training data for EM to estimate m values for every comparison.

**Fix (production):** provide labeled holdout data:
```python
linker.training.estimate_m_from_label_column("ground_truth_pid")
```

**In the demo:** warning is benign. The model still produces good predictions because u estimates carry the heavy lifting.

### F6. ER pipeline runs in stub mode

**Symptom:** log shows `Splink not installed; running in STUB mode` and the cluster_id is just the ground_truth_pid.

**Cause:** `splink` package not installed. The stub is a deterministic fallback so the rest of the pipeline (waterfall + audit dashboard) still runs.

**Fix:**
```bash
pip install splink[duckdb]
```

---

## Recovery procedures

### R1. Re-run the full pipeline

```bash
bash scripts/end_to_end_smoketest.sh
```

This regenerates everything in `apps/entity/data/` and `apps/entity/reports/`.

### R2. Re-run just ER (without regenerating the dataset)

```bash
python -m apps.entity.src.er_engine
```

### R3. Re-run just the audit dashboard

```bash
python -m apps.entity.src.audit_dashboard
```

### R4. Reset everything from scratch

```bash
rm -rf apps/entity/data apps/entity/reports
bash scripts/end_to_end_smoketest.sh
```

---

## Production hardening checklist

If you fork this PoC into a real system, here's the path:

- [ ] Replace synthetic tier prices with real negotiated vendor pricing
- [ ] Wire the simulator into an Airflow DAG (replace `simulate()` with DAG tasks)
- [ ] Add a labeled-holdout evaluation step (estimate_m_from_label_column)
- [ ] Add a Suppression / opt-out pipeline (GDPR-grade; the core matching
      stays the same, just don't link records whose source has a removal flag)
- [ ] Add an Elasticsearch serving layer (index `clusters.parquet` for
      low-latency lookup)
- [ ] Add a Neo4j / Memgraph identity-graph store (cluster → relationships)
- [ ] Add libpostal address normalization (drop-in for the raw
      `address_full` field)
- [ ] Add LangSmith-style LLM-eval hooks (even though ER is rule-based,
      you'll want a feedback loop on which "review queue" entries get
      human verdict)
- [ ] Run on a held-out vendor cross-section (real LexisNexis vs. real Pipl)
      to validate the FPR against your real labeled corpus
- [ ] Wire the per-vendor scorecard to a monthly cron that emails the team

---

## Operational metrics worth tracking

| Metric | Healthy | Source |
|---|---|---|
| FPR (on labeled holdout) | < 0.5% | `er_summary.json` |
| n_clusters / n_universe | 0.95-1.05 | `er_summary.json` |
| Cost-per-resolved profile | declines over time | `waterfall_summary.json` |
| Waterfall tier-depth median | 2-3 (most queries resolve at cheap tiers) | `waterfall_summary.json` |
| Vendor freshness (median record age) | < 90 days | per-vendor scorecard (production) |

---

## License

MIT. Synthetic data + Splink integration is fully runnable. Vendor pricing
and hit-rates in the waterfall are placeholders.
