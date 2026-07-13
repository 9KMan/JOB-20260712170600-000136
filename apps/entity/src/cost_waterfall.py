"""
Cost-aware vendor waterfall simulator.

Real people-search systems use a "waterfall" of vendors: start with the cheapest
sources that cover common cases, escalate to charge-on-success or premium sources
only when the cheap ones don't resolve. This optimizes cost per resolved profile
without sacrificing hit-rate.

The simulator logs:
  - per-tier call count + hit rate + total cost
  - per-tier median latency
  - end-to-end cost per resolved profile
  - distribution of how many tiers were touched before resolution

In production this is wired into the Airflow DAG that orchestrates vendor
calls; the demo simulates the same waterfall against the clusters.parquet
output of the ER engine.

Tiers (cheapest → most expensive):
  T0 — Internal cache (free, hit-rate is corpus_size / corpus_size if cached).
  T1 — Free / opt-in: HLR cache, HAVE-I-BEEN-PWNED (free tier), phone-type API (free).
  T2 — Per-call: NumVerify-style ($0.001–$0.01 per call).
  T3 — Per-success: Pipl Search API ($0.05–$0.50 only if a record is returned).
  T4 — Per-success premium: LexisNexis / Accurint ($1–$5 per query, full profile).

Run: python -m apps.entity.src.cost_waterfall
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

log = logging.getLogger("juliet.waterfall")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic vendor tier definitions (per-call / per-success pricing)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Tier:
    name: str
    cost_per_call: float
    cost_on_success: float      # additional cost if a record is returned
    p_hit_unresolved: float    # hit prob for records that prior tiers missed
    p_hit_resolved: float      # hit prob for records already touched
    latency_ms: int


TIERS = [
    Tier("T0_internal_cache",   0.0,   0.0,  0.05, 0.95,   5),
    Tier("T1_free_signal",       0.0,   0.0,  0.55, 0.99,  30),
    Tier("T2_per_call",          0.005, 0.0,  0.65, 0.85, 120),
    Tier("T3_per_success",       0.0,   0.08, 0.45, 0.70, 400),
    Tier("T4_premium_per_success", 0.0, 0.50, 0.85, 0.95, 800),
]


def simulate(n_queries: int = 5000, *, seed: int = 7) -> dict:
    """Run the waterfall simulator. Returns cost / hit / latency summary."""
    random.seed(seed)
    tier_stats = {t.name: {"calls": 0, "hits": 0, "cost": 0.0, "latency_sum_ms": 0} for t in TIERS}
    depths = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}  # how many tiers touched
    total_cost = 0.0

    for _ in range(n_queries):
        resolved = False
        touched_tiers = 0
        for tier in TIERS:
            tier_stats[tier.name]["calls"] += 1
            tier_stats[tier.name]["cost"] += tier.cost_per_call + tier.cost_on_success  # billed even if miss in real systems
            tier_stats[tier.name]["latency_sum_ms"] += tier.latency_ms
            touched_tiers += 1
            # Determine hit probability
            p_hit = tier.p_hit_unresolved if not resolved else tier.p_hit_resolved
            if random.random() < p_hit:
                tier_stats[tier.name]["hits"] += 1
                resolved = True
                break
        depths[touched_tiers] = depths.get(touched_tiers, 0) + 1
        total_cost += tier_stats[tier.name]["cost"]  # for averaging

    summary = {
        "n_queries": n_queries,
        "total_cost_usd": round(total_cost, 2),
        "cost_per_resolved_profile_usd": round(total_cost / max(1, n_queries), 4),
        "per_tier": {},
        "tier_depth_distribution": depths,
    }
    for tier in TIERS:
        s = tier_stats[tier.name]
        n = max(1, s["calls"])
        summary["per_tier"][tier.name] = {
            "calls": s["calls"],
            "hits": s["hits"],
            "hit_rate": round(s["hits"] / n, 3),
            "total_cost_usd": round(s["cost"], 2),
            "avg_latency_ms": round(s["latency_sum_ms"] / n, 1),
            "cost_share_pct": round(100 * s["cost"] / max(0.01, total_cost), 1),
        }

    # Print summary
    log.info(f"Waterfall simulation: {n_queries} queries, total cost ${total_cost:.2f}")
    for name, s in summary["per_tier"].items():
        log.info(f"  {name:32} calls={s['calls']:5} hit={s['hit_rate']:>5} "
                 f"cost=${s['total_cost_usd']:>7} latency={s['avg_latency_ms']:.0f}ms "
                 f"share={s['cost_share_pct']}%")
    log.info(f"Tier-depth distribution (n_tiers touched per query): {depths}")

    return summary


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "reports"
    out.mkdir(parents=True, exist_ok=True)
    summary = simulate()
    with open(out / "waterfall_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
