# Baseline Prioritization Summary (v2)

## Decisions
- **Unit of prioritization:** product × issue (264 groups)
- **Growth-rate method:** last 12 months vs prior 12 months, with +1 smoothing on denominator
- **Materiality floors (100k sample):** MIN_TOTAL_VOLUME = 50, MIN_PRIOR_VOLUME = 20. Groups below either threshold get `growth_z_floored = 0` — they can still rank on volume or rate signals.
- **Weights (v2):** volume_z 0.30 · growth_z_floored 0.35 · untimely 0.20 · monetary_relief 0.10 · recency 0.05

## Why the weight rebalance (v1 → v2)
- v1 correlation heatmap showed `growth_rate ↔ recency_weight` = 0.62 — partial redundancy.
- Cut `recency_weight` from 0.10 → 0.05; freed weight to `untimely_rate` (0.15 → 0.20), an under-weighted regulatory risk signal with low correlation to the others.
- `volume_z`, `growth_z_floored`, `monetary_relief_rate` unchanged.

## Top-20 sanity check
- Top-20 groups with volume < 50: **2** (target ≤3)
- Top-20 groups with volume < 100: **6**
- Top-20 median volume: **353**

## Top 5 priorities (v2)

1. **Credit reporting or other personal consumer reports — Incorrect information on your report**  
   volume=35209, prior_volume=10902, growth=+96.8%, untimely=0.1%, monetary_relief=0.0%, recency=60.9%, score=4.627

2. **Credit reporting or other personal consumer reports — Improper use of your report**  
   volume=15733, prior_volume=5748, growth=+42.8%, untimely=0.1%, monetary_relief=0.0%, recency=52.2%, score=1.998

3. **Debt collection — Took or threatened to take negative or legal action**  
   volume=769, prior_volume=103, growth=+320.2%, untimely=5.3%, monetary_relief=0.5%, recency=56.7%, score=1.849

4. **Credit reporting or other personal consumer reports — Problem with a company's investigation into an existing problem**  
   volume=11584, prior_volume=3992, growth=+59.4%, untimely=0.1%, monetary_relief=0.1%, recency=54.9%, score=1.604

5. **Credit reporting, credit repair services, or other personal consumer reports — Incorrect information on your report**  
   volume=7184, prior_volume=0, growth=+0.0%, untimely=0.3%, monetary_relief=0.1%, recency=0.0%, score=0.808

## Data caveats
- Scored on a 100k-row sample of the full 14.35M-row CFPB file. Rankings on the full file may shift.
- `consumer_complaint_narrative` (74% null) and `consumer_disputed` (94.6% null) are deliberately excluded from this baseline.
- Very small groups can still rank if they score highly on `untimely_rate` or `monetary_relief_rate` — this is acceptable (a small group with 100% monetary relief is genuinely worth investigating).
- Weights will be re-balanced once BERTopic themes, urgency scoring, and sentiment are added (Notebook 03+).
- When porting to the full 14.35M-row file, scale materiality thresholds ~143x (≈ MIN_TOTAL_VOLUME=500, MIN_PRIOR_VOLUME=200).
