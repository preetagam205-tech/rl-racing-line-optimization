# Experimental benchmark results

## Protocol

This project evaluated four controllers on five circuit families: **ellipse, wavy, complex, spline, hard spline**. Each family was tested on **20 held-out procedurally generated circuits**, with trial IDs 10000–10019. Each controller therefore ran **100 episodes**.

The final PPO / evolutionary comparison reused the same `RacingEnv` vehicle dynamics, track generation, start index, checkpoint logic, and lap completion rules.

| Controller | Description |
|---|---|
| Tuned Pure Pursuit | Centerline geometry with curvature-aware target speed |
| Broad Domain PPO | PPO trained across varied circuit layouts |
| Hard Five Family PPO | PPO model trained for hard/mixed circuit families |
| Evolved geometric | Genetic search over geometric controller parameters, evaluated in the same vehicle physics |

### Lap completion (%)

| Controller | Ellipse | Wavy | Complex | Spline | Hard spline | All circuits |
|---|---:|---:|---:|---:|---:|---:|
| Tuned Pure Pursuit | 100 | 100 | 100 | 100 | 100 | 100 |
| Broad Domain PPO | 100 | 100 | 100 | 70 | 30 | 80 |
| Hard Five Family PPO | 100 | 85 | 50 | 55 | 40 | 66 |
| Evolved geometric | 35 | 60 | 90 | 100 | 100 | 77 |

### Mean lap time (s; *completed laps only*)

| Controller | Ellipse | Wavy | Complex | Spline | Hard spline |
|---|---:|---:|---:|---:|---:|
| Tuned Pure Pursuit | 20.762 | 20.317 | 23.170 | 28.660 | 31.465 |
| Broad Domain PPO | 15.363 | 15.893 | 15.810 | 16.979 | 16.100 |
| Hard Five Family PPO | 16.905 | 17.015 | 16.675 | 18.568 | 17.475 |
| Evolved geometric | 15.429 | 15.188 | 16.344 | 19.238 | 20.170 |

### Matched-circuit comparison: Broad Domain PPO and tuned Pure Pursuit

Both drivers had to **complete the same circuit** for it to enter this comparison.

| Family | Paired successful circuits | Mean lap-time reduction from baseline | Mean path-length change |
|---|---:|---:|---:|
| Ellipse | 20 | 25.90% | -1.98% |
| Wavy | 20 | 21.79% | -1.09% |
| Complex | 20 | 31.40% | -1.51% |
| Spline | 14 | 38.28% | -2.08% |
| Hard spline | 6 | 46.09% | -1.12% |

The per-circuit percentage differences were averaged; these values can differ slightly from percentages calculated from mean lap times.

## What the results support

- **Pure Pursuit:** strongest lap-completion reliability (100/100).
- **Broad Domain PPO:** 80/100 completions and fast completed laps, particularly on ellipse/wavy/complex circuits.
- **Evolved geometric controller:** strong results on spline and hard-spline families (40/40 completed), but poor ellipse and wavy completion.
- **Steering smoothness:** the evolved geometric controller typically has smaller average changes to steering actions than PPO in these trials.
- **Geometric efficiency:** Broad Domain's completed matched laps travel about 1–2% less distance than Pure Pursuit's completed paths.

## Limitations

- The evolved geometric policy uses the **full centerline**; PPO uses local observations. A fair comparison of *algorithms alone* would require controlling the information provided to each policy.
- Optimization and training budgets are different.
- The Pure Pursuit baseline uses heuristic speed limits; differences in lap time also measure **speed planning and control**, not just racing-line geometry.
- Completion selection matters: PPO lap-time averages on hard circuits include only successful cases.
- Each family has only 20 trials and some matched subsets are small.
- Track novelty is restricted to held-out seeds in known track-generator families.
- **No globally optimal trajectory benchmark** is provided.
- The raw benchmark CSVs, complete code, and policy checkpoints must be committed or linked separately for independent replication.

These qualifications are part of the reported findings, not afterthoughts.
