# Section 03 — Plasticity and Recurrence

Replication package for Section 03 (*Plasticity and recurrence*) of **Forecasting Through Drift** (Adapt-1 Research Series, Part 2).

This folder packages the step-by-step traces, per-seed summaries, and audited `scorecard.json` for the two complementary benchmarks that map the plasticity–stability–recurrence tradeoff:

1. **Exp 96B — Recurring Nonlinear Regimes (`K = 6`, Controlled Synthetic Testbed, Live API vs. 8 Online Peers):** Tests whether latent regime archiving (`latent_belief` with `recurrence_enabled: true`) lets an online nonlinear policy (`sequential_q_mlp`) immediately reactivate an archived regime (`A1 -> B1 -> C -> A2`) without losing stability, and exposes the boundary when a reward law flips silently with zero context shift (`B2*`).
2. **Exp 79A & Paired Flush Ablation — 2021–2023 Stock–Bond Correlation Inversion (`SPY/TLT/HYG/GLD`, Replayed Empirical Market Data + Simulated 5 bps Overlay):** Tests smooth history retention (`W = 128`) against hard drift-triggered 21-day sample flushes and external probabilistic peers as 60-day `SPY/TLT` correlation flips from `-0.42` to `+0.32`.

> **Provenance Separation:**
> - **Exp 96B** is a controlled synthetic `K = 6` nonlinear bandit testbed (evaluated live against the hosted Adapt-1 API alongside 8 deterministic offline online bandit peers).
> - **Exp 79A** uses 100% real historical daily Yahoo Finance adjusted total-return prices (`SPY`, `TLT`, `HYG`, `GLD`; `2021-10-01` to `2023-06-30`, `n = 439` trading days; calibration strictly pre-evaluation `2020-01-02` to `2021-09-30`, `n = 441` days). Portfolio weights, tail-budget exposure scaling, correlation tilt, and 5 bps one-way transaction costs form an identical, strictly causal **simulated execution overlay** across all models.

---

## Files in This Folder

| File | Rows | Description |
|---|---:|---|
| `exp96b_n15_step_trace.csv` | 4,500 | `n=15` OOS seeds (`96001..96015`) × 300 steps: inputs, optima, live Adapt-1 Native & regime telemetry, and 8 online peers. |
| `exp96b_live_ablations_step_trace.csv` | 9,000 | Paired `n=8` live ablations (`adapt1_lb_native`, `adapt1_lb_off`, `adapt1_lb_policy_reset`) + `n=6` `D=4` (`adapt1_lb_delayed_d4`), × 300 steps. |
| `exp96b_per_seed_summary.csv` | 234 | Per-seed metrics for `n15_primary` (10×15), `n8_paired_ablations` (3×8), and `n6_delayed_d4` (10×6). |
| `exp79a_spy_tlt_flip_trace.csv` | 439 | Daily `2021-10-01..2023-06-30` returns, 60d `SPY/TLT` corr, `ES_4D`, 5 bps equity/DD, `VaR99` breaches (Smooth, 21d Flush, 60/40, Q-HGBR, Chronos-2, TimesFM-3.0, DCC-GARCH-FHS, EWMA). |
| `scorecard.json` | — | Audited metrics, Holm–Bonferroni `p_H`, operating notes, column specs, and verification targets. |

---

## Audited Headline Results

### 1. Exp 96B — Recurring Nonlinear Regimes (`n = 15` Primary Seeds, `T = 300`)

| Model | Overall | A1–A2 | A2-10 | A2 Full | B2* L20 | Regret | p_H (A2-10) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Adapt-1 Native** (`latent_belief` ON) | **38.4%** | **47.8%** | **73.3%** | **78.7%** | 1.0% | 136.70 | **ANCHOR** |
| ADWIN Online HGBR | 36.6% | 40.2% | 52.7% | 55.8% | 35.7% | 136.37 | 0.0016 |
| Online NeuralUCB | 34.8% | 36.1% | 46.0% | 60.9% | 45.3% | 144.76 | 0.0036 |
| EWMA Epsilon-Greedy | 29.3% | 29.1% | 23.3% | 31.4% | 36.3% | 147.29 | 0.0005 |
| Stationary LinUCB | 29.3% | 32.8% | 41.3% | 43.8% | 23.3% | 151.94 | 0.0005 |
| ADWIN Online Ridge | 28.9% | 31.9% | 36.0% | 40.7% | 27.7% | 151.74 | 0.0005 |
| Sliding-Window LinUCB | 28.6% | 30.9% | 20.0% | 31.0% | 31.0% | 149.72 | 0.0005 |
| Discounted TS | 27.6% | 28.2% | 20.0% | 27.6% | 30.7% | 152.83 | 0.0005 |
| EXP3.S | 16.6% | 16.7% | 16.7% | 16.4% | 19.0% | 186.50 | 0.0005 |

- **Delayed Feedback (`D = 4` steps, `n = 6` seeds `96001..96006`):** `A2` first-10 accuracy holds at **65.0%** (vs. `Online_NeuralUCB` `38.3%`, `ADWIN_Online_HGBR` `31.7%`).
- **Paired Live Ablations (`n = 8` seeds `96001..96008`):**
  - Native (`latent_belief` ON): **78.8%** `A2` first-10 (`82.5%` full `A2`, `0.0%` `B2*` last-20, `136.57` regret)
  - Regime tracking OFF (`latent_belief.enabled = false`): **57.5%** `A2` first-10 (`57.7%` full `A2`, `0.0%` `B2*` last-20, `156.69` regret)
  - Intermediate blanket policy resets (`POST /policy/reset`, 16 resets across 8 seeds): **23.8%** `A2` first-10 (`37.1%` full `A2`, **1.2%** `B2*` last-20, `163.30` regret)

> **Reactivation Step-Indexing & Update Cadence Audit (Zero Spin):**
> In 0-indexed trace coordinates, Phase `A2` spans `step_t = 180..239` (1-indexed steps `181..240`). Because `POST /query` sets `update_memory_state: false` and `POST /feedback` follows each query:
> - On `step_t = 180` (1-indexed step 181, first query of `A2`), Adapt-1 immediately selects the optimal Regime-A route on **14/15 seeds (93.3%)**, while stored `most_likely_regime` reflects pre-feedback state (`3/15` already in `A1` pool).
> - Immediately after ingesting `step_t = 180` feedback, at `step_t = 181` (first post-feedback query of `A2`), **15/15 seeds (100%)** reactivate an archived `A1` regime (`mlr_in_a1_pool == 1`) and stay in archived `A1` regimes for `53–60 / 60` steps of `A2` (`13/15` at the final `step_t = 239` snapshot).
> - Under `D = 4` delayed feedback, all `6/6` seeds similarly flip into an archived `A1` regime at `step_t = 184` as soon as `step_t = 180` feedback matures.

### 2. Exp 79A — 2021–2023 `SPY/TLT` Correlation Flip (`n = 439` Days, Simulated 5 bps Overlay)

| Model | ES 4D | VaR99 (p_cc) | Max DD (5bp) | Net (5bp) | Notes |
|---|---:|---:|---:|---:|---|
| **Adapt-1 Smooth (W=128)** | **1.34525** | **5/439 (0.906 PASS)** | **-19.61%** | **-2.25%** | Zero flushes; preserves 99% tail support |
| Adapt-1 21d Flush (8 rot.) | 1.36707 | 8/439 (0.258 PASS) | -22.55% | -5.95% | Degrades `ES_4D` (`p_H=0.0189`, wkly NW HAC) |
| Static 60/40 | n/a | n/a | -28.74% | -9.72% | Unhedged stock–bond flip |
| Online Q-HGBR (Rolling) | 1.36179 | 23/439 (6.45e-10 FAIL) | -22.86% | -3.17% | Chronic 99% tail under-dispersion |
| Chronos-2 (TO GaussCop) | 1.36251 | 9/439 (0.127 PASS) | -21.58% | -4.00% | Frozen TSFM + Gaussian copula |
| TimesFM-3.0 (TO GaussCop) | 1.34467 | 8/439 (0.258 PASS) | -20.75% | -2.10% | Frozen TSFM + Gaussian copula |
| DCC-GARCH-FHS | 1.36487 | 3/439 (0.763 PASS) | -19.79% | -3.13% | Classical dynamic conditional corr. |
| EWMA Empirical Residuals | 1.33821 | 1/439 (0.146 PASS) | -18.55% | -1.34% | Conservative tail (`p_pof=0.0498`) |

---

## Exact Synthetic `K = 6` Generator (Exp 96B)

Anyone can regenerate the exact Exp 96B episodes bit-for-bit using only `hashlib` and `numpy`:

### Deterministic SHA-256 Seed Rule

```python
import hashlib
import numpy as np

def deterministic_seed(*parts) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest()[:4], "big")

# Initialize each episode RNG with exact arguments ("arena96b_seed", seed, 60):
# rng = np.random.default_rng(deterministic_seed("arena96b_seed", seed, 60))
# Primary OOS seeds (n = 15): [96001, 96002, ..., 96015]
# Paired live ablations (n = 8): [96001, ..., 96008]
# Delayed-feedback D=4 slice (n = 6): [96001, ..., 96006]
# Calibration DEV seeds (n = 5): [96901, ..., 96905]
```

### Nonlinear Zero-Mean Quadrant + Radial Geometry

Each step draws a 4D context vector `(x0, x1, reg_load, reg_budget)` and a latent nonlinear zone `z in {0, 1, 2}`:

- **Zone ordering within each 60-step phase:** `z = j % 3` for `j < 18` (balanced warm-up), then `z ~ Uniform({0, 1, 2})` via `rng.integers(0, 3)` for `j = 18..59`.
- **Zone `Z0` (inner disk):** `r = sqrt(Uniform(0, 0.18^2))`, `theta ~ Uniform(0, 2*pi)`, `(x0, x1) = (round(r*cos(theta), 4), round(r*sin(theta), 4))`.
- **Zone `Z1` (outer positive-XOR quadrants Q1/Q3):** `r ~ Uniform(0.38, 0.50)`, `q in {0, 2}` with prob `0.5`, `theta ~ Uniform(0.28, pi/2 - 0.28) + q * pi/2`.
- **Zone `Z2` (outer negative-XOR quadrants Q2/Q4):** `r ~ Uniform(0.38, 0.50)`, `q in {1, 3}` with prob `0.5`, `theta ~ Uniform(0.28, pi/2 - 0.28) + q * pi/2`.
- Because `E[x0 | Z_m] = E[x1 | Z_m] = 0` for all `m in {0, 1, 2}`, every linear and context-blind bandit is mathematically bounded at `<= 1/3 = 33.3%` accuracy (empirical `SEALED_OOS` means: `Z0 = [-0.0026, -0.0026]`, `Z1 = [-0.0095, -0.0146]`, `Z2 = [-0.0026, -0.0023]`).

### 5-Phase Schedule & Reward Law (`T = 300`, `phase_len = 60`)

For each seed, draw a seed-specific permutation `perm = rng.permutation(6).tolist()` to eliminate arm-index bias, and define optimal route mappings `(Z0, Z1, Z2) -> route_k`:

| Phase | Steps | Center `(reg_load, reg_budget)` + `U(-0.08, 0.08)` | Optimal Map `(Z0, Z1, Z2)` |
|---|---|---|---|
| `A1` | 0..59 | (-2.0, -2.0) | `(perm[0], perm[1], perm[2])` |
| `B1` | 60..119 | (+2.0, -2.0) | `(perm[3], perm[4], perm[5])` |
| `C` | 120..179 | (-2.0, +2.0) | `(perm[2], perm[5], perm[0])` |
| `A2` | 180..239 | (-2.0, -2.0) | `(perm[0], perm[1], perm[2])` (exact return to `A1`) |
| `B2_silent` (`B2*`) | 240..299 | (-2.0, -2.0) | `(perm[4], perm[3], perm[5])` (silent same-context inversion) |

At each step `t`, counterfactual rewards for all `a in {0..5}` are sampled independently and clipped to `[0, 1]` (`round(..., 4)`):
- Optimal arm (`a == opt_a`): `Normal(0.92, 0.055^2)`
- Same-regime wrong-zone arm (`a in active_set \ {opt_a}`): `Normal(0.24, 0.055^2)`
- Off-regime arm (`a not in active_set`): `Normal(0.12, 0.055^2)`

### Adapt-1 API Configuration (Exp 96B)

Endpoint base: `https://rei-neuroadapt-api.reilabs.org/api/v1` (configure via `ADAPT_API_BASE` and `ADAPT_API_KEY`; see `scripts/adapt1_client.py`).

- **Protocol:** Uniform cold-start exploration on steps `0..11` (2 pulls per arm, identical across all models), then 100% hands-off `POST /domains/{id}/query` (`selection_mode: "auto"`, `allow_exploration: true`, `update_memory_state: false`) followed by `POST /domains/{id}/feedback` (or delayed by `D = 4` steps for the delayed slice).
- **Domain config excerpt:**
  - `learning.policy`: `exploration_mode: "ucb"`, `exploration_strength: 1.25`, `model_max_weight: 0.85`, `min_context_observations: 2`, `decay_half_life_seconds: 0`
  - `learning.latent_belief`: `enabled: true` (`false` for `LB_OFF`), `recurrence_enabled: true`, `max_regimes: 12`, `novelty_threshold: 0.45`, `change_point_threshold: 0.25`, `change_point_patience: 2`, `recurrence_margin: 0.10`, `recurrence_min_support: 2`, `recurrence_reward_threshold: 0.20`, `historical_regime_weight: 0.05`, `temperature: 0.25`
  - `learning.training`: `model_type: "sequential_q_mlp"`, `min_samples: 12`, `retrain_interval: 6`, `epochs: 120`, `learning_rate: 0.04`
  - `learning.credit_assignment`: `mode: "eligibility_trace"`, `delay_path: "values.delay_steps"`, `discount: 0.90`

---

## Yahoo Finance `SPY/TLT/HYG/GLD` Overlay Protocol (Exp 79A & Hard-Flush Ablation)

### Data & Causal Split

- **Public market source:** Daily Yahoo Finance adjusted total-return prices for `SPY` (US equities), `TLT` (20+ year US Treasuries), `HYG` (high-yield credit), and `GLD` (gold), expressed as daily percent returns `(r_spy, r_tlt, r_hyg, r_gld)`.
- **Calibration window (frozen pre-evaluation):** `2020-01-02` to `2021-09-30` (`n = 441` trading days).
- **Evaluation window:** `2021-10-01` to `2023-06-30` (`n = 439` trading days, `92` ISO weeks), spanning the 2022 inflation rate-hike cycle as rolling 60-day `SPY/TLT` correlation inverted from `-0.4233` to `+0.3154`.
- **Targets & state features:** 4D volatility-standardized returns `z_k(t) = r_k(t) / ewma_sigma_k(t)` (where `ewma_sigma_k(t)` uses strictly `< t` returns), conditioned on 6 strictly causal state features (`f_port_vol`, `f_corr_sb`, `f_cred_vol`, `f_vov`, `f_dd_spy`, `f_corr_sg`). Joint stress indicator `y_stress_realized = 1` when at least 2 of 4 assets move `>= 1.0` calibration standard deviation (`std_1sigma_cal`: `SPY 1.6871%`, `TLT 1.1617%`, `HYG 0.8427%`, `GLD 1.1072%`).

### Simulated 5 bps Risk-Parity Overlay Rules (Identical Across All Models)

Before each day `t` realizes, every probabilistic model emits a weighted 4D predictive return cloud `{(pts_j, p_j)}`. Using parameters frozen strictly on the calibration window (`b_cvar99_target_pct = 2.28664`, `expo_min = 0.25`, `expo_max = 1.00`, `corr_tilt_cap = 0.50`, `stress_base_rate_cal = 0.174603`, `stress_throttle_max = 0.35`):

1. **Stock–bond correlation inversion tilt:** Start from base weights `w_base = [0.60, 0.20, 0.10, 0.10]` (`[SPY, TLT, HYG, GLD]`). When forecast correlation `corr_hat(SPY, TLT) > 0`, shift `frac = min(1, corr_hat / 0.50) * 0.50` of the `TLT` sleeve out of `TLT`, moving half into `GLD` and half into cash (`0%` yield).
2. **CVaR99 tail-budget exposure scaling:** Compute left-tail `CVaR99_hat` of the `60/20/10/10` portfolio under the predictive cloud and set `expo_cvar = clip(2.28664 / max(|CVaR99_hat|, 0.25), 0.25, 1.00)`.
3. **Joint-stress throttle:** With cloud stress probability `p_stress`, set `throttle = 1 - 0.35 * clip((p_stress - 0.174603) / (1 - 0.174603), 0, 1)` and final weights `w_t = w_tilt * max(0.25, expo_cvar * throttle)`.
4. **Transaction costs:** Net daily return = `dot(w_t, r_t) / 100 - sum(|w_t - w_{t-1}|) * 0.0005` (`5 bps` one-way).

### Adapt-1 Smooth Retention vs. Hard 21-Day Flush

- **Smooth Retention (`W = 128`):** Single live `futures` domain (`conditioning: "kernel"`, `numeric_transport: "none"`, `bandwidth: 0.35`, `max_samples: 128`, `max_support: 128`, `max_branches: 128`) retaining the rolling 128-day history throughout the 439-day window.
- **Hard 21-Day Flush:** Identical `futures` domain plus `learning.structure` (`drift_detection_enabled: true`, `drift_patience: 3`, `representation.maximum_lag: 2`); whenever live drift detection fires (8 rotations on `2021-10-29`, `2021-11-30`, `2021-12-30`, `2022-02-25`, `2022-03-28`, `2022-05-06`, `2022-06-07`, `2023-06-07`), the domain rotates and reseeds only the latest 21 trading days. In the `+1..+20d` post-rotation cold-start slice (`n = 156` days), truncating support degrades `ES_4D` by `+0.05237` (`p = 3.27e-07`), driving the full-window degradation (`1.34525 -> 1.36707`, weekly Holm `p = 0.0189`) and return drag (`-2.25% -> -5.95%`).

---

## Regime and Retention Comparisons

1. **Silent Same-Context Lock-In (`B2*`):** When optimal actions invert while context stays unchanged (`Exp 96B Phase B2_silent`), hands-off latent belief stays locked on the archived Regime-A policy (`1.0%` last-20 accuracy across `n = 15`). Blanket resets (`POST /policy/reset` on same-context regime spawns) also remain at `1.2%` on `B2*` while erasing Regime-A memory on `A2` (`78.8% -> 23.8%`). In the companion controlled 2-arm noiseless inversion benchmark (`n = 10` seeds), explicit failure-triggered `POST /policy/reset` (`154 ms`) cures one-way silent lock-in (`0.0% -> 97.3%` late accuracy, `10/10` recovered in `5.6` steps) at the same archive-wipe cost.
2. **Full 5-Phase Regret Parity vs. `ADWIN_Online_HGBR`:** Because `B2*` lock-in adds regret in phase 5, Adapt-1 Native ties `ADWIN_Online_HGBR` on 5-phase cumulative regret (`136.70` vs. `136.37`, `p_H = 0.9341`), while sweeping all 8/8 peers across the four observable recurring phases `A1..A2` (`89.25` vs. `101.00`, `p_H = 0.0076`).
3. **Hard Flush Variance on Noisy Markets:** Hard 21-day sample flushes help on abrupt noiseless structural breaks but starve 99% tail resolution on noisy financial returns, where smooth history retention (`W = 128`) dominates both `ES_4D` and drawdown-adjusted return.

---

## Recompute All Headline Metrics (Python 3 Stdlib Only)

Run from this directory (`python3 -c "..."`) to independently verify every headline metric directly from the CSVs:

```python
import csv
from collections import defaultdict

# 1. Exp 96B primary n=15 trace
rows96 = list(csv.DictReader(open("exp96b_n15_step_trace.csv")))
assert len(rows96) == 4500

a2_t180_opt = sum(int(r["adapt1_native_correct"]) for r in rows96 if int(r["step_t"]) == 180)
a2_t181_rec = sum(int(r["adapt1_native_mlr_in_a1_pool"]) for r in rows96 if int(r["step_t"]) == 181)
a2_f10 = [r for r in rows96 if 180 <= int(r["step_t"]) < 190]
a2_all = [r for r in rows96 if r["phase"] == "A2"]
b2_l20 = [r for r in rows96 if 280 <= int(r["step_t"]) < 300]

print(f"Exp 96B A2 t=180 opt hits: {a2_t180_opt}/15 | t=181 A1 pool recall: {a2_t181_rec}/15")
print(f"Exp 96B A2 first-10 acc  : Adapt-1={sum(int(r['adapt1_native_correct']) for r in a2_f10)/len(a2_f10):.4f} "
      f"ADWIN-HGBR={sum(int(r['adwin_hgbr_correct']) for r in a2_f10)/len(a2_f10):.4f} "
      f"NeuralUCB={sum(int(r['online_neuralucb_correct']) for r in a2_f10)/len(a2_f10):.4f}")
print(f"Exp 96B A2 full acc      : {sum(int(r['adapt1_native_correct']) for r in a2_all)/len(a2_all):.4f}")
print(f"Exp 96B B2* last-20 acc  : {sum(int(r['adapt1_native_correct']) for r in b2_l20)/len(b2_l20):.4f}")

# 2. Exp 96B paired n=8 ablations & n=6 delayed-D4
abls = list(csv.DictReader(open("exp96b_live_ablations_step_trace.csv")))
for arm in ("adapt1_lb_native", "adapt1_lb_off", "adapt1_lb_policy_reset", "adapt1_lb_delayed_d4"):
    sub = [r for r in abls if r["arm"] == arm and 180 <= int(r["step_t"]) < 190]
    b2s = [r for r in abls if r["arm"] == arm and 280 <= int(r["step_t"]) < 300]
    print(f"Ablation {arm:22s}: A2_f10={sum(int(r['is_correct']) for r in sub)/len(sub):.4f} "
          f"B2_last20={sum(int(r['is_correct']) for r in b2s)/len(b2s):.4f}")

# 3. Exp 79A 439-day SPY/TLT flip + smooth vs hard 21-day flush
rows79 = list(csv.DictReader(open("exp79a_spy_tlt_flip_trace.csv")))
assert len(rows79) == 439
for label, es_c, dd_c, eq_c in [
    ("Adapt-1 Smooth W128", "adapt1_smooth_es4d", "adapt1_smooth_dd5_pct", "adapt1_smooth_eq5"),
    ("Adapt-1 Hard 21d Flush", "adapt1_flush21d_es4d", "adapt1_flush21d_dd5_pct", "adapt1_flush21d_eq5"),
    ("Online Q-HGBR", "q_hgbr_online_es4d", "q_hgbr_online_dd5_pct", "q_hgbr_online_eq5"),
    ("Chronos-2", "chronos2_es4d", "chronos2_dd5_pct", "chronos2_eq5"),
]:
    es = sum(float(r[es_c]) for r in rows79) / len(rows79)
    mdd = min(float(r[dd_c]) for r in rows79)
    ret = (float(rows79[-1][eq_c]) - 1.0) * 100.0
    print(f"Exp 79A {label:22s}: ES={es:.5f} MaxDD={mdd:.2f}% Ret5={ret:+.2f}%")
print(f"Exp 79A Static 60/40          : MaxDD={min(float(r['static_60_40_dd5_pct']) for r in rows79):.2f}% "
      f"Ret5={(float(rows79[-1]['static_60_40_eq5']) - 1.0) * 100.0:+.2f}%")
```
