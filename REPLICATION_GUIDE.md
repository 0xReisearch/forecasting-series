# Replication Guide — Adapt-1 Research Series (Part 2: *Forecasting Through Drift*)

This guide documents how to (1) **verify every headline metric offline** from the committed traces in under 30 seconds, and (2) **replicate each of the five sections live** against the hosted Adapt-1 REST API using public datasets and deterministic synthetic generators.

---

## 1. Environment Setup & API Wire Discipline

### 1.1 Offline Verification & Diagnostic Plotting

Offline trace verification requires only **Python 3.9+** (standard library only; zero `pip` installs):

```bash
python3 verify_traces.py
python3 scripts/adapt1_client.py --self-test
```

To regenerate the six diagnostic PNG figures in [`figures/`](./figures/) directly from the committed CSV traces and scorecards, install optional plotting dependencies (`matplotlib` and `numpy`) and run:

```bash
python3 scripts/plot_figures.py
```

For full mathematical baseline implementations, scoring equations, and stylized overlay definitions extracted directly from the benchmark runner code, see the companion reference manuals in `docs/`:
- [`docs/BASELINES_FORECASTING.md`](./docs/BASELINES_FORECASTING.md) — Sections 01, 03 & 04 comparators (`GJR-DCC-GARCH`, TSFMs, `ACI`, `K=6` generator & 8 online bandits, 19-yr macro & 311 survival baselines).
- [`docs/BASELINES_VALUE_AND_BELIEF.md`](./docs/BASELINES_VALUE_AND_BELIEF.md) — Sections 02 & 05 comparators (all 20/9/14 probability peers, `Exp 100A` lockout setup, simulated K8s POMDP/DAG environments, `SlotFactoredPPO`, `Exact VoI`, `POMCP`, and QMDP analysis).
- [`docs/METHODS_AND_METRICS.md`](./docs/METHODS_AND_METRICS.md) — Exact continuous/mixed Energy Score, binary probability metrics, Newey-West HAC / Holm-Bonferroni / sign-flip permutation, Basel MAR33 / Kupiec / Christoffersen / Acerbi-Székely tests, and all stylized causal decision overlay equations.

### 1.2 Live Adapt-1 API Environment

Configure two environment variables before running any live replication script. Never commit credentials or tokens to version control:

```bash
export ADAPT_API_BASE="https://rei-neuroadapt-api.reilabs.org/api/v1"
export ADAPT_API_KEY="<your-api-key>"

# Read-only pre-flight check (GET /domains, zero state mutation)
python3 scripts/adapt1_client.py --check
```

### 1.3 Mandatory Production Wire Rules

The reference client [`scripts/adapt1_client.py`](./scripts/adapt1_client.py) enforces all six operational rules verified across every live evaluation in Part 2:

1. **Authentication & Headers:** Send `Authorization: Bearer ${ADAPT_API_KEY}` (`X-API-Key` returns `HTTP 403`), a custom non-default `User-Agent` (default HTTP library user agents trigger gateway `HTTP 403`), `Accept: application/json`, and `Content-Type: application/json`.
2. **Strict RFC-8259 JSON:** Always serialize with `allow_nan=False`, check `bool` strictly before `int`, and reject `None` or non-finite values client-side before `POST /events` or `POST /batch` (the API silently drops incomplete event payloads with `accepted: false`).
3. **One Active Domain per API Key:** Isolate concurrent experiments across separate API keys to avoid shared-index latency interference.
4. **Domain Rotation on Long Streams:** Rotate active `futures` domains every `1,000–3,000` ingested events via `POST /batch` (`<= 50` operations per call) to keep `evidence_memories` bounded and latency flat.
5. **Idempotency-Safe Retries:** Retry `(429, 500, 502, 503, 504)` on idempotent reads/DELETEs, but retry **only** `(429, 502, 503, 504)` on state-mutating `POST`s (fail fast on `4xx` and `500` to prevent duplicate observations).
6. **Verified `try / finally` Teardown:** Always wrap domain lifetimes in `Adapt1Client.domain_session(...)` so cleanup executes `DELETE /domains/{domain_id}` followed by `GET /domains/{domain_id} -> HTTP 404`.

---

## 2. Strict Causal Score-Before-Ingest Protocol

Every benchmark in Part 2 enforces strict online prequential causality from an **empty task state**:

1. **Zero Future Leakage:** For every evaluation step $t$, issue the forecast or action request (`POST /query` or `POST /value/predict`) using only information causally available at or before decision time $t$.
2. **Chronological Maturation:** Record and score the issued forecast/decision against the true outcome $y_t$ **before** ingesting $y_t$ via `POST /events`, `POST /feedback`, or `POST /value/observe`. Multi-step horizons (`H > 1`) and delayed operational tickets hold outcomes in a pending queue until maturation time $t_{\text{mature}}$ before ingestion.
3. **Pre-Locked Calibration Splits & Overlays:** All bandwidths, regularization settings, rank-PIT transforms, neural capacity caps, and stylized decision overlay parameters (fees, leverage caps, VoLL penalties, SLA maps) are locked strictly on chronological calibration splits prior to the sealed out-of-sample evaluation window.
4. **Equal-Budget vs. Unconstrained Ceilings:** Matched-window comparators receive the exact same history window and experience budget as Adapt-1, kept strictly separate from unconstrained multi-year offline ceilings (e.g., expanding GJR-GARCH, long-context `ctx=1,024–2,048` TSFMs, or 20,000-episode PPO).
5. **Strictly Append-Only Operational Logs:** Progressive ticket triage (Section 04) requires append-only checkpoint snapshots (`0h`, `+1h`, `+6h`) with hard survival truncation so retroactively edited ticket fields never leak future closure states.

---

## 3. Public Data Sources & Date Ranges

All empirical benchmarks derive from publicly accessible data portals; synthetic testbeds use deterministic generators documented in each section's `README.md`.

| Benchmark | Public Source / Generator | Universe & Freq | OOS Scope & Window | Tier |
| :--- | :--- | :--- | :--- | :--- |
| **01** `Exp 98A`<br>DE Power Scarcity | **SMARD** (`smard.de`) day-ahead power & grid drivers (res. load, wind/solar) | 4 DE-LU blocks (EUR/MWh), daily | `n=450`d (`2025-08-27..2026-11-19`) | Telemetry + 1%-fee BESS/hedge |
| **01** `Exp 98B`<br>Catalyst Vol | Public equities/vol closes + FOMC/CPI calendar | Daily return & implied-vol shocks | Summary in Sec 01 scorecard | Telemetry |
| **01** `Exp 79B/90C`<br>Crypto Cascade | **Binance** spot & 1h klines | `BTC, ETH, SOL, XRP`, event/1h | `n=1,149` (`2025-09..2026-11`, `65w`) | Telemetry + 3× `5/15 bps` |
| **01** `Exp 89B`<br>EU Dunkelflaute | **SMARD** / **Fraunhofer ISE** 4-TSO wind + macro energy | 4 DE TSOs (`50Hertz`, `Amprion`, `TenneT`, `TransnetBW`), daily | `n=323`d (`ctx=1,024` TSFMs) | Telemetry + VoLL reserve |
| **01** `Exp 90A`<br>US Wind | EIA / ISO regional wind generation | 4 US ISOs, daily (`ctx=1,536`) | Summary in Sec 01 scorecard | Telemetry |
| **02** `Exp 103A`<br>Perp Funding | **Binance** 8h perp funding & OHLCV | `BTC, ETH, SOL`, 8h bars | `n=16,077` (`256w`, `2022–2026`) | Telemetry |
| **02** `Exp 103B`<br>Macro Tails | **Yahoo Finance** / **FRED** closes & VIX term | 8 macro ETFs/indices (pre-2020 `DEV_CAL`) | `n=13,856` (`360w`, `2020–2026`, `6.6%–7.0%`) | Telemetry |
| **02** `Exp 102C`<br>Power Extremes | **SMARD** DE day-ahead power (4 pooled blocks) | 4 daily power blocks, pooled | `n=1,800` (`65w`, `W=2,048`) | Telemetry |
| **02** `Exp 100A`<br>Lockout Audit | Realized vol & SMARD power blocks | `42` vol + `18` power blocks | Lockout frequency audit | Telemetry |
| **03** `Exp 96B`<br>Recurring Regimes | Deterministic `K=6` quadrant+radial (`A1→B1→C→A2→B2*`) | 6 actions, zero-linear-mean zones | `300` steps × `n=15` (+ `n=8` abl.) | Synthetic testbed |
| **03** `Exp 79A`<br>SPY/TLT Flip | **Yahoo Finance** daily ETF returns | `SPY, TLT, HYG, GLD`, daily | `n=439`d (`2021–2023`) | Telemetry + `5 bps` |
| **04** `Exp 16`<br>19-Yr Macro | **Yahoo Finance** / **FRED** weekly returns & stress | 5 global macro assets, weekly | `n=890`w (`2008–2026`, `1,024w` arch.) | Telemetry (ES & FRTB) |
| **04** `Exp 53`+NYC<br>311 Revisions | Socrata **311 Open Data** (NYC Lower Man., Chicago, SF) | Tickets at `0h/+1h/+6h` (NYC, SF) & `0h/+6h/+48h` (CHI) | `11,326` ckpts (NYC `4,330`/`n=826`; CHI `2,621`; SF `4,375`) | Telemetry (PII stripped) |
| **04** `Exp 54/59`<br>Survival Logs | **Mozilla Bugzilla** & **SF Fire/EMS** dispatch | Heavy-tailed high-cardinality logs | Summary in Sec 04 scorecard | Telemetry |
| **05** `Exp 93A/B/C`<br>K8s Triage | Simulated K8s POMDP & 6-node RPC DAG | `120`w (`93A`), `78,480`w (`93B`), `6`-node DAG (`93C`, `30` camp.) | Ep-0 rollouts + `204/204` parity | Simulated testbed |

---

## 4. Per-Section Adapt-1 API Configuration Recipes

Below are the canonical Adapt-1 REST payload shapes corresponding to each of the five sections. Exact per-experiment feature names and calibrated hyperparameters are preserved in each section's `scorecard.json` and `README.md`.

### Discovery Mode Option (Viable Across Sections 01–05)

All five sections can also be executed in **Discovery Mode** without hand-authored feature lists or explicit domain ontologies:
- **Transition Discovery (`futures`):** Leave `learning.transition.input_paths: []` and enable `learning.transition.autonomous_projection` (`enabled: true`, `minimum_observations: 24`, `minimum_availability: 0.8`, `maximum_input_paths: 8`, `maximum_buffered_events: 128`). Pre-seed via `POST /batch` (`<= 50` ops/chunk), echo both `future_id` and `decision_id` on every settled `POST /events` call, and inspect selected paths via `POST /domains/{id}/explain`. On sealed German power (`Exp 98A`, `n = 450`), Transition Discovery selected `8/11` raw candidates and scored **`60.84` EUR/MWh** 4D ES (beating all `12/12` GJR-GARCH variants at `63.08`).
- **Structure & Temporal Representation Discovery:** Set `learning.structure.feature_paths: []` with `learning.structure.representation.enabled: true` (`maximum_lag: 2`, `maximum_order: 2`) over strictly causal discrete tokens (`tok_*`) to mine `@lag[1..2]` symbolic rules and live drift alerts (`0.8065` AUROC / `92.4%` coverage on `Exp 98A` evening peak spikes).
- **Two-Stage Discovery $\to$ `/value/*`:** Run pre-OOS Transition Discovery, extract `autonomous_projection.input_paths` from `POST /explain`, and declare those discovered signals in the `/value/*` domain. On `Exp 102C` Power Pool (`n = 1,800`), pruning `20 -> 8` features cut Brier from `0.1801` to `0.1493` and lifted AUROC from `0.7618` to `0.8440`. **Best practice:** in pooled multi-entity panels (`Exp 103A`) or before rare structural correlation flips (`Exp 79A`), pin entity-ID flags and known slow-moving regime switches while letting `autonomous_projection` prune noisy covariates.
- **Empirical Topology & Sequential Discovery:** For factorized DAGs (`Exp 93C`), `empirical_topology_inference: true` with empty `effect_catalog` and `objective_model: "none"` recovers `74.2%` of full authored return (`+3.092` vs `+4.168`, `0.0` broken closes). Note that `autonomous_projection` operates on `learning.transition`, not `learning.context` / `latent_belief` (contextual bandits like `Exp 96B` still require explicit `learning.context.feature_paths`). Full cross-section audit: [`part-2-forecasting-through-drift/06_autonomous_discovery/`](./part-2-forecasting-through-drift/06_autonomous_discovery/).

```json
{
  "learning": {
    "enabled": true,
    "transition": {
      "enabled": true,
      "event_types": ["outcome"],
      "input_paths": [],
      "autonomous_projection": {
        "enabled": true,
        "minimum_observations": 24,
        "minimum_availability": 0.8,
        "maximum_input_paths": 8,
        "maximum_buffered_events": 128,
        "discover_causal_bindings": true,
        "minimum_causal_variables": 2
      }
    }
  }
}
```

### Section 01 — Joint Tails, `linear_residual` Transport & TSFM Residual Wrapping

Use `learning.transition.futures` with `max_samples == max_support <= 128` and `max_branches == 128` so `futures.branches` exports the lossless joint scenario cloud (`omitted_probability < 1e-12`).

- **German Power (`Exp 98A`, upward scarcity drift):** `conditioning: "kernel"`, `numeric_transport: "linear_residual"`, `numeric_ridge: 15.0` (swept `5.0..40.0` on calibration split), `bandwidth: 0.35`, `W = 128`. Compress heavy-tailed inputs into compact causal rolling rank-PIT coordinates (`d <= 5`) before enabling `linear_residual` transport, and pre-seed `>= max(32, 2*(d+1)+2)` complete rows via `POST /batch`.
- **Crypto Cascade (`Exp 79B/90C`) & EU Wind Residual Airbag (`Exp 89B`):** `numeric_transport: "none"`, `W = 128`, `max_branches = 128`. For residual wrapping, ingest the base TSFM's realized residual (or Gaussian `PIT-Z` error) conditioned on current context plus base TSFM scale `(\mu_t^{\text{base}}, \sigma_t^{\text{base}})`, then reconstruct `F_t^{\text{wrap}} = \mu_t^{\text{base}} + R_t^{\text{A1}}` client-side.

```json
{
  "domain_id": "part2-sec01-power-linres",
  "session_id": "sess-sec01",
  "schema": {
    "signals": ["f_res_load", "f_wind_fcst", "f_solar_fcst", "f_lag_spread", "t_blk1", "t_blk2", "t_blk3", "t_blk4"],
    "event_types": ["outcome"]
  },
  "learning": {
    "enabled": true,
    "transition": {
      "enabled": true,
      "event_types": ["outcome"],
      "input_paths": ["values.f_res_load", "values.f_wind_fcst", "values.f_solar_fcst", "values.f_lag_spread"],
      "targets": [
        {"path": "values.t_blk1", "type": "number"},
        {"path": "values.t_blk2", "type": "number"},
        {"path": "values.t_blk3", "type": "number"},
        {"path": "values.t_blk4", "type": "number"}
      ],
      "max_samples": 128,
      "max_distance": 1.0,
      "required_support": 6,
      "numeric_ridge": 15.0,
      "numeric_min_skill": 0.0,
      "futures": {
        "enabled": true,
        "conditioning": "kernel",
        "numeric_transport": "linear_residual",
        "bandwidth": 0.35,
        "prior_strength": 2.0,
        "max_support": 128,
        "max_branches": 128,
        "max_pending_forecasts": 1024,
        "future_id_path": "metadata.future_id"
      }
    }
  }
}
```

### Section 02 — Stateful Online Probability Models (`/value/*`)

Configure a projected value domain (`POST /value/configure`), stream observations in chunks of `<= 50` (`POST /value/observe`) with concurrent cross-sectional rows grouped under a shared `episode_id` (`bar_<t>` for 8h crypto, `day_<t>` for daily macro), train on sliding `observation_ids` (`POST /value/train` with `install: false` to avoid validation-gated staleness lockouts, then score with `model_source: "candidate"` or promote), and query via `POST /value/predict`.

- **Regime-Shifting Crypto Funding (`Exp 103A`, `W = 4,608`):** `model_type: "mlp_v2"`, `dimensions: 64`, `dense_hidden_dimensions: 64`, `epochs: 120`, `learning_rate: 0.003`, `loss_function: "log_loss"`, `warm_start: true`, `validation_mode: "episode_hash"`, `dense_inputs: true`.
- **Stationary Rare Macro Tails (`Exp 103B`, `6.6%–7.0%` base rate):** `model_type: "mlp_v2"`, `dimensions: 16`, `dense_hidden_dimensions: 8`, `epochs: 45`, `learning_rate: 0.0015`, `loss_function: "log_loss"`, `warm_start: false`, `validation_mode: "chronological"`, `dense_inputs: true`.
- **Seasonally Drifting Pooled Power (`Exp 102C`, `W = 2,048`):** swap `model_type` to `"linear"` on the exact same projected basis (`loss_function: "log_loss"`, `warm_start: false`, `validation_mode: "episode_hash"`), or feed Two-Stage pre-OOS Discovery (`8/20` features) into `mlp_v2` (`16/8`, `epochs: 80`, `learning_rate: 0.0015`, `warm_start: false`, `validation_mode: "episode_hash"`).

```json
{
  "session_id": "sess-sec02",
  "request_id": "retrain-block-042",
  "install": false,
  "observation_ids": ["obs-0001", "obs-0002"],
  "options": {
    "model_type": "mlp_v2",
    "dimensions": 16,
    "dense_hidden_dimensions": 8,
    "dense_inputs": true,
    "epochs": 45,
    "learning_rate": 0.0015,
    "loss_function": "log_loss",
    "warm_start": false,
    "validation_mode": "chronological"
  }
}
```

### Section 03 — Plasticity, `latent_belief` Recurrence & Smooth Retention

- **Observable Recurring Nonlinear Regimes (`Exp 96B`):** Create a policy + sequential domain with lowercase action tokens (`when: []`, `relation: "controls"`), `training.model_type: "sequential_q_mlp"`, and `latent_belief` enabled with `recurrence_enabled: true`. Submit execution outcomes via `POST /feedback` (never double-post `/events`), and inspect active regime telemetry via context-aware `POST /explain` (passing `relation: "controls"` and non-empty `context.values`). Do **not** call `POST /policy/reset` during observable recurring cycles—reserve `POST /policy/reset` strictly for confirmed silent same-context payoff inversions (`B2*`).
- **Continuous Stock-Bond Correlation Flip (`Exp 79A`):** Use `learning.transition.futures` with smooth rolling retention (`W = 128`, `numeric_transport: "none"`); do **not** apply hard 21-day flushes on noisy continuous financial markets.

```json
{
  "latent_belief": {
    "enabled": true,
    "max_regimes": 12,
    "change_point_patience": 2,
    "change_point_threshold": 0.25,
    "historical_regime_weight": 0.05,
    "novelty_threshold": 0.45,
    "recurrence_enabled": true,
    "recurrence_margin": 0.10,
    "recurrence_min_support": 2,
    "recurrence_reward_threshold": 0.20,
    "temperature": 0.25
  }
}
```

### Section 04 — Deep-Archive Similarity & Progressive Open-Case Conditioning

- **19-Year Global Macro Archive (`Exp 16`):** Configure `learning.transition.futures` with deep archive retention (`max_samples: 512..1024`, `max_support: 512..1024`, `conditioning: "kernel"`, `numeric_transport: "none"`, `bandwidth: 0.20..0.30`). Keep `bw >= 0.20` (too-tight `bw = 0.10` starves tail support and fails 99% ES calibration).
- **Multi-Stage Municipal 311 Revisions (`Exp 53` + NYC):** Include elapsed ticket age and newly arrived stage indicators in `input_paths`, share one `context.metadata.future_id` across progressive checkpoints (`0h`, `+1h`, `+6h`) for the same ticket, and resolve the single `future_id` via `POST /events` only after the ticket reaches closure or its terminal truncation horizon. Raise `max_pending_forecasts` (e.g., `2048–4096`) to accommodate concurrent open tickets.

### Section 05 — Exogenous 1-Cloud Scoring vs. Active Belief Collapse (`bound_transition`)

- **Exogenous Decisions (Eq. 1):** Query `futures` **once** per decision step and optimize $(1-\lambda)\mathbb{E}_{z \sim F_t}[u(a,z)] + \lambda\,\mathrm{CVaR}_\alpha(u(a,z))$ subject to $\Pr(u(a,z) < \tau) \le p_{\max}$ client-side over the returned weighted cloud, eliminating per-action sample fragmentation (`27×` fewer sequential API calls).
- **Endogenous Partial Observability (`Exp 93A/B/C`):** Author the complete `learning.sequential.bound_transition` triad (`effect_catalog`, `topology_model: "factorized_conjunctive"`, `objective_model: "linear"`) and keep `allow_exploration: true` on `/query`:
  - **2-Axis Exact Belief Search (`Exp 93A`, `120` prior worlds):** `belief_planning_mode: "exact"`, `belief_exact_horizon: 4`, `belief_planning_horizon: 6`, `belief_max_worlds: 256`.
  - **3-Axis QMDP Scaling (`Exp 93B`, `78,480` prior worlds):** override the five scaling keys below to avoid the `40 s` exact-search timeout while preserving sub-second latency (`563 ms` p50):

```json
{
  "belief_planning_mode": "qmdp",
  "belief_max_worlds": 64,
  "belief_qmdp_max_worlds": 64,
  "belief_beam_width": 16,
  "belief_planning_horizon": 3
}
```

---

## 5. Statistical Testing & Calibration Protocol

All significance tests and tail backtests are computed strictly on out-of-sample forecast errors (see [`docs/METHODS_AND_METRICS.md`](./docs/METHODS_AND_METRICS.md) for full derivations, exact clipping constants, lag rules, and stylized overlay equations):

1. **Strictly Proper Energy Score (`ES`):** For a forecast distribution $F_t$ represented by weighted branches $\{(p_i, z_i)\}_{i=1}^M$ and realized vector $y_t \in \mathbb{R}^d$:
   $$\mathrm{ES}(F_t, y_t) = \sum_{i=1}^M p_i \|z_i - y_t\|_2 - \frac{1}{2} \sum_{i=1}^M \sum_{j=1}^M p_i p_j \|z_i - z_j\|_2$$
   Lower is better. Mixed categorical+continuous targets (Section 04) use the standardized mixed Euclidean+one-hot distance documented in Section 04's `README.md`.
2. **Newey-West HAC & Holm-Bonferroni (`n >= 30`):** Paired loss differentials $d_t = L_{\text{model}}(t) - L_{\text{peer}}(t)$ are aggregated at non-overlapping weekly or daily blocks and tested with two-sided Newey-West heteroskedasticity- and autocorrelation-consistent (HAC) standard errors; all multi-comparator contrasts within each benchmark family are adjusted via Holm-Bonferroni (`p_H`).
3. **Exact Sign-Flip Permutation Tests (`n < 30`):** Small-sample seed or campaign comparisons (e.g., `n = 15` or `n = 8` in Section 03) use exact paired sign-flip permutation `p`-values with Holm correction.
4. **Basel FRTB & Tail Calibration Backtests:**
   - **Traffic-Light Exceedance Zones:** Peak/VaR99 breach count classification (`GREEN`, `YELLOW`, `RED`) over the evaluation window.
   - **Kupiec POF (`p_pof`):** Unconditional coverage likelihood-ratio test on `VaR95` / `VaR99` exceedances.
   - **Christoffersen CC (`p_cc`):** Joint unconditional coverage + independence test against volatility-clustered breaches.
   - **Acerbi-Székely ES Test:** Tail-severity backtest on Expected Shortfall beyond `VaR97.5` / `VaR99`.
5. **Binary Probability Scoring (Section 02):**
   - **Brier Score:** $\frac{1}{n}\sum_i (\hat{p}_i - y_i)^2$
   - **Log-Loss:** $-\frac{1}{n}\sum_i [y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)]$
   - **Expected Calibration Error (`ECE`):** Weighted absolute gap between predicted probability and empirical frequency across calibration bins, plus calibration ratio (`CalR = \bar{\hat{p}} / \bar{y}`).
   - **Discrimination:** ROC-AUC (`AUROC`).

---

## 6. Configuration and Evaluation Notes

Use these settings and checks when running the five sections:

1. **Joint support (Section 01):** Keep joint tail scoring within `W <= 128`, matching the `128`-scenario export cap. Use `futures.marginals` for full-support 1D marginals up to `W = 1,024`; increasing `max_support` alone does not expand the joint export.
2. **Residual wrapping and costs (Section 01):** Check tail calibration before wrapping a base forecast, and apply the same transaction costs to every arm. On the 1,536-day US wind study (`Exp 90A`), standalone long-context TSFMs lead and wrapping raises breaches (`3 -> 5`). On FOMC/CPI implied-vol catalysts (`Exp 98B`), return-only GJR scores `1.139` versus Adapt-1's `1.145`. At `15 bps` fees, levered `3×` crypto books breach the survival floor.
3. **Validation and model capacity (Section 02):** Track candidate and installed models separately and retain the specified split and window settings. Default `install=True` promotion retained calm weights on `76%–86%` of volatility blocks (`Exp 100A`). On shifting crypto funding (`Exp 103A`), `validation_mode="chronological"` withholds the newest 20% and ranks `#17–#19/21`. Compact rare-tail regularization (`Exp 103B`, `W=4,096`, `10` models/target) reduces LeftTail AUROC (`0.5871 -> 0.5688`) while achieving `#1/10` LeftTail ECE (`0.00081`, `#3/10` Brier `0.06445`), `#2/10` ExtremeMove Brier (`0.06045`, ECE `0.00402`), and `0/108` Holm losses; on seasonal power (`Exp 102C`, `15` models), warm-start 20-feature `mlp_v2` ranks `#12/15` (`0.18012`), `linear` ranks `#6/15` (`0.16078`), pooled `HGBR` ranks `#3/15` (`0.15627`) / `HGBR_Disc8` `#2/15` (`0.15431`), and Two-Stage Discovery (`8/20` features) → `mlp_v2` ranks `#1/15` (`0.14933`).
4. **Regime tracking and retention (Section 03):** Evaluate observable regime changes separately from unchanged-context payoff flips. Native tracking reaches `1.0%` late accuracy on `B2*` (`Exp 96B`). The companion two-arm inversion study tests explicit resets, which also erase archived regimes. On noisy financial markets (`Exp 79A`), retaining history scores better than hard 21-day flushes: ES `1.345` versus `1.367` (`p_H = 0.0189`).
5. **Survival conditioning (Section 04):** Check cutoff eligibility and tail calibration when choosing bandwidth. In the heavy-tailed Mozilla Bugzilla and SF Fire/EMS studies (`Exp 54`, `Exp 59`), similarity weighting retains mass from closed cases across survival cutoffs; quantile random forests and boosted trees lead. The `bw = 0.10` setting in `Exp 16` fails 99% ES calibration.
6. **Planning and response time (Section 05):** Choose the planning mode according to the value of diagnostic probes. QMDP assumes full observability after one step; on pure-probe tasks it scores `-0.631` clean / `-0.255` flaky versus `+1.420` for exact VoI. Recorded production-API inference is `< 900 ms` for standard calls and approximately `2 s` for deep archives or value retrains.
