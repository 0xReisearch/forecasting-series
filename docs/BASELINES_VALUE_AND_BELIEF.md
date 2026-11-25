# Technical Baseline Specification — Sections 02 & 05

This document specifies the exact mathematical formulations, feature pipelines, windowing and refit schedules, hyperparameter tuning protocols, optimization solvers, and environment dynamics for every comparator evaluated in **Section 02 (Online Probability Models)** and **Section 05 (Belief Collapse & Active Inference)** of *Adapt-1 Research Series — Part 2: Forecasting Through Drift*.

Every specification below is extracted directly from the benchmark implementations that produced the committed traces in `part-2-forecasting-through-drift/02_online_probability_models/` and `part-2-forecasting-through-drift/05_belief_collapse/`. No comparator is artificially handicapped: offline peers share identical causal splits, identical refit cadences, matched sliding windows, multi-start MLE convergence checks (`0` frozen initializations), and strictly causal pre-OOS hyperparameter selection.

---

## Part I — Section 02: Online Probability Models (`Exp 103A`, `Exp 103B`, `Exp 102C`, `Exp 100A`)

### 1. Shared Causal Protocol & Numerical Integrity

1. **Score-Before-Ingest Walk-Forward Execution:** At every refit block $[t_0, t_1)$, all models (Adapt-1 and every offline baseline) train strictly on records whose outcomes settled at or before $t_0$. Predictions over $[t_0, t_1)$ are generated and frozen before any outcome in $[t_0, t_1)$ is revealed.
2. **Pre-OOS Feature Standardization:** Continuous features are standardized using sample mean and standard deviation estimated exclusively on the pre-OOS warmup partition (dropping initial rolling-lag warmup bars/days) and clipped to $[-6.0, 6.0]$. Binary indicators and trigonometric calendar encodings ($\sin, \cos$ day-of-year) are kept unscaled. Perturbing all OOS rows produces exactly $0.0$ maximum absolute difference on pre-OOS features and thresholds.
3. **Deterministic Execution:** Single-threaded BLAS (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) and deterministic 32-bit seeds derived from `int(sha256("|".join(parts))[:8], 16)` guarantee exact reproducibility across runs.
4. **Probability Clipping & Scoring:** All predicted probabilities are clipped identically to $[10^{-6}, 1 - 10^{-6}]$ when computing Brier score, binary Log-Loss, 10-bin equal-width ECE, 10-bin quantile ECE (`ece_q10`), and exact tie-averaged Mann-Whitney U AUROC.

### 2. Shared Hardened Multi-Start Mean-NLL MLE GJR-GARCH(1,1) Solver

Used by `AR1_GJR_SignProb` (`W=512` and `W=1,024` in Exp 103A) and `Real_MLE_GJR_GARCH_TailProb` (`W=512` in Exp 103B), replacing fragile single-start fits with a variance-targeted multi-start optimizer that achieves **zero frozen initializations** across all refits:

- **Conditional Variance Equation:** Given zero-mean innovations $e_1, \dots, e_n$ in a sliding window of length $n$,
  $$h_t = \max\!\Bigl(\omega + \bigl(\alpha + \gamma \mathbf{1}\{e_{t-1} < 0\}\bigr) e_{t-1}^2 + \beta h_{t-1},\; 10^{-8}\Bigr), \quad h_0 = \max\!\Bigl(\frac{1}{\min(40, n)}\sum_{i=1}^{\min(40, n)} e_i^2,\; 10^{-6}\Bigr).$$
- **Variance Targeting & Stationarity Constraint:** Letting $\bar{v} = \max(\overline{e^2}, 10^{-6})$ and $\kappa = \operatorname{clip}(\overline{\mathbf{1}\{e_t < 0\}}, 0.15, 0.85)$, the intercept is pinned to $\omega = \max\bigl((1 - \alpha - \kappa\gamma - \beta)\bar{v},\, 10^{-10}\bigr)$ subject to
  $$\alpha \in [0.002, 0.45], \quad \gamma \in [0.0, 0.65], \quad \beta \in [0.40, 0.990], \quad \alpha + \kappa\gamma + \beta < 0.996.$$
- **Objective (Gaussian QML Mean Negative Log-Likelihood):**
  $$\mathcal{L}(\alpha, \gamma, \beta) = \frac{1}{2(n - 1)} \sum_{t=2}^{n} \left(\log h_t + \frac{e_t^2}{h_t}\right).$$
- **Multi-Start Optimization Pipeline (`scipy.optimize.minimize`):**
  1. Evaluate candidate starts: warm start from the previous block $(\hat{\alpha}_{b-1}, \hat{\gamma}_{b-1}, \hat{\beta}_{b-1})$ (clipped to interior bounds) plus three diverse anchors $(0.03, 0.06, 0.88)$, $(0.10, 0.14, 0.74)$, and $(0.06, 0.01, 0.91)$.
  2. Run bounded **L-BFGS-B** (`maxiter=80`, `ftol=1e-10`, `eps=1e-5`) from the lowest-NLL start and a second distinct start separated by $\max|\Delta \theta| > 0.05$.
  3. Run mandatory **Nelder-Mead** refinement (`maxiter=180`, `xatol=1e-4`, `fatol=1e-6`) from the best L-BFGS-B solution.
  4. Verify non-frozen convergence: $\max|\hat{\theta} - (0.04, 0.08, 0.84)| > 10^{-4}$ on **100% of refits** (`0` frozen refits across all 36×3=108 crypto refits and all 28×8=224 macro refits).
- **Filtered Historical Simulation (FHS) Probability Conversion:** Standardized in-window residuals $z_i = \operatorname{clip}(e_i / \sqrt{\max(h_i, 10^{-8})}, -8, 8)$ are combined with the strictly causal filtered 1-step-ahead conditional mean $\mu_t$ and volatility $\sigma_t = \sqrt{h_t}$ rolled forward step-by-step across the OOS block using observed innovations $e_{t-1}$. Event probabilities use Laplace-smoothed empirical counts over the $n$ FHS draws:
  $$\hat{p}_t = \operatorname{clip}\!\left(\frac{0.5 + \sum_{i=1}^{n} \mathbf{1}\{\mathcal{E}(\mu_t + \sigma_t z_i)\}}{n + 1},\; 10^{-4},\; 1 - 10^{-4}\right).$$

---

### 3. Exp 103A — 5-Year Crypto Perpetual Funding Sign-Flips (`BTC/ETH/SOL`, 21 Models)

- **Panel & Split:** `6,782` 8-hour bars (`2020-09-13` to `2026-11-22`), pre-2022 warmup `1,423` bars (first `25` bars excluded from scaling fit), sealed OOS `2022-01-01` to `2026-11-22` (`5,359` bars $\times 3 = 16{,}077$ forecasts, `256` ISO weeks, `22.1%` negative-funding rate).
- **Target & Refit Cadence:** $y_{t,a} = \mathbf{1}\{f_{t+1,a} < 0\}$. Every model refits every **`150` 8h bars (`36` blocks)**.
- **Feature Sets:**
  - *Per-Coin (`16` features):* Own funding `f_lag0`, `f_lag1`, `f_lag2`, 9-bar rolling mean `f_ma9`, 21-bar rolling std `f_std21`, 1-bar diff `f_diff1`, indicators `is_hi_clamp` ($\mathbf{1}\{f \ge 0.99\}$), `is_neg_now` ($\mathbf{1}\{f < 0\}$), cross-coin funding (`btc_f8`, `eth_f8`, `sol_f8`), 7-day log realized vol (`btc_log_rv7d`, `sol_log_rv7d`), and 3 cross-coin spreads (`f_spread_vs_ew`, `f_spread_btc_sol`, `rv_spread_btc_sol`).
  - *Pooled (`19` features):* Same `16` features plus 3 coin one-hots (`is_btc`, `is_eth`, `is_sol`); Adapt-1 packs the first `16` into `context` and the `3` cross-coin spreads into `policy_features`.
- **Pre-2022 Causal Hyperparameter Tuning (`Online_HGBR_Cls`):** Evaluated strictly on the last `450` pre-2022 bars (`2021-08`..`2021-12`, 3 blocks) over the grid `max_depth` $\in \{3, 4\}$, `l2_regularization` $\in \{2.0, 5.0\}$, `min_samples_leaf` $\in \{25, 30\}$, `window` $\in \{512, 1024\}$ bars (`max_iter=150`, `learning_rate=0.04`). Both pooled and per-coin locked `(max_depth=3, l2_regularization=2.0, min_samples_leaf=25)` at `W=512`; Exp 103A then added window-matched `W=1,024` (`3,072` pooled rows) and `W=1,536` (`4,608` pooled rows; `4,194` rows on Block 0 due to warmup start `t=25`, `4,608` on Blocks 1–35) peers using the exact same locked tree hyperparameters.

| Model (21 Total) | Estimator | Window | Hyperparameters / Spec |
| :--- | :--- | :--- | :--- |
| `Adapt1_CUDA_MLPv2_`<br>`FixEp_EpHash_W4608` *(Anchor)* | Adapt-1 `/value/*`<br>`mlp_v2` | Pooled `4,608`<br>(`1,536` bars) | `dims=64`, `dense_h=64`, `ep=120`, `lr=3e-3`, `log_loss`, `warm_start=True`, `episode_id=bar_t`, `validation_mode="episode_hash"`, `interaction_selection="evidence"` |
| `Adapt1_CUDA_MLPv2_`<br>`FixEp_EpHash_W3072` | Adapt-1 `/value/*`<br>`mlp_v2` | Pooled `3,072`<br>(`1,024` bars) | Identical to Anchor at `W=3,072` rows |
| `Adapt1_CUDA_MLPv2_`<br>`FixEp_Chron_W4608` | Adapt-1 `/value/*`<br>`mlp_v2` | Pooled `4,608`<br>(`1,536` bars) | Identical to Anchor with `validation_mode="chronological"` (holds out newest 20% = 307 bars / 102d) |
| `Adapt1_CUDA_MLPv2_`<br>`FixEp_Chron_W3072` | Adapt-1 `/value/*`<br>`mlp_v2` | Pooled `3,072`<br>(`1,024` bars) | Identical to Anchor at `W=3,072` with `validation_mode="chronological"` (holds out newest 204 bars / 68d) |
| `Adapt1_CUDA_MLPv2_`<br>`Slide3072` / `Slide1536` | Adapt-1 `/value/*`<br>`mlp_v2` | Pooled `3,072`<br>/ `1,536` | Prior ungrouped per-row `episode_id=e_j`, `episode_hash`, `warm_start=True` |
| `Online_HGBR_Cls_`<br>`Pooled_W1536` / `W1024` / `W512` | `sklearn.ensemble`<br>`HistGradientBoosting`<br>`Classifier` | Pooled `4,608`<br>/ `3,072` / `1,536` | `max_iter=150`, `max_depth=3`, `lr=0.04`, `min_samples_leaf=25`, `l2_regularization=2.0` (pre-2022 DEV-locked) |
| `Online_HGBR_Cls_`<br>`PerCoin_W1024` / `W512` | `sklearn.ensemble`<br>`HistGradientBoosting`<br>`Classifier` | Per-coin<br>`1,024` / `512` | Same pre-2022 DEV-locked hyperparameters; window-mean fallback if `<2` classes |
| `ExtraTrees_Cls_`<br>`Pooled_W1536` / `W1024` | `sklearn.ensemble`<br>`ExtraTreesClassifier` | Pooled `4,608`<br>/ `3,072` | `n_estimators=200`, `max_depth=10`, `min_samples_leaf=8`, `n_jobs=1` |
| `LogisticRegression_L2_`<br>`Pooled_W1536` / `W1024` / `W512` | `sklearn.linear_model`<br>`LogisticRegressionCV` | Pooled `4,608`<br>/ `3,072` / `1,536` | `penalty="l2"`, `Cs=logspace(-2, 2, 9)`, `cv=4`, `scoring="neg_log_loss"`, `max_iter=500` |
| `LogisticRegression_L2_`<br>`PerCoin_W1024` / `W512` | `sklearn.linear_model`<br>`LogisticRegressionCV` | Per-coin<br>`1,024` / `512` | Same 4-fold log-loss CV grid; empirical mean fallback if `<4` positives or negatives |
| `AR1_GJR_SignProb_`<br>`W1024` / `W512` | OLS AR(1) + MLE<br>GJR-GARCH(1,1)-FHS | Per-coin<br>`1,024` / `512` | OLS $f_{t+1}=c_0+\phi f_t+e_t$ ($\lvert\phi\rvert\le 0.98$) + Sec I.2 GJR-MLE (`0/108` frozen; med $\phi\in[0.68,0.74]$, $\alpha\in[0.14,0.20]$, $\beta\in[0.68,0.83]$ at `W=1,024`) |
| `Persistence_Sign` | Empirical 2-state<br>Markov chain | Per-coin `512` | $\hat{p}_{t,a}=\hat{P}(y_{t,a}=1\mid f_{t,a}<0)$ if $f_{t,a}<0$ else $\hat{P}(y_{t,a}=1\mid f_{t,a}\ge 0)$, clipped to $[10^{-4},1-10^{-4}]$ |

---

### 4. Exp 103B — 7-Year 8-Asset Daily Macro Tail-Shocks (10 Models per Target)

- **Panel & Split:** `8` assets (`SPY, TLT, HYG, GLD, USO, UUP, USDJPY, KRE`), `4,248` trading days inner-joined with lagged CBOE implied volatility (`VIX`, `VIX3M`, `VVIX`). Pre-2020 `DEV_CAL` spans `2018-01-02..2019-12-31` (`503` days $\times 8 = 4{,}024$ forecasts for Adapt-1 candidate locking; last `500` pre-2020 days for `HGBR` tuning). Sealed OOS spans `2020-01-02..2026-11-20` (`1,732` days $\times 8 = 13{,}856$ forecasts, `360` ISO weeks). Stress slice `VIX_INVERSION` marks days with lagged $\text{VIX}/\text{VIX3M} > 1.0$ (`123` days, `984` forecasts, `45` ISO weeks).
- **Targets & Refit Cadence:** Using strictly lagged 22-day RMS realized volatility $\text{rv22}_{t-1} = \sqrt{\frac{1}{22}\sum_{k=1}^{22} r_{t-k}^2}$:
  1. `LeftTail`: $y_t^{\text{LT}} = \mathbf{1}\{r_t < -1.5\,\text{rv22}_{t-1}\}$ (`6.96%` OOS base rate).
  2. `ExtremeMove`: $y_t^{\text{EM}} = \mathbf{1}\{|r_t| > 2.0\,\text{rv22}_{t-1}\}$ (`6.57%` OOS base rate).
  All models refit every **`63` trading days (`28` blocks)** on sliding window **`W = 512` days** (`4,096` pooled rows / `512` per-asset rows).
- **Feature Sets:**
  - *Per-Asset (`13` features):* Own lagged returns/vol (`r_lag1`, `rv1`, `rv5`, `rv22`, `rv1_over_rv22`, `neg_r_lag1`, `r_lag2`, `rv5_over_rv22`, `vix_over_rv22`) + market state (`f_log_vix`, `f_vix_term`, `f_log_vvix`, `xs_neg_frac_lag1`, the lagged cross-sectional fraction of negative-return assets).
  - *Pooled (`21` features):* `8` primary features + `8` asset one-hots (`16` in `context`) + `5` extra features (`policy_features`).

| Model (10 per Target) | Estimator | Window | Hyperparameters / Pre-2020 Locked Config |
| :--- | :--- | :--- | :--- |
| `Adapt1_103B_C3_`<br>`MLPv2_Compact_LogLoss` *(Anchor)* | Adapt-1 `/value/*`<br>`mlp_v2` | Pooled `4,096`<br>(`512`d) | Locked on 2018–2019 `DEV_CAL` (lowest Brier `0.06608` among `C1..C6`): `dims=16`, `dense_h=8`, `dense_inputs=True`, `ep=45`, `lr=1.5e-3`, `log_loss`, `warm_start=False`, `episode_id=day_t`, `validation_mode="chronological"` (`#1/10` LeftTail ECE `0.00081`, `#3/10` Br `0.06445`; `#2/10` ExtremeMove Br `0.06045`, ECE `0.00402`) |
| `Adapt1_103B_C2_`<br>`Linear_Chron_Squared` | Adapt-1 `/value/*`<br>`linear` | Pooled `4,096`<br>(`512`d) | Locked linear candidate on `DEV_CAL` (`0.06720` Brier): `dims=64`, `ep=100`, `lr=3e-3`, `squared`, `warm_start=False`, `episode_id=day_t`, `validation_mode="chronological"` |
| `Adapt1_CUDA_MLPv2_`<br>`LeftTail` / `ExtremeMove` | Adapt-1 `/value/*`<br>`mlp_v2` | Pooled `4,096`<br>(`512`d) | Unfixed baseline (`dims=64`, `dense_h=64`, `ep=120`, `lr=3e-3`, `warm_start=True`, per-row `episode_id=e_j`, `episode_hash`) |
| `Real_MLE_GJR_GARCH_`<br>`TailProb` | Multi-Start MLE<br>GJR-GARCH(1,1)-FHS | Per-asset<br>`512`d | Sec I.2 solver on demeaned returns $e_t=r_t-\bar{r}_{\text{win}}$; `0/224` frozen refits; exact FHS exceedance $\hat{P}(\bar{r}+\sigma_t z<-1.5\,\text{rv22}_{t-1})$ & $\hat{P}(\lvert\bar{r}+\sigma_t z\rvert>2.0\,\text{rv22}_{t-1})$ |
| `LogisticRegression_L2_`<br>`Pooled` / `PerAsset` | `sklearn.linear_model`<br>`LogisticRegressionCV` | Pooled `4,096`<br>/ Per-asset `512` | `penalty="l2"`, `Cs=logspace(-2, 2, 9)`, `cv=4`, `scoring="neg_log_loss"`, `max_iter=500` (base-rate fallback if `<5` positives) |
| `Online_HGBR_Cls_`<br>`Pooled` / `PerAsset` | `sklearn.ensemble`<br>`HistGradientBoosting`<br>`Classifier` | Pooled `4,096`<br>/ Per-asset `512` | `max_iter=120`, `lr=0.04`, grid `{(3,2.0,25), (3,8.0,40), (4,5.0,30)}` tuned on pre-2020 `DEV_CAL`: LT Pooled `(3,2.0,25)`, LT PerAsset & both EM `(3,8.0,40)` |
| `ExtraTrees_Cls_`<br>`Pooled` | `sklearn.ensemble`<br>`ExtraTreesClassifier` | Pooled `4,096`<br>(`512`d) | `n_estimators=200`, `max_depth=10`, `min_samples_leaf=10`, `n_jobs=1` |
| `Trailing_BaseRate` | Laplace-smoothed<br>empirical rate | Per-asset `512`d | $(0.5+\sum y_{\text{win}})/(512+1)$, clipped to $[10^{-4},1-10^{-4}]$ |

*Stage 1 Pre-2020 `DEV_CAL` Candidate Grid (`C1..C6`, `LeftTail`, `n = 4,024`, zero 2020–2026 peeking):* Screened `C1` (`linear`, `log_loss`, `64`, `100ep`), `C2` (`linear`, `squared`, `64`, `100ep` $\to$ **locked linear**), `C3` (`mlp_v2`, `16/8`, `45ep`, `log_loss`, `ws=False` $\to$ **locked `mlp_v2`**), `C4` (`mlp_v2`, `16/8`, `squared`, `ws=True`, `frw=0.1`), `C5` (`mlp_v2`, `16/8`, `squared`, `ws=False`), and `C6` (`mlp_v2`, `16/8`, `log_loss`, `ws=True`, `frw=0.1`). Live wire probing confirmed `dense_inputs=True` is mandatory for numeric value models and `function_retention_weight` requires `warm_start=True` (HTTP 422 otherwise).

---

### 5. Exp 102C — Pooled 4-Block German Day-Ahead Power Extremes (15 Models)

- **Panel & Split:** German day-ahead power auction blocks (`P_EVE`, `P_MIDDAY`, `P_NIGHT`, `P_SPREAD`), pre-OOS buffer `604` days (`2024-01-01..2025-08-26`, including `DEV_CAL` `2024-10-31..2025-05-18`, `n = 200` days), sealed OOS `2025-08-27..2026-11-19` (`450` days $\times 4 = 1{,}800$ forecasts, `65` ISO weeks).
- **Target & Refit Cadence:** $y_{t,b} = \mathbf{1}\{p_{t,b} > Q_{0.85}^{\text{DEV}}(b) \;\lor\; p_{t,b} < 0\}$, with strictly causal `DEV_CAL` thresholds $Q_{0.85}^{\text{DEV}} = \{171.32, 124.15, 109.11, 174.73\}\text{ EUR/MWh}$. Refit every **`25` days (`18` blocks)** on sliding window **`W = 512` days** (`2,048` pooled rows / `512` per-block rows).
- **Feature Sets (Strictly Pre-Auction):**
  - *Per-Block (`16` features):* `10` shared drivers (residual load proxy, evening load forecast, wind `w100`, temperature, `D-1` evening/midday/spread prices, 30-day forward evening price, weekend flag, $\cos(\text{doy})$), `2` own-block lags (`D-1`, `D-7`), and `4` extra features (`p_std7`, $\sin(\text{doy})$, `p_eve_lag7`, `p_midday_lag7`).
  - *Pooled (`20` features):* Same `16` features + `4` block one-hots (`is_p_eve..is_p_spread`), packed as `16` in `context` + `4` in `policy_features`.
  - *Pre-OOS Discovery Subset (`8/20` features, strictly `< 2025-08-27`, `n=512` pre-OOS rows via `autonomous_projection`, `max_input_paths=8`):* `own_block_lag1`, `is_p_midday`, `is_p_spread`, `is_p_night`, `is_p_eve`, `r0_p_midday_prev`, `r0_is_wknd_D`, `own_block_lag7`.

| Model (15 Total) | Estimator | Window | Hyperparameters / `DEV_CAL` Locked Config |
| :--- | :--- | :--- | :--- |
| `Adapt1_CUDA_MLPv2_`<br>`Disc8_PowerPool2048` | Adapt-1 `/value/*`<br>`mlp_v2` (`8/20` Disc8) | Pooled `2,048` | `dims=16`, `dense_h=8`, `ep=80`, `lr=1.5e-3`, `log_loss`, `ws=False`, `episode_hash`, `preserve_binary_inputs=True`, `interaction_selection="evidence"`<br>OOS: **`#1/15`** Brier `0.14933` (`BSS 32.86%`), LL `0.46342`, AUROC `0.8440`, `#2/15` ECE `0.05297`, `#1/15` `P_EVE` `0.12937` (`9/14` Holm Brier & LL wins, `2/3` matched `Disc8`, `0` losses) |
| `Adapt1_CUDA_Linear_`<br>`PowerPool2048` | Adapt-1 `/value/*`<br>`linear` (20f) | Pooled `2,048` | `representation="projected"`, `dims=64`, `ep=120`, `lr=3e-3`, `log_loss`, `warm_start=False`, `validation_mode="episode_hash"` (`#6/15` Brier `0.16078`, `#3/15` `P_EVE` `0.14004`) |
| `Adapt1_CUDA_MLPv2_`<br>`PowerPool2048` | Adapt-1 `/value/*`<br>`mlp_v2` (20f) | Pooled `2,048` | `representation="projected"`, `dims=64`, `dense_h=64`, `ep=120`, `lr=3e-3`, `log_loss`, `warm_start=True`, `validation_mode="episode_hash"` (`#12/15` Brier `0.18012`, `#11/15` `P_EVE` `0.15353`) |
| `Online_HGBR_Cls_`<br>`Pooled` (`_Disc8`) / `PerBlock` | `sklearn.ensemble`<br>`HistGradientBoosting`<br>`Classifier` | Pooled `2,048`<br>/ Per-blk `512` | `max_iter=120`, `lr=0.04`, `DEV_CAL` grid `{(3,3.0,15), (4,1.0,15), (3,8.0,25)}`: 20f Pooled & PerBlock locked `(3,8.0,25)` (`#3/15` `0.15627` / `#5/15` `0.15963`); matched `Disc8` locked `(3,3.0,15)` (`#2/15` Brier `0.15431`) |
| `ExtraTrees_Cls_`<br>`Pooled` (`_Disc8`) / `PerBlock` | `sklearn.ensemble`<br>`ExtraTreesClassifier` | Pooled `2,048`<br>/ Per-blk `512` | `n_estimators=200`, `max_depth=10`, `min_samples_leaf=4`, `n_jobs=1` (`Disc8` `#10/15` Brier `0.17588`) |
| `LEAR_LASSO_Logit_`<br>`Pooled` / `PerBlock` | `sklearn.linear_model`<br>`LogisticRegressionCV` | Pooled `2,048`<br>/ Per-blk `512` | Sparse L1 logistic ARX (`penalty="l1"`, `solver="liblinear"`, `Cs=logspace(-2, 2, 9)`, `cv=4`, `scoring="neg_log_loss"`, `max_iter=500`) |
| `LogisticRegression_L2_`<br>`Pooled` (`_Disc8`) / `PerBlock` | `sklearn.linear_model`<br>`LogisticRegressionCV` | Pooled `2,048`<br>/ Per-blk `512` | Ridge logistic (`penalty="l2"`, `solver="lbfgs"`, `Cs=logspace(-2, 2, 9)`, `cv=4`, `scoring="neg_log_loss"`, `max_iter=500`; `Disc8` `#14/15` Brier `0.18996`) |
| `Trailing_BaseRate` | Laplace-smoothed<br>empirical rate | Per-blk `512`d | $(0.5+\sum y_{\text{win}})/(n_{\text{win}}+1)$ |

---

### 6. Exp 100A — Validation-Gated Install Lockout Audit Setup

Exp 100A audited Adapt-1's server-side model promotion gate under non-stationary regime shifts across two empirical domains (recorded per block in `exp100a_install_lockout_blocks.csv`):

- **Shared `/value/configure` Promotion Config:** `mode="prediction"`, `minimum_cases=16`, `required_improvement=0.0`, `maximum_prediction_regression=1.0`, `representation="projected"`, `validation_mode="chronological"`, `epochs=120`, `dimensions=64`, `dense_hidden_dimensions=64`, `learning_rate=0.003`.
- **Arena 100A-1 (Realized Volatility, `42` blocks of `42` trading days, `2020-01-02..2026-11-20`):** `4` assets (`SPY, TLT, HYG, GLD`), `16` pooled features (`12` HAR-RV + leverage + implied-vol/catalyst + `4` one-hots), `loss_function="squared"`, targets `abs_r` ($|r_t|$, `1,732`d) and `rv5_fwd` ($\sqrt{\frac{1}{5}\sum_{k=0}^4 r_{t+k}^2}$, `1,728`d, availability lagged by `+5` days for zero overlap leakage). Targets mapped causally from pre-2020 $[0, 1.5\max_{\text{pre}}]$ to $[0.02, 0.98]$.
  - *Expanding (`install=True`):* Growing causal buffer starting `2016-01-01` (`4,024` to `10,912` rows), `warm_start=True`, `warm_start_source="installed"`, scored via `model_source="installed"`. Returned `status="retained"` on **`36/42`** (`abs_r`) and **`32/42`** (`rv5_fwd`) blocks.
  - *Sliding Candidate (`install=False`):* Block 0 seeds `install=True` on trailing `W = 512 × 4 = 2,048` rows; Blocks 1–41 pass explicit `observation_ids[-2048:]`, `install=False`, `warm_start_source="candidate"`, `expected_initial_model_sha256`, and score via `model_source="candidate"`. Returned **`0/42`** `status="retained"` lockouts across both targets (42/42 distinct weight versions).
- **Arena 100A-2 (German Power, `18` blocks of `25` days, `2025-08-27..2026-11-19`):** `14` strictly pre-auction features (`r0_*` + `D-7` lags + $\sin(\text{doy})$), targets `ev_peak_spike_200` (`log_loss`), `ev_neg_midday` (`log_loss`), and `p_peak` (`squared`, pre-OOS bounds `[0.0, 614.28] EUR/MWh` $\to [0.02, 0.98]$).
  - *Expanding (`install=True`):* Growing buffer starting `2024-01-01` (`597` to `1,022` rows). Returned `status="retained"` on **`14/18`** (`ev_peak_spike_200`, 12 consecutive retained blocks `1..12` + `3` `rejected` + `2` retained), **`16/18`** (`p_peak`), and **`17/18`** (`ev_neg_midday`).
  - *Sliding Candidate (`install=False`, `W = 512` rows):* Returned **`0/18`** `status="retained"` lockouts across all three targets.

---

## Part II — Section 05: Belief Collapse & Active Inference (`Exp 93A`, `Exp 93B`, `Exp 93C`)

### 1. Simulated Trace-Parameterized Kubernetes POMDP Environments

**Data and Simulation Setup:** Service dependency graphs (Online Boutique Kubernetes manifests: 12 Deployments, 16 RPC edges), fault taxonomy, and non-uniform incident frequency / cascade / reopening skews (`233` archived Statuspage incidents from GitHub, Anthropic, OpenAI, Cloudflare, Vercel, and GCP split chronologically into `116` `DEV_CAL` and `117` `SEALED_OOS`) are derived from real archived records. Discrete state axes, closed-loop remediation dynamics, and dollar SLA schedules are **design-authored simulation environments** in which every Adapt-1 arm makes 100% native live API decisions with zero client-side overrides.

**Strict Disjoint Seed Partitions (Zero Train / Calibration / OOS Overlap):**

| Arena | `DEV_CAL` ($n=15$) | `SEALED_OOS` ($n=30$) | RL Pre-Train ($20\text{k}$ eps) | Bandit ($500$ eps) |
| :--- | :--- | :--- | :--- | :--- |
| **93A** | `1100..1114` | `1000..1029` | `100,000..119,999` (DEV: `95,000..95,399`) | `98,000..98,499` |
| **93B-Clean** | `2100..2114` | `2000..2029` | `200,000..219,999` (DEV: `195,000..195,399`) | `198,000..198,499` |
| **93B-Flaky15** | `2100..2114` | `2200..2229` | `230,000..249,999` (DEV: `225,000..225,399`) | `228,000..228,499` |
| **93C** | `3100..3114` | `3000..3029` | `300,000..319,999` (DEV: `295,000..295,399`) | `298,000..298,499` |

#### 1.1 Arena 93A — Heterogeneous 5-Service $\times$ 2-Axis Kubernetes Root-Cause Hunter
- **Services & Asymmetric SLA Tiers (Per-Episode Slot-Shuffled):**
  `paymentservice` (`$250,000`), `checkoutservice` (`$150,000`), `frontend` (`$80,000`), `cartservice` (`$35,000`), `currencyservice` (`$10,000`). Total SLA = `$525,000`; **mean SLA normalization scale** $\overline{\text{SLA}}_{93A} = \$105{,}000$.
- **Runbooks & Direct Probe Profiles (Per-Episode Slot-Shuffled):**
  `pod-cpu-throttle-reset` (`1.5 min`, `$1,500`), `netem-delay-reroute` (`2.5 min`, `$2,500`), `oom-heap-recycle` (`3.5 min`, `$4,000`), `packet-loss-mesh-failover` (`5.0 min`, `$6,000`).
- **State, Action, & Latent World Spaces:**
  - *State:* Each service $s \in \{0..4\}$ has coordinate $c_s \in \{-1, +1\}^2$, closed indicator $u_s \in \{0, 1\}$, and observable health objective $o_s = g^\top c_s \in \{-1.0, 0.0, +1.0\}$. Initial coordinates are drawn with tier-skewed critical double-fault probabilities $(0.80, 0.68, 0.52, 0.36, 0.22)$ guaranteeing $\ge 2$ critical ($o_s = -1$) and $\ge 1$ partial ($o_s = 0$) services.
  - *Actions ($|\mathcal{A}| = 26$, Horizon $H = 16$):* `1` standby (`no_op`, 2.0 min), `5` certifications (`commit(s)`), and `5 × 4 = 20` diagnostic/remediation probes (`apply_tool(s, r)`).
  - *Latent World & Prior Count (`120` worlds):* Hidden bijective permutation $\pi \in S_4$ ($4! = 24$) assigning the 4 runbooks to orthogonal axis flips $\{(+2,0), (-2,0), (0,+2), (0,-2)\}$ with coordinate clamping $c'_s = \operatorname{clip}(c_s + \Delta_{\pi(r)}, -1, 1)$, plus hidden healthy polarity $g \in \{\pm\frac{1}{2}\}^2$ (deducible from $(c_s, o_s)$ at $t=0$). Under Adapt-1's `topology_model="factorized_conjunctive"` with `topology_require_connected=True`, the 2-cube has $5$ connected axis-aligned spanning subgraphs (full 4-cycle + 4 spanning trees), giving $24 \times 5 = \mathbf{120}$ joint prior worlds.

#### 1.2 Arena 93B — 2-Cluster $\times$ 3-Axis $\times$ 6-Playbook POMDP (`78,480` Prior Worlds; Clean & Flaky-15%)
- **Clusters & SLA Tiers (Shuffled):** `us-east-edge-ingress` (`$200,000`), `eu-central-core-mesh` (`$120,000`). Total SLA = `$320,000`; mean SLA scale $\overline{\text{SLA}}_{93B} = \$160{,}000$.
- **Axes & Playbooks:** 3 orthogonal fault layers (`compute_cpu_fault`, `network_delay_fault`, `storage_loss_fault`) on $\{-1,+1\}^3$; 6 playbooks (`cpu-quota-rebalance`, `pod-evict-reschedule`, `bgp-anycast-reroute`, `ebpf-tc-flush`, `wal-compaction-reset`, `nvme-iops-failover`) with latencies `(1.5, 2.0, 2.5, 3.0, 4.0, 5.0)` min and direct costs `($1.5k, $2.0k, $2.5k, $3.5k, $4.5k, $6.0k)`.
- **Initial State & Action Space ($|\mathcal{A}| = 15$, $H = 14$):** Exactly one cluster initialized at the triple-fault corner ($o_s = -1.0$) and one at a double-fault corner ($o_s = -\frac{1}{3}$), randomly assigned. Actions: `1` `no_op`, `2` `commit`, `2 × 6 = 12` `apply_tool`.
- **Latent World & Prior Count (`78,480` worlds):** Hidden bijective permutation $\pi \in S_6$ ($6! = 720$) over $\{\pm 2 e_0, \pm 2 e_1, \pm 2 e_2\}$ and goal $g \in \{\pm\frac{1}{3}\}^3$. Under `factorized_conjunctive` + `topology_require_connected=True`, the 3-cube admits $109$ connected topologies, yielding $720 \times 109 = \mathbf{78{,}480}$ joint prior worlds.
- **Stochastic Observation / Transition Model (`93B-Flaky15`):** On every `apply_tool(s, r)`, a pre-drawn paired uniform variate $U_t \sim \mathrm{Uniform}(0,1)$ (common random numbers across all agents) triggers a transient timeout whenever $U_t < p_{\text{flaky}} = 0.15$:
  $$c'_s = \begin{cases} c_s & \text{with probability } p_{\text{flaky}} \text{ (timeout / no-op)}, \\ \operatorname{clip}(c_s + \Delta_{\pi(r)}, -1, 1) & \text{with probability } 1 - p_{\text{flaky}}. \end{cases}$$
  Notice that if $\operatorname{clip}(c_s + \Delta_{\pi(r)}, -1, 1) = c_s$ (boundary clamp), the no-op observation occurs with probability $1$ regardless of flake; otherwise a no-op occurs with probability $p_{\text{flaky}}$ and the shifted corner occurs with probability $1 - p_{\text{flaky}}$.

#### 1.3 Arena 93C — 6-Node Online Boutique RPC DAG (Non-Bijective, 2-Wave Transfer)
- **DAG Topology & SLA Tiers:** 6 real Online Boutique nodes with fixed SLA values: `0: paymentservice` (`$200k`), `1: checkoutservice` (`$150k`), `2: frontend` (`$120k`), `3: cartservice` (`$60k`), `4: redis-cart` (`$80k`), `5: currencyservice` (`$30k`); total SLA = `$640,000`, mean scale $\overline{\text{SLA}}_{93C} = \$106{,}666.67$. Directed RPC fault-propagation edges: $(0\!\to\!1), (1\!\to\!2), (4\!\to\!3), (3\!\to\!1), (3\!\to\!2), (5\!\to\!1), (5\!\to\!2)$.
- **2-Wave Cascading Protocol ($|\mathcal{A}| = 31$, $H = 18$):** Partitioned into two 3-node subgraphs: Path A $\{0,1,2\}$ and Path B $\{4,3,5\}$, randomly ordered as Wave 1 and Wave 2. Each wave initializes its root node at `(blast=1, pressure=1)`, direct caller at `(1,1)` w.p. `0.70` (else `(1,0)` or `(0,1)`), and 2-hop caller at `(1,0)` (`0.40`), `(0,1)` (`0.40`), or `(1,1)` (`0.20`), with objective $o_s = 1 - 0.5\,\text{blast}_s - 0.5\,\text{pressure}_s \in \{0.0, 0.5, 1.0\}$. Wave 2 activates seamlessly inside the same episode once all 3 Wave-1 nodes are closed, keeping the exact same hidden playbook mapping to test online transfer.
- **Non-Bijective Effect Space (`200` valid mappings):** 4 playbooks (`1.5, 2.5, 3.5, 4.5` min; `$1.5k, $2.5k, $3.5k, $5.0k`) each map independently to the catalog $\{(-1,-1), (-1,0), (0,-1), (0,0)\}$ (clamped to $[0,1]^2$). Excluding all $4!=24$ bijections, the $31$ unsolvable mappings (neither $(-1,-1)$ nor both single-axis effects), and the trivial all-full mapping $(0,0,0,0)$ yields exactly $4^4 - 24 - 31 - 1 = \mathbf{200}$ strictly non-bijective solvable mappings.

#### 1.4 Unified Economic Reward, Ongoing SLA Burn, & Cost-Ratio Sensitivity
Let $\mathcal{D}_t = \{s : u_s = 0 \land o_s < 0.99\}$ denote open degraded services at step $t$, with active outage exposure $E_t = \sum_{s \in \mathcal{D}_t} \text{SLA}_s$.
- **Commit Payout Fraction $\phi(o_s)$:**
  - *2-axis (93A):* $\phi(1.0) = 1.0$, $\phi(0.0) = 0.20$, $\phi(-1.0) = 0.0$.
  - *3-axis (93B):* $\phi(1.0) = 1.0$, $\phi(+\tfrac{1}{3}) = 0.25$, $\phi(-\tfrac{1}{3}) = 0.05$, $\phi(-1.0) = 0.0$.
  - *DAG (93C):* $\phi(1.0) = 1.0$, $\phi(0.5) = 0.20$, $\phi(0.0) = 0.0$.
- **Step Net Dollar Value & Normalized Return:**
  $$r_t^{\$} = \mathbf{1}\{a_t = \text{commit}(s)\}\,\phi(o_s)\,\text{SLA}_s - c_{\text{direct}}(a_t) - \rho \cdot \Delta \tau(a_t) \cdot E_t, \qquad R_{\text{ep}} = \frac{\sum_{t=0}^{T-1} r_t^{\$}}{\overline{\text{SLA}}}.$$
  Commit actions incur zero latency ($\Delta \tau = 0$, $c_{\text{direct}} = 0$) and immediately remove $\text{SLA}_s$ from future burn $E_{t+1}$. Standby (`no_op`) incurs $\Delta \tau = 2.0\text{ min}$ and $c_{\text{standby}}$.
- **Cost Sensitivity Regimes:** Primary **`1:50`** ($\rho = 0.012/\text{min}$, probe cost multiplier `1.0`, standby `$1,000`); high-burn **`1:25`** ($\rho = 0.024/\text{min}$, mult `2.0`, standby `$2,000`); low-burn **`1:100`** ($\rho = 0.006/\text{min}$, mult `0.5`, standby `$500`).

---

### 2. Exact Comparator Architectures, Tuning Grids, & Training Budgets

Every offline comparator maintains only strictly causal within-episode public observations (`0.0` future-perturbation drift across all 14 tested learners). All hyperparameters are tuned exclusively on `DEV_CAL` seeds.

#### 2.1 Bayesian World-Belief Solvers (`Oracle`, `Exact_1Step_VoI`, `Exact_2Step_VoI`, `POMCP`)
- **Vectorized Exact Belief Filter (`GridBayesBelief` / `DAGBayesBelief93C`):** Maintains exact posterior $b_t(w)$ over all $|\mathcal{W}| = 24$ (93A), $720$ (93B), or $200$ (93C) mappings. Under noise assumption $p_f \in \{0.0, 0.15\}$, observing transition $(c_{\text{pre}}, r) \to c_{\text{post}}$ applies likelihood:
  $$P(c_{\text{post}} \mid c_{\text{pre}}, r, w) = \begin{cases} 1 & \text{if } c_{\text{post}} = c_{\text{pre}} = f_w(c_{\text{pre}}, r), \\ p_f & \text{if } c_{\text{post}} = c_{\text{pre}} \ne f_w(c_{\text{pre}}, r), \\ 1 - p_f & \text{if } c_{\text{post}} \ne c_{\text{pre}} \text{ and } c_{\text{post}} = f_w(c_{\text{pre}}, r), \\ 10^{-12} & \text{otherwise (with uniform reset if } Z \le 10^{-15}\text{).} \end{cases}$$
- **Immediate Probe Utility & Shannon Mutual Information:** For probe $(s, r)$,
  $$v_1(b, s, r) = \frac{\text{SLA}_s}{n_{\text{ax}}\,\overline{\text{SLA}}}\,\mathbb{E}[\Delta_{\text{fixed}}] + 0.35\,\frac{\text{SLA}_s}{\overline{\text{SLA}}}\,P(o'_s = 1) - \frac{c_r + \rho\,\tau_r\,E_t}{\overline{\text{SLA}}}, \qquad I(W; O \mid b, s, r) = H(O) - H(O \mid W).$$
- **`Exact_1Step_VoI`:** Immediately commits the highest-SLA healthy open service; otherwise maximizes $v_1(b, s, r) + \lambda_{\text{VoI}}\,\eta_{\text{urg}}\,I(W; O \mid b, s, r)$ (with urgency damping $\eta_{\text{urg}} = 0.45$ when remaining steps $\le$ minimum flips+commits needed on grids).
- **`Exact_2Step_VoI`:** Exact 2-ply expectimax over all distinct observation branches $(q_o, b'_o)$:
  $$Q_2(b, s, r) = v_1(b, s, r) + 0.50\,\lambda_{\text{VoI}}\,I(W; O \mid b, s, r) + 0.85 \sum_{o} q_o \max_{a_2} V_1(b'_o, a_2).$$
- **`Ideal_Bayes_World_Oracle` / `NoiseAware_Bayes_Oracle` / `Det_Bayes_World_Oracle`:** Exact 2-ply belief-tree search using Smith's $c/\mu$ rule ($\text{SLA}_s / \mathbb{E}[T_{\text{fix},s}]$ ordering under posterior $b'_o$, scaled by $(1-p_f)^{-1}$ when noise-aware) minus remaining world-entropy penalty ($0.18\,H(b')$ for 2-axis, $0.12\,H(b')$ for 3-axis). `Det_Bayes_World_Oracle` sets $p_f = 0$ on 93B-Flaky15 to measure model-misspecification fragility.
- **`POMCP_256` & `POMCP_1000` (Silver & Veness 2010):** Pure-NumPy particle-filter UCT with `256` particles (`93A`/`93B`) or `144` particles (`93C`), systematic resampling after each observation (+ `5%` uniform world reinvigoration when $p_f > 0$), UCT tree depth $\min(3, \text{rem\_steps})$, exploration constant $c_{\text{UCT}}$ tuned on `DEV_CAL`, `256` or `1,000` simulations/step, and a domain-informed rollout policy (certify healthy high-SLA services; else probe highest-SLA open service).

#### 2.2 Slot-Factored Neural RL (`PPO_Factored`) & Contextual Bandits / TD Learners
All parametric RL and bandit peers receive the exact same slot-factored public feature matrix $F(s_t) \in \mathbb{R}^{|\mathcal{A}| \times d_{\text{act}}}$ constructed by `GridEpisodeMemory` / `DAGEpisodeMemory93C` (zero hidden world information):
- **Feature Dimension $d_{\text{act}}$:**
  - *93A ($d_{\text{act}} = 25$)* and *93B ($d_{\text{act}} = 28 = 19 + 3 n_{\text{ax}}$):* Bias, normalized step, remaining fraction, open-service fraction, `any_healthy`, `last_step`, open degraded SLA fraction, 3 operation-type one-hots (`no_op`, `commit`, `apply_tool`), target service objective / healthy / critical flags, normalized service SLA weight, normalized probe latency & direct cost, runbook tried / effect-known / tried-at-corner flags, service coordinates ($n_{\text{ax}}$), observed runbook delta ($n_{\text{ax}}$), and deduced goal vector ($n_{\text{ax}}$).
  - *93C ($d_{\text{act}} = 22$):* Same global, node (`blast`, `pressure`, `objective`, `sla_weight`), playbook latency/cost, and observed $(\Delta\text{blast}, \Delta\text{pressure})$ memory features plus active-wave indicator.
- **`SlotFactoredPPO` (`0 / 200 / 1,000 / 5,000 / 20,000` Disjoint Episodes):**
  - *Architecture (Pure NumPy):* Shared 2-layer `tanh` encoder $h_1 = \tanh(F W_1 + b_1)$, $h_2 = \tanh(h_1 W_2 + b_2)$ with `hidden_dim = 24`, Xavier initialization ($W_1, W_2$) and $0.2\times$ Xavier output-head scaling ($w_p, w_v$). Policy logits $\ell_a = h_{2,a}^\top w_p + b_p$ are masked to $-\infty$ ($-10^9$) on illegal actions before softmax; state value $V(s) = \bigl(\frac{1}{|\mathcal{A}_{\text{valid}}|}\sum_{a \in \mathcal{A}_{\text{valid}}} h_{2,a}\bigr)^\top w_v + b_v$.
  - *Optimization & Schedules:* Discount $\gamma = 0.98$, GAE $\lambda = 0.95$, advantage standardization, clipped surrogate $\epsilon = 0.2$, value loss coefficient `0.5`, batch update every `20` episodes, `3` PPO epochs per batch, global $L_2$ gradient norm clip `1.0`, Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$). Smooth cosine LR decay $\eta(e) = \eta_0 \bigl(0.15 + 0.425(1 + \cos(\pi e / E))\bigr)$ and cosine entropy schedule $c_{\text{ent}}(e) = c_0 \bigl(0.4 + 1.05(1 + \cos(\pi e / E))\bigr)$.
  - *Gradient Verification:* Analytical backpropagation verified against central finite differences ($\delta = 10^{-6}$) to max error $6.91\times 10^{-11}$ (93A), $7.03\times 10^{-11}$ (93B), and $7.75\times 10^{-11}$ (93C).
  - *Note on 93B Convergence Plateau:* On 93B (`78,480` joint worlds, 15 actions, dense negative probe burn), PPO and all 500-ep bandits converge to the safe local optimum of immediately committing the double-fault cluster for $+0.05\times\text{SLA}$ (`+0.052` return, `$8,267`) rather than risking multi-step exploratory probe burn — illustrating the exploration barrier of model-free RL without structured belief tracking.
- **`Factored_RidgeQ_UCB` (`0ep` Online & `500ep` Pre-Trained):** Recursive Sherman-Morrison Ridge TD(0) regression $A^{-1} \leftarrow A^{-1} - \frac{A^{-1}x x^\top A^{-1}}{1 + x^\top A^{-1}x}$, $b \leftarrow b + y_t x_t$, $y_t = r_t + \gamma_{\text{TD}} \operatorname{clip}(\max_{a'} \theta^\top x', -5, 6)$ ($\gamma_{\text{TD}}=0.95$), UCB action bonus $+\alpha\sqrt{x^\top A^{-1} x}$.
- **`LinUCB` & `Thompson_Sampling` (`0ep` & `500ep`):** Episodic discounted return-to-go ($G_t = \sum_{k \ge t} 0.95^{k-t} r_k$) contextual ridge regression; LinUCB adds $+\alpha\sqrt{x^\top A^{-1} x}$, while Thompson samples $\tilde{\theta} \sim \mathcal{N}(\hat{\mu},\, v_{\text{scale}}^2 A^{-1})$ via Cholesky (with eigendecomposition fallback).
- **`Heuristic_Greedy_SRE`:** Rule-based on-call scanner (certify any healthy service $\to$ apply highest-progress known runbook $\to$ scan untried runbooks on lowest-objective service); ignores heterogeneous SLA tiers and probe latencies.
- **Adapt-1 Live Schema Ablations:** `Live_Adapt1_UnauthoredBT` sets `effect_catalog=[]`, `effect_assignment="independent"`, `reversible_effects=False`, `topology_model="none"`, `objective_model="none"`, `empirical_topology_inference=True`; `Live_Adapt1_CatalogOnly` supplies `effect_catalog` while leaving `topology_model="none"` and `objective_model="none"`; `Live_Adapt1_PolicySeq` disables `bound_transition` completely (`enabled=False`).

#### 2.3 Exact `DEV_CAL`-Locked Hyperparameters Across All Four Section 05 Arenas

| Comparator (`DEV_CAL` Grid) | `93A` | `93B-Clean` | `93B-Flaky15` | `93C` |
| :--- | :--- | :--- | :--- | :--- |
| `Exact_1Step_VoI` ($\lambda_{\text{VoI}} \in \{0.15, 0.35, 0.65, 1.0\}$) | `0.35` | `0.15` | `0.65` | `0.15` |
| `Exact_2Step_VoI` ($\lambda_{\text{VoI}} \in \{0.15, 0.35, 0.65, 1.0\}$) | `0.15` | `0.65` | `1.00` | `0.15` |
| `POMCP` ($c_{\text{UCT}} \in \{0.5, 1.0, 1.5, 2.0\}$) | `0.5` | `2.0` | `1.5` | `1.5` |
| `Factored_RidgeQ_UCB` ($\alpha \in \{0.25, 0.5, 1.0\}, \lambda \in \{0.5, 1.0, 2.0\}$) | `(0.25, 2.0)` | `(0.25, 2.0)` | `(0.25, 1.0)` | `(1.0, 2.0)` |
| `LinUCB` ($\alpha \in \{0.25, 0.5, 1.0\}, \lambda \in \{0.5, 1.0, 2.0\}$) | `(0.25, 0.5)` | `(1.0, 1.0)` | `(0.5, 1.0)` | `(0.5, 0.5)` |
| `Thompson` ($v_{\text{scale}} \in \{0.10, 0.25, 0.50\}, \lambda \in \{0.5, 1.0, 2.0\}$) | `(0.25, 0.5)` | `(0.10, 2.0)` | `(0.10, 0.5)` | `(0.25, 1.0)` |
| `PPO_Factored` (`lr` $\in \{3\mathrm{e}{-3}, 5\mathrm{e}{-3}\}$, `ent` $\in \{0.01, 0.025\}$, `h=24`) | `(3e-3, 0.01)` | `(3e-3, 0.01)` | `(3e-3, 0.01)` | `(3e-3, 0.01)` |

---

### 3. Mathematical Analysis of QMDP (`finite_world_qmdp`) vs. Exact Belief Search

In a finite-world POMDP with observable state $s_t$, latent world $w \in \mathcal{W}$, and posterior belief $b_t(w) = P(W = w \mid h_t)$, the exact finite-horizon Bellman recursion over belief states is:
$$Q^*(b_t, a) = \sum_{w \in \mathcal{W}} b_t(w)\,R(s_t, a, w) + \gamma \sum_{o \in \mathcal{O}} P(o \mid b_t, a)\, V^*\!\bigl(\tau(b_t, a, o)\bigr), \qquad V^*(b) = \max_{a \in \mathcal{A}} Q^*(b, a),$$
where $\tau(b_t, a, o)$ is the Bayes-updated posterior after observing $o$.

#### 3.1 The QMDP 1-Step Full-Observability Approximation
The **QMDP** heuristic (Littman, Cassandra, & Kaelbling, 1995) replaces the belief-space continuation value $V^*(\tau(b_t, a, o))$ with the fully observable MDP value $V_{\text{MDP}}^*(s_{t+1}, w)$ inside each fixed world $w$:
$$Q_{\text{QMDP}}(b_t, a) = \sum_{w \in \mathcal{W}} b_t(w)\, Q_{\text{MDP}}^*(s_t, a, w) = \sum_{w \in \mathcal{W}} b_t(w)\left[R(s_t, a, w) + \gamma \sum_{s'} P(s' \mid s_t, a, w)\,V_{\text{MDP}}^*(s', w)\right].$$
In Adapt-1's `finite_world_qmdp` planner (configured on Arena 93B with `belief_planning_mode="qmdp"`, `belief_planning_horizon=3`, `belief_qmdp_max_worlds=64`, `belief_beam_width=16`, `belief_information_weight=0.25`, switching to exact search only once surviving worlds drop to $\le 16$ via `belief_exact_world_limit=16`), the expectation is evaluated over at most $M \le 64$ active worlds rather than expanding an exact belief tree.

#### 3.2 Complexity Collapse: Why QMDP Runs in `563 ms` vs. `40 s` Exact Timeout
- **Exact Belief-Tree Search (`belief_planning_mode="exact"`):** At each ply $d = 1, \dots, H_b$, the planner branches over $|\mathcal{A}|$ actions and up to $|\mathcal{O}|$ distinct observation outcomes, updating posteriors across $|\mathcal{W}|$ worlds at every node:
  $$\mathcal{C}_{\text{exact}} = \mathcal{O}\!\Bigl(|\mathcal{W}| \cdot (|\mathcal{A}| \cdot |\mathcal{O}|)^{H_b}\Bigr).$$
  On 2-axis 93A ($|\mathcal{W}| = 120$), exact search finishes in `442–655 ms` median. On 3-axis 93B ($|\mathcal{W}| = 78{,}480$, $654\times$ larger world space, $|\mathcal{A}| = 15$, up to $7$ observation outcomes per probe), exact tree expansion exceeds the `40,157 ms` HTTP 504 gateway wall.
- **QMDP (`belief_planning_mode="qmdp"`):** Eliminates observation branching completely. Each of the $M \le 64$ active worlds is solved independently as a deterministic/fully-observable short-horizon MDP ($H_{\text{QMDP}} = 3$, beam `16`), yielding linear complexity in $M$:
  $$\mathcal{C}_{\text{QMDP}} = \mathcal{O}\!\Bigl(M \cdot |\mathcal{A}| \cdot H_{\text{QMDP}}\Bigr).$$
  This yields **zero HTTP 504 timeouts across `828/828` live 93B queries**, reducing median query latency to **`563 ms`** on the production parity benchmark (`733–765 ms` across full 30-episode clean/flaky runs — a **$54.8\times$–$71.3\times$ latency reduction** vs. the 40 s exact timeout) while collapsing surviving worlds from `78,480` to a median of `61.5` (`95.0` mean) by step 6.

#### 3.3 Why QMDP Systematically Undervalues Pure Information-Gathering Probes
By Jensen's inequality (commuting $\max$ and $\mathbb{E}$), the fully observable continuation value strictly upper-bounds the true partially observable value:
$$V^*(b_{t+1}) = \max_{\pi} \mathbb{E}_{w \sim b_{t+1}}\!\bigl[V^\pi(s_{t+1}, w)\bigr] \;\le\; \mathbb{E}_{w \sim b_{t+1}}\!\bigl[V_{\text{MDP}}^*(s_{t+1}, w)\bigr] \;\triangleq\; V_{\text{FO}}(b_{t+1}).$$
Define the **expected value of information** of action $a$ as the reduction in the partial-observability gap $\Delta_{\text{VOI}}(b_t, a) = \gamma\,\mathbb{E}_o\bigl[V^*(\tau(b_t,a,o))\bigr] - \gamma\,\max_{a'}\mathbb{E}_{w \sim b_t}\bigl[\mathbb{E}_{s'}[Q^*(s', a', w)]\bigr]$. Under QMDP, the continuation value is replaced by $V_{\text{FO}}$, which satisfies by the law of iterated expectations:
$$\sum_{o \in \mathcal{O}} P(o \mid b_t, a)\sum_{w \in \mathcal{W}} \tau(b_t, a, o)(w)\,V_{\text{MDP}}^*(s', w) \;=\; \sum_{w \in \mathcal{W}} b_t(w) \sum_{s'} P(s' \mid s_t, a, w)\,V_{\text{MDP}}^*(s', w).$$
The posterior update $\tau(b_t, a, o)$ **cancels out algebraically**: QMDP behaves as if an oracle will reveal the true hidden world $w$ at step $t+1$ *regardless of whether action $a$ resolves uncertainty or not*. Consequently:
1. Any diagnostic probe $a_{\text{probe}}$ whose immediate reward is negative ($-c_{\text{probe}} - \text{burn} < 0$) and whose value lies in collapsing world entropy $H(W \mid b_{t+1})$ receives **zero intrinsic belief-collapse credit** in $Q_{\text{QMDP}}$ beyond the small additive `belief_information_weight` heuristic evaluated on a `64`-world subsample of `78,480` worlds.
2. In 3-axis 93B-Clean, this causes `Live_Adapt1_QMDP_Native` to under-explore diagnostic probes and accumulate `40.1 min` of downtime (`-0.631` return, `0.43/2` healthy) relative to exact belief-updating `Exact_1Step_VoI` (`+1.420`, `2.00/2` healthy), `Exact_2Step_VoI` (`+1.416`), and `POMCP_1000` (`+1.349`). This trade-off directly reflects the classic theoretical boundary of QMDP: orders-of-magnitude tractability at scale, paired with myopia toward pure information-gathering actions until surviving worlds shrink below the exact-planning threshold.
