# Forecasting & Online-Adaptation Baselines Technical Specification (Sections 01, 03, and 04)

This document specifies the exact mathematical formulations, optimization routines, convergence fallbacks, quantile grids, copula constructors, drift detectors, random-seeding rules, survival-truncation logic, and calibration-locked hyperparameters for every comparator evaluated in **Section 01** (*Joint tails and drift*), **Section 03** (*Plasticity and recurrence*), and **Section 04** (*Conditioning across time*) of *Forecasting Through Drift (Part 2)*.

---

## 0. Common Causal, Calibration, and Reproducibility Protocol

Every comparator across Sections 01, 03, and 04 obeys four uniform audit invariants:

1. **Strict Score-Before-Ingest Causality ($\mathcal{F}_{t-H}$ Embargo):** At decision time $t$, all model parameters, feature scalers, residual banks, copula correlation matrices, and conformal scales are estimated solely from observations whose outcomes have matured strictly before $t$ (i.e., $s \le t - H$ for fixed-horizon forecasting at lead $H \ge 1$, or $t_{\mathrm{resolve}}(j) < t_{\mathrm{query}}(i)$ for asynchronous operational queues). Every baseline family passes an automated future-perturbation test verifying $\max_{t < t_{\mathrm{cut}}} \|\hat{P}_t(D) - \hat{P}_t(D_{\mathrm{pert}})\|_{\infty} = 0.0$ when all inputs and outcomes at $t \ge t_{\mathrm{cut}}$ are overwritten.
2. **Disjoint Calibration (`DEV_CAL`) vs. Sealed Evaluation (`SEALED_OOS`):** All baseline hyperparameters (window widths, ridge penalties, tree depths, EWMA decays, ACI step sizes, bandit exploration bonuses, and foundation-model context/covariate modes) are grid-selected strictly on pre-evaluation `DEV_CAL` slices (or disjoint `DEV_CAL` synthetic seeds). `SEALED_OOS` observations are excluded from RAM during hyperparameter locking.
3. **Deterministic Seeding:** Stochastic baselines (Monte Carlo copulas, Student-$t$ sampling, NeuralUCB, Thompson Sampling, EXP3.S, and mixed-atom resampling) derive 64-bit integer seeds via `SHA-256` hashing of `(label, step_index)` or `(label, seed_id)` passed to NumPy `PCG64` (`np.random.default_rng`), yielding bit-reproducible clouds across runs.
4. **Canonical Predictive Cloud Interface:** Every probabilistic forecasting baseline emits an explicit weighted atom cloud $\mathcal{C}_t = \{(y_t^{(k)}, p_{t,k})\}_{k=1}^M$ with $p_{t,k} \ge 0$ and $\sum_{k=1}^M p_{t,k} = 1$, scored identically by the multivariate Energy Score, 1D CRPS, regulatory VaR/ES backtests, and downstream portfolio/hedge overlays.

---

## 1. Section 01 Baselines — Joint Tails and Drift (`Exp 98A`, `Exp 79B`, `Exp 89B`, `Exp 90A`, `Exp 90C`)

### 1.1 Rolling & Expanding GJR-DCC-GARCH(1,1) Suite (All 12 Variants — `Exp 98A`)

On the 4-dimensional German day-ahead power block price panel (`Exp 98A`, $m = 4$: Night, Morning, Solar-Midday, Evening Peak; `DEV_CAL` $n = 200$, `SEALED_OOS` $n = 450$), all 12 econometric comparators are implemented directly in NumPy/SciPy (custom variance-targeted QML/MLE rather than an opaque wrapper) to guarantee strict `< t` causality, deterministic seeding, and complete optimizer telemetry.

#### A. Mean Equations (Two Specifications)

Let $y_t \in \mathbb{R}^m$ denote the realized 4-block price vector on delivery day $t$.

1. **Strictly Causal EWMA Mean (`mean_model="ewma"`):**
   $$\hat{\mu}_0 = \mathbf{0}, \qquad \hat{\mu}_1 = y_0, \qquad \hat{\mu}_t = \lambda_\mu \hat{\mu}_{t-1} + (1 - \lambda_\mu) y_{t-1} \quad (t \ge 2), \qquad r_t = y_t - \hat{\mu}_t.$$
   `DEV_CAL` grid sweep over $\lambda_\mu \in \{0.90, 0.94, 0.97\}$ locked $\lambda_\mu = 0.90$ (mean `DEV_CAL` 4D ES `67.97` vs. `68.84` and `71.88`).

2. **Per-Block Rolling / Expanding OLS AR(1) Mean (`mean_model="ar1"`):**
   At each refit step $t$ with training window $[s_{\mathrm{ar}}, t)$, for each dimension $j \in \{1,\dots,m\}$ we regress $y_{j,s}$ on $y_{j,s-1}$ for $s \in [s_{\mathrm{ar}}+1, t)$:
   $$\hat{\phi}_j = \mathrm{clip}\!\left(\frac{\sum_s (y_{j,s-1} - \bar{y}_{j,-1})(y_{j,s} - \bar{y}_{j,0})}{\sum_s (y_{j,s-1} - \bar{y}_{j,-1})^2 + 10^{-6}},\; -0.98,\; 0.98\right), \qquad \hat{c}_j = \bar{y}_{j,0} - \hat{\phi}_j \bar{y}_{j,-1},$$
   giving strictly causal one-step predictions $\hat{\mu}_{j,s} = \hat{c}_j + \hat{\phi}_j y_{j,s-1}$ and innovations $r_{j,s} = y_{j,s} - \hat{\mu}_{j,s}$.

#### B. Step 1 — Variance-Targeted Asymmetric GJR-GARCH(1,1) QML

For each coordinate $j$ over the $n_{\mathrm{fit}}$ training innovations $\{r_{j,0}, \dots, r_{j,n_{\mathrm{fit}}-1}\}$, let $\bar{v}_j = \max(\frac{1}{n_{\mathrm{fit}}}\sum_{i=0}^{n_{\mathrm{fit}}-1} r_{j,i}^2,\; 10^{-4})$ and let the empirical negative-innovation fraction be
$$\kappa_j = \mathrm{clip}\!\left(\frac{1}{n_{\mathrm{fit}}} \sum_{i=0}^{n_{\mathrm{fit}}-1} \mathbb{I}[r_{j,i} < 0],\; 0.15,\; 0.85\right).$$
With persistence $P_j = \alpha_j + \kappa_j \gamma_j + \beta_j$ and variance-targeted intercept $\omega_j = \max((1 - P_j)\bar{v}_j,\; 10^{-6})$, the conditional variance recursion is initialized at $h_{j,0} = \max(\frac{1}{\min(40, n_{\mathrm{fit}})}\sum_{i=0}^{\min(40,n_{\mathrm{fit}})-1} r_{j,i}^2,\; 10^{-4})$ and evolves as:
$$h_{j,i} = \max\!\left(\omega_j + \bigl(\alpha_j + \gamma_j \mathbb{I}[r_{j,i-1} < 0]\bigr) r_{j,i-1}^2 + \beta_j h_{j,i-1},\; 10^{-6}\right), \qquad i = 1,\dots,n_{\mathrm{fit}}-1.$$
Parameters $(\alpha_j, \gamma_j, \beta_j)$ minimize the Gaussian quasi-negative log-likelihood:
$$\mathcal{L}_{\mathrm{GJR}}(\alpha_j, \gamma_j, \beta_j) = \frac{1}{2} \sum_{i=1}^{n_{\mathrm{fit}}-1} \left(\log h_{j,i} + \frac{r_{j,i}^2}{h_{j,i}}\right)$$
subject to box constraints $\alpha_j \in [0.002, 0.50]$, $\gamma_j \in [0.0, 0.60]$, $\beta_j \in [0.30, 0.985]$, and stationarity $P_j < 0.995$ (infeasible evaluations return penalty $10^{12}$).

- **Two-Stage Optimizer & Convergence Fallbacks:** Each refit warm-starts from the previous refit's $(\alpha_j, \gamma_j, \beta_j)$ (initial default $(0.08, 0.06, 0.78)$).
  - *Stage 1:* Bounded `L-BFGS-B` (`maxiter=60, ftol=1e-7`).
  - *Stage 2 (Refinement / Fallback):* If Stage 1 does not report convergence or improves NLL by less than $10^{-4}$ over the warm start, `Nelder-Mead` (`maxiter=140, xatol=1e-3, fatol=1e-2`) runs from the best point and is accepted whenever it lowers NLL.
  - *Post-Fit Stability Guard:* If the final persistence satisfies $P_j \ge 0.995$, all three parameters are proportionally rescaled by $0.990 / P_j$.
- **Standardized Residuals:** $e_{j,i} = \mathrm{clip}\!\left(r_{j,i} / \sqrt{\max(h_{j,i}, 10^{-6})},\; -8.0,\; 8.0\right)$.

#### C. Step 2 — Engle DCC(1,1) Correlation QML & Fallbacks

Given the $n_{\mathrm{fit}} \times m$ standardized innovation matrix $E$, the unconditional correlation target is shrunk and re-normalized to unit diagonal:
$$\tilde{Q} = 0.97 \left(\frac{1}{n_{\mathrm{fit}}} E^\top E\right) + 0.03 I_m, \qquad \bar{Q} = D_{\tilde{Q}}^{-1/2} \tilde{Q} D_{\tilde{Q}}^{-1/2}.$$
Initializing $Q_0 = \bar{Q}$, the DCC(1,1) recursion is:
$$Q_i = (1 - a_{\mathrm{dcc}} - b_{\mathrm{dcc}})\bar{Q} + a_{\mathrm{dcc}} e_{i-1} e_{i-1}^\top + b_{\mathrm{dcc}} Q_{i-1}, \qquad R_i = 0.995\, D_{Q_i}^{-1/2} Q_i D_{Q_i}^{-1/2} + 0.005 I_m.$$
Parameters $(a_{\mathrm{dcc}}, b_{\mathrm{dcc}})$ minimize the correlation QML objective
$$\mathcal{L}_{\mathrm{DCC}}(a_{\mathrm{dcc}}, b_{\mathrm{dcc}}) = \frac{1}{2} \sum_{i=1}^{n_{\mathrm{fit}}-1} \left(\log |R_i| + e_i^\top R_i^{-1} e_i\right)$$
over $a_{\mathrm{dcc}} \ge 0.002$, $b_{\mathrm{dcc}} \ge 0.35$, $a_{\mathrm{dcc}} + b_{\mathrm{dcc}} < 0.992$ via warm-started `Nelder-Mead` (`maxiter=70, xatol=2e-3, fatol=5e-2`, initial $(0.04, 0.88)$). Any non-SPD $R_i$ during optimization returns penalty $10^{12}$. At forecast/whitening time, Cholesky factorization $R_t = L_t L_t^\top$ falls back to symmetric eigendecomposition $L_t = U \operatorname{diag}(\sqrt{\max(\lambda, 10^{-6})})$ if Cholesky fails.

#### D. Step 3 — Predictive Cloud Construction (FHS vs. Multivariate Student-$t$ MLE)

1. **Filtered Historical Simulation (`FHS`):** At each refit, the trailing `fhs_win` standardized vectors are causally whitened via $u_s = L_s^{-1} e_s$ using each historical step's own factor $L_s$. Between refits, as each day $t-1$ matures, the variance $h_t$, correlation $Q_t$, and whitened innovation $u_{t-1} = L_{t-1}^{-1} e_{t-1}$ update strictly online (retaining the trailing `fhs_win` bank $U_t$). Forecast scenarios are projected with uniform weights $p_k = 1 / |U_t|$:
   $$y_t^{(k)} = \hat{\mu}_t + \sigma_t \odot (L_t u_k), \qquad \sigma_{t,j} = \sqrt{\max(h_{t,j}, 10^{-6})}.$$

2. **Continuous Multivariate Student-$t$ (`StudentT`):** Conditioning on the fitted DCC quadratic forms $q_i = e_i^\top R_i^{-1} e_i$ ($i=1,\dots,n_{\mathrm{fit}}-1$), the degrees of freedom $\nu$ is estimated by bounded scalar MLE (`scipy.optimize.minimize_scalar(..., bounds=(3.1, 60.0), method="bounded", xatol=1e-2)`, fallback $\nu = 8.0$) minimizing the exact variance-normalized $m$-variate Student-$t$ NLL:
   $$\mathcal{L}_t(\nu) = -\sum_{i=1}^{n_{\mathrm{fit}}-1} \left[\log \Gamma\!\left(\frac{\nu+m}{2}\right) - \log \Gamma\!\left(\frac{\nu}{2}\right) - \frac{m}{2}\log(\pi(\nu - 2)) - \frac{\nu + m}{2}\log\!\left(1 + \frac{q_i}{\nu - 2}\right)\right].$$
   At each forecast step $t$, $M = 256$ independent draws $g_k \sim \mathcal{N}(0, I_m)$ and $\chi_k \sim \chi^2_\nu$ (SHA-256 seeded) produce uniform-weight ($1/256$) scenarios:
   $$y_t^{(k)} = \hat{\mu}_t + \sigma_t \odot \left(L_t g_k \sqrt{\frac{\nu - 2}{\max(\chi_k, 10^{-6})}}\right).$$

#### E. Complete Roster of All 12 GJR-DCC-GARCH Variants (`Exp 98A`)

| Variant | Mean | Fit Window ($n_{\mathrm{fit}}$) | Cloud ($M$) | Refit (OOS) | GJR / DCC Conv. |
|---|---|---|---:|---:|---:|
| GJR-DCC FHS Roll128 r21 | EWMA ($\lambda_\mu=0.90$) | Roll 128 | FHS (128) | 21d (22) | 100% / $\ge 86\%$ |
| GJR-DCC FHS Roll256 r21 | EWMA ($\lambda_\mu=0.90$) | Roll 256 | FHS (256) | 21d (22) | 100% / $\ge 86\%$ |
| GJR-DCC FHS Roll512 r21 | EWMA ($\lambda_\mu=0.90$) | Roll 512 | FHS (512) | 21d (22) | 100% / $\ge 86\%$ |
| GJR-DCC FHS Expand r21 | EWMA ($\lambda_\mu=0.90$) | Exp (604→1045) | FHS (256) | 21d (22) | 100% / 95.5% |
| GJR-DCC FHS Roll256 r5 | EWMA ($\lambda_\mu=0.90$) | Roll 256 | FHS (256) | 5d (90) | 100% / $\ge 86\%$ |
| AR1 GJR-DCC FHS Roll128 r21 | OLS AR(1) | Roll 128 | FHS (128) | 21d (22) | 100% / $\ge 86\%$ |
| AR1 GJR-DCC FHS Roll256 r21 | OLS AR(1) | Roll 256 | FHS (256) | 21d (22) | 100% / 100% |
| AR1 GJR-DCC FHS Roll512 r21 | OLS AR(1) | Roll 512 | FHS (512) | 21d (22) | 100% / $\ge 86\%$ |
| AR1 GJR-DCC FHS Expand r21 | OLS AR(1) | Exp (603→1044) | FHS (256) | 21d (22) | 100% / 86.4% |
| AR1 GJR-DCC FHS Roll256 r5 | OLS AR(1) | Roll 256 | FHS (256) | 5d (90) | 100% / $\ge 86\%$ |
| AR1 GJR-DCC StudentT Roll256 r21 | OLS AR(1) | Roll 256 | Student-$t$ (256) | 21d (22) | 100% / 100% |
| AR1 GJR-DCC StudentT Expand r21 | OLS AR(1) | Exp (603→1044) | Student-$t$ (256) | 21d (22) | 100% / 86.4% |

All 12 variants are programmatically audited for non-constant MLE parameter trajectories across refits and strictly increasing $n_{\mathrm{fit}}$ on expanding arms.

---

### 1.2 Exogenous Linear, Local-Linear, Tree, and Dynamic-Covariance Peers

#### A. Offline Local Linear Regression (`Exp 98A`)

- **`Offline Local Linear Regression` (`trace_tag: local_linres_nw`):** Rolling-window local linear regression fit over the trailing window $\mathcal{W}_t = \{t - W, \dots, t - 1\}$ ($W = 128$) on the identical pre-auction covariates with Ridge regularization ($\lambda = 40.0$) and local smoothing bandwidth $h = 0.20$ (`DEV_CAL` grid $h \in \{0.20, 0.25, 0.35, 0.50\} \times \lambda \in \{5, 15, 40\}$ locked $h = 0.20, \lambda = 40.0$).

#### B. Two-Stage Heteroskedastic Ridge (`Tier2_HeteroRidge` — `Exp 98A`, `Exp 89B`)

Over trailing window $W$ with standardized design matrix $U = [\mathbf{1}, Z]$:
1. **Stage 1 (Conditional Mean):** Solve $B_{\mu} = (U^\top U + \operatorname{diag}(0, r_{\mathrm{mean}}\mathbf{1}_p))^{-1} U^\top Y_{\mathcal{W}_t}$, obtaining residuals $r_i = y_i - u_i^\top B_{\mu}$ and query mean $\hat{\mu}_t = u_t^\top B_{\mu}$.
2. **Stage 2 (Conditional Log-Variance):** Regress $v_{i,j} = \log(r_{i,j}^2 + c)$ on $U$ with ridge penalty $r_{\mathrm{var}}$ ($c = 1.0$, log-variance clipped to $[-6, 12]$ in `Exp 98A`; $c = 10^{-4}$, clipped to $[-8, 4]$ in `Exp 89B`). Form standardized residuals $\tilde{r}_{i,j} = \mathrm{clip}(r_{i,j} / \hat{\sigma}_{i,j}, -C, C)$ ($C = 8$ in `Exp 98A`, $C = 6$ in `Exp 89B`) and emit uniform-weight ($1/W$) rescaled scenarios $y_t^{(i)} = \hat{\mu}_t + \tilde{r}_i \odot \hat{\sigma}_t$ (clipped to $[0, 1]$ for wind capacity factors in `Exp 89B`).

#### C. Online Quantile HistGradientBoosting + Causal PIT-Z Gaussian Copula (`Online_Q_HGBR_GaussCop`)

For each target coordinate $j \in \{1,\dots,m\}$ and each quantile level $\tau \in \mathcal{Q}$, an independent `sklearn.ensemble.HistGradientBoostingRegressor(loss="quantile", quantile=tau)` is fit on the trailing $W$ matured rows and periodically refit every $R_{\mathrm{refit}}$ steps. Crossed quantile curves are repaired via isotonic forward accumulation (`np.maximum.accumulate`).

- **Quantile Grids:**
  - `Exp 98A`, `Exp 89B`, `Exp 90A`, `Exp 90C`: 9-point grid $\mathcal{Q}_9 = \{0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95\}$.
  - `Exp 79A`, `Exp 79B`: 9-point tail-extended grid $\mathcal{Q}_{9,\mathrm{tail}} = \{0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975\}$.
- **Causal PIT-Z Gaussian Copula:** As each step $s \le t - H$ matures, its marginal probability integral transform $u_{s,j} = \mathrm{clip}(\hat{F}_{s,j}(y_{s,j}), 10^{-4}, 1 - 10^{-4})$ (computed by piecewise-linear CDF inversion over $\hat{q}_{s,j}$) is transformed to normal scores $z_{s,j} = \mathrm{clip}(\Phi^{-1}(u_{s,j}), -4.5, 4.5)$. Over the trailing `copula_win` matured PIT-Z vectors, the empirical correlation matrix is shrunk toward identity:
  $$R_t = 0.95\, \hat{R}_t + 0.05 I_m, \qquad R_t = L_t L_t^\top$$
  (with eigendecomposition fallback if Cholesky fails). Drawing $M = 256$ standard normals (antithetic $[Z_{128}; -Z_{128}]$ in `Exp 89B/90A/90C`, SHA-256 seeded), correlated uniform draws $U_t = \Phi(Z L_t^\top)$ are inverted coordinate-by-coordinate through $\hat{q}_{t,j}$ (linear interpolation inside $[\tau_{\min}, \tau_{\max}]$ and linear tail slope extrapolation outside, clipped to $[0, 1]$ for bounded wind capacity factors).

#### D. Online Ridge, EWMA Empirical Copula, Fixed-Parameter DCC-GARCH-FHS, and Prior Counterfactual

- **`Online_Ridge`:** Rolling-window standardized multi-output Ridge regression with residual resampling over the trailing window $W$ (using standardized residuals in `Exp 89B`).
- **`EWMA_EmpResid_Copula`:**
  - *Coordinate-scale variant (`Exp 98A`, `Exp 89B`, `Exp 90A`, `Exp 90C`):* Strictly causal recursive EWMA mean $\hat{\mu}_t = \lambda \hat{\mu}_{t-1} + (1 - \lambda) y_{t-1}$ and diagonal variance $\hat{v}_t = \lambda \hat{v}_{t-1} + (1 - \lambda)(y_{t-1} - \hat{\mu}_{t-1})^2$; trailing $W$ standardized residual vectors $e_s = \mathrm{clip}((y_s - \hat{\mu}_s)/\sqrt{\hat{v}_s}, -6, 6)$ are rescaled by $\sqrt{\hat{v}_t}$ with uniform weights $1/W$, preserving the empirical cross-asset dependence.
  - *Multivariate covariance-whitened variant (`Exp 79B`):* Tracks full $m \times m$ EWMA covariance $\Sigma_t = \lambda \Sigma_{t-1} + (1 - \lambda)(y_{t-1} - \hat{\mu}_{t-1})(y_{t-1} - \hat{\mu}_{t-1})^\top$ ($\lambda = 0.94, W = 256$), whitens innovations via symmetric eigendecomposition $z_s = \mathrm{clip}(\Sigma_s^{-1/2}(y_s - \hat{\mu}_s), -6, 6)$, and projects $y_t^{(k)} = \hat{\mu}_t + \Sigma_t^{1/2} z_k$.
- **Fixed-Parameter Causal `DCC_GARCH_FHS` (`Exp 79A`, `Exp 79B`):** Distinct from `Exp 98A`'s full rolling/expanding MLE GJR-DCC suite, `Exp 79` uses a strictly causal RiskMetrics-style fixed-parameter GARCH(1,1) ($\alpha = 0.08, \beta = 0.88$, variance-targeted $\omega_j = (1 - \alpha - \beta)\bar{v}_j$) coupled with Engle DCC(1,1) ($a_{\mathrm{dcc}} = 0.05, b_{\mathrm{dcc}} = 0.92$, EWMA demean $\lambda_\mu = 0.94$), symmetric eigendecomposition whitening ($z_s$ clipped to $[-6, 6]$), and a trailing $W = 256$ FHS innovation bank.
- **`Own_PriorCF_W128`:** Unconditioned empirical cloud assigning uniform mass $1/W$ to the exact trailing $W = 128$ matured target vectors.

#### E. Locked `DEV_CAL` Hyperparameters Across Section 01 Experiments

| Exp | Q-HGBR (GaussCop) | HeteroRidge | Online Ridge | EWMA EmpResid |
|---|---|---|---|---|
| **Exp 98A** (DE Power, 4D) | W=256, d=3, l2=5.0, iter=60, lr=0.06, leaf=12, refit=21, cop=128 | W=256, r_mean=5.0, r_var=40.0 | — (Local Linear: W=128, h=0.20, λ=40.0) | W=128, λ=0.85 (grid 0.85..0.97) |
| **Exp 79B** (Binance 4-Asset) | W=512, d=2, l2=25.0, iter=50, leaf=10, refit=25, cop=256 | — | W=256, α ∈ {1, 5, 25, 100, 500} | W=256, λ=0.94 (full cov) |
| **Exp 89B** (EU 4-TSO Wind) | W=384, d=3, l2=5.0, iter=70, lr=0.07, refit=32, cop=256, clip[0,1] | W=128, r_mean=5.0, r_var=75.0 | W=128, r=1.0 (std. resid) | W=128, λ=0.80 |
| **Exp 90A** (US Wind) | W=512, d=2, l2=5.0, refit=32, cop=256, clip[0,1] | — | W=512, r=1.0, clip[0,1] | W=128, α=0.15 (λ=0.85) |
| **Exp 90C** (Binance Long-Ctx) | W=512, refit=32, cop=256 | — | W=256, r=1.0 | W=128, α=0.15 (λ=0.85) |

---

### 1.3 Pretrained Time-Series Foundation Models & Adaptive Conformal Inference (`ACI`)

#### A. Pretrained Checkpoints, Context Ladders, and Attention Audits

Two pretrained zero-shot probabilistic foundation models are evaluated strictly causally in PyTorch:

- **`google/timesfm-3.0-pytorch`** (`TimesFM3Forecaster`, native maximum context length `15,360` tokens).
- **`amazon/chronos-2`** (`Chronos2Pipeline`, native maximum context length `8,192` tokens).

At the longest evaluated context (`ctx = 2,048` in `Exp 90C`), both models pass two explicit diagnostic audits:
1. **Non-Truncated Long-Context Utilization:** Perturbing only the earliest window `[0:512]` of a `2,048`-step input changes the forecast quantile outputs by mean absolute delta `0.1533` (`TimesFM-3.0`) and `0.0902` (`Chronos-2`), confirming active attention across the full context.
2. **Exact Causality:** Perturbing future targets/covariates beyond $t$ produces exact `0.0` change in all pre-cutoff predictions across all context lengths and covariate modes.

Evaluated context ladders:
- `Exp 89B`: $\mathrm{ctx} \in \{256, 512, 1024\}$ (actual available history in OOS ranges `729..1,024` days at `ctx=1,024`).
- `Exp 90A`: $\mathrm{ctx} \in \{256, 512, 1024, 1536\}$.
- `Exp 90C`: $\mathrm{ctx} \in \{256, 512, 1024, 2048\}$ (yielding `32` raw TSFM configurations across `2` models $\times$ `4` contexts $\times$ `2` covariate modes $\times$ `2` copula constructors, of which `8` operate at `ctx=2,048`).

#### B. Covariate Wiring Modes (`TargetOnly` vs. `CovExog`)

- **`TargetOnly`:** Only historical target trajectories $Y[t-\mathrm{ctx} : t]$ are supplied (for `Chronos-2`, jointly across all $m$ target channels; for `TimesFM-3.0`, target channel $j$ as the primary series and the remaining $m - 1$ target channels as `past_only_covariates`).
- **`CovExog`:** Exogenous drivers known strictly at issue time $t$ are additionally supplied: `Chronos-2` receives `past_covariates = X[st:t]` and `future_covariates = X[t:t+1]` jointly; `TimesFM-3.0` receives `past_future_covariates = X[st:t+1]` alongside the other $m - 1$ targets as `past_only_covariates`.

#### C. Canonical 21-Point Quantile Grid & TimesFM-3.0 Split-Gaussian Tail Expansion

All foundation models are standardized to a common 21-point marginal quantile grid:
$$\mathcal{Q}_{21} = \{0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99\}.$$
- **`Chronos-2`** outputs all 21 quantile levels natively via `predict_quantiles(..., quantile_levels=Q_GRID_21)`.
- **`TimesFM-3.0`** natively outputs the 9 deciles $\{0.10, 0.20, \dots, 0.90\}$ (after dropping the leading point-mean slot when 10 outputs are returned). Interior points in $[0.10, 0.90]$ are linearly interpolated; extreme tails $\tau \in \{0.01, 0.05\}$ and $\tau \in \{0.95, 0.99\}$ are extrapolated via a split-Gaussian approximation matched to the predicted median and outer deciles:
  $$\sigma_L = \frac{\max(\hat{q}_{0.50} - \hat{q}_{0.10},\; 10^{-8})}{\Phi^{-1}(0.90)}, \qquad \sigma_R = \frac{\max(\hat{q}_{0.90} - \hat{q}_{0.50},\; 10^{-8})}{\Phi^{-1}(0.90)},$$
  $$\hat{q}_\tau = \begin{cases} \hat{q}_{0.50} + \sigma_L \Phi^{-1}(\tau), & \tau < 0.10, \\ \hat{q}_{0.50} + \sigma_R \Phi^{-1}(\tau), & \tau > 0.90, \end{cases}$$
  followed by monotone enforcement `np.maximum.accumulate`.

#### D. Multivariate Cloud Constructors (`Indep` vs. `GaussCop`)

From the 21-quantile marginal forecast $\hat{q}_{t,j}$ ($j = 1,\dots,m$), $M = 256$ uniform-weight ($1/256$) joint scenarios are generated by two constructors:
1. **`Indep`:** Independent uniform draws $u_{k,j} \sim \mathrm{Unif}(10^{-4}, 1 - 10^{-4})$ inverted piecewise-linearly through $\hat{q}_{t,j}$ (with linear end-segment slope extrapolation).
2. **`GaussCop`:** Uses the exact causal rolling PIT-Z correlation matrix over the trailing `COPULA_WIN = 256` matured residuals ($z_{s,j} = \mathrm{clip}(\Phi^{-1}(\mathrm{PIT}(y_{s,j}, \hat{q}_{s,j})), -4.5, 4.5)$, shrinkage $R_t = 0.95 \hat{R}_t + 0.05 I_m$, Cholesky factor $L_t$ with eigendecomposition fallback), antithetic normal draws $Z = [Z_{128}; -Z_{128}]$, and $U = \Phi(Z L_t^\top)$ inverted dimension-by-dimension.

#### E. Adaptive Conformal Inference Wrapper (`Online_Conformal_Airbag` / `ACI`)

To test whether a nonparametric conformal recalibrator suffices to repair foundation-model tails without regime conditioning, `Exp 89B`, `Exp 90A`, and `Exp 90C` implement a strictly causal multiplicative Gibbs & Candès (2021)-style ACI wrapper over the trailing $W = 128$ matured residual vectors $H_t$:

- **Two Residual Coordinate Systems:**
  1. `pit_z`: Normal-score residual $r_{m,j} = \mathrm{clip}(\Phi^{-1}(\mathrm{PIT}(y_{m,j}, \hat{q}_{m,j})), -4.5, 4.5)$, reconstructed via $y_t^{(k)} = \hat{F}_t^{-1}\!\bigl(\Phi(\mathrm{clip}(s_t r_{k}, -4.5, 4.5))\bigr)$.
  2. `studentized_add`: Spread-normalized additive residual $r_{m,j} = (y_{m,j} - \hat{q}_{0.50,m,j}) / \max(\hat{q}_{0.90,m,j} - \hat{q}_{0.10,m,j}, \epsilon)$, reconstructed via $y_t^{(k)} = \hat{q}_{0.50,t} + s_t r_k \odot \max(\hat{q}_{0.90,t} - \hat{q}_{0.10,t}, \epsilon)$.
- **Multiplicative Scale Update:** Initialized at $s_0 = 1.0$. Whenever a previous forecast from step $m = t - H$ matures, the empirical coordinate-averaged 90% interval miscoverage rate is computed from the wrapped predictive bounds $[\hat{q}_{0.05,m,j}^{\mathrm{wrap}}, \hat{q}_{0.95,m,j}^{\mathrm{wrap}}]$:
  $$\mathrm{err}_m = \frac{1}{m}\sum_{j=1}^m \mathbb{I}\!\left[y_{m,j} < \hat{q}_{0.05,m,j}^{\mathrm{wrap}} \;\lor\; y_{m,j} > \hat{q}_{0.95,m,j}^{\mathrm{wrap}}\right],$$
  $$s_t = \mathrm{clip}\!\left(s_{t-1} \exp\!\bigl(\gamma_{\mathrm{ACI}} (\mathrm{err}_m - 0.10)\bigr),\; 0.50,\; 2.50\right).$$
  The wrapped predictive cloud assigns uniform weight $1 / |H_t|$ to the scaled residual reconstructions. Setting $\gamma_{\mathrm{ACI}} = 0$ reduces identically to the unscaled empirical residual wrapper (`Wrap_PriorCF`).

- **DEV_CAL-Locked ACI Parameters:**
  - `Exp 89B` (grid $\gamma_{\mathrm{ACI}} \in \{0.0, 0.008, 0.015, 0.030\}$): `TimesFM-3.0 (CovExog, ctx=1024)` locked `studentized_add`, $\gamma_{\mathrm{ACI}} = 0.0$; `Chronos-2 (CovExog, ctx=1024)` locked `pit_z`, $\gamma_{\mathrm{ACI}} = 0.0$.
  - `Exp 90A` (grid $\gamma_{\mathrm{ACI}} \in \{0.0, 0.005, 0.015, 0.050\}$): `TimesFM-3.0 (CovExog, ctx=1536)` locked `studentized_add`, $\gamma_{\mathrm{ACI}} = 0.0$; `Chronos-2 (CovExog, ctx=1536)` locked `pit_z`, $\gamma_{\mathrm{ACI}} = 0.0$.
  - `Exp 90C` (grid $\gamma_{\mathrm{ACI}} \in \{0.0, 0.005, 0.015, 0.050\}$): `TimesFM-3.0 (TargetOnly, ctx=1024)` locked `pit_z`, $\gamma_{\mathrm{ACI}} = 0.0$; `Chronos-2 (TargetOnly, ctx=2048)` locked `studentized_add`, $\gamma_{\mathrm{ACI}} = 0.0$.

---

## 2. Section 03 Baselines — Plasticity and Recurrence (`Exp 96B`, `Exp 79A`)

### 2.1 Synthetic $K=6$ Nonlinear Quadrant+Radial Recurring-Regime Generator (`Exp 96B`)

`Exp 96B` is a controlled $K = 6$ contextual routing bandit (`route_0, ..., route_5`) over horizon $T = 300$ partitioned into five consecutive 60-step regimes: **`A1` ($t \in [0, 59]$) $\to$ `B1` ($[60, 119]$) $\to$ `C` ($[120, 179]$) $\to$ `A2` ($[180, 239]$) $\to$ `B2_silent` ($[240, 299]$)**.

#### A. Disjoint Seeds & Arm-Index Permutation

- **Calibration seeds (`DEV_CAL`, $n = 5$):** `96901, 96902, 96903, 96904, 96905`.
- **Sealed evaluation seeds (`SEALED_OOS`, $n = 15$):** `96001..96015` (paired ablation subset `96001..96008`, $n = 8$; delayed feedback $D = 4$ stress slice `96001..96006`, $n = 6$).
- Each seed draws a deterministic SHA-256 permutation $\pi$ of $\{0, 1, 2, 3, 4, 5\}$ to eliminate arm-index ordering bias.

#### B. Context Geometry & Nonlinear Zero-Mean Zones

Each step presents a 4D context vector $x_t = (x_{0,t}, x_{1,t}, \text{reg\_load}_t, \text{reg\_budget}_t) \in \mathbb{R}^4$:

1. **Macro-Regime Coordinates $(\text{reg\_load}_t, \text{reg\_budget}_t)$:** Drawn as $\mu_{\mathrm{reg}} + \eta_t$ with $\eta_t \sim \mathrm{Unif}(-0.08, 0.08)^2$, where:
   - `A1`, `A2`, and `B2_silent`: $\mu_{\mathrm{reg}} = (-2.0, -2.0)$ (exact coordinate return in `A2`, and **context-blind silent drift** in `B2_silent`).
   - `B1`: $\mu_{\mathrm{reg}} = (+2.0, -2.0)$.
   - `C`: $\mu_{\mathrm{reg}} = (-2.0, +2.0)$.

2. **Concentric + XOR Nonlinear Zone Coordinates $(x_{0,t}, x_{1,t}) \in [-0.5, 0.5]^2$:** Within each 60-step phase, the first 18 steps deterministically cycle $z_t = (t \bmod 60) \bmod 3$ to guarantee balanced initial zone exposure; the remaining 42 steps draw $z_t \sim \mathrm{Unif}\{0, 1, 2\}$. Conditional on zone $z_t$:
   - **$Z_0$ (Inner Disk):** $r = \sqrt{u}$ with $u \sim \mathrm{Unif}(0, 0.18^2)$, $\theta \sim \mathrm{Unif}(0, 2\pi)$, $(x_0, x_1) = (r\cos\theta, r\sin\theta)$.
   - **$Z_1$ (Outer XOR$+$ Ring, Quadrants 1 & 3):** $r \sim \mathrm{Unif}(0.38, 0.50)$, quadrant $q \sim \mathrm{Unif}\{0, 2\}$, $\theta \sim \mathrm{Unif}(0.28, \frac{\pi}{2} - 0.28) + q\frac{\pi}{2}$.
   - **$Z_2$ (Outer XOR$-$ Ring, Quadrants 2 & 4):** $r \sim \mathrm{Unif}(0.38, 0.50)$, quadrant $q \sim \mathrm{Unif}\{1, 3\}$, $\theta \sim \mathrm{Unif}(0.28, \frac{\pi}{2} - 0.28) + q\frac{\pi}{2}$.

   **Linear Accuracy Ceiling Proof:** By radial and antipodal quadrant symmetry, $\mathbb{E}[x_0 \mid Z_m] = \mathbb{E}[x_1 \mid Z_m] = 0$ for all $m \in \{0, 1, 2\}$. Consequently, for any linear predictor $f_a(x) = w_{a,0} + w_{a,1} x_0 + w_{a,2} x_1 + w_{a,3}\text{reg\_load} + w_{a,4}\text{reg\_budget}$, $\mathbb{E}[f_a(x) \mid Z_m, \mathrm{regime}]$ is invariant to $m$, capping every linear or context-blind bandit at an asymptotic optimal-arm accuracy of at most $1/3 = 33.3\%$.

#### C. Phase-to-Zone Optimal Action Maps & Stochastic Payoffs

Each phase maps zones $(Z_0, Z_1, Z_2)$ to a 3-arm active subset:
- `A1` and `A2` (exact recurrence): $(\pi_0, \pi_1, \pi_2)$.
- `B1`: $(\pi_3, \pi_4, \pi_5)$.
- `C`: $(\pi_2, \pi_5, \pi_0)$.
- `B2_silent` (same context as `A2`, silently flipped optimum): $(\pi_4, \pi_3, \pi_5)$.

Realized rewards are generated as $R_{t,a} = \mathrm{clip}(\mu_{t,a} + \varepsilon_{t,a},\; 0,\; 1)$ with $\varepsilon_{t,a} \sim \mathcal{N}(0, 0.055^2)$ and expected payoffs:
$$\mu_{t,a} = \begin{cases} 0.92, & a = a^*(t) \;\text{(optimal arm for current phase and zone)}, \\ 0.24, & a \in \mathcal{A}_{\mathrm{phase}} \setminus \{a^*(t)\} \;\text{(same-regime wrong-zone arm)}, \\ 0.12, & a \notin \mathcal{A}_{\mathrm{phase}} \;\text{(off-regime arm)}. \end{cases}$$

For delayed-feedback evaluations ($D = 4$), selected triplets $(x_t, a_t, r_t)$ enter a FIFO queue and are delivered to `agent.update()` strictly at step $t + D$.

---

### 2.2 All 8 Online Bandit Peers & Drift Detector (`Exp 96B`)

All 8 offline peers share an identical uniform cold-start (`warmup_per_action = 2`, i.e., 12 initial round-robin pulls) and pass automated exact `0.0` future-perturbation causality verification.

#### A. Pure-NumPy ADWIN Drift Detector (Bifet & Gavaldà, 2007)

Used by both `ADWIN_Online_HGBR` and `ADWIN_Online_Ridge` over the stream of realized rewards $\{r_s\} \subset [0, 1]$ (stored in a bounded FIFO buffer up to `max_window = 240`, `min_window = 8`). On every update (subject to a minimum spacing of `16` steps between resets), all split indices $n_0 \in [\text{min\_window}, n - \text{min\_window}]$ ($n_1 = n - n_0$) are tested with harmonic sample size $m = (1/n_0 + 1/n_1)^{-1}$, adjusted confidence $\delta' = \delta / \max(\ln n, 1)$, and pooled window variance $\hat{\sigma}_W^2$:
$$\varepsilon_{\mathrm{Bern}} = \sqrt{\frac{2}{m} \hat{\sigma}_W^2 \ln\!\frac{2}{\delta'}} + \frac{2}{3m} \ln\!\frac{2}{\delta'}, \qquad \varepsilon_{\mathrm{Hoeff}} = \sqrt{\frac{1}{2m} \ln\!\frac{2}{\delta'}}, \qquad \varepsilon_{\mathrm{cut}} = \min(\varepsilon_{\mathrm{Bern}}, \varepsilon_{\mathrm{Hoeff}}).$$
Whenever $|\bar{r}_0 - \bar{r}_1| \ge \varepsilon_{\mathrm{cut}}$, ADWIN drops the older prefix (choosing the split with the largest normalized gap), increments `n_resets`, and triggers the downstream model to rebuild on the surviving suffix.

#### B. Algorithm Formulations

1. **`Online_NeuralUCB` (Zhou, Li & Gu, 2020 style):**
   - *Architecture:* 2-layer $\tanh$ MLP mapping $\tilde{z}(x, a) = [\mathrm{zstd}(x),\; e_a] \in \mathbb{R}^{d_{\mathrm{ctx}}+K}$ to hidden representation $\phi(x, a) = \tanh(W_1 \tilde{z} + b_1) \in \mathbb{R}^H$ ($H = 48$) and scalar reward $\hat{r}(x, a) = w_2^\top \phi(x, a) + b_2$. Online feature standardization uses causal running mean/std with $\sigma$ floor `0.15` and clipping to $[-3, 3]$. Analytical backpropagation gradients are verified against finite differences ($\max \text{rel. err} < 10^{-6}$).
   - *Training:* On each update, pushes $(x_t, a_t, r_t)$ into a FIFO replay buffer of length `replay_window` and executes `train_steps_per_update = 20` Adam steps (`lr = 0.02`, $\beta_1 = 0.9, \beta_2 = 0.999$, `l2_reg = 1e-4`, global gradient-norm clip `5.0`) on minibatches of size `48` combining the `8` most recent transitions with `40` uniform replay draws.
   - *Discounted Last-Layer UCB:* Maintains discounted Gram matrix $A_t = \gamma A_{t-1} + (1 - \gamma)I_H + \phi_t \phi_t^\top$ ($\gamma = 0.99$) and selects $\arg\max_a \bigl(\hat{r}(x_t, a) + \alpha_{\mathrm{ucb}} \sqrt{\phi(x_t, a)^\top A_t^{-1} \phi(x_t, a)}\bigr)$.

2. **`ADWIN_Online_HGBR`:** Pairs `PureNumpyADWIN` on realized rewards with per-arm `HistGradientBoostingRegressor` (`max_depth=4, max_iter=35, min_samples_leaf=2, l2_regularization=0.1, learning_rate=0.15`), refitting arm $a$'s tree ensemble every `refit_every = 4` pulls once $n_a \ge 6$ (using arm mean for $n_a < 6$). On ADWIN drift, discards the dropped prefix and clears all arm models. Selects:
   $$a_t = \arg\max_a \left(\hat{\mu}_a(x_t) + c_{\mathrm{ucb}} \sqrt{\frac{\ln(n_{\mathrm{since\_reset}} + 2)}{n_a + 1}}\right).$$

3. **`ADWIN_Online_Ridge`:** Pairs `PureNumpyADWIN` with disjoint linear Ridge LinUCB on augmented features $\tilde{x} = [1, x] \in \mathbb{R}^{d_{\mathrm{ctx}}+1}$ ($A_a = \lambda I + \sum \tilde{x}\tilde{x}^\top$, $b_a = \sum r\tilde{x}$, score $\hat{\theta}_a^\top \tilde{x} + \alpha\sqrt{\tilde{x}^\top A_a^{-1} \tilde{x}}$). Upon ADWIN drift, rebuilds $(A_a, b_a)$ exactly from the surviving suffix.

4. **`Discounted_TS` (Exponentially Discounted Linear Thompson Sampling):** Maintains per-arm discounted precision $A_a \leftarrow \gamma A_a + (1 - \gamma)\lambda I + \tilde{x}\tilde{x}^\top$ and $b_a \leftarrow \gamma b_a + r\tilde{x}$ (with decay applied across all arms at each step); draws $\tilde{\theta}_a \sim \mathcal{N}(A_a^{-1} b_a,\; v_{\mathrm{scale}}^2 A_a^{-1})$ and selects $\arg\max_a \tilde{\theta}_a^\top \tilde{x}_t$.

5. **`SlidingWindow_LinUCB` (Cheung, Simchi-Levi & Zhu, 2019):** Recomputes exact disjoint Ridge sufficient statistics $(A_a, b_a)$ strictly from the trailing $W$ global transitions and selects $\arg\max_a (\hat{\theta}_a^\top \tilde{x}_t + \alpha\sqrt{\tilde{x}_t^\top A_a^{-1} \tilde{x}_t})$.

6. **`Stationary_LinUCB`:** Standard non-forgetting disjoint Ridge LinUCB over all history ($\gamma = 1$, infinite window).

7. **`EXP3_S` (Auer et al., 2002):** Context-blind adversarial switching bandit maintaining log-stabilized weights $w_a$:
   $$p_a = (1 - \gamma_{\mathrm{exp}})\frac{w_a}{\sum_k w_k} + \frac{\gamma_{\mathrm{exp}}}{K}, \qquad w'_a = w_a \exp\!\left(\frac{\gamma_{\mathrm{exp}}}{K}\hat{x}_a\right) + \frac{e\,\alpha_{\mathrm{mix}}}{K}\sum_{k=1}^K w_k, \qquad \hat{x}_a = \frac{r_t}{p_a}\mathbb{I}[a = a_t].$$

8. **`EWMA_EpsilonGreedy`:** Context-blind exponentially weighted action-value estimator $Q_{a_t} \leftarrow (1 - \eta) Q_{a_t} + \eta r_t$ with $\epsilon$-greedy uniform exploration.

#### C. `DEV_CAL` Tuning Grids & Locked Hyperparameters (`Exp 96B`)

| Peer | DEV_CAL Grid ($n=5$) | Locked | DEV Regret |
|---|---|---|---:|
| ADWIN Online HGBR | $\delta \in \{0.05, 0.15\} \times c_{\mathrm{ucb}} \in \{0.20, 0.40\}$ (depth=4, iter=35, leaf=2, refit=4) | $\delta=0.15, c_{\mathrm{ucb}}=0.20$ | 130.20 |
| Online NeuralUCB | $\alpha_{\mathrm{ucb}} \in \{0.10, 0.20\} \times W_{\mathrm{rep}} \in \{120, 240, 300\}$ (H=48, lr=0.02, steps=20) | $\alpha_{\mathrm{ucb}}=0.10, W_{\mathrm{rep}}=120$ | 138.21 |
| Sliding-Window LinUCB | $W \in \{24, 48, 80\} \times \alpha \in \{0.25, 0.75\}$ ($\lambda=1.0$) | $W=80, \alpha=0.25, \lambda=1.0$ | 146.10 |
| EWMA Epsilon-Greedy | $\eta \in \{0.15, 0.30\} \times \epsilon \in \{0.05, 0.12\}$ | $\eta=0.30, \epsilon=0.05$ | 146.67 |
| Discounted TS | $\gamma \in \{0.88, 0.94, 0.98\} \times v_{\mathrm{scale}} \in \{0.15, 0.30\}$ ($\lambda=1.0$) | $\gamma=0.94, v_{\mathrm{scale}}=0.15, \lambda=1.0$ | 147.26 |
| Stationary LinUCB | $\alpha \in \{0.25, 0.50, 1.0\}$ ($\lambda=1.0$) | $\alpha=1.0, \lambda=1.0$ | 149.05 |
| ADWIN Online Ridge | $\delta \in \{0.05, 0.15\} \times \alpha \in \{0.25, 0.75\}$ ($\lambda=1.0$) | $\delta=0.15, \alpha=0.75, \lambda=1.0$ | 149.13 |
| EXP3.S | $\gamma_{\mathrm{exp}} \in \{0.05, 0.12\} \times \alpha_{\mathrm{mix}} \in \{0.01, 0.04\}$ | $\gamma_{\mathrm{exp}}=0.12, \alpha_{\mathrm{mix}}=0.04$ | 181.30 |

---

### 2.3 `Exp 79A` SPY/TLT 2022 Correlation-Flip Comparators & Hard 21-Day Flush Ablation

`Exp 79A` evaluates daily total-return forecasting and a frozen simulated `5 bps` risk-parity tail-budget overlay over 4 macro ETFs (`SPY, TLT, HYG, GLD`): pre-evaluation calibration `2020-01-02..2021-09-30` ($n = 441$ trading days) and sealed evaluation `2021-10-01..2023-06-30` ($n = 439$ trading days), spanning the 60-day stock-bond correlation flip from `-0.4233` to `+0.3154`.

#### A. Frozen Risk-Parity Tail-Budget Overlay Parameters (Identical Across All Forecasting Models)

Calibrated strictly on the pre-evaluation window: 99% CVaR target budget $b_{\mathrm{CVaR99}} = 2.28664\%$, gross exposure bounds $[0.25, 1.00]$, correlation tilt cap `0.50`, calibration stress base rate `0.174603`, maximum stress throttle `0.35`, zero daily cash yield, and `5 bps` one-way transaction fee (`0 bps` and `15 bps` also reported for sensitivity).

#### B. Comparator Roster

- **Static Allocation Benchmarks:** `Static_60_40` (60% SPY / 40% TLT, daily rebalanced) and `Static_60_20_10_10` (60% SPY / 20% TLT / 10% HYG / 10% GLD).
- **`Q_HGBR_Static_PreFlip`:** 9-quantile `HistGradientBoostingRegressor` ($\mathcal{Q}_{9,\mathrm{tail}} = \{0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975\}$, `max_depth=2, max_iter=50, min_samples_leaf=10, l2_reg=100.0`) fit **once** on pre-flip history and frozen across the entire evaluation window, paired with a frozen pre-flip PIT-Z Gaussian copula.
- **`Q_HGBR_Online_Roll`:** Identical 9-quantile HGBR architecture refit every `21` trading days on trailing `W = 512` days with a causal rolling `256`-day PIT-Z Gaussian copula ($M = 256$).
- **`Chronos2_TO_GaussCop` & `TimesFM3_TO_GaussCop`:** Target-only zero-shot foundation models converted via `Q_GRID_21` and the causal `256`-day PIT-Z Gaussian copula (Section 1.3).
- **`DCC_GARCH_FHS`:** Fixed-parameter causal GARCH(1,1) ($\alpha = 0.08, \beta = 0.88$) + Engle DCC(1,1) ($a = 0.05, b = 0.92$, EWMA demean $\lambda_\mu = 0.94$) + eigendecomposition-whitened FHS (`W = 256`).
- **`EWMA_EmpResid`:** Vol-standardized RiskMetrics EWMA FHS (`W = 512`).
- **`Online_Ridge` & `Own_PriorCF`:** Rolling standardized Ridge with empirical residuals and uniform trailing empirical window.

#### C. Smooth Retention vs. Hard 21-Day Drift-Gated Flush Ablation

To isolate whether hard forgetting helps or hurts under gradual correlation drift, `Exp 79A` pairs two identical Adapt-1 configurations (`max_samples = 128`, `bandwidth = 0.35`, identical `volstd` scaling and `5 bps` overlay):

1. **`Adapt1_Smooth_Retention_W128` (`rotation_mode="none"`):** Retains the continuous trailing `W = 128` observation buffer with smooth kernel reweighting as correlations shift.
2. **`Adapt1_Hard_21d_Flush_DriftReset` (`rotation_mode="drift_gated"`):** Enables live drift detection (`drift_detection_enabled = true`, `drift_patience = 3`). After each step $k$ is observed, if `steps_since_rot >= cooldown_steps` (`21` trading days) and either the autonomous drift counter increments (`cur_drift_det > base_drift_det`) or a superseded change-point retirement triggers (`superseded_cp and cur_retired > base_retired`), the current domain is closed and replaced by a fresh domain re-seeded strictly with the trailing `reseed_k_post_drift = 21` matured observations. Any rare immediate post-flush cold-start abstention falls back causally to uniform $1/K$ weighting over the active domain's buffer. Across the 439 evaluation days, this triggered **8 rotations** (after steps `20, 41, 62, 101, 122, 150, 171, 422`), degrading 5 bps return (`-5.95%` vs. `-2.25%`), max drawdown (`-22.55%` vs. `-19.61%`), and 99% VaR breaches (`8` vs. `5`) because discarding 107 days of tail history starves the empirical tail quantile estimator.

---

## 3. Section 04 Baselines — Conditioning Across Time (`Exp 16`, NYC 311, Chicago & SF 311 `Exp 53`)

### 3.1 `Exp 16` — 19-Year Weekly 5-Asset Global Macro Comparators (2008–2026)

`Exp 16` evaluates weekly non-overlapping 5-trading-day joint forecasts over $m = 5$ volatility-standardized global macro returns (`z_eq, z_y10, z_vix, z_oil, z_usd`, total 930 weeks `2008-04-02..2026-09-18`, primary evaluation window `idx >= 40`, $n = 890$ weeks `2009-01-15..2026-09-18`).

#### A. Two-Step MLE Engle (2002) `DCC-GARCH(1,1)` (`Gauss` and `FHS`)

- **Step 1 — Univariate GARCH(1,1) QML:** On each asset $j$, mean $\hat{\mu}_j$ is the training sample mean, $e_{j,s} = y_{j,s} - \hat{\mu}_j$, $v_{j,0} = \max(\mathrm{Var}(e_j), 10^{-4})$, and variance recursion $h_{j,s} = \omega_j + \alpha_j e_{j,s-1}^2 + \beta_j h_{j,s-1}$ minimizes $\frac{1}{2}\sum_s (\log h_{j,s} + e_{j,s}^2 / h_{j,s})$ subject to $\omega_j > 10^{-7}$, $\alpha_j \ge 10^{-4}$, $\beta_j \ge 0$, $\alpha_j + \beta_j < 0.995$. Optimization uses 3-start `Nelder-Mead` (`maxiter=300, xatol=1e-4, fatol=1e-4`) from `[(0.06*v0, 0.10, 0.84), (0.03*v0, 0.06, 0.90), (0.15*v0, 0.15, 0.70)]`.
- **Step 2 — Engle DCC(1,1) QML:** Standardized innovations $u_{s,j} = e_{s,j} / \sqrt{\max(h_{s,j}, 10^{-6})}$ form $\bar{Q} = \frac{1}{T_{\mathrm{tr}}} U^\top U + 10^{-5} I_5$ (normalized to unit diagonal in the FRTB tail audit). The DCC recursion $Q_s = (1 - a - b)\bar{Q} + a u_{s-1} u_{s-1}^\top + b Q_{s-1}$ ($a \ge 0, b \ge 0, a + b < 0.990$) is minimized via an 8-point coarse grid search `{(0.01,0.95), (0.03,0.92), (0.05,0.90), (0.08,0.85), (0.12,0.80), (0.02,0.80), (0.005,0.90), (0,0)}` followed by `Nelder-Mead` polish (`maxiter=120, xatol=1e-3, fatol=1e-3`).
- **SPD Factorization & Causal Whitening:** Predictive covariance $\Sigma_t = D_t R_t D_t$ is factored via Cholesky with an escalating jitter ladder $\{10^{-6}, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$ and symmetric eigendecomposition fallback ($\lambda_{\min} = 10^{-4}$). Strictly after scoring week $t$, the realized innovation is whitened by $z_t = L_t^{-1}(y_t - \hat{\mu})$ and $L_2$-norm clipped to $\|z_t\|_2 \le 8.0$.
- **Sampling & Two Evaluation Protocols:**
  - *5D Energy Score Suite (`start_idx=40`):* Rolling window `window = 512` weeks, refit every `26` weeks. `DCC-GARCH(1,1)-Gauss` uses $M = 512$ fixed antithetic normal draws $[Z_{256}; -Z_{256}]$ (`seed = 2026`); `DCC-GARCH(1,1)-FHS` projects the trailing `512` causally whitened innovations $z_s$ with uniform weights.
  - *Full-Distribution Basel III / FRTB Tail Audit (`start_idx=40`):* Expanding training window $Y[:t]$, refit every `13` weeks, trailing `256` whitened FHS innovations. For `DCC-GARCH(1,1)-Gauss`, marginal quantiles, continuous PIT, and left/right Expected Shortfall (`ES_05`, `ES_01`) use exact closed-form Gaussian expressions ($\mu_j - \sigma_j \phi(\Phi^{-1}(\alpha)) / \alpha$); for `DCC-GARCH(1,1)-FHS`, empirical Expected Shortfall uses the exact Acerbi-Tasche boundary-atom-split estimator.

#### B. `RiskMetrics EWMA` (`Gauss` and `FHS`)

Evaluated at weekly decays $\lambda \in \{0.75, 0.88\}$ ($\lambda = 0.75 \approx 0.94^5$ daily-to-weekly equivalent; $\lambda = 0.88$ is the primary FRTB comparator). Strictly after scoring week $t$, updates $\hat{\mu}_t = \lambda \hat{\mu}_{t-1} + (1 - \lambda) y_t$ (or expanding mean in the FRTB tail audit) and $\Sigma_t = \lambda \Sigma_{t-1} + (1 - \lambda) e_t e_t^\top$, appending the causally whitened, $\|z_t\|_2 \le 8.0$ clipped innovation $z_t = L_t^{-1} e_t$ to the FHS bank (`512` draws for ES; `256` draws for FRTB tails). Gaussian clouds use the same $512$ antithetic normal draws (and exact closed-form Gaussian tail moments in the FRTB audit).

#### C. Frozen-After-Setup and One-Pass Streaming Macro Comparators (`POST_SETUP_200`, $n = 730$, & `FULL_19Y`, $n = 890$)

To benchmark unattended deployment without periodic GARCH refitting, four additional comparators are evaluated on the paired `POST_SETUP_200` ($n = 730$) and `FULL_19Y` ($n = 890$) windows:
- **Frozen-After-Setup (fit once on initial 200 setup weeks, 0 refits):** `Frozen_Gauss_Cov_19Y` (static 5D sample mean and covariance) and `Frozen_HGBR_Resid_5D` (frozen per-dimension HGBR mean predictors plus frozen out-of-fold empirical residual bank).
- **One-Pass Streaming ($O(1)/O(d^2)$ updates, 0 batch refits):** `Online_RLS_FF_5D` (Sherman-Morrison recursive least squares with exponential forgetting factor plus recursive EWMA innovation covariance) and `Online_EWMA_Empirical_Conformal` (recursive EWMA location/scale plus rolling empirical conformal residual quantiles).

---

### 3.2 Municipal 311 Progressive Survival Comparators (`Exp 53` Chicago & SF 311; NYC Lower Manhattan 311)

Municipal 311 benchmarks predict joint mixed targets $y_i = (c_i, d_i)$ — discrete ticket disposition $c_i \in \{0,\dots,C-1\}$ (including a designated 7-day SLA `overrun` class $c_{\mathrm{ov}}$) and scaled log-duration $d_i = \log(1 + \Delta t_i) / s_{\mathrm{dur}} \in [0, y_{\max}]$ — across three progressive revision checkpoints per ticket:
- **NYC Lower Manhattan 311** (`0h / +1h / +6h`, SLA `168h`, $n = 4,330$ predicted checkpoints; sealed 2026 OOS $n = 3,873$, identical 3-stage cohort $n = 826$).
- **Chicago 311 (`Exp 53`)** (`0h / +6h / +48h`, SLA `168h`, $n = 2,621$ live post-2026-09 checkpoints: `1,106 / 855 / 660`).
- **San Francisco 311 (`Exp 53`)** (`0h / +1h / +6h`, SLA `168h`, $n = 4,375$ live post-2026-09 checkpoints: `1,722 / 1,545 / 1,108`).

#### A. Asynchronous Causal History & Survival-Truncation Risk-Set Rule

At ticket $i$'s revision stage $r \in \{0, 1, 2\}$ queried at timestamp $t_{\mathrm{query}}(i, r)$:
1. **Strict Resolution Cutoff:** Candidate history is restricted to tickets already closed before the query:
   $$\mathcal{H}_{i,r}^{(0)} = \left\{j : t_{\mathrm{resolve}}(j) < t_{\mathrm{query}}(i, r)\right\},$$
   ordered chronologically by `resolve_hr` and truncated to the most recent $W$ resolved tickets ($W = 1024$ in `Exp 53`; $W = 512$ in NYC 311).
2. **Hard Survival-Truncation Conditioning (Stages `rev_1` and `rev_2`):** Because querying at checkpoint $\tau_r > 0$ reveals that ticket $i$ has survived open past elapsed time $\tau_r$, all classical survival comparators (`OLHR`, `Refit HGBR`, `Refit QRF`, and `Survival_PriorCF`) hard-filter the training history to the exact surviving risk set:
   $$\mathcal{H}_{i,r} = \left\{j \in \mathcal{H}_{i,r}^{(0)} : \Delta t_j > \tau_r \right\},$$
   and enforce the physical lower duration bound $d_{\min, r} = \log(1 + \tau_r) / s_{\mathrm{dur}}$ on all predicted duration atoms (skipping only if $|\mathcal{H}_{i,r}| < 20$). In contrast, **`Own_PriorCF`** uses the unfiltered $\mathcal{H}_{i,r}^{(0)}$ at all stages to quantify pure unconditioned prior skill.

#### B. Coherent Mixed Categorical + Duration Atom Sampling ($M = 256$)

To convert parametric classifier probabilities $\hat{p}_{\mathrm{cat}} \in \Delta^{C-1}$ and regression outputs into valid joint mixed clouds scored by the exact mixed Energy Score, `OLHR`, `HGBR`, and the `QRF` fallback use `sample_coherent_mixed_atoms` ($M = 256$, deterministic seed):
1. Allocate discrete atom counts $n_c$ proportionally to $\hat{p}_{\mathrm{cat}}$ via deterministic rounding with remainder assigned to largest fractional parts ($\sum_c n_c = 256$).
2. For the SLA overrun class $c = c_{\mathrm{ov}}$, all $n_{c_{\mathrm{ov}}}$ atoms are pinned exactly to $d = y_{\max}$ (preserving logical coherence between disposition and 7-day SLA breach).
3. For each non-overrun class $c \ne c_{\mathrm{ov}}$, durations are sampled inside $[d_{\min,r}, y_{\max} - 10^{-4}]$ from class-conditional historical durations shifted by $0.5(\hat{\mu}_d - \bar{d}_{\mathrm{hist}})$ whenever class support in the risk set is $\ge 4$, and otherwise from clipped regression residual draws $\hat{\mu}_d + r_k$ (or clipped tree predictions in the QRF fallback). Marginal disposition Brier/log-loss and duration CRPS are additionally evaluated directly from $\hat{p}_{\mathrm{cat}}$ and the exact residual marginal cloud.

#### C. Periodic-Refit Classical & Event-Time TSFM Comparators

1. **`OLHR` (Online Linear Hazard Regression / `Online_Logit_HeteroRidge`):**
   - *Disposition Model:* $L_2$-regularized multinomial logistic regression on standardized features (unpenalized intercept, regularization $\lambda$, `scipy.optimize.minimize(..., method="L-BFGS-B", options={"maxiter": 80})`), refit every **12** steps (`step_seen % 12 == 0`).
   - *Duration Model:* Exact closed-form 1D standardized Ridge regression (same penalty $\lambda$), refit at **every** step (`1`-step cadence).
   - *Group-Heteroskedastic Residual Pool:* Filters the Ridge residuals to historical tickets matching the query's agency/category group `grp_q` whenever group support is $\ge 8$, falling back to the full risk-set residual pool otherwise. (NYC 311 additionally reports homoskedastic **`Online_Logit_Ridge`** using global residuals only.)

2. **`Refit HGBR` (`HGBR_Clf_Reg`):**
   - Coupled `HistGradientBoostingClassifier` + `HistGradientBoostingRegressor` (`learning_rate = 0.06`, `random_state = 53 + r` in `Exp 53`, `26 + r` in NYC 311), refit every **20** steps in `Exp 53` (`step_seen % 20 == 0`) and every **24** steps in NYC 311. Unseen classes in the local window receive probability floor $10^{-6}$ before renormalization; marginal duration cloud is $\mathrm{clip}(\hat{\mu}_d + r_{\mathrm{train}}, d_{\min,r}, y_{\max})$.

3. **`Refit QRF` (Exact Meinshausen 2006 Quantile Random Forest — `Exp 53`):**
   - Coupled `RandomForestClassifier` + `RandomForestRegressor` (`n_estimators = 64`, `random_state = 530 + r`), refit every **20** steps (`step_seen % 20 == 0`).
   - *Exact Leaf-Empirical Conditional Distribution:* Using terminal leaf assignments $L_b(x)$ across all $B = 64$ regression trees (`rf_reg.apply`), each historical risk-set ticket $j \in \mathcal{H}_{i,r}$ receives the exact Meinshausen leaf weight:
     $$w_j(x_q) = \frac{1}{B} \sum_{b=1}^B \frac{\mathbb{I}[L_b(x_j) = L_b(x_q)]}{\max\!\left(\sum_{k \in \mathcal{H}_{i,r}} \mathbb{I}[L_b(x_k) = L_b(x_q)],\; 1\right)}.$$
     Positive-weight training pairs $(c_j, d_j)$ form the joint empirical atom cloud directly (with overrun durations pinned to $y_{\max}$ and non-overrun durations clipped to $[d_{\min,r}, y_{\max} - 10^{-4}]$). If $\sum_j w_j \le 10^{-12}$, falls back to individual tree predictions paired with `sample_coherent_mixed_atoms`.

4. **Event-Time Foundation Models (`TimesFM3_CovExog_Indep`, `Chronos2_CovExog_Indep` — NYC 311 `rev_0`):**
   - Evaluated strictly causally at ticket arrival (`rev_0`) by encoding the 4-class one-hot disposition vector plus scaled log-duration as a 5-channel event-time target series alongside intake covariates.

5. **Unconditioned Priors:** `Own_PriorCF` (uniform empirical distribution over unfiltered trailing resolved tickets $\mathcal{H}_{i,r}^{(0)}$) and `Survival_PriorCF` (uniform empirical distribution over the checkpoint-filtered survival risk set $\mathcal{H}_{i,r}$; identical to `Own_PriorCF` at `rev_0`).

#### D. Frozen-After-Setup and One-Pass Streaming 311 Comparators (`Exp 53`)

To separate the contribution of periodic batch retraining from online single-pass adaptation, `Exp 53` also evaluates:
- **Frozen-After-Setup (0 refits):** `Frozen_QRF_Mixed`, `Frozen_HGBR_Mixed`, and `Frozen_Logit_Ridge_Mixed`, trained **once** strictly on pre-2026 resolved history (`3,814` tickets in Chicago; `4,187` tickets in San Francisco) and frozen across 2026 evaluation (exhibiting `+17%` to `+26%` Energy Score drift vs. refit peers in San Francisco).
- **One-Pass Streaming ($O(1)/O(d^2)$ per ticket, 0 batch refits):** `Online_FTRL_Logit_RLS_Mixed` (FTRL-Proximal multinomial log-loss classifier coupled with Sherman-Morrison recursive least squares duration regression) and `Online_SGD_Mixed_Survival` (single-pass streaming stochastic gradient mixed-survival classifier/regressor).

#### E. Locked Hyperparameters & Refit Cadences Across Municipal 311 Benchmarks

| Benchmark (W, Stages) | OLHR | Refit HGBR | Refit QRF | Frozen / 1-Pass Peers |
|---|---|---|---|---|
| **NYC Lower Manhattan** ($W=512$, 0h/+1h/+6h) | $\lambda=1.0$; logit 12-step, Ridge 1-step | d=3, iter=50, lr=0.06; refit 24 | — | TimesFM-3.0 & Chronos-2 (`CovExog_Indep`, `rev_0`) |
| **Chicago 311** (Exp 53; $W=1024$, 0h/+6h/+48h) | $\lambda=50.0$; logit 12-step, Ridge 1-step | d=3, iter=50, l2=10.0, lr=0.06; refit 20 | trees=64, depth=None, leaf=8; refit 20 | Frozen (3,814 pre-2026); FTRL+RLS, `SGD_Survival` (1-pass) |
| **San Francisco 311** (Exp 53; $W=1024$, 0h/+1h/+6h) | $\lambda=5.0$; logit 12-step, Ridge 1-step | d=2, iter=50, l2=1.0, lr=0.06; refit 20 | trees=64, depth=6, leaf=5; refit 20 | Frozen (4,187 pre-2026); FTRL+RLS, `SGD_Survival` (1-pass) |

> **Operating-Boundary Note (`Exp 54` Mozilla Bugzilla & `Exp 59` SF Fire/EMS):** On hard-survival operational queues with high-cardinality categorical regimes, risk-set-filtered `QRF`, `HGBR`, and `OLHR` outperform single-domain similarity conditioning when the survival cutoff partitions closed-before-checkpoint items sharply (`Exp 54`, `rev_1/rev_2`) or when cross-station interactions dominate early dispatch (`Exp 59`, `rev_0/rev_1`). All operational comparators require strictly append-only event logs with retroactive snapshot fields (such as final priority or final owner) stripped before feature construction.
