# Methods, Scoring Rules, Statistical Tests, and Causal Decision Overlays

> **Technical Reference for Part 2 (*Forecasting Through Drift*) of `forecasting-series`**
>
> This specification defines every proper scoring rule, paired inference procedure, regulatory tail-risk backtest, and stylized causal decision overlay used to generate and audit the committed out-of-sample traces across Sections 01–05. All equations, constants, tolerances, and edge-case guards match the production evaluation implementations identically.

---

## Global Notation & Evaluation Invariants

Every forecast issued at decision step $t$ is represented either as:

- A **discrete weighted predictive cloud** $F_t = \{(p_{t,i}, z_{t,i})\}_{i=1}^{M_t}$ with support points $z_{t,i} \in \mathbb{R}^d$ (or mixed space $\mathcal{C} \times \mathbb{R}$) and weights $p_{t,i} \ge 0$, or
- A **binary event probability** $\hat{p}_t \in [0, 1]$ for an indicator outcome $y_t \in \{0, 1\}$.

All evaluation harnesses enforce four non-negotiable invariants before scoring:

1. **Strict Probability-Mass Validation (Zero Silent Renormalization):** Every discrete predictive cloud is validated against
   $$p_{t,i} \ge -10^{-12}, \qquad \forall i \in \{1,\dots,M_t\}, \qquad \left|1 - \sum_{i=1}^{M_t} p_{t,i}\right| \le 10^{-5},$$
   with all $p_{t,i}$ and $z_{t,i}$ strictly finite (`math.isfinite`). Clouds failing mass conservation raise an assertion error rather than silently rescaling truncated probability mass. After validation, floating-point drift within $10^{-5}$ is normalized via $\tilde{p}_i = \max(p_i, 0) / \sum_j \max(p_j, 0)$.
2. **Strict Causal Score-Before-Ingest Ordering:** At step $t$, forecast $F_t$ (or $\hat{p}_t$) depends strictly on observations matured before decision time $t$. Realized outcomes $y_t$ are scored against $F_t$ **before** ingestion into model state. For asynchronous survival streams (Section 04), history at query time $t_q$ contains strictly $\{j : t_{\text{resolve},j} < t_q\}$.
3. **Unrounded `float64` Inference:** All paired hypothesis tests, calibration tests, and step-down Holm–Bonferroni adjustments operate on raw unrounded IEEE-754 `float64` arrays. Asymptotic underflow is formatted as `<1e-300` rather than literal `0.00e+00`.
4. **Deterministic Seeding:** All Monte Carlo permutations and bootstrap procedures derive 32-bit seeds deterministically from `SHA-256` hashes of contrast metadata rather than process-salted RNG state.

---

## 1. Strictly Proper Scoring Rules

### 1.1 Continuous Multivariate Energy Score ($\mathrm{ES}$)

For a weighted empirical cloud $F_t = \{(p_i, z_i)\}_{i=1}^{M}$ in $\mathbb{R}^d$ and realized vector $y_t \in \mathbb{R}^d$, the multivariate Energy Score (Gneiting & Raftery, 2007) is:

$$\mathrm{ES}(F_t, y_t) = \mathbb{E}_{Z \sim F_t}\!\left[\|Z - y_t\|_2\right] - \frac{1}{2}\,\mathbb{E}_{Z, Z' \stackrel{\text{i.i.d.}}{\sim} F_t}\!\left[\|Z - Z'\|_2\right]$$

Evaluated directly on the weighted branches:

$$\mathrm{ES}(F_t, y_t) = \sum_{i=1}^{M} p_i \|z_i - y_t\|_2 - \frac{1}{2}\sum_{i=1}^{M}\sum_{j=1}^{M} p_i p_j \|z_i - z_j\|_2 = \mathbf{p}^\top \mathbf{d}_1 - \frac{1}{2}\,\mathbf{p}^\top D\,\mathbf{p}$$

where $(\mathbf{d}_1)_i = \|z_i - y_t\|_2$ and $D_{ij} = \|z_i - z_j\|_2$ is the $M \times M$ pairwise Euclidean distance matrix (for $M = 1$, $\mathrm{ES}(F_t, y_t) = \|z_1 - y_t\|_2$).

- **Orientation & Propriety:** Lower is better. $\mathrm{ES}$ is strictly proper on the class of Borel probability measures with finite first moment and simultaneously rewards marginal sharpness, bias accuracy, and joint cross-asset dependence structure.
- **Complexity:** $\mathcal{O}(M^2 d)$ time and $\mathcal{O}(M^2)$ memory per forecast step (computed via `scipy.spatial.distance.cdist` or `squareform(pdist)` to avoid catastrophic cancellation in Gram-matrix expansions $\|z_i\|^2 + \|z_j\|^2 - 2 z_i^\top z_j$). For lossless joint clouds ($M \le 128$, Sections 01 & 03) or full archive clouds ($M \le 1024$, Section 04), evaluation is exact with zero Monte Carlo approximation.

### 1.2 Univariate CRPS, Weighted Tail Quantiles, Expected Shortfall, and $\mathrm{FZ0}$

When $d = 1$, the Energy Score specializes to the Continuous Ranked Probability Score ($\mathrm{CRPS}$). Sorting the scalar support points $v_{(1)} \le v_{(2)} \le \dots \le v_{(M)}$ with associated weights $p_{(1)}, \dots, p_{(M)}$ and cumulative distribution $C_k = \sum_{j=1}^k p_{(j)}$ reduces the quadratic pairwise sum to an exact $\mathcal{O}(M \log M)$ identity:

$$\mathrm{CRPS}(F_t, y_t) = \sum_{k=1}^{M} p_{(k)} \left|v_{(k)} - y_t\right| - \sum_{k=1}^{M-1} C_k (1 - C_k)\left(v_{(k+1)} - v_{(k)}\right)$$

From the same sorted empirical CDF, exact predictive quantiles and tail Expected Shortfalls are computed without parametric smoothing:

- **Left-tail quantile ($q_\alpha$) and Expected Shortfall ($\widehat{\mathrm{ES}}_\alpha$, $\alpha \in (0,1)$):** Let $k^* = \min\{k \in \{1,\dots,M\} : C_k \ge \alpha\}$ (capped at $M$), with $C_0 = 0$. Then:
  $$q_\alpha = v_{(k^*)}, \qquad \widehat{\mathrm{ES}}_\alpha = \min\!\left(q_\alpha,\; \frac{1}{\alpha}\left[\sum_{k=1}^{k^*-1} p_{(k)} v_{(k)} + \max\!\left(0,\, \alpha - C_{k^*-1}\right) v_{(k^*)}\right]\right)$$
- **Right-tail quantile ($q_{1-\alpha}$) and upper Expected Shortfall ($\widehat{\mathrm{ES}}^+_{1-\alpha}$):** Evaluated via exact reflection on $-v$:
  $$q_{1-\alpha}(v, p) = -q_\alpha(-v, p), \qquad \widehat{\mathrm{ES}}^+_{1-\alpha}(v, p) = -\widehat{\mathrm{ES}}_\alpha(-v, p)$$
- **Pinball (Quantile) Loss:** For quantile level $\tau \in (0, 1)$,
  $$\rho_\tau(y, \hat{q}) = \max\!\left(\tau(y - \hat{q}),\; (\tau - 1)(y - \hat{q})\right)$$
- **Zero-Homogeneous Joint $(\mathrm{VaR}_\alpha, \mathrm{ES}_\alpha)$ Loss ($\mathrm{FZ0}$, `Exp 79A`):** Following Patton, Ziegel & Chen (2019), for left-tail forecast $(v, e) = (q_\alpha, \widehat{\mathrm{ES}}_\alpha)$ with negative-domain guards $e_s = \min(e, -0.05)$ and $v_s = \min(v, e_s + 10.0)$:
  $$\mathrm{FZ0}_\alpha(y; v, e) = -\frac{1}{\alpha e_s}\mathbb{I}[y \le v_s](v_s - y) + \frac{v_s}{e_s} + \ln(-e_s) - 1$$

### 1.3 Mixed Continuous + Categorical Energy Score (`Exp 53` & NYC 311)

Municipal 311 survival forecasts jointly predict a discrete terminal disposition $y_c \in \{0, \dots, K-1\}$ ($K = 4$ in NYC & SF; $K = 3$ in Chicago) and a capped continuous resolution duration $D_{\text{hr}} \in [0, H_{\text{SLA}}]$ ($H_{\text{SLA}} = 168\text{ h}$ for 7-day 311 SLAs; $720\text{ h}$ for `Exp 54`; $60\text{ min}$ for `Exp 59`).

1. **Continuous Target Scaling:** Resolution duration is log-compressed so its empirical standard deviation matches the one-hot categorical Euclidean scale approximately $1:1$:
   $$y_{\text{dur}} = \frac{\ln\!\left(1 + \min(D_{\text{hr}}, 168.0)\right)}{2.0} \in \left[0,\; \frac{\ln 169}{2}\right] \approx [0,\, 2.5649]$$
   (In `Exp 59` EMS dispatch, duration in minutes is normalized onto $[0,1]$ via $\ln(1 + \min(D_{\text{min}}, 180.0)) / \ln(181.0)$.)
2. **Joint Mixed Embedding:** Each weighted scenario $(c_i, v_i, p_i)$ is embedded into $\mathbb{R}^{K+1}$ by concatenating the $K$-dimensional one-hot disposition vector $\mathbf{e}(c_i) \in \{0,1\}^K$ with the scalar scaled duration $v_i$:
   $$\tilde{z}_i = \begin{bmatrix}\mathbf{e}(c_i) \\ v_i\end{bmatrix} \in \mathbb{R}^{K+1}, \qquad \tilde{y}_t = \begin{bmatrix}\mathbf{e}(y_c) \\ y_{\text{dur}}\end{bmatrix} \in \mathbb{R}^{K+1}$$
   The **Mixed Energy Score** is the exact Euclidean multivariate Energy Score on these embedded vectors:
   $$\mathrm{ES}_{\text{mixed}}(F_t, \tilde{y}_t) = \sum_{i=1}^M p_i \|\tilde{z}_i - \tilde{y}_t\|_2 - \frac{1}{2}\sum_{i=1}^M\sum_{j=1}^M p_i p_j \|\tilde{z}_i - \tilde{z}_j\|_2$$
3. **Component & Operational Triage Diagnostics:** With marginal disposition probabilities $p_c = \sum_{i: c_i = c} p_i$ and overrun index $c_{\text{ov}}$ (`SLA_Overrun_gt7d`):
   - **Multiclass Disposition Brier & Log-Loss:**
     $$\mathrm{Brier}_{\text{disp}} = \sum_{c=0}^{K-1}\left(p_c - \mathbb{I}[y_c = c]\right)^2, \qquad \mathrm{LogLoss}_{\text{disp}} = -\ln\!\left(\max(p_{y_c}, 10^{-6})\right)$$
   - **7-Day Overrun Brier & Marginal Duration $\mathrm{CRPS}$ / $\mathrm{Cov}_{90}$:**
     $$\mathrm{Brier}_{\text{ov}} = \left(p_{c_{\text{ov}}} - \mathbb{I}[y_c = c_{\text{ov}}]\right)^2, \qquad \mathrm{Cov}_{90} = \mathbb{I}\!\left[q_{0.05} - 10^{-9} \le y_{\text{dur}} \le q_{0.95} + 10^{-9}\right]$$
   - **Asymmetric SLA Overrun Triage Cost (`ratio_r`, $r \in \{3, 5, 8\}$):** Escalating when $p_{c_{\text{ov}}} \ge 1/r$ (Bayes-optimal decision threshold under $1:r$ false-alarm vs. missed-overrun cost):
     $$C_r = \mathbb{I}\!\left[p_{c_{\text{ov}}} \ge \frac{1}{r}\right] + r\,\mathbb{I}\!\left[p_{c_{\text{ov}}} < \frac{1}{r}\right]\mathbb{I}\!\left[y_c = c_{\text{ov}}\right]$$
   - **3-Action × 4-State Joint Cost Matrix (`triage_cost_mat`):** Choosing $a^* = \arg\min_{a \in \{0,1,2\}} (\mathbf{M}\,\mathbf{p}_{\text{state}})_a$ where $\mathbf{M} = \begin{bmatrix}0 & 0 & 3.5 & 10.0\\ 1.8 & 1.8 & 2.3 & 6.3\\ 4.2 & 5.7 & 4.2 & 5.2\end{bmatrix}$ over states `Fast_or_Admin`, `No_Violation_Gone`, `Delayed_Action_gt24h`, and `SLA_Overrun_gt7d`.

### 1.4 Binary Probabilistic Scores (`Exp 102C`, `Exp 103A`, `Exp 103B`)

Given $n$ binary outcomes $y_i \in \{0,1\}$ and raw probability predictions $p_i$, `prob_metrics` (`exp102_common.py`, imported identically by all `Exp 103` arenas) clips probabilities to $\hat{p}_i = \mathrm{clip}(p_i, \varepsilon, 1-\varepsilon)$ with **$\varepsilon = 10^{-6}$** prior to computing every binary score:

- **Brier Score:**
  $$\mathrm{Brier} = \frac{1}{n}\sum_{i=1}^{n}\left(\hat{p}_i - y_i\right)^2$$
- **Binary Log-Loss (Cross-Entropy):**
  $$\mathrm{LogLoss} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \ln(\hat{p}_i) + (1 - y_i)\ln(1 - \hat{p}_i)\right]$$
- **Equal-Width Expected Calibration Error ($\mathrm{ECE}_{10}$, headline ECE):** Partition $[0, 1]$ into $B = 10$ uniform intervals $I_b = [\frac{b-1}{10}, \frac{b}{10})$ for $b = 1,\dots,9$ and $I_{10} = [0.9, 1.0]$ (closed right endpoint on the final bin). Let $S_b = \{i : \hat{p}_i \in I_b\}$ and $n_b = |S_b|$:
  $$\mathrm{ECE}_{10} = \sum_{b=1}^{10} \frac{n_b}{n}\left|\frac{1}{n_b}\sum_{i \in S_b} y_i - \frac{1}{n_b}\sum_{i \in S_b} \hat{p}_i\right|$$
- **Equal-Mass (Quantile-Binned) Calibration Error ($\mathrm{ECE}_{q10}$, companion rare-tail diagnostic):** Sort observations by $\hat{p}_i$ (stable mergesort), partition into $10$ equal-count slices $Q_1, \dots, Q_{10}$ via `np.array_split`, and evaluate $\sum_{b=1}^{10} \frac{|Q_b|}{n}|\bar{y}_{Q_b} - \bar{\hat{p}}_{Q_b}|$.
- **Calibration Ratio ($\mathrm{CalR}$):**
  $$\mathrm{CalR} = \frac{\bar{\hat{p}}}{\bar{y}} = \frac{\frac{1}{n}\sum_{i=1}^n \hat{p}_i}{\frac{1}{n}\sum_{i=1}^n y_i} \qquad (\text{ideal} = 1.0,\; \text{defined for } \bar{y} > 10^{-12})$$
- **ROC-AUC ($\mathrm{AUROC}$):** Exact tie-averaged Mann–Whitney $U$ statistic with midranks $r_i = \mathrm{rankdata}(\hat{\mathbf{p}})_i$ over $n_1 = \sum_i y_i > 0$ positives and $n_0 = n - n_1 > 0$ negatives:
  $$\mathrm{AUROC} = \frac{\sum_{i: y_i = 1} r_i - \frac{1}{2}n_1(n_1 + 1)}{n_1 n_0}$$
- **Spiegelhalter (1986) Calibration Test:** Under $H_0: \mathbb{E}[y_i \mid \hat{p}_i] = \hat{p}_i$,
  $$z_{\text{Spieg}} = \frac{\sum_{i=1}^n (y_i - \hat{p}_i)(1 - 2\hat{p}_i)}{\sqrt{\sum_{i=1}^n (1 - 2\hat{p}_i)^2\,\hat{p}_i(1 - \hat{p}_i)}}, \qquad p_{\text{Spieg}} = 2\,\Phi(-|z_{\text{Spieg}}|)$$

---

## 2. Causal Inference & Multiple-Testing Statistics

### 2.1 Block-Aggregated Paired Loss Differentials

To eliminate within-week or intraday serial dependence before hypothesis testing, raw per-step scores $L_A(t), L_B(t)$ are grouped into chronologically ordered non-overlapping clusters $B_1, \dots, B_G$ (ISO calendar weeks `YYYY-Www` in Sections 01–03, daily blocks where specified, or per-seed / per-episode means in controlled testbeds):

$$d_g = \frac{1}{|B_g|}\sum_{t \in B_g} \left(L_A(t) - L_B(t)\right), \qquad g = 1, \dots, G$$

For lower-is-better losses (`ES`, `Brier`, `LogLoss`, reserve cost), $d_g < 0$ favors model $A$; for higher-is-better metrics (daily net EUR/MWh, return, episode reward), the sign flips or advantage is defined as $L_A(t) - L_B(t) > 0$.

### 2.2 Newey–West HAC Variance Estimator (Bartlett Kernel)

Let $\mathbf{d} = (d_1, \dots, d_n)^\top$ be a chronologically ordered paired differential series with sample mean $\bar{d} = \frac{1}{n}\sum_{t=1}^n d_t$ and centered residuals $u_t = d_t - \bar{d}$.

- **Minimum Sample Guard (Hard Ban for $n < 30$):** Asymptotic Newey–West HAC inference is **strictly disabled whenever $n < 30$** (`MIN_HAC_N = 30`). In all small-sample slices ($n < 30$), `hac_eligible = False` and the primary inference switches automatically to the exact paired sign-flip permutation test (Section 2.4).
- **Automatic Bartlett Bandwidth (Lag Rule):**
  $$L(n) = \max\!\left(1,\; \min\!\left(10,\; \left\lfloor 4\left(\frac{n}{100}\right)^{2/9} \right\rfloor\right)\right)$$
  *(In `exp102`/`exp103`, the upper cap is $n - 1$ without the 10-lag ceiling; in `exp79`, $L = 10$ for $n \ge 60$ and $\max(1, \lfloor 4(n/100)^{2/9}\rfloor)$ for $30 \le n < 60$; in `Exp 16`, lag-10 `t_10` is reported directly across $n = 890$ weekly blocks.)*
- **Bartlett Long-Run Variance, $t$-Statistic, and Two-Sided $p$-Value:**
  $$\hat{\gamma}_\ell = \frac{1}{n}\sum_{t=\ell+1}^{n} u_t u_{t-\ell}, \qquad \hat{\Omega}_{\text{NW}} = \hat{\gamma}_0 + 2\sum_{\ell=1}^{L} \left(1 - \frac{\ell}{L + 1}\right)\hat{\gamma}_\ell$$
  $$\mathrm{SE}_{\text{NW}} = \sqrt{\frac{\max(\hat{\Omega}_{\text{NW}},\, 10^{-18})}{n}}, \qquad t_{\text{HAC}} = \frac{\bar{d}}{\mathrm{SE}_{\text{NW}}}, \qquad p_{\text{raw}} = 2\,\Phi\!\left(-|t_{\text{HAC}}|\right) = \operatorname{erfc}\!\left(\frac{|t_{\text{HAC}}|}{\sqrt{2}}\right)$$

### 2.3 Full-Family Step-Down Holm–Bonferroni Adjustment ($p_H$)

Whenever an anchor model is contrasted against $m$ comparators within an experimental family, **all $m$ raw unrounded `float64` $p$-values** are retained (no post-hoc pruning of comparators). Sort the family in ascending order $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$:

$$p_H^{(r)} = \min\!\left(1.0,\; \max_{1 \le j \le r} \left\{(m - j + 1)\, p_{(j)}\right\}\right), \qquad r = 1, \dots, m$$

A contrast is declared statistically significant at family-wise error rate $\alpha = 0.05$ if and only if $p_H^{(r)} < 0.05$.

```text
Algorithm 1: Full-Family Step-Down Holm-Bonferroni Adjustment
Input:  Map of unrounded float64 p-values {k_1: p_1, ..., k_m: p_m}
1. m <- number of comparators in the complete pre-specified family
2. Sort valid pairs (k_r, p_(r)) ascending by p_(r) (tie-break lexicographically by k_r)
3. running_max <- 0.0
4. For r = 0, 1, ..., m - 1:
     adj <- min(1.0, max(running_max, p_(r) * (m - r)))
     running_max <- adj
     p_H[k_r] <- adj
5. Return {k_r: p_H[k_r]} with significance flag (p_H[k_r] < 0.05)
```

### 2.4 Exact Paired Sign-Flip Permutation Test ($n < 30$) & Companion Rank Tests

For small-sample seed, campaign, or short-horizon comparisons (e.g., `Exp 96B` $n = 15$ and $n = 8$; `Exp 93` $n = 30$; post-Sep short slices with $< 30$ ISO weeks), inference uses the two-sided paired sign-flip permutation test under $H_0: d_i \stackrel{d}{=} -d_i$:

- **Observed Statistic & Scale-Relative ULP Guard:** $\bar{d}_{\text{obs}} = |\frac{1}{n}\sum_{i=1}^n d_i|$, with floating-point tie tolerance $\delta_{\text{tol}} = 10^{-12}\max(1.0,\, \bar{d}_{\text{obs}})$ (or $10^{-12}\max(1.0,\, \max_i |d_i|,\, \bar{d}_{\text{obs}})$ in `exp93_stats.py`).
- **Exact Enumeration ($n \le n_{\max}$, where $n_{\max} = 18$ in `exp98a`/`exp89`/`exp93`/`exp96`/`exp79` and $n_{\max} = 20$ in `exp102`/`exp103`):**
  $$p_{\text{exact}} = \frac{1}{2^n}\sum_{\mathbf{s} \in \{-1, +1\}^n} \mathbb{I}\!\left(\left|\frac{1}{n}\sum_{i=1}^n s_i d_i\right| \ge \bar{d}_{\text{obs}} - \delta_{\text{tol}}\right)$$
  *(For $n = 8$, the minimum possible two-sided $p$-value is $2/2^8 = 0.0078125$; for $n = 15$, $2/2^{15} = 6.10 \times 10^{-5}$.)*
- **Deterministic Monte Carlo Sign-Flip ($n > n_{\max}$):** Draw $M$ i.i.d. Rademacher sign vectors $\mathbf{s}^{(m)} \sim \mathrm{Unif}(\{-1,+1\}^n)$ ($M = 100{,}000$ in `exp98a`/`exp89`/`exp93`/`exp102`/`exp103`; $M = 50{,}000$ for $18 < n < 30$ and $20{,}000$ for $n \ge 30$ in `exp79`/`exp96`) using a `SHA-256` seed and apply the conservative Phipson–Smyth estimator:
  $$p_{\text{MC}} = \frac{1 + \sum_{m=1}^{M} \mathbb{I}\!\left(\left|\frac{1}{n}\sum_{i=1}^n s_i^{(m)} d_i\right| \ge \bar{d}_{\text{obs}} - \delta_{\text{tol}}\right)}{M + 1}$$
- **Companion Exact / Tie-Exact Wilcoxon Signed-Rank Test (`exp93_stats.py`):** Drops exact/near-zero differences ($|d_i| \le 10^{-12}\max(1, \max_j |d_j|)$), runs exact Wilcoxon when untied, and falls back to exact $2^k$ or $100{,}000$-draw sign-flip permutation over midranks $R_i = \mathrm{rankdata}(|d_i|)$ whenever ties occur.

---

## 3. Tail-Risk & Basel FRTB Backtesting Suite

Let $I_t \in \{0, 1\}$ denote the binary tail exceedance indicator at nominal tail level $\alpha \in (0, 1)$ over $T$ out-of-sample steps: $I_t = \mathbb{I}[y_t < q_{\alpha,t}]$ for left-tail loss backtests (`VaR95` $\alpha = 0.05$, `VaR99` $\alpha = 0.01$) or $I_t = \mathbb{I}[y_t > q_{1-\alpha,t}]$ for right-tail spike backtests (e.g., evening power scarcity spikes or VIX spikes). Let $x = \sum_{t=1}^T I_t$ and $\hat{p} = x / T$.

### 3.1 Basel Traffic-Light Exceedance Zones & Sample-Size Scaling

Under the Basel Committee (BCBS MAR33) traffic-light framework at $\alpha = 0.01$, zone boundaries for arbitrary sample length $n$ are governed by the exact cumulative binomial probability $F_{\text{Bin}}(k; n, 0.01) = \sum_{j=0}^{k} \binom{n}{j} 0.01^j 0.99^{n-j}$ (`basel_traffic_light_zone` in `exp98a_core.py`):

$$\text{Zone}(k, n) = \begin{cases} \text{GREEN}, & k = 0 \;\text{ or }\; F_{\text{Bin}}(k; n, 0.01) < 0.95 \\ \text{YELLOW}, & 0.95 \le F_{\text{Bin}}(k; n, 0.01) < 0.9999 \\ \text{RED}, & F_{\text{Bin}}(k; n, 0.01) \ge 0.9999 \end{cases}$$

The equivalent 250-day breach count $k_{\text{eq}} = \mathrm{round}(250 k / n)$ maps to the Basel MAR33 capital multiplier $m_c = 3.0 + k_+$, where $k_+ \in \{0.00, 0.40, 0.50, 0.65, 0.75, 0.85, 1.00\}$ for $k_{\text{eq}} \in \{\le 4, 5, 6, 7, 8, 9, \ge 10\}$.

| Horizon | Exp. ($0.01n$) | **GREEN** ($F < 0.95$) | **YELLOW** ($0.95 \le F < 0.9999$) | **RED** ($F \ge 0.9999$) |
| :--- | ---: | :---: | :---: | :---: |
| Basel 1-Yr (`n=250`) | 2.5 | 0–4 (`F(4)=0.8922`) | 5–9 (`F(5)=0.9588`, `F(9)=0.99975`) | ≥10 (`F(10)=0.99995`) |
| Exp 98A Power (`n=450`) | 4.5 | 0–7 (`F(7)=0.9144`) | 8–13 (`F(8)=0.9606`, `F(13)=0.99977`) | ≥14 (`F(14)=0.99993`) |
| Exp 16 Macro (`n=890`) | 8.9 | 0–13 (`F(13)=0.9321`) | 14–21 (`F(14)=0.9625`, `F(21)=0.99986`) | ≥22 (`F(22)=0.99995`) |

### 3.2 Kupiec Proportion of Failures ($\mathrm{LR}_{\text{POF}}, p_{\text{pof}}$)

Tests unconditional coverage $H_0: \Pr(I_t = 1) = \alpha$ via the binomial log-likelihood ratio (with $0 \ln 0 := 0$ and probability floor $10^{-15}$):

$$\mathrm{LR}_{\text{POF}} = \max\!\left(0,\; -2\left[\left((T - x)\ln(1 - \alpha) + x \ln \alpha\right) - \left((T - x)\ln(1 - \hat{p}) + x \ln \hat{p}\right)\right]\right)$$

Under $H_0$, $\mathrm{LR}_{\text{POF}} \stackrel{d}{\to} \chi^2_1$, giving $p_{\text{pof}} = 1 - F_{\chi^2_1}(\mathrm{LR}_{\text{POF}})$ and pass criterion $p_{\text{pof}} \ge 0.05$.

### 3.3 Christoffersen Independence & Conditional Coverage ($\mathrm{LR}_{\text{IND}}, \mathrm{LR}_{\text{CC}}, p_{\text{cc}}$)

Let $n_{ab} = \sum_{t=2}^T \mathbb{I}[I_{t-1} = a \land I_t = b]$ for $a, b \in \{0, 1\}$ count consecutive first-order Markov transitions (strictly within contiguous sequences on panel datasets), with $n_0 = n_{00} + n_{01}$, $n_1 = n_{10} + n_{11}$, $\hat{\pi}_{01} = n_{01}/n_0$, $\hat{\pi}_{11} = n_{11}/n_1$, and pooled transition probability $\hat{\pi} = (n_{01} + n_{11})/(n_0 + n_1)$.

When $n_0 > 0$, $n_1 > 0$, and $0 < \hat{\pi} < 1$:

$$\mathrm{LR}_{\text{IND}} = \max\!\left(0,\; -2\left[\ell_0(\hat{\pi}) - \ell_1(\hat{\pi}_{01}, \hat{\pi}_{11})\right]\right)$$

where:

$$\ell_0(\hat{\pi}) = (n_{00} + n_{10})\ln(1 - \hat{\pi}) + (n_{01} + n_{11})\ln\hat{\pi}$$
$$\ell_1(\hat{\pi}_{01}, \hat{\pi}_{11}) = n_{00}\ln(1 - \hat{\pi}_{01}) + n_{01}\ln\hat{\pi}_{01} + n_{10}\ln(1 - \hat{\pi}_{11}) + n_{11}\ln\hat{\pi}_{11}$$

(If $n_0 = 0$ or $n_1 = 0$, $\mathrm{LR}_{\text{IND}} = 0$ and $p_{\text{ind}} = 1.0$.) Combining unconditional coverage and Markov independence yields the joint **Conditional Coverage** statistic:

$$\mathrm{LR}_{\text{CC}} = \mathrm{LR}_{\text{POF}} + \mathrm{LR}_{\text{IND}} \stackrel{d}{\to} \chi^2_2, \qquad p_{\text{cc}} = 1 - F_{\chi^2_2}(\mathrm{LR}_{\text{CC}})$$

Pass criteria: $p_{\text{ind}} \ge 0.05$ and $p_{\text{cc}} \ge 0.05$. Positive breach clustering is flagged specifically when $p_{\text{ind}} < 0.05$ and $\hat{\pi}_{11} > \hat{\pi}_{01}$.

### 3.4 Acerbi–Székely Expected Shortfall ($Z_2$) & PIT Uniformity

Following Acerbi & Székely (2014, Eq. 6), the per-step ES score $z_t$ tests joint $(\mathrm{VaR}_\alpha, \mathrm{ES}_\alpha)$ adequacy without requiring conditional distribution inversion:

- **Left tail (returns / PnL, negative = loss; $I_t = \mathbb{I}[y_t < q_{\alpha,t}]$):**
  $$z_t = \frac{y_t\, I_t}{\alpha\,\max(-\widehat{\mathrm{ES}}_{\alpha,t},\, 10^{-8})} + 1$$
- **Right tail (positive spikes, e.g., VIX; $I_t = \mathbb{I}[y_t > q_{1-\alpha,t}]$):**
  $$z_t = 1 - \frac{y_t\, I_t}{\alpha\,\max(\widehat{\mathrm{ES}}^+_{1-\alpha,t},\, 10^{-8})}$$

The test statistic is $Z_2 = \frac{1}{T}\sum_{t=1}^T z_t$. Under exact tail calibration $H_0$, $\mathbb{E}[z_t] = 0$; $Z_2 < 0$ indicates **tail-risk underestimation** (either too many breaches or losses deeper than predicted ES), whereas $Z_2 > 0$ indicates conservative tail protection. One-sided underestimation significance ($H_1: Z_2 < 0$) is assessed via both:

1. **Newey–West HAC (`hac_lag = 4`):** $t_{\text{HAC}} = Z_2 / \mathrm{SE}_{\text{NW}}(z)$, $p_{\text{HAC}} = \Phi(t_{\text{HAC}})$.
2. **Stationary Circular Block Bootstrap (`n_boot = 5,000`, block length $b = \max(1, \min(L, \lfloor T/10 \rfloor))$):** Resample centered blocks from $z_t - Z_2$ to obtain empirical means $\bar{z}^{*(k)}$ and compute $p_{\text{boot}} = (1 + \sum_{k=1}^{B}\mathbb{I}[\bar{z}^{*(k)} \le Z_2])/(B + 1)$; passes non-underestimation iff $p \ge 0.05$.

**Continuous & Discrete PIT Uniformity:**
For continuous Probability Integral Transforms $u_t = \hat{F}_t(y_t) \in [0, 1]$, uniformity against $\mathrm{Uniform}(0,1)$ is audited via the one-sample Kolmogorov–Smirnov test ($D_T = \sup_{u \in [0,1]}|\hat{F}_T(u) - u|$, pass iff $p_{\text{KS}} \ge 0.05$) and Cramér–von Mises test ($\omega^2$). On quantile-ladder baselines lacking continuous densities, discrete PIT bins defined by sorted quantiles $\tau_1 < \dots < \tau_K$ are tested via Pearson $\chi^2_{K}$ goodness-of-fit and Total Variation Distance $\mathrm{TVD} = \frac{1}{2}\sum_{k=0}^{K}|\hat{\pi}_k - (\tau_{k+1}-\tau_k)|$.

**Basel FRTB 3/3 Pass Bundle (`Exp 16`):** A model passes `3/3` regulatory tail checks on the 19-year equity target iff:
1. $p_{\text{cc}}(\mathrm{VaR}_{99}) \ge 0.05$,
2. $p_{\text{underest}}(Z_{2,99}) \ge 0.05$, and
3. $p_{\text{KS}}(\text{Continuous PIT}) \ge 0.05$.

---

## 4. Exact Mathematical Formulation of Stylized Causal Decision Overlays

> **Data and Simulation Protocol:** Every decision overlay below is a **stylized deterministic economic rule** evaluated causally on real public empirical telemetry (SMARD power prices/wind, Binance 1h klines, Yahoo Finance ETFs). Real market prices and returns are never altered; overlay PnL, reserve cost, and survival equity curves map each model's identically formatted predictive cloud $F_t = \{(p_i, z_i)\}_{i=1}^M$ through pre-locked formulas with zero model-specific tuning.

### 4.1 `Exp 98A` — German Power 1%-Fee BESS Arbitrage & Peak-Scarcity Hedge Overlay

All models output a 4D day-ahead German price cloud $(z_{i,0}, z_{i,1}, z_{i,2}, z_{i,3}) = (p_{\text{eve},i}, p_{\text{mid},i}, p_{\text{ngt},i}, p_{\text{spread},i})$ in EUR/MWh with weights $p_i$. Consistent with exogenous single-cloud scoring (Eq. 1 / Eq. 2), each decision leg is optimized client-side over the **same single queried predictive cloud** across a fee sensitivity grid $f \in \{0\%, 1\%, 2\%\}$ (headline $f = 0.01$):

**Leg 1 — 1 MWh Grid BESS Arbitrage Dispatch:**
With round-trip efficiency $\eta = 0.88$, cycle degradation wear $w_{\text{wear}} = 15.0\text{ EUR/MWh}$, and left-tail CVaR risk aversion $\lambda_{\text{bess}} = 0.25$, define branch-wise PnL for three candidate actions $a \in \{\text{IDLE}, \text{MID}\!\to\!\text{EVE}, \text{NGT}\!\to\!\text{EVE}\}$:

$$\mathrm{PnL}_{\text{idle},i} = 0$$
$$\mathrm{PnL}_{\text{mid},i} = \eta\, p_{\text{eve},i} - p_{\text{mid},i} - w_{\text{wear}} - f\!\left(|p_{\text{mid},i}| + \eta |p_{\text{eve},i}|\right)$$
$$\mathrm{PnL}_{\text{ngt},i} = \eta\, p_{\text{eve},i} - p_{\text{ngt},i} - w_{\text{wear}} - f\!\left(|p_{\text{ngt},i}| + \eta |p_{\text{eve},i}|\right)$$

Each day $t$ before the day-ahead auction clears, the desk selects the risk-adjusted maximizer:

$$a_t^* = \arg\max_{a \in \{\text{idle},\, \text{mid},\, \text{ngt}\}} \left(\mathbb{E}_{F_t}\!\left[\mathrm{PnL}_a\right] - \lambda_{\text{bess}}\max\!\left(0,\; -\widehat{\mathrm{ES}}_{0.05}(\mathrm{PnL}_a)\right)\right)$$

with strict tie-break ordering (`mid >= ngt` and `> 0`, else `ngt > 0`, else `idle`), settling realized PnL $\mathrm{PnL}_t^{\text{BESS}}$ against realized SMARD prices $(y_{\text{eve},t}, y_{\text{mid},t}, y_{\text{ngt},t})$.

**Leg 2 — 1 MWh Utility Peak-Scarcity Forward-vs-Spot Hedge:**
Against the causal 30-day rolling evening-peak forward benchmark $K_{\text{fwd},t}$, the utility selects a discrete hedge ratio $h_t^* \in \mathcal{H} = \{0, 0.25, 0.50, 0.75, 1.0\}$ minimizing expected procurement cost plus right-tail 95% CVaR penalty ($\lambda_{\text{hedge}} = 0.35$):

$$C_{t,i}(h) = h\, K_{\text{fwd},t}(1 + f) + (1 - h)\, p_{\text{eve},i}$$
$$h_t^* = \arg\min_{h \in \mathcal{H}} \left(\mathbb{E}_{F_t}\!\left[C_t(h)\right] + \lambda_{\text{hedge}}\,\widehat{\mathrm{ES}}^+_{0.95}\!\left(C_t(h)\right)\right)$$

Realized hedge saving relative to unhedged spot procurement is $S_t^{\text{hedge}} = y_{\text{eve},t} - \left[h_t^* K_{\text{fwd},t}(1+f) + (1-h_t^*) y_{\text{eve},t}\right]$.

**Total Desk Net Economic Score (EUR/MWh/day):**

$$\mathrm{Net}_t = \mathrm{PnL}_t^{\text{BESS}} + S_t^{\text{hedge}}, \qquad \mathrm{Sharpe}_{\text{ann}} = \frac{\overline{\mathrm{Net}}}{s_{\mathrm{Net}}}\sqrt{365}$$

### 4.2 `Exp 79B / 90C` — Digital-Asset 3× Dynamic Leverage Survival Book

Evaluated on $n = 1{,}149$ held-out Binance event-clock steps (`2025-09..2026-11`) across 4 major digital assets (`BTC`, `ETH`, `SOL`, `XRP`). Between consecutive event bars $h_k$ and $h_{k+1}$, realized equal-weight 4-coin close-to-close return $r_k^{\text{cc}}$ and worst intrabar low excursion $r_k^{\text{lo}}$ are computed strictly from post-query bars $(h_k+1 \dots h_{k+1})$:

$$r_k^{\text{cc}} = \frac{1}{4}\sum_{c=1}^{4} \frac{\mathrm{Close}_{c,h_{k+1}} - \mathrm{Close}_{c,h_k}}{\mathrm{Close}_{c,h_k}}, \qquad r_k^{\text{lo}} = \frac{1}{4}\sum_{c=1}^{4} \frac{\min_{u \in \{h_k+1,\dots,h_{k+1}\}} \mathrm{Low}_{c,u} - \mathrm{Close}_{c,h_k}}{\mathrm{Close}_{c,h_k}}$$

Every model outputs a 5D predictive cloud over $(\ln(1+\Delta t_{\text{next}}),\, \text{mag}_{\text{BTC}},\, \text{mag}_{\text{ETH}},\, \text{mag}_{\text{SOL}},\, \text{mag}_{\text{XRP}})$ where $\text{mag}_c = \ln(1 + \text{GK}_{c,\text{pct}})$. All controller constants are frozen strictly on `DEV_CAL`: base leverage $L_{\max} = 3.0$, target single-interval loss budget $\tau_{\text{loss}} = 12.0\%$, intrabar excursion multiplier $\kappa_{\text{gk}} = 1.35$, calibration volatility surge threshold $q_{75}^{\text{cal}}$ and base rate $\bar{p}_s$, 3-hour burst base rate $\bar{p}_b$, and 20th-percentile basis threshold $b_{20}^{\text{cal}}$.

**Three-Stage Causal Leverage Controller:**

1. **Joint 99% Tail-Volatility Leverage Cap:** Map cloud magnitudes to mean 4-coin Garman–Klass volatility $\mathrm{gk}_i = \frac{1}{4}\sum_{c=1}^{4}\mathrm{expm1}(\mathrm{clip}(z_{i,c}, -2, 6))$ and compute $q_{99,k}^{\text{gk}} = Q_{0.99}(\mathrm{gk}, p_k)$:
   $$L_{\text{vol},k} = \mathrm{clip}\!\left(\frac{12.0}{\max(1.35\, q_{99,k}^{\text{gk}},\, 1.0)},\; 0.0,\; 3.0\right)$$
2. **Crash-Regime Probability Throttle:** Compute cloud probabilities $p_{\text{surge},k} = \Pr_{F_k}(\frac{1}{4}\sum_{c=1}^4 z_{i,c} \ge q_{75}^{\text{cal}})$ and $p_{\text{burst},k} = \Pr_{F_k}(z_{i,0} \le \ln(1+3))$, then form the excess stress score:
   $$S_k = \frac{1}{2}\,\mathrm{clip}\!\left(\frac{p_{\text{surge},k} - \bar{p}_s}{1 - \bar{p}_s},\, 0,\, 1\right) + \frac{1}{2}\,\mathrm{clip}\!\left(\frac{p_{\text{burst},k} - \bar{p}_b}{1 - \bar{p}_b},\, 0,\, 1\right)$$
3. **Basis-Flip & Negative-Momentum Brake:** Using strictly causal market state at event $k$ close:
   $$B_k = \begin{cases} 0.50, & \text{if } \mathrm{basis}_k < b_{20}^{\text{cal}} \;\land\; \mathrm{ret}_{3\text{h},k} < 0 \\ 1.00, & \text{otherwise} \end{cases}$$
4. **Final Leverage Allocation:**
   $$L_k = \mathrm{clip}\!\left(L_{\text{vol},k}\,(1 - 0.70\, S_k)\, B_k,\; 0.0,\; 3.0\right)$$

**Fee Schedule & 5% Forced-Liquidation Ruin Floor:**
For each transaction-cost level $c_{\text{tc}} \in \{0, 5, 15\}\text{ bps}$ (headline `5 bps`, boundary `15 bps`), net interval return is $r_k^{\text{net}} = L_k r_k^{\text{cc}} - c_{\text{tc}} \cdot 10^{-4} |L_k - L_{k-1}|$ (with $L_{-1} = 0$). Starting from $E_0 = 1.0$, equity evolves with an absorbing maintenance-margin floor $E_{\text{floor}} = 0.05$:

```text
Algorithm 2: Intrabar-Checked Equity Evolution with 5% Forced-Liquidation Floor
Initialize E_0 = 1.0, ruined = False, n_margin_calls = 0
For k = 0, 1, ..., n - 1:
  If ruined:
    E_{k+1} = 0.0; continue
  E_low = E_k * (1.0 + L_k * r_k^lo)               # Intrabar worst-low margin check
  If E_low <= 0.05:
    ruined = True; n_margin_calls += 1; E_{k+1} = 0.0; continue
  If L_k * r_k^lo <= -0.25:                        # Severe >=25% single-interval warning
    n_margin_calls += 1
  E_next = E_k * (1.0 + r_k^net)                   # Close-to-close fee-adjusted check
  If E_next <= 0.05:
    ruined = True; E_{k+1} = 0.0
  Else:
    E_{k+1} = E_next
```

### 4.3 `Exp 89B` — European 4-TSO Evening-Peak Balancing-Reserve & VoLL Overlay

Models output a 4D scaled cloud over German evening residual load, North/South wind shortfalls, and log peak price (`n = 323` held-out days). The residual-load coordinate $z_{i,0}$ is inverted statelessly to physical megawatts:

$$R_{i,\text{MW}} = 1000\cdot\max\!\left(0,\; 35.0 + 10.0\, z_{i,0}\right)$$

For the 4-hour evening peak block ($H = 4\text{ h}$, 16:00–20:00 UTC), a balancing responsible party commits thermal/peaker standby reserve $b_t \ge 0$ (MW) at standby cost $c_{\text{over}} = 50.0\text{ EUR/MWh}$ against Value of Lost Load (VoLL) emergency scarcity penalty $c_{\text{short}} = \rho\, c_{\text{over}}$ over $\rho \in \{10, 20, 50\}$ (headline **$\rho = 20 \implies c_{\text{short}} = 1{,}000\text{ EUR/MWh}$**).

By the exact newsvendor optimality condition, the Bayes-optimal reserve commitment under predictive cloud $F_t$ is the **upper $\tau^*$ quantile**:

$$\tau^* = \frac{\rho}{1 + \rho} = \frac{20}{21} \approx 0.95238, \qquad b_t^* = Q_{\tau^*}\!\left(R_{\text{MW}},\, p_t\right)$$

Realized daily balancing cost (EUR), annualized system cost (EUR M/yr), and severe unserved-load outage count ($\ge 2.5\text{ GW}$ deficit threshold) are:

$$\mathrm{Cost}_t = 4.0 \cdot \left[\rho\, c_{\text{over}}\max(0,\, R_t - b_t^*) + c_{\text{over}}\max(0,\, b_t^* - R_t)\right]$$
$$\mathrm{Cost}_{\text{ann}}\;(\text{EUR M/yr}) = \frac{365}{10^6}\cdot \frac{1}{n}\sum_{t=1}^{n} \mathrm{Cost}_t, \qquad \mathrm{Outages}_{\ge 2.5\text{GW}} = \sum_{t=1}^{n} \mathbb{I}\!\left[\max(0,\, R_t - b_t^*) \ge 2{,}500\text{ MW}\right]$$

*(Companion `Exp 89A` 4-TSO wind schedule bidding uses the mirrored lower-tail quantile $\tau^* = 1/(1+\rho)$ over national aggregate wind $\sum_{k=1}^4 \mathrm{Cap}_k\,\mathrm{CF}_{i,k}$ with capacities $[18.0, 12.5, 22.0, 1.8]\text{ GW}$ and a $2{,}000\text{ MW}$ severe shortfall threshold.)*

### 4.4 `Exp 79A` — 4-Asset `SPY / TLT / HYG / GLD` 5 bps Risk-Overlay Allocation Rule

Evaluated on $n = 439$ daily steps spanning the 2021–2023 stock–bond correlation flip (`2021-10-01..2023-06-30`, calibration strictly `2020-01-02..2021-09-30`). Let $r_t = (r_{\text{SPY},t}, r_{\text{TLT},t}, r_{\text{HYG},t}, r_{\text{GLD},t})^\top$ be daily percentage returns, with base allocation $w_{\text{base}} = (0.60, 0.20, 0.10, 0.10)^\top$ (compared against Static 60/40 $w_{60/40} = (0.60, 0.40, 0, 0)^\top$). Calibration constants frozen strictly on the pre-evaluation split are the rolling 128-day median baseline `CVaR99` risk budget $B_{\text{CVaR99}}$, per-asset 1-sigma scales $\sigma_j^{\text{cal}}$, and multi-asset stress base rate $\bar{p}_{\text{stress}}$.

From each day's queried 4D return cloud $(Z_t, p_t)$:

1. **Forecast Stock–Bond Correlation Tilt (`TLT -> GLD + Cash`):** Compute the cloud-weighted correlation $\hat{\rho}_{sb,t} = \mathrm{Corr}_{F_t}(Z_{\text{SPY}}, Z_{\text{TLT}}) \in [-1, 1]$. Starting from $w_{\text{tilt}} = w_{\text{base}}$, if $\hat{\rho}_{sb,t} > 0$:
   $$\Delta_t = w_{\text{base},\text{TLT}} \cdot 0.50\,\min\!\left(1.0,\; \frac{\hat{\rho}_{sb,t}}{0.50}\right), \qquad w_{\text{tilt},\text{TLT}} \leftarrow w_{\text{base},\text{TLT}} - \Delta_t, \qquad w_{\text{tilt},\text{GLD}} \leftarrow w_{\text{base},\text{GLD}} + 0.50\,\Delta_t$$
   with the remaining $0.50\,\Delta_t$ held in 0%-yield cash.
2. **CVaR99 Tail-Budget Exposure Scaling:** Let $\widehat{\mathrm{CVaR}}_{99,t} = \left|\min\!\left(\widehat{\mathrm{ES}}_{0.01}(Z_t w_{\text{base}}, p_t),\, 0\right)\right|$:
   $$e_{\text{cvar},t} = \mathrm{clip}\!\left(\frac{B_{\text{CVaR99}}}{\max(\widehat{\mathrm{CVaR}}_{99,t},\, 0.25)},\; 0.25,\; 1.00\right)$$
3. **Multi-Asset Stress Throttle:** Let $p_{\text{stress},t} = \Pr_{F_t}\!\left(\sum_{j=1}^4 \mathbb{I}\!\left[|Z_{i,j}| / \sigma_j^{\text{cal}} \ge 1.0\right] \ge 2\right)$:
   $$\theta_t = 1.0 - 0.35\,\mathrm{clip}\!\left(\frac{p_{\text{stress},t} - \bar{p}_{\text{stress}}}{1 - \bar{p}_{\text{stress}}},\; 0.0,\; 1.0\right)$$
4. **Portfolio Weights & Fee-Adjusted Net Return:**
   $$w_t = w_{\text{tilt},t} \cdot \max\!\left(0.25,\; e_{\text{cvar},t}\,\theta_t\right), \qquad r_t^{\text{net}} = \frac{w_t^\top r_t}{100} - c_{\text{tc}} \cdot 10^{-4}\,\|w_t - w_{t-1}\|_1$$
   evaluated across $c_{\text{tc}} \in \{0, 5, 15\}\text{ bps}$ (headline **$5\text{ bps}$**), with $w_{-1} = \mathbf{0}$, equity $E_{t+1} = E_t(1 + r_t^{\text{net}})$, drawdown $\mathrm{DD}_t = E_{t+1}/\max_{s \le t+1} E_s - 1$, and annualized Sharpe ($\sqrt{252}$).

---

## References

1. Gneiting, T. & Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. *Journal of the American Statistical Association*, 102(477), 359–378.
2. Newey, W. K. & West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703–708.
3. Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
4. Phipson, B. & Smyth, G. K. (2010). Permutation P-values Should Never Be Zero: Calculating Exact P-values When Permutations Are Randomly Drawn. *Statistical Applications in Genetics and Molecular Biology*, 9(1).
5. Kupiec, P. H. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models. *Journal of Derivatives*, 3(2), 73–84.
6. Christoffersen, P. F. (1998). Evaluating Interval Forecasts. *International Economic Review*, 39(4), 841–862.
7. Acerbi, C. & Székely, B. (2014). Back-testing Expected Shortfall. *Risk*, December 2014.
8. Basel Committee on Banking Supervision (2019). *Minimum Capital Requirements for Market Risk (MAR33)*. Bank for International Settlements.
9. Patton, A. J., Ziegel, J. F. & Chen, R. (2019). Dynamic Semiparametric Models for Expected Shortfall (and Value-at-Risk). *Journal of Econometrics*, 211(2), 388–413.
10. Spiegelhalter, D. J. (1986). Probabilistic Prediction in Patient Management and Clinical Trials. *Statistics in Medicine*, 5(5), 421–433.
