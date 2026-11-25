# Section 02 — Online Probability Models (`/value/*`)

Replication traces, audited scorecard, and API recipes for **Section 02 ("Online probability models")** of the Rei Labs Adapt-1 Research Series, Part 2 (*Forecasting Through Drift*).

All four studies evaluate Adapt-1's native stateful `/value/*` binary probability learners directly on **empirical market telemetry** under strict score-before-ingest walk-forward protocols — with zero client-side ensembling, zero post-hoc Platt or isotonic recalibration, and zero future peeking.

---

## Directory Contents

| File | Rows | Scope | Key Verifiable Results |
| :--- | ---: | :--- | :--- |
| [`exp103a_crypto_funding_oos_trace.csv`](exp103a_crypto_funding_oos_trace.csv) | 16,077 | 103A: `BTC/ETH/SOL` 8h (`2022-01..2026-11`, 256w, 21m) | `#2/21` Br `0.12108`, `#1/21` ECE `0.01091`, `2026` `#1/21`, chron-split `#17/#19` |
| [`exp103b_macro_left_tail_oos_trace.csv`](exp103b_macro_left_tail_oos_trace.csv) | 13,856 | 103B `LeftTail` (`2020-01..2026-11`, 360w, 10m, 15 cols, `W=4,096`) | Compact `C3` cuts ECE `-94.7%` to `#1/10` (`0.00081`), `#3/10` Br `0.06445` (within `0.00015` of MLE GJR), AUC `0.5871→0.5688` |
| [`exp103b_macro_extreme_move_oos_trace.csv`](exp103b_macro_extreme_move_oos_trace.csv) | 13,856 | 103B `ExtremeMove` (`2020-01..2026-11`, 360w, 10m, 15 cols, `W=4,096`) | Compact `C3` cuts ECE `-79.2%` (`0.00402`), `#2/10` Br `0.06045`, `VIX_INV` `#1/10` (`0.12048`), `2026` `#2/10` Br & `#1/10` ECE (`0.00619`) |
| [`exp102c_german_power_oos_trace.csv`](exp102c_german_power_oos_trace.csv) | 1,800 | 102C: 4-block DE power (`2025-08..2026-11`, 65w, 15m, 19 cols) | `Disc8` `#1/15` Br `0.14933`/LL `0.46342`/AUC `0.8440`/`P_EVE 0.12937` (vs `HGBR_Disc8` `0.15431`, `Linear` `#6/15` `0.16078`) |
| [`exp100a_install_lockout_blocks.csv`](exp100a_install_lockout_blocks.csv) | 414 | 100A: Block retrain audit — vol (42/arm) & power (18/arm) | Expanding `install=True` retained `32/42`, `36/42` vol & `14–17/18` power vs `0/42`, `0/18` sliding |
| [`scorecard.json`](scorecard.json) | — | Machine-readable RFC-8259 audited scorecard | Leaderboards, Holm `p_H`, pre-2020 `DEV_CAL` lock, telemetry, operating notes, column specs |

---

## Results and Configuration Comparisons

### 1. Exp 103A — 5-Year Crypto Perpetual Funding Sign-Flips (`BTC/ETH/SOL`, 8h)
- **Public Data Source:** Binance USDT-M Perpetual Futures 8-hour funding rates and realized volatility (`BTCUSDT`, `ETHUSDT`, `SOLUSDT`), `2020-09-13` to `2026-11-22`. Pre-2022 warmup (`1,423` bars) fixes all feature standardization parameters; sealed OOS spans `2022-01-01` to `2026-11-22` (`5,359` bars × 3 coins = **`16,077` forecasts**, `256` ISO weeks, `22.1%` negative-funding rate).
- **Target:** Binary `y_neg_funding = 1{funding_{t+1} < 0}`. Refit every `150` 8h bars (`36` blocks), identical across all 21 models.
- **Headline Result (`Adapt1_CUDA_MLPv2_FixEp_EpHash_W4608`):**
  - **Full 5-Year OOS (`256` weeks):** Ranks **`#2/21` Brier (`0.12108`)** and **`#1/21` ECE (`0.01091`)** — `2.4×` lower ECE than `ExtraTrees_Cls_Pooled_W1536` (`0.02634`) and `2.5×` lower than `LogisticRegression_L2_Pooled_W1536` (`0.02773`). Mean retrain time: **`2.28s`** (`~2.3s/retrain`, `0` rejected blocks).
  - **Holm-Bonferroni Contrasts (`m = 20`):** **`10` Holm-significant Brier wins / `0` losses** on Full OOS (`24` wins / `0` losses across all 3 slices and both metrics), including beating matched-window `AR1_GJR_SignProb_W1024` (`0.12633`, `p_H = 0.0049`) and tying `#1` `Online_HGBR_Cls_Pooled_W1536` (`0.12049`, `p_H = 1.00`, NS).
  - **2026 Live Slice (`47` weeks, `n = 2,928`):** Ranks **`#1/21` on Brier (`0.14814`), Log-Loss (`0.45387`), and AUROC (`0.8453`)**.
  - **Recent Slice and Validation Split:** On the 12-week `POST_2026_09` slice (`n = 741`), Adapt-1 ranks `#14/21` (`0.13721`, NS vs `#1` HGBR `0.13193`, `p_H = 0.3433`). Switching `validation_mode` from `"episode_hash"` to `"chronological"` withholds the newest 20% (`68–102` days) from weight fitting and gives Full-OOS Brier **`0.12862` (`#17/21` at `W = 4,608`, `p_H = 3.93e-4`)** and **`0.13266` (`#19/21` at `W = 3,072`, `p_H = 0.0038`)**.

### 2. Exp 103B — 7-Year 8-Asset Daily Macro Tail-Shock Exceedance
- **Public Data Source:** Daily returns across 8 liquid ETF/FX instruments (`SPY, TLT, HYG, GLD, USO, UUP, USDJPY, KRE`) paired with lagged CBOE volatility term-structure indices (`VIX`, `VIX3M`, `VVIX`) from Yahoo Finance / FRED / CBOE (`2010–2026`). Strictly causal pre-2020 `DEV_CAL` (`2018-01-02..2019-12-31`, `n = 4,024` forecasts on `LeftTail`) locked the compact architecture with zero 2020–2026 peeking. Sealed OOS spans `2020-01-02..2026-11-20` (`1,732` trading days × 8 assets = **`13,856` forecasts**, `360` ISO weeks).
- **Targets:** `LeftTail = 1{r_t < -1.5 * rv22_{t-1}}` (`6.96%` base rate) and `ExtremeMove = 1{|r_t| > 2.0 * rv22_{t-1}}` (`6.57%` base rate), refit every `63` trading days (`28` blocks, equal-budget `W = 4,096` pooled rows / `512`d per-asset; `10` models per target, `15` CSV cols).
- **Headline Result (`Adapt1_103B_C3_MLPv2_Compact_LogLoss`):**
  - **`LeftTail` Full OOS:** Cuts ECE **`-94.7%`** (`0.01510 -> 0.00081`, **`#1/10`** across all models including `Trailing_BaseRate`) and climbs to **`#3/10` Brier (`0.06445`)**, within `0.00015` of `Real_MLE_GJR_GARCH_TailProb` (`0.06430`, `p_H = 1.00`, NS).
  - **`ExtremeMove` Full OOS:** Cuts ECE **`-79.2%`** (`0.01929 -> 0.00402`) and climbs to **`#2/10` Brier (`0.06045`)**, beating `LogisticRegression_L2_Pooled`, `Online_HGBR_Cls_Pooled`, and `Real_MLE_GJR_GARCH_TailProb`.
  - **Stress & Live Slices (`ExtremeMove`):** Sweeps **`#1/10` on `VIX_INVERSION` stress** (`45` weeks, `n = 984`, Brier **`0.12048`**, Log-Loss **`0.40562`**) and ranks **`#2/10` Brier (`0.06808`) & `#1/10` ECE (`0.00619`)** in `2026` (`47` weeks, `n = 1,792`).
  - **Holm-Bonferroni Tally:** **`0/108` Holm-significant losses** across both targets, 3 slices, 2 metrics, and all 9 peers (down from `11` losses in the unregularized Exp 102B baseline).
  - **Documented Trade-Off:** Compressing `mlp_v2` to `16/8` hidden units for `45` epochs traded `LeftTail` ranking AUROC (`0.5871 -> 0.5688`, `-0.0184`) for the `-94.7%` calibration gain, while holding `ExtremeMove` AUROC nearly unchanged (`0.6269 -> 0.6266`).

*CSV inventory: `15` columns (`5` metadata + `10` equal-budget `W = 4,096` / `512`d models); full 10-model leaderboards and 108-contrast Holm `p_H` tables live in `scorecard.json`.*

### 3. Exp 102C — Pooled 4-Block German Power Extremes
- **Public Data Source:** German day-ahead electricity auction prices (`DE-LU` bidding zone, Bundesnetzagentur **SMARD**) aggregated into 4 intraday delivery blocks (`P_EVE`, `P_MIDDAY`, `P_NIGHT`, `P_SPREAD`), combined with pre-auction load proxies and **Open-Meteo** wind/temperature forecasts. Upper `Q85_dev` thresholds (`171.32 / 124.15 / 109.11 / 174.73 EUR/MWh`) are frozen causally on `DEV_CAL` (`2024-10-31..2025-05-18`). Sealed OOS spans `2025-08-27..2026-11-19` (`450` days × 4 blocks = **`1,800` forecasts**, `65` ISO weeks, `W = 2,048` rows, refit every `25` days; `15` models: original `11` + `Adapt1_CUDA_MLPv2_Disc8_PowerPool2048` + `3` matched `Disc8` peers, `19` CSV cols).
- **Target:** Two-sided extreme event `y_extreme = 1{block_price > Q85_dev OR block_price < 0}`.
- **Model Comparison (`15` models):**
  - **`Adapt1_CUDA_MLPv2_Disc8_PowerPool2048` (Two-Stage pre-OOS Discovery `8/20` features → `/value/*` `mlp_v2` `16×8`, `80ep`, `warm_start=False`):** Ranks **`#1/15` overall** on FULL Brier (**`0.14933`**, `BSS 32.86%`), Log-Loss (**`0.46342`**), and AUROC (**`0.8440`**), `#2/15` ECE (`0.05297`), and **`#1/15` on `P_EVE`** (`0.12937` Brier, `#1/15` LL `0.40908`, `#1/15` ECE `0.04908`) — beating matched `Online_HGBR_Disc8` (`0.15431`) and full-20 `Online_HGBR` (`0.15627`) with **`9/14` Holm Brier wins, `9/14` Holm Log-Loss wins, and `0` Holm losses** (`2/3` Holm Brier & LL wins in matched `Disc8` `m = 3` pool: `LogReg_Disc8` & `ExtraTrees_Disc8`, `0` losses).
  - **`Adapt1_CUDA_Linear_PowerPool2048` (`model_type="linear"` on Adapt-1's projected basis):** Ranks **`#6/15` overall (`#4/11` orig.) Brier (`0.16078`)**, **`#3/15` (`#1/11` orig.) on evening peaks (`P_EVE` Brier `0.14004`)**, beats **all 4 raw-feature LEAR/L2-logistic baselines** (`0.16728–0.18921`), and Holm-significantly beats the warm-start neural head (`p_H = 0.0136` Brier, `p_H = 5.42e-4` Log-Loss).
  - **`Adapt1_CUDA_MLPv2_PowerPool2048` (warm-start 20f):** Scores `0.18012` Brier (**`#12/15`**, `#9/11` orig.; `#11/15` `P_EVE` `0.15353`) on upward seasonal drift with `W = 2,048`.
  - **Strongest Comparator Peers:** `Online_HGBR_Cls_Pooled_Disc8` ranks `#2/15` (`0.15431` Brier, `#2/15` `P_EVE` `0.13916`) and full-20 `Online_HGBR_Cls_Pooled` ranks `#3/15` (`0.15627` Brier, `#6/15` `P_EVE` `0.14538`).

| Model (`Exp 102C`, `n=1,800`, `65`w) | FULL Brier | FULL LL | AUROC | ECE | P_EVE Br | p_H vs Disc8 (Br/LL) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Adapt-1 Disc8 MLPv2 (8f)** | **0.14933 (#1)** | **0.46342** | **0.8440** | 0.05297 | **0.12937 (#1)** | **ANCHOR** |
| Online HGBR Pooled (Disc8) | 0.15431 (#2) | 0.47799 | 0.8368 | 0.07230 | 0.13916 (#2) | 0.243 / 0.254 |
| Online HGBR Pooled (20f) | 0.15627 (#3) | 0.47763 | 0.8410 | 0.07207 | 0.14538 (#6) | 0.243 / 0.358 |
| **Adapt-1 Linear Projected (20f)** | 0.16078 (#6) | 0.52264 | 0.8096 | 0.06910 | 0.14004 (#3) | 0.243 / 0.0308* |
| ExtraTrees Pooled (Disc8) | 0.17588 (#10) | 0.52111 | 0.8327 | 0.10794 | 0.17016 (#14) | 0.0029* / 0.0032* |
| Adapt-1 Warm MLPv2 (20f) | 0.18012 (#12) | 0.68892 | 0.7618 | 0.09962 | 0.15353 (#11) | 0.00265* / 0.00063* |
| LogReg L2 Pooled (Disc8) | 0.18996 (#14) | 0.56220 | 0.7791 | 0.10583 | 0.16844 (#13) | <1e-5* / <1e-7* |

*`p_H` = paired weekly NW-HAC Holm (`m = 14`, FULL slice; `*` = Holm-sig win for `Adapt-1 Disc8`, `9/14` Br & LL wins, `0` losses; `m = 3` matched `Disc8`: `2/3` Holm Br & LL wins vs `ExtraTrees_Disc8` & `LogReg_Disc8`, `0` losses). CSV columns (`19` cols = `4` meta + `15` models): `Adapt1_CUDA_MLPv2_Disc8_PowerPool2048`, `Adapt1_CUDA_Linear_PowerPool2048`, `Adapt1_CUDA_MLPv2_PowerPool2048`, `Online_HGBR_Cls_Pooled_Disc8`, `ExtraTrees_Cls_Pooled_Disc8`, `LogisticRegression_L2_Pooled_Disc8`, plus original 20f peers.*

### 4. Exp 100A — Validation-Gated Install Lockout Audit
- **Finding:** On drifting volatility (`42` blocks of 42 trading days) and German power (`18` blocks of 25 days) streams, calling `POST /value/train` with `install=True` on an expanding buffer caused the server's promotion gate to return `status="retained"` — keeping calm-regime Block-0 weights — whenever incoming high-volatility regimes increased irreducible validation loss:
  - **Realized Volatility (`42` blocks):** `abs_r` retained **`36/42` (`85.7%`)** blocks; `rv5_fwd` retained **`32/42` (`76.2%`)** blocks.
  - **German Power (`18` blocks):** `ev_peak_spike_200` retained **`14/18`** blocks (`12` consecutive retained blocks `1..12`, then `3` `rejected` blocks `13..15`, then `2` retained blocks `16..17` — zero post-Block-0 installs); `p_peak` retained **`16/18`** blocks; `ev_neg_midday` retained **`17/18`** blocks. These counts come from `exp100a_install_lockout_blocks.csv`; the `12`-block figure is the initial consecutive retained streak for `ev_peak_spike_200`.
- **Fix Verified in Trace:** Switching to sliding `observation_ids` with `install=False`, `warm_start_source="candidate"`, and scoring via `model_source="candidate"` reduced `status="retained"` lockouts to **`0/42`** on volatility and **`0/18`** on German power.

---

## Adapt-1 `/value/*` API Configuration Recipes

Set environment variables before calling the hosted API:

```bash
export ADAPT_API_BASE="https://rei-neuroadapt-api.reilabs.org/api/v1"
export ADAPT_API_KEY="<YOUR_API_KEY>"
```

All calls send `Authorization: Bearer <ADAPT_API_KEY>` (never hardcode credentials) and follow the strict **score-before-ingest** causal loop:
1. `POST /domains/{domain_id}/value/configure` — set `"representation": "projected"` and `"max_samples": 20000`.
2. `POST /domains/{domain_id}/value/observe` — ingest only observations whose outcomes have already settled (`obs_available <= block_start`), assigning a shared `episode_id` across contemporaneous assets (`bar_{t:06d}` in 103A, `day_{t:06d}` in 103B).
3. `POST /domains/{domain_id}/value/train` — train Block 0 with `"install": true`, then subsequent sliding windows with `"install": false` and explicit `"observation_ids"` of length `W`.
4. Poll `POST /domains/{domain_id}/value/state` until training completes.
5. `POST /domains/{domain_id}/value/predict` — score the upcoming block with `"model_source": "candidate"` (or `"installed"` on Block 0) **before** observing its outcomes.
6. Clean up via `DELETE /domains/{domain_id}` inside a `try/finally` block and verify `GET /domains/{domain_id}` returns HTTP `404`.

### Exact `options` Payloads

**Exp 103A (Crypto Funding Sign-Flips — Regime-Shifting Panel, `W = 4,608`, `episode_id = "bar_{t:06d}"`):**
```json
{
  "model_type": "mlp_v2",
  "dimensions": 64,
  "dense_hidden_dimensions": 64,
  "epochs": 120,
  "learning_rate": 0.003,
  "loss_function": "log_loss",
  "interaction_selection": "evidence",
  "warm_start": true,
  "validation_mode": "episode_hash"
}
```

**Exp 103B (Rare Macro Tail-Shocks — `6.6–7.0%` Base Rate, `W = 4,096`, `episode_id = "day_{t:06d}"`):**
```json
{
  "model_type": "mlp_v2",
  "dimensions": 16,
  "dense_hidden_dimensions": 8,
  "dense_inputs": true,
  "epochs": 45,
  "learning_rate": 0.0015,
  "loss_function": "log_loss",
  "interaction_selection": "evidence",
  "warm_start": false,
  "validation_mode": "chronological"
}
```

**Exp 102C (`Disc8` Two-Stage `16×8`, `80ep`, on `8/20` pre-OOS discovered features, `W = 2,048`):**
```json
{
  "model_type": "mlp_v2",
  "dimensions": 16,
  "dense_hidden_dimensions": 8,
  "epochs": 80,
  "learning_rate": 0.0015,
  "loss_function": "log_loss",
  "interaction_selection": "evidence",
  "preserve_binary_inputs": true,
  "warm_start": false,
  "validation_mode": "episode_hash"
}
```

**Exp 102C (German Power 4-Block Extremes — Linear Readout on Projected Basis, `W = 2,048`):**
```json
{
  "model_type": "linear",
  "dimensions": 64,
  "epochs": 120,
  "learning_rate": 0.003,
  "loss_function": "log_loss",
  "warm_start": false,
  "validation_mode": "episode_hash"
}
```

---

## Independent Metric Recomputation (Python 3 Stdlib Only)

Every headline Brier score, Log-Loss, 10-bin equal-width ECE, AUROC, and lockout count can be recomputed offline directly from the CSV traces with zero third-party packages:

```python
import csv, math

def score_binary(rows, y_col, p_col):
    yt = [float(r[y_col]) for r in rows]
    pp = [min(1.0 - 1e-6, max(1e-6, float(r[p_col]))) for r in rows]
    n = len(yt)
    brier = sum((p - y) ** 2 for y, p in zip(yt, pp)) / n
    ll = -sum(y * math.log(p) + (1.0 - y) * math.log(1.0 - p) for y, p in zip(yt, pp)) / n
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10.0, (b + 1) / 10.0
        idx = [i for i, p in enumerate(pp) if (lo <= p <= hi if b == 9 else lo <= p < hi)]
        if idx:
            ece += (len(idx) / n) * abs(sum(yt[i] for i in idx) / len(idx) - sum(pp[i] for i in idx) / len(idx))
    return {"n": n, "brier": round(brier, 5), "log_loss": round(ll, 5), "ece": round(ece, 5)}

rows_103a = list(csv.DictReader(open("exp103a_crypto_funding_oos_trace.csv")))
print("103A Full:", score_binary(rows_103a, "y_neg_funding", "Adapt1_CUDA_MLPv2_FixEp_EpHash_W4608"))
print("103A Y2026:", score_binary([r for r in rows_103a if r["ts_utc"] >= "2026-01-01"], "y_neg_funding", "Adapt1_CUDA_MLPv2_FixEp_EpHash_W4608"))

rows_103b_lt = list(csv.DictReader(open("exp103b_macro_left_tail_oos_trace.csv")))
print("103B LeftTail C3:", score_binary(rows_103b_lt, "y_left_tail", "Adapt1_103B_C3_MLPv2_Compact_LogLoss"))

rows_103b_em = list(csv.DictReader(open("exp103b_macro_extreme_move_oos_trace.csv")))
print("103B ExtremeMove C3:", score_binary(rows_103b_em, "y_extreme_move", "Adapt1_103B_C3_MLPv2_Compact_LogLoss"))
print("103B ExtremeMove C3 VIX_INV:", score_binary([r for r in rows_103b_em if r["vix_inversion"] == "1"], "y_extreme_move", "Adapt1_103B_C3_MLPv2_Compact_LogLoss"))

rows_102c = list(csv.DictReader(open("exp102c_german_power_oos_trace.csv")))
print("102C Disc8:", score_binary(rows_102c, "y_extreme", "Adapt1_CUDA_MLPv2_Disc8_PowerPool2048"))
print("102C Linear:", score_binary(rows_102c, "y_extreme", "Adapt1_CUDA_Linear_PowerPool2048"))
```

From the repository root, run `python3 verify_traces.py` to check SHA-256 checksums, strict RFC-8259 syntax, zero-leak hygiene, and all trace-to-scorecard reconciliation assertions.
