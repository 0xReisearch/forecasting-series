# Section 01 — Joint Tails and Drift

Out-of-sample evaluation traces, audited scorecard, and replication protocol for Section 01 (*"Joint tails and drift"*) of Rei Labs' Part 2 essay, **Forecasting Through Drift**.

This folder covers three primary empirical benchmarks and two additional evaluations of Adapt-1's probabilistic `futures` surface:

1. **German Day-Ahead Power Scarcity (`Exp 98A`)** — `linear_residual` transport on paired empirical clouds under upward scarcity drift and negative midday solar prices, plus the daily options/FOMC/CPI catalyst boundary (`Exp 98B`).
2. **Digital-Asset Microstructure Cascade (`Exp 79B` + `Exp 90C`)** — event-clock joint tail survival under a 15-month bear cascade and flash crash vs. uncapped long-context time-series foundation models (`ctx = 256..2,048`), quantile trees, and GARCH/EWMA peers.
3. **European Dunkelflaute & Geo-Energy Residual Wrapping (`Exp 89B`)** — state-conditional Adapt-1 residual wrapping ("tail-risk airbag") over uncapped `ctx = 1,024` `Chronos-2` and `TimesFM-3.0`, plus the smooth long-context US wind boundary (`Exp 90A`).

---

## Files in This Folder

| File | Rows | Scope & Contents |
| :--- | ---: | :--- |
| [`exp98a_smard_power_oos_trace.csv`](./exp98a_smard_power_oos_trace.csv) | 450 | `2025-08-27..2026-11-19`: 4D ES, 1%-fee BESS+hedge PnL, peak `VaR99` breaches (Adapt-1 LinRes/Kernel, 12 GJR-GARCH, Q-HGBR, Local LinReg, HeteroRidge, EWMA, PriorCF). |
| [`exp79b_crypto_cascade_oos_trace.csv`](./exp79b_crypto_cascade_oos_trace.csv) | 1,149 | `2025-09-01..2026-11-23`: 5D ES (12 models), dynamic leverage, 5 bps 3× equity curves incl. Oct 10–12 2025 flash-crash window. |
| [`exp90c_longctx_tsfm_survival_summary.csv`](./exp90c_longctx_tsfm_survival_summary.csv) | 45 | Long-context (`ctx=256..2,048`) survival audit: 32/32 raw TSFMs (`8/8` at `ctx=2,048`), Q-HGBR, standalone & wrapped Adapt-1, Conformal, EWMA, Ridge, statics + ruin timestamps. |
| [`exp89b_eu_dunkelflaute_oos_trace.csv`](./exp89b_eu_dunkelflaute_oos_trace.csv) | 323 | `2026-01-01..2026-11-19`: 4D ES (25 models incl. Discovery `es_adapt1_airbag_disc_chronos2`), 20× VoLL cost, outages (`>=2.5 GW`), `VaR95` breaches (Airbags, `ctx=1,024` TSFMs, ACI). |
| [`scorecard.json`](./scorecard.json) | — | Audited metrics, comparators, operating notes, API configs, and trace reconciliation targets. |

---

## 1. Public Data Sources & Provenance Separation

We explicitly separate **raw empirical telemetry** from **stylized causal decision overlays**:

| Benchmark | Telemetry Source | Universe & Sealed OOS | Decision Overlay (Identical Across Models) |
| :--- | :--- | :--- | :--- |
| **Exp 98A** | SMARD (`smard.de`) DE-LU prices & forecasts + Open-Meteo 100m wind/temp | 4 blocks (`p_eve/midday/night/spread`, EUR/MWh); `n=450` (`2025-08-27..2026-11-19`) | 1%-fee (0/1/2%) 1 MWh BESS (`eta=0.88`, wear `15`, `lam=0.25`) + 1 MWh hedge (`lam=0.35`, 30d fwd) |
| **Exp 98B** *(catalyst)* | ETFs (`SPY/TLT/HYG/GLD`), CBOE (`VIX/VIX3M/VVIX/OVX/GVZ`), FOMC/CPI | 4D daily returns; `n=726` (`2024-01-02..2026-11-20`, `55` FOMC/CPI) | Pure probabilistic (`ES_4D`, `CRPS_60`, `FZ0_99`) + FRTB VaR/ES backtests |
| **Exp 79B / 90C** | Binance 1h spot & perp OHLCV + basis | `BTC/ETH/SOL/XRP` 5D event-clock (`log dt` + 4 GK); `n=1,149` (`2025-09-01..2026-11-23`) | 3× lev (`99%` tail-vol, crash & basis brakes), `5%` liq. floor, `0/5/15 bps` fees (pre-OOS locked) |
| **Exp 89B** | SMARD 4-TSO (`50Hertz/Amprion/TenneT/TransnetBW`, Fraunhofer ISE) + Open-Meteo + OVX + ECB FX + GPRD | 4D eve-peak (`resload`, N/S wind, `log p_eve`); `n=323` (`2026-01-01..2026-11-19`) | 4h reserve (`50 EUR/MWh` standby, `20×` VoLL `1,000` [10/20/50×], outage `>=2.5 GW`) |
| **Exp 90A** *(smooth wind)* | EIA-930 hourly US wind | 4 ISOs (`ERCO/CISO/PJM/MISO`); `n=318` (`2026-01-01..2026-11-22`), `ctx=1,536` | 4h spinning reserve (`$30/MWh`, `20×` VoLL) + `VaR95/99` backtests |

---

## 2. Strict Causal Score-Before-Ingest Protocol

Every benchmark runs strictly online with zero future leakage:

1. **Exp 98A (Pre-Auction Cutoff):** All post-auction columns (`r1_*`, `r2_*`) are dropped before feature construction (`post_auction_columns_included = 0`). Features use only pre-auction `r0_*` grid/weather forecasts and prior-day settled prices, standardized via strictly causal expanding mean/std (`<= t - 1`). Hyperparameters (`bw = 0.20`, `numeric_ridge = 40.0`) are locked strictly on `DEV_CAL` (`2024-10-31..2025-05-18`, `n = 200`), separated from `SEALED_OOS` (`2025-08-27..2026-11-19`, `n = 450`) by a 100-day causal buffer.
2. **Exp 79B / 90C (Event-Clock Maturation):** Each cascade event `k` queries `POST /domains/{id}/query` using only realized prior-bar microstructure features (`f_dt_prev`, `f_hw24`, `f_hw168`, `f_cur_mag`, `f_ofi_sell`, `f_ewma_vol`), scores the returned joint cloud against the realized next event, updates the 3× leverage overlay using only the pre-locked `DEV_CAL` rules, and only then ingests the realized outcome via `POST /events`.
3. **Exp 89B (10-Hour Maturation Gap):** Delivery day `D` evening peak (`16:00–20:00 UTC`) is queried at `D 06:00:00Z` using `D 05:00:00Z` wind/meteo observations, `D - 1` evening price (`(D-1) 20:00:00Z`), and `<= D - 1` OVX/FX/GPRD closes. `DEV_CAL` (`2024-09-01..2025-12-31`, `n = 487`) strictly precedes `SEALED_OOS` (`2026-01-01..2026-11-19`, `n = 323`). For residual wrapping, Adapt-1 ingests the realized base TSFM residual (`pit_z` for `Chronos-2`, `add_resid` for `TimesFM-3.0`) after each evening settles, conditioned on physical state plus base TSFM uncertainty (`tsfm_iqr_rank`).

---

## 3. Exact Adapt-1 API Configuration Recipes

Set credentials via environment variables (never hardcode keys or hostnames):

```bash
export ADAPT_API_BASE="https://rei-neuroadapt-api.reilabs.org/api/v1"
export ADAPT_API_KEY="<your-api-key>"
```

All live evaluations enforce lossless joint cloud export (`max_samples == max_support == max_branches == 128`) and verified `DELETE -> GET 404` domain teardown.

### 3.1 Exp 98A — German Power (`Live_Adapt1_W128_LinRes`)

```json
{
  "domain_id": "sec01-exp98a-power-linres",
  "session_id": "sess-sec01-exp98a",
  "schema": {
    "signals": [
      "z_r0_fc_resload_proxy_gw", "z_r0_fc_load_eve_D_gw",
      "z_r0_om_w100_mean", "z_r0_om_temp_mean",
      "z_r0_p_eve_prev", "z_r0_p_midday_prev",
      "z_r0_is_wknd_D", "z_r0_cos_doy_D",
      "raw_p_eve_eur", "raw_p_midday_eur", "raw_p_night_eur", "raw_p_spread_eur"
    ],
    "event_types": ["outcome"]
  },
  "learning": {
    "enabled": true,
    "transition": {
      "enabled": true,
      "event_types": ["outcome"],
      "input_paths": [
        "values.z_r0_fc_resload_proxy_gw", "values.z_r0_fc_load_eve_D_gw",
        "values.z_r0_om_w100_mean", "values.z_r0_om_temp_mean",
        "values.z_r0_p_eve_prev", "values.z_r0_p_midday_prev",
        "values.z_r0_is_wknd_D", "values.z_r0_cos_doy_D"
      ],
      "targets": [
        {"path": "values.raw_p_eve_eur", "type": "number"},
        {"path": "values.raw_p_midday_eur", "type": "number"},
        {"path": "values.raw_p_night_eur", "type": "number"},
        {"path": "values.raw_p_spread_eur", "type": "number"}
      ],
      "required_support": 6,
      "max_samples": 128,
      "max_distance": 1.0,
      "numeric_min_skill": 0.0,
      "numeric_ridge": 40.0,
      "futures": {
        "enabled": true,
        "conditioning": "kernel",
        "numeric_transport": "linear_residual",
        "bandwidth": 0.2,
        "prior_strength": 2.0,
        "max_support": 128,
        "max_branches": 128,
        "max_pending_forecasts": 64
      }
    }
  }
}
```

### 3.2 Exp 79B — Crypto Cascade (`Adapt1_Live_W128_Futures`)

Identical `learning.transition.futures` structure with 6 bounded microstructure inputs (`f_dt_prev`, `f_hw24`, `f_hw168`, `f_cur_mag`, `f_ofi_sell`, `f_ewma_vol`), 5 numeric targets (`t_log_dt_next`, `t_mag_btc`, `t_mag_eth`, `t_mag_sol`, `t_mag_xrp`), `conditioning: "kernel"`, `numeric_transport: "linear_residual"`, `bandwidth: 0.18`, `numeric_ridge: 50.0`, `max_samples: 128`, `max_support: 128`, `max_branches: 128`. (In the Exp 90C companion audit, standalone `Live_Adapt1_W128` used `bandwidth: 0.24`, `numeric_ridge: 30.0`, `numeric_transport: "linear_residual"`.)

### 3.3 Exp 89B — EU Dunkelflaute Residual Wrapper (`Adapt-1 Airbag`)

- **Wrapping `Chronos-2 (ctx = 1,024)` (`pit_z` residual targets):** 5 inputs (`x_resload_da_s`, `x_wshort_05`, `x_log_ovx_t`, `x_eurusd_stress_t`, `tsfm_iqr_rank`), 4 residual targets (`r_y_resload_s`, `r_y_wshort_north_s`, `r_y_wshort_south_s`, `r_y_log_peve_s`), `conditioning: "kernel"`, `numeric_transport: "none"`, `bandwidth: 0.22`, `max_samples: 128`, `max_support: 128`, `max_branches: 128`.
- **Wrapping `TimesFM-3.0 (ctx = 1,024)` (`add_resid` residual targets):** Same 5 inputs and 4 residual targets with `conditioning: "kernel"`, `numeric_transport: "linear_residual"`, `bandwidth: 0.24`, `numeric_ridge: 30.0`.

---

## 4. Results and Configuration Comparisons

### 4.1 Exp 98A — German SMARD Day-Ahead Power (`n = 450`, `2025-08-27..2026-11-19`)

| Model | 4D ES (EUR/MWh) | 1%-Fee Net (EUR/MWh/d) | Peak VaR99 | Basel Zone |
| :--- | ---: | ---: | ---: | :--- |
| Offline Local Linear Reg. | 57.80 | 71.98 | 15/450 | RED |
| **Adapt-1 LinRes (W=128)** | **58.76** | **71.83** *(61.04 BESS + 10.79 Hedge)* | **11/450** | **YELLOW** |
| Tier-2 HeteroRidge *(offline)* | 59.26 | 73.47 | 9/450 | YELLOW |
| Best GJR-GARCH (FHS Roll128) | 63.08 | 64.91 | 14/450 | RED |
| All 12 GJR-GARCH *(range)* | 63.08–64.47 | 63.20–67.44 | 6–21/450 | 1 GREEN (6/450), 8 YELLOW, 3 RED |
| Adapt-1 Kernel (W=128) | 65.26 | 66.09 | 11/450 | YELLOW |
| Online Q-HGBR (GaussCop) | 66.00 | 65.08 | 30/450 | RED |

- **Exp 98B Catalyst Boundary (`n = 726`, `2024-01-02..2026-11-20`):** On daily multi-asset returns (`SPY/TLT/HYG/GLD`) with options implied vol and scheduled FOMC/CPI dates, rolling MLE `GJR_DCC_GARCH_FHS_Roll256_r21` edges `Live_Adapt1_ImpliedVol_W128_Kernel` on full-OOS `ES_4D` (`1.139` vs `1.145`, NS `p_H = 1.00`), and `GJR_StudentT_Roll256_r21` leads `EX_ANTE_FOMC_CPI` (`1.277` vs `1.288`).

### 4.2 Exp 79B + Exp 90C — Binance 4-Asset Digital-Asset Cascade (`n = 1,149`, 3× 5 bps Overlay)

- **Unlevered `Static_1x_Spot` (`BTC/ETH/SOL/XRP`):** fell `-56.5%` (`-56.53%`) to `0.43×` (`0.4347×`) equity (`MaxDD -65.75%`).
- **Adapt-1 (`Exp 79B`, `Adapt1_Live_W128_Futures`):** ranked `#1/12` on `ES_5D` (`0.697` / `0.69698`), cut Oct 10 2025 flash-crash leverage to `0.08×` (`0.0792×`), held Oct 2025 crash DD to `-28.4%` (`-28.37%`), and survived the full 15-month bear cascade at `0.09×` (`0.0881×`) final equity (`-91.19%`, `MaxDD -94.29%`).
- **Surviving Classical Peers (`5 bps`):** `EWMA_EmpResid` (`0.090×` in 79B; `0.076×` in 90C), `DCC_GARCH_FHS` (`0.082×` in 79B), and `Online_Ridge` (`0.107×`) also survived at `5 bps`.
- **Liquidation Ruin (`5%` Floor, Jun–Nov 2026):** `32/32` raw TSFMs across `ctx = 256..2,048` (`8/8` at `ctx = 2,048`, ruined `2026-06-05..2026-11-20`), `Q-HGBR` (`2026-11-14` in 90C / `2026-11-20` in 79B), and `Static_3x_Unhedged` (`2026-06-02`) hit the 5% forced-liquidation floor (`-100%`).
- **Fee and Calibration Results:** At `15 bps` one-way fees, turnover drag pushes `Adapt1_Live_W128_Futures` (step 1,117) and `DCC_GARCH_FHS` past the 5% floor while lower-turnover `EWMA_EmpResid` (`-94.21%`) survives. In Exp 90C, wrapping rescued `Chronos-2 ctx=2,048` to survival (`0.0766×`), whereas wrapped `TimesFM-3.0 ctx=1,024` still liquidated at step 1,138.

### 4.3 Exp 89B + Exp 90A — EU Dunkelflaute Wrapping (`n = 323`, `ctx = 1,024`) & US Wind Boundary (`n = 318`, `ctx = 1,536`)

| Model (`Exp 89B`, `ctx=1,024`) | 4D ES (Rank/25) | 20× VoLL (EUR M/yr) | Outages (`>=2.5 GW`) | VaR95 (`p_cc`) |
| :--- | ---: | ---: | ---: | :--- |
| **Adapt-1 Disc. Airbag (Chronos-2)** *(6/9 covs)* | **0.76981 (#1/25)** | **991.87 (#1/25)** | **7** | **5.88% (0.236 PASS)** |
| **Adapt-1 Authored Airbag (Chronos-2)** | **0.77152 (#2/25)** | **1,000.79 (#2/25)** | **7** | 6.19% (0.169 PASS) |
| ACI Airbag (Chronos-2) | 0.77206 (#3/25) | 1,026.32 (#4/25) | 9 | 5.6% (0.309) |
| Chronos-2 `ctx=1,024` *(raw CovExog GC)* | 0.77886 (#5/25) | 1,125.28 (#10/25) | 14 | 10.5% (2.04e-4) |
| **Adapt-1 Airbag (TimesFM-3.0)** | **0.81002 (#9/25)** | **1,111.64 (#7/25)** | **9** | 5.9% (0.563) |
| ACI Airbag (TimesFM-3.0) | 0.82078 (#10/25) | 1,010.23 (#3/25) | 2 | 5.6% (0.309) |
| TimesFM-3.0 `ctx=1,024` *(raw CovExog GC)* | 0.92394 (#21/25) | 1,811.38 (#23/25) | 47 | 27.6% (3.41e-39) |

*CSV column key (`exp89b_eu_dunkelflaute_oos_trace.csv`):* Discovery Chronos-2 Airbag = `es_adapt1_airbag_disc_chronos2`, `cost20x_adapt1_airbag_disc_chronos2`, `outage20x_adapt1_airbag_disc_chronos2`, `var95_br_adapt1_airbag_disc_chronos2`; authored Airbags = `*_adapt1_airbag_c2*`, `*_adapt1_airbag_t3*`; ACI = `*_aci_c2*`, `*_aci_t3*`; raw TSFMs = `*_c2_ctx1024_cov_gc*`, `*_t3_ctx1024_cov_gc*`.

- Wrapping cuts `TimesFM-3.0` outages **`47 -> 9`** (`EUR 1,811M -> 1,112M/yr`) and `Chronos-2` outages **`14 -> 7`** (`EUR 1,125M -> 992M/yr` with full-323d Transition Discovery [`6/9` covariates incl. `GPRD`, `#1/25` `ES_4D = 0.769808`, `VaR95 = 5.88%` `19/323` `p_cc = 0.236`, `21/24` Holm ES wins / `0` losses], `1,001M/yr` authored [`#2/25` `0.771524`, `VaR95 = 6.19%` `20/323`]), while statistically tying `ACI` on 20× weekly dispatch cost (Discovery `p_H = 0.142` vs `ACI(Chronos-2)`, `0.679` vs `ACI(TimesFM-3.0)`; authored `p_H = 0.88` [`0.8796`] against both ACI variants; `ES_4D` vs `ACI(Chronos-2)` `p_H = 0.803` Discovery / `1.00` authored).
- **Exp 90A US Wind Boundary (`n = 318`, `ctx = 1,536`):** On smooth 4-ISO US wind where raw long-context TSFMs are already well-calibrated, wrapping degrades accuracy and increases `VaR99` breaches: `TimesFM-3.0 ctx=1,536` shifts `3 -> 4` breaches (`ES 0.16644 -> 0.16942`, `-1.79%`, `p_H = 0.0155`) and `Chronos-2 ctx=1,536` shifts `2 -> 6` breaches (`ES 0.16614 -> 0.16788`, `-1.04%`, `p_H = 0.3087`) — an average shift from `2.5 -> 5.0` breaches (summarized as `3 -> 5` in the essay). Only wrap base models when tail calibration tests confirm under-dispersion.

---

## 5. Offline Metric Recomputation (Python 3 Standard Library Only)

Run directly inside this directory to verify every headline metric from the CSV traces:

```python
import csv

def load(fn):
    with open(fn, encoding="utf-8") as f:
        return list(csv.DictReader(f))

# 1. Exp 98A (n=450)
r98 = load("exp98a_smard_power_oos_trace.csv")
assert len(r98) == 450
mean = lambda rows, col: sum(float(r[col]) for r in rows) / len(rows)
tot  = lambda rows, col: sum(float(r[col]) for r in rows)
print(f"Exp 98A Adapt-1 ES_4D={mean(r98,'es_adapt1_linres'):.2f}, Best GJR={mean(r98,'es_ar1_gjr_fhs_r128_r21'):.2f}, Desk1%={mean(r98,'desk1_adapt1_linres'):.2f}, VaR99={int(tot(r98,'var99_br_adapt1_linres'))}/450")

# 2. Exp 79B (n=1,149) + Exp 90C (n=45)
r79 = load("exp79b_crypto_cascade_oos_trace.csv")
assert len(r79) == 1149
oct_row = [r for r in r79 if r["is_oct2025_crash_step"] == "1"][0]
print(f"Exp 79B Adapt-1 ES_5D={mean(r79,'es5d_adapt1'):.3f}, Oct25 lev={float(oct_row['lev_adapt1']):.2f}x, Final eq={float(r79[-1]['eq5_adapt1']):.2f}x, 1x Spot={float(r79[-1]['eq5_static_1x_spot']):.2f}x")
r90 = load("exp90c_longctx_tsfm_survival_summary.csv")
raw_ruined = sum(1 for r in r90 if r["is_raw_tsfm"] == "1" and r["survived_5bps"] == "0")
c2048_ruined = sum(1 for r in r90 if r["is_raw_tsfm"] == "1" and r["context_length"] == "2048" and r["survived_5bps"] == "0")
print(f"Exp 90C Raw TSFMs ruined: {raw_ruined}/32 (ctx=2048: {c2048_ruined}/8)")

# 3. Exp 89B (n=323)
r89 = load("exp89b_eu_dunkelflaute_oos_trace.csv")
assert len(r89) == 323
c20 = lambda col: mean(r89, col) * 365.0 / 1e6
print(f"Exp 89B Disc Airbag(C2) ES={mean(r89,'es_adapt1_airbag_disc_chronos2'):.6f}, Cost={c20('cost20x_adapt1_airbag_disc_chronos2'):.2f}M, Outages={int(tot(r89,'outage20x_adapt1_airbag_disc_chronos2'))}, VaR95={int(tot(r89,'var95_br_adapt1_airbag_disc_chronos2'))}/323")
print(f"Exp 89B Auth Airbag(C2) ES={mean(r89,'es_adapt1_airbag_c2'):.6f}, Cost={c20('cost20x_adapt1_airbag_c2_eur'):.2f}M, Outages C2 {int(tot(r89,'outage20x_c2_ctx1024_cov_gc'))}->{int(tot(r89,'outage20x_adapt1_airbag_c2'))}, T3 {int(tot(r89,'outage20x_t3_ctx1024_cov_gc'))}->{int(tot(r89,'outage20x_adapt1_airbag_t3'))}")
```
