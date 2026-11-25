# Section 04 — Conditioning Across Time

> **Part 2: *Forecasting Through Drift*** · **Provenance Tier:** Raw public empirical telemetry (strictly causal prequential evaluation; zero synthetic overlays; zero PII).

This directory packages the out-of-sample evaluation traces, audited [`scorecard.json`](./scorecard.json), API configuration recipes, and independent recomputation protocol for **Section 04 ("Conditioning across time")** of Rei Labs' Part 2 research essay.

Section 04 evaluates Adapt-1's conditional similarity surface (`learning.transition.futures`) across two opposite time horizons:

1. **Multi-decade archive recall (`Exp 16`, 19-year weekly 5-asset global macro, 2008–2026, `n = 890` evaluated weeks):** Retaining up to `1,024` weeks lets state-conditional similarity pull dormant 2008 crisis analogues back to the foreground only when volatility structure calls for them (`0.67×` in calm 2017 and `0.47×` in late 2026 vs. `2.70×` during Feb 2018 Volmageddon, `2.34×` at Mar 2020 COVID onset / `1.62×` across the 13-week COVID peak, and `2.62×` during the Apr 2025 tariff shock). All `9/9` configurations with `W >= 128` and `bw >= 0.20` pass all `3/3` Basel FRTB tail/PIT checks and beat `DCC-GARCH(1,1)-FHS` (`2.3336`) on 5D Energy Score (`5/9` significantly), led by `W = 512, bw = 0.30` at `2.3208` (`t = -3.78`).
2. **Intra-case progressive revisions (`Exp 53` + NYC Lower Manhattan, `11,326` checkpoints across 3 US cities):** Sharing a single `context.metadata.future_id` across progressive checkpoints of an open municipal 311 ticket (`0h` -> `+1h` -> `+6h` in NYC/SF; `0h` -> `+6h` -> `+48h` in Chicago) refines joint disposition + duration forecasts as survival information arrives without retraining per stage.
3. **Documented operating boundaries:** Tight bandwidth `bw = 0.10` in `Exp 16` fails 99% Expected Shortfall calibration; high-cardinality heavy-tailed survival logs (`Exp 54` Mozilla Bugzilla Thunderbird and `Exp 59` SF Fire/EMS dispatch) expose soft similarity bleed across hard survival cutoffs where risk-set-filtered quantile forests and boosted trees lead; upstream operational logs must stay strictly append-only.

---

## Headline Audited Results

### 1. 19-Year Weekly 5-Asset Global Macro (`Exp 16`, `n = 890` Evaluated Weeks)

| Model | W / bw | 5D ES | t_10 vs DCC | VaR99 p_cc | ES99 Z2 p | PIT KS p | FRTB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Adapt-1 kern (sup=512)** | 512 / 0.30 | **2.3208** | **-3.78** | 0.3577 | 0.1459 | 0.6541 | **PASS** |
| **Adapt-1 kern (sup=512)** | 1024 / 0.20 | **2.3226** | **-2.64** | 0.5190 | 0.1650 | 0.5462 | **PASS** |
| Adapt-1 kern (sup=256) | 256 / 0.30 | 2.3225 | -2.79 | 0.6903 | 0.2251 | 0.9299 | **PASS** |
| Adapt-1 kern (sup=1024) | 1024 / 0.30 | 2.3226 | -3.10 | 0.5190 | 0.1838 | 0.4134 | **PASS** |
| Adapt-1 kern (sup=512) | 1024 / 0.30 | 2.3250 | -2.42 | 0.3577 | 0.1473 | 0.5055 | **PASS** |
| *Adapt-1 kern (sup=512, tight bw)* | 1024 / 0.10 | 2.3272 | -0.78 | 0.1342 | **0.0378** | 0.4666 | **FAIL** |
| DCC-GARCH(1,1)-FHS | — | 2.3336 | — | 0.9115 | 0.3911 | 0.8626 | **PASS** |
| RiskMetrics EWMA-FHS (lam=0.88) | — | — | — | 0.5190 | 0.1603 | 0.8321 | **PASS** |
| DCC-GARCH(1,1)-Gauss | — | 2.3504 | — | 0.5190 | 0.0859 | **0.0002** | **FAIL** |
| RiskMetrics EWMA-Gauss (lam=0.88) | — | — | — | **0.0006** | **0.0007** | **0.0063** | **FAIL** |

**2008 Crisis Archive Weight Recall at `W = 1024, bw = 0.20` (`cond_mass / prior_mass`):**

| Period / Event | Dates | Cond. 2008 | Uniform | Ratio |
| :--- | :--- | ---: | ---: | ---: |
| Calm 2017 (full year, `n=51` wks) | 2017-01..2017-12 | 12.75% | 19.12% | **0.67×** (0.6671) |
| Late 2026 calm trough week | 2026-08-13 | 4.64% | 9.77% | **0.47×** (0.4749) |
| Feb 2018 Volmageddon shock | 2018-02-06 | 48.51% | 17.94% | **2.70×** (2.7034) |
| Mar 2020 COVID onset week | 2020-03-03 | 34.67% | 14.84% | **2.34×** (2.3354) |
| 13-week COVID peak (`idx 598..610`) | 2020-02-18..2020-05-13 | 24.00% | 14.83% | **1.62×** (1.6185) |
| Apr 2025 tariff shock | 2025-04-04 | 27.65% | 10.55% | **2.62×** (2.6212) |

### 2. Multi-City Municipal 311 Progressive Revisions (`11,326` Checkpoints)

| City & Scope | Checkpoints & Stages | Adapt-1 Mixed ES | Calibration & Comparators |
| :--- | :--- | :--- | :--- |
| **NYC Lower Manhattan** (Exp 26, `zip_10006_all`, OOS 2026-01..11) | 4,330 (3,873 OOS); 0h / +1h / +6h | Cohort (`n=826`): **0.6187 → 0.5534 → 0.5308** (`t_10=-9.20`) | 7d overrun within **±0.5 pp** all stages (0h: 16.65% vs 17.15%; +1h: 30.27% vs 30.36%; +6h: 40.72% vs 40.44%). |
| **Chicago 311** (Exp 53, post-Sep 2026) | 2,621 (1,106 / 855 / 660); 0h / +6h / +48h | **0.4716 → 0.4008 → 0.2579** | **#1/7 all 3 stages**; 0h **-18.9%** vs OLHR (0.5818, `p_H=7.24e-11 < 1e-4`), beats Refit QRF (0.4998, `p_H=0.0033`) & HGBR (0.5920, `p_H<1e-4`). |
| **San Francisco 311** (Exp 53, post-Sep 2026) | 4,375 (1,722 / 1,545 / 1,108); 0h / +1h / +6h | **0.6459 → 0.6298 → 0.5602** | #2/7 at 0h (vs Refit QRF 0.6361, NS `p_H=0.099`); **#1/7 at +1h & +6h**. Frozen pre-2026 drift **+17%–24%**: QRF 0.7714 / 0.7443 / 0.6769, HGBR 0.7987 / 0.7663 / 0.7193, Logit-Ridge 0.7964 / 0.7602 / 0.6977. |

> **Provenance note on NYC 311:** The `4,330`-checkpoint / `n = 826` identical-ticket NYC Lower Manhattan progressive-revision study comes from `Exp 26` (`zip_10006_all`, `2025-11-12..2026-11-12`, `4,330` predicted checkpoints + `10` initial burn-in abstentions), distinct from the separate 4-borough asynchronous-mask triage benchmark (`Exp 31`). Notice also that the exact `n = 826` cohort `+6h` ES is `0.530845` (reported as `0.531` / `0.5309` when rounded from the 5-decimal summary `0.53085`).

---

## Public Data Sources

All traces are derived strictly from public, keyless portals:

- **19-Year Weekly 5-Asset Global Macro (`Exp 16`, `2008-04-02` .. `2026-09-18`):**
  - **Equities:** Ken French 49 Industry Portfolios equal-weight US equity composite (contiguous trading calendar).
  - **Macro / Vol / Rates / FX / Oil (FRED):** `VIXCLS` (CBOE VIX), `VXVCLS` (CBOE 3-month VIX), `DGS10` (10-Year US Treasury constant maturity yield), `DCOILWTICO` (WTI crude oil spot, 5-day log return symmetrically winsorized at `±30%`), and `DTWEXBGS` (Trade-Weighted US Dollar Index).
  - **Conditioning feature:** Single causal volatility term-structure ratio `vix_term = log(VIXCLS / VXVCLS)`. Targets are causal EWMA volatility-standardized 5-trading-day changes (`z_eq`, `z_y10`, `z_vix`, `z_oil`, `z_usd`, clipped at `±6.0`).
- **Municipal 311 Service Requests (Socrata Open Data, zero API key required):**
  - **NYC Lower Manhattan (`Exp 26`):** `https://data.cityofnewyork.us/resource/erm2-nwe9.json` (ZIP `10006` dispatch queue, `2025-11-12` .. `2026-11-12`, 7-day SLA).
  - **Chicago 311 (`Exp 53`):** `https://data.cityofchicago.org/resource/v6vf-nfxy.json` (4 smallest-population Community Areas: Fuller Park, Burnside, Oakland, Riverdale; post-Sep 2026 live window through `2026-11-12`, 7-day SLA).
  - **San Francisco 311 (`Exp 53`):** `https://data.sfgov.org/resource/vw6y-z8j6.json` (3 urban residential/mixed neighborhoods; post-Sep 2026 live window through `2026-11-12`, 7-day SLA).
- **Documented Boundary Benchmarks (offline summary only in `scorecard.json`):**
  - **Mozilla Bugzilla (`Exp 54`):** Public Bugzilla REST API (`Thunderbird` `/bug`, `/history`, `/comment`, 30-day SLA, checkpoints at `0h / +24h / +168h`).
  - **San Francisco Fire/EMS (`Exp 59`):** `https://data.sfgov.org/resource/nuek-vuh3.json` (4-battalion dispatch cohort, 60-minute turnaround SLA, checkpoints at `0m / +10m / +25m`).

---

## Adapt-1 API Configuration Recipes

All live runs target `ADAPT_API_BASE="https://rei-neuroadapt-api.reilabs.org/api/v1"` with `Authorization: Bearer ${ADAPT_API_KEY}` and clean teardown (`DELETE /domains/{domain_id}` verified by `GET -> HTTP 404`).

### Exp 16 — Deep-Archive 5-Asset Macro Domain (`W = 1024, bw = 0.20` or `W = 512, bw = 0.30`)

```json
{
  "domain_id": "part2-sec04-macro-19y",
  "session_id": "sess-sec04-macro",
  "schema": {
    "signals": ["vix_term", "z_eq", "z_y10", "z_vix", "z_oil", "z_usd"],
    "event_types": ["outcome"]
  },
  "learning": {
    "enabled": true,
    "transition": {
      "enabled": true,
      "event_types": ["outcome"],
      "input_paths": ["values.vix_term"],
      "targets": [
        {"path": "values.z_eq", "type": "number"},
        {"path": "values.z_y10", "type": "number"},
        {"path": "values.z_vix", "type": "number"},
        {"path": "values.z_oil", "type": "number"},
        {"path": "values.z_usd", "type": "number"}
      ],
      "required_support": 6,
      "max_samples": 1024,
      "max_distance": 1.0,
      "numeric_min_skill": 0.0,
      "futures": {
        "enabled": true,
        "conditioning": "kernel",
        "numeric_transport": "none",
        "bandwidth": 0.20,
        "max_support": 512,
        "max_branches": 128,
        "max_pending_forecasts": 1024
      }
    }
  }
}
```

> Switch `"max_samples": 512, "max_support": 512, "bandwidth": 0.30` for the best-ES sweep configuration (`2.3208`, `t = -3.78`). Avoid `bandwidth <= 0.10`, which starves tail support and fails 99% Expected Shortfall calibration.

### Exp 53 & NYC — Progressive Multi-Stage 311 Survival Domain (Shared `future_id`)

```json
{
  "domain_id": "part2-sec04-311-revisions",
  "session_id": "sess-sec04-311",
  "schema": {
    "signals": [
      "agency_grp", "service_grp", "source_grp", "entity_grp",
      "hour_sin", "log_ag_backlog_t0",
      "open_r1", "log_ag_backlog_r1",
      "open_r2", "log_ag_backlog_r2",
      "t_disposition_7d", "t_log_dur_sc"
    ],
    "event_types": ["outcome"]
  },
  "learning": {
    "enabled": true,
    "transition": {
      "enabled": true,
      "event_types": ["outcome"],
      "input_paths": [
        "values.agency_grp", "values.service_grp", "values.source_grp", "values.entity_grp",
        "values.hour_sin", "values.log_ag_backlog_t0",
        "values.open_r1", "values.log_ag_backlog_r1",
        "values.open_r2", "values.log_ag_backlog_r2"
      ],
      "targets": [
        {"path": "values.t_disposition_7d", "type": "categorical"},
        {"path": "values.t_log_dur_sc", "type": "number"}
      ],
      "required_support": 6,
      "max_samples": 1024,
      "max_distance": 1.0,
      "numeric_min_skill": 0.0,
      "futures": {
        "enabled": true,
        "conditioning": "kernel",
        "numeric_transport": "none",
        "bandwidth": 0.16,
        "max_support": 1024,
        "max_branches": 128,
        "max_pending_forecasts": 1024,
        "max_scores": 1024,
        "max_calibration_groups": 1024
      }
    }
  }
}
```

---

## Causal Score-Before-Ingest & Append-Only Log Protocol

1. **Non-Overlapping Causal Maturation (`Exp 16`):** Each weekly query at date $t$ sees only `vix_term` observed at market close on $t$ and outcomes matured strictly before $t$. The 5-trading-day outcome at $t + 5$ is scored first and only then ingested via `POST /events`.
2. **Asynchronous Event-Clock Maturation (`Exp 26` & `Exp 53`):** Tickets arrive continuously and remain open for up to 7 days (`168h` SLA cap). Queries at `rev 0`, `rev 1` (if still open), and `rev 2` (if still open) share one `context.metadata.future_id` and omit future stage signals not yet observed. The ticket's single terminal outcome (`t_disposition_7d`, `t_log_dur_sc`) is ingested only at `min(closed_ts, created_ts + 168h)`—never at intake.
3. **Zero Right-Censoring:** All Socrata pulls end at `2026-11-12T23:59:59`, leaving a `>= 10`-day maturation buffer before snapshot extraction so `0%` of 7-day SLA outcomes are right-censored.
4. **Strictly Append-Only Upstream Requirement:** Operational columns overwritten in place after dispatch (for example, `final_priority` in SF Fire/EMS, which changes on `24.69%` of dispatches) leak resolution state and must be excluded or reconstructed solely from immutable revision timestamps.

---

## Bandwidth and Survival Conditioning

1. **Bandwidth Tail Starvation (`Exp 16`, `bw = 0.10`):** Narrowing `bandwidth` to `0.10` shrinks tail support and fails the 99% Expected Shortfall Acerbi–Székely calibration test (`Z2 = -0.8709`, `p = 0.0378`). Keep `W >= 128` and `bw >= 0.20`.
2. **Soft Similarity Bleed (`Exp 54` Mozilla Bugzilla, offline boundary):** On Thunderbird bugs (`30d` SLA, `0h / +24h / +168h`), locking `bw = 0.20` across variable revision dimensionality lets closed-before-cutoff bugs bleed `~6–10 pp` of similarity weight into surviving open-case predictions (`Rev 2` predicted overrun `70.5%` vs. actual `81.0%`), causing Adapt-1 (`0.4063 / 0.2415`) to trail risk-set-filtered `QRF` (`0.3795 / 0.2118`), `HGBR` (`0.3818 / 0.2139`), and `OLHR` (`0.3884 / 0.2188`, all `p_H <= 0.0002`).
3. **High-Cardinality Cross-Station Dilution (`Exp 59` SF Fire/EMS, offline boundary):** On 4-battalion EMS dispatches (`60m` SLA, `0m / +10m / +25m`), similarity-based conditioning dilutes support when a new station launches, trailing `HGBR` and `OLHR` at `Rev 0` (`0.3476` vs. `0.3360 / 0.3371`) and `Rev 1` (`0.3284` vs. `0.3179 / 0.3219`) before reaching statistical parity at `Rev 2` (`0.3061` vs. `0.3059`, `p_H = 1.00`).
4. **Operational Scope:** Adapt-1 provides unified empty-start REST governance, stateful multi-revision `future_id` audit lineage, and progressive open-case conditioning across revision checkpoints without per-stage model-zoo retraining pipelines.

---

## Files & Independent Offline Recomputation

| File | Rows | Description |
| :--- | ---: | :--- |
| [`exp16_macro_w1024_bw020_trace.csv`](./exp16_macro_w1024_bw020_trace.csv) | 930 (890 eval) | Weekly 5D ES, FRTB quantiles/PIT, 2008 GFC recall (`W=1024, bw=0.20`) |
| [`exp16_macro_w0512_bw030_trace.csv`](./exp16_macro_w0512_bw030_trace.csv) | 930 (890 eval) | Best-ES weekly trace (`W=512, bw=0.30`, `ES=2.3208`) |
| [`exp16_sweep_summary.csv`](./exp16_sweep_summary.csv) | 16 | 12 Adapt-1 sweep arms + 4 GARCH/EWMA peers with 5D ES & FRTB 3/3 flags |
| [`nyc_311_lower_manhattan_revisions_trace.csv`](./nyc_311_lower_manhattan_revisions_trace.csv) | 4,330 | NYC Lower Manhattan 311 checkpoints (3,873 OOS, `n=826` 3-stage cohort) |
| [`chicago_311_post2026_09_revisions_trace.csv`](./chicago_311_post2026_09_revisions_trace.csv) | 2,621 | Chicago 311 post-Sep 2026 checkpoints (1,106 / 855 / 660) |
| [`sf_311_post2026_09_revisions_trace.csv`](./sf_311_post2026_09_revisions_trace.csv) | 4,375 | San Francisco 311 post-Sep 2026 checkpoints (1,722 / 1,545 / 1,108) |
| [`scorecard.json`](./scorecard.json) | — | Audited metrics, comparators, operating notes, column specs, assertions |

Run the snippet below from this directory (Python 3 standard library only) to independently recompute every headline metric:

```python
import csv
from collections import defaultdict

def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

# 1. Exp 16: 5D ES and 2008 archive weight recall
w1024 = [r for r in load_csv("exp16_macro_w1024_bw020_trace.csv") if r["is_eval_890"] == "1"]
w0512 = [r for r in load_csv("exp16_macro_w0512_bw030_trace.csv") if r["is_eval_890"] == "1"]
print("Exp16 W=1024 bw=0.20 ES:", round(sum(float(r["energy_score"]) for r in w1024) / len(w1024), 4))
print("Exp16 W=512  bw=0.30 ES:", round(sum(float(r["energy_score"]) for r in w0512) / len(w0512), 4))

r2017 = [r for r in w1024 if r["date"].startswith("2017")]
rcovid13 = [r for r in w1024 if 598 <= int(r["idx"]) <= 610]
print("2008 share ratio (2017):", round(sum(float(r["gfc_mass_cond"]) for r in r2017) / sum(float(r["gfc_mass_prior"]) for r in r2017), 2))
print("2008 share ratio (COVID 13w):", round(sum(float(r["gfc_mass_cond"]) for r in rcovid13) / sum(float(r["gfc_mass_prior"]) for r in rcovid13), 2))
for dt in ("2026-08-13", "2018-02-06", "2020-03-03", "2025-04-04"):
    row = next(r for r in w1024 if r["date"] == dt)
    print(f"2008 share ratio ({dt}):", round(float(row["gfc_ratio"]), 2))

# 2. NYC identical-ticket 3-stage trajectory (n=826) & overrun calibration
nyc_oos = [r for r in load_csv("nyc_311_lower_manhattan_revisions_trace.csv") if r["is_oos"] == "1"]
by_tkt = defaultdict(dict)
for r in nyc_oos:
    by_tkt[r["ticket_hash"]][int(r["rev"])] = r
c826 = [k for k, d in by_tkt.items() if {0, 1, 2}.issubset(d)]
print("NYC identical cohort n:", len(c826), "ES:", [round(sum(float(by_tkt[k][rv]["es_mixed"]) for k in c826) / len(c826), 4) for rv in (0, 1, 2)])
for rv in (0, 1, 2):
    sub = [r for r in nyc_oos if int(r["rev"]) == rv]
    print(f"NYC rev{rv} (n={len(sub)}) 7d overrun pred={100*sum(float(r['p_overrun_7d']) for r in sub)/len(sub):.2f}% act={100*sum(float(r['y_overrun_7d']) for r in sub)/len(sub):.2f}%")

# 3. Chicago & SF post-Sep 2026 ES by stage
for label, fn in [("Chicago", "chicago_311_post2026_09_revisions_trace.csv"), ("SF", "sf_311_post2026_09_revisions_trace.csv")]:
    rows = load_csv(fn)
    stats = [(rv, len(s), round(sum(float(r["es_mixed"]) for r in s) / len(s), 4)) for rv in (0, 1, 2) for s in [[r for r in rows if int(r["rev"]) == rv]]]
    print(label, stats)
```
