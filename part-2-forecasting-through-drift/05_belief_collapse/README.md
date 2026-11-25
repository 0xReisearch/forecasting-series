# Section 05 — Belief Collapse Under Partial Observability

Replication artifacts, live/offline traces, and audited scorecard for Section 05 (**"Belief collapse"**) of Rei Labs' *Forecasting Through Drift* (Adapt-1 Research Series, Part 2).

> **Explicit Simulation & Provenance Notice:** All three POMDP arenas (`Exp 93A`, `Exp 93B`, `Exp 93C`) are **simulated closed-loop Kubernetes incident-triage environments**, not live production cluster fault injections. Service topologies (Online Boutique: 12 Kubernetes Deployments, 16 RPC edges), fault taxonomies (RCAEval: 202 indexed paths), and non-uniform component incident rates (233 archived Statuspage incidents split chronologically into 116 development calibration and 117 sealed out-of-sample records) are calibrated from archived real-world operational records; discrete fault axes, interactive remediation transitions, and asymmetric SLA dollar tiers (`$10k–$250k`) are design-authored around **100% live Adapt-1 API decisions** with zero client-side action overrides or commit guards.

---

## 1. Headline Findings & Direct Trace Reconciliation

| Exp | Arena & Strategy | Belief / Learning | Result (1:50 SLA) | Comparators |
|---|---|---|---|---|
| **93A** | 5×2×4 K8s (26 act, **120 worlds**, $10k–$250k, n=30); exact search (0 train) | 120 → 49.6 (s1) → 8.43 (s4) → 3.33 / **3.0 med** (s6, n=3) | **+2.253** (+$236,578 / +$237k; 3.30/5 ok; 23.6m DT) | Beats PPO ≤5k (-5.091 0ep, +0.232 1k, +0.641/+$67k 5k, p_H=2e-4), bandits (≤+0.232), unauthored (-0.758); trails PPO 20k (+3.657/+$384k), Exact VoI (+3.926/+$412k), POMCP (+3.777). |
| **93B** | 2×3×6 K8s (15 act, **78,480 worlds**, Clean 0% & Flaky 15%, n=30); QMDP (`max_worlds=64`) | 78,480 → 29,136 (s1) → 3,146 (s3) → **95.0 / 61.5 med** (s6) | Clean **-0.631** (-$100,925); Flaky **-0.255** (-$40,775); **0 504s** | Exact timed out (40,157 ms 504); QMDP 562.7 ms prod p50 (733.0 ms ref); 1-step full-obs undervalues probes vs Exact 1-Step VoI (+1.420 clean / +0.921 flaky). |
| **93-PROD** | Cross-deployment parity (10 93B-Clean + 5 93A seeds); Ref vs Prod | 140/140 (93B) + 64/64 (93A) steps | **204/204 (100.0%)** bit-exact; 15/15 traces; 0.0 diff | Prod p50: **562.72 ms** (~563 ms, 93B QMDP), **442.50 ms** (93A exact); 0 HTTP 504s; 2/2 DELETE→404. |
| **93C** | 6-node DAG, 4 non-bijective playbooks, 2×3 waves (n=30 = 60); factorized belief (topo=none, indep.) | **30/30** learned (max 2.37/4; 26/30 planned; W1 2.83/3, W2 1.87/3) | **+3.619** (+$385,985 / +$386k; 4.70/6 ok) | Beats PPO 0/200ep (-1.120, +0.323, p_H=2e-4), ties PPO 1k (+3.202, p_H=0.321), trails PPO 5k/20k (+5.204, +5.244) & Exact VoI (+5.236). |

---

## 2. Simulated K8s POMDP Environments & Authored vs. Unauthored Topology

### Arena 93A — Heterogeneous `5 × 2 × 4` Root-Cause Triage (`120` Prior Worlds)
- **Services & asymmetric simulated SLA tiers:** `paymentservice` (`$250k`), `checkoutservice` (`$150k`), `frontend` (`$80k`), `cartservice` (`$35k`), `currencyservice` (`$10k`), slot-shuffled per episode so no positional bias exists.
- **Fault state & runbooks:** Each service has `2` hidden binary fault axes (`{-1, +1}^2`). Four Kubernetes runbooks (`pod-cpu-throttle-reset`, `netem-delay-reroute`, `oom-heap-recycle`, `packet-loss-mesh-failover`) apply an unknown bijective permutation of axis transitions (`±2.0`) with heterogeneous execution times (`1.5, 2.5, 3.5, 5.0 min`) and probe costs (`$1.5k–$6.0k`), plus active SLA burn (`1.2%/min` at baseline `1:50` ratio; swept across `1:25`, `1:50`, `1:100`) on unresolved services.
- **Authored vs. unauthored contrast:** Supplying the declarative `effect_catalog` (`bijective`), `topology_model="factorized_conjunctive"`, and `objective_hypotheses` activates `finite_world_belief_search` on `30/30` episodes (`+2.253` / `+$236,578` from episode zero). Stripping that domain model (`Live_Adapt1_UnauthoredBT`: `topology_model="none"`, empty `effect_catalog`) drops mean return to **`-0.758`** (`-$79,605`, `p_H = 2.0e-4`), isolating the value of online belief collapse over declarative fault worlds.
- **Note on step-6 world-count sampling in 93A:** `POST /explain` telemetry was queried on all `30/30` episodes at steps `1` (`mean = 49.60`, `median = 72.0`) and `4` (`mean = 8.43`, `median = 6.0`), and on `3` representative episodes (`seeds 1000..1002`) at steps `2`, `3`, and `6` (`mean = 3.333`, `median = 3.0`, range `[3, 4]`). In Arena 93B-Clean, step `6` was polled on all `30/30` episodes (`mean = 95.033`, `median = 61.5`, range `[17, 392]`).

### Arena 93B — Scaling to `78,480` Prior Worlds (`2 × 3 × 6`) & the QMDP Boundary
- **State space:** Two cloud ingress/mesh clusters (`$200k` / `$120k`), `3` orthogonal fault layers (`compute_cpu_fault`, `network_delay_fault`, `storage_loss_fault` in `{-1, +1}^3`), and `6` cross-layer playbooks (`6! = 720` permutations across connected topologies = **`78,480` joint prior worlds**).
- **Timeout wall broken:** Exact belief search (`belief_planning_mode="exact"`) timed out after `40,157 ms` (`HTTP 504`) in the 3-axis regime. Switching to `belief_planning_mode="qmdp"` (`belief_max_worlds=64`, `belief_qmdp_max_worlds=64`) eliminates all timeouts (`0` HTTP 504s across `828` live reference queries and `140` live production parity queries; `562.72 ms` production p50, `732.96 ms` reference p50).
- **Documented QMDP boundary:** QMDP computes action values assuming full state observability will arrive after one step. In a `78,480`-world 3D POMDP where pure diagnostic probes have negative immediate reward and only pay off over multi-step information gain, QMDP systematically undervalues pure probes—scoring **`-0.631`** (`93B-Clean`) and **`-0.255`** (`93B-Flaky15`, `15%` transient probe timeouts) versus **`+1.420`** clean / **`+0.921`** flaky for local `Exact_1Step_VoI` (and **`+0.662`** / **`+0.156`** for `Live_Adapt1_UnauthoredBT`).

### Arena 93C — Topology-Free `factorized_effect_belief` (`6`-Node Service DAG)
- **No topology prior required:** Six Online Boutique services (`paymentservice`, `checkoutservice`, `frontend`, `cartservice`, `redis-cart`, `currencyservice`) with `4` **non-bijective** playbooks drawn independently from `{(-1,-1), (-1,0), (0,-1), (0,0)}` over `(blast, pressure)`. Each campaign runs two consecutive 3-node cascading waves under the same hidden playbook mapping.
- **Online signature learning:** With `topology_model="none"` and `effect_assignment="independent"`, Adapt-1 learns playbook effect signatures online on **`30/30` campaigns** (mean max `2.37/4` assigned signatures, `26/30` reaching `bap_status="planned"`), transferring knowledge from Wave 1 (`2.83/3` healthy) to Wave 2 (`1.87/3` healthy) to earn **`+3.619`** (`+$385,985`). It beats `PPO_Factored` at `0` and `200` training episodes (`p_H = 2.0e-4`), statistically ties `PPO_Factored_1000ep` (`+3.202`, `p_H = 0.321`), and trails `PPO_Factored_5000ep` (`+5.204`) / `20000ep` (`+5.244`) and offline `Exact_2Step_VoI` (`+5.236`).

---

## 3. Equation 1: Exogenous vs. Endogenous Control Fork & Latency Profile

Adapt-1 separates sequential decision problems along a single causal boundary (Equation 1 of the post):

1. **Exogenous control (`P(Y_{t+1} | X_t, a_t) = P(Y_{t+1} | X_t)`):** When candidate actions do not alter future environmental outcomes (e.g., portfolio sizing, hedging, risk caps), issue **one** joint forecast query per step and evaluate all candidate actions externally against the returned joint scenario distribution. In our cross-vertical exogenous control A/B (`Exp 24`, `3,625` scored steps), 1-cloud external scoring cut sequential API evaluations by **`26.98x` (`~27x`)** across a 9-knob risk frontier (`1,694` forecast calls vs. `45,702` per-action planning calls; `3.0x` for a single risk knob, `1,694` vs. `5,078`) while outperforming per-action native planning (`+12.61` vs. `+11.94` bps utility on Arena 1, `t_HAC = +2.88`).
2. **Endogenous POMDPs (`P(S_{t+1}, O_{t+1} | S_t, a_t)`):** Reserve Adapt-1's stateful `sequential.bound_transition` belief search (`finite_world_belief_search`, `finite_world_qmdp`, `factorized_effect_belief`) for problems like Arenas 93A–93C where actions actively probe hidden fault states or mutate service state.
3. **Measured live REST latency regimes:**
   - **Standard regime (`< 900 ms` p50):** Across all 10 live Exp 93 arms, median query latency spans `437.7–764.6 ms` (`442.5 ms` production p50 on 93A exact belief search; `562.7 ms` production p50 on 93B 78,480-world QMDP); standard rolling archives (`[0, 1024)` events in Exp 28 soak) run at `527.5 ms` median / `893.8 ms` p90.
   - **Deep archives & async value retrains (`~2 s`):** Long multi-thousand-event archives exhibit `+284.1 ms / 1k` post-saturation latency growth (Exp 28 soak), and asynchronous neural value-model retrains (`mlp_v2`) complete in **`2.016 s`** wall time (`Exp 84A`).

---

## 4. Adapt-1 API Configuration Recipe

Use environment variables `ADAPT_API_BASE=https://rei-neuroadapt-api.reilabs.org/api/v1` and `ADAPT_API_KEY` (`Authorization: Bearer $ADAPT_API_KEY`). Always clean up ephemeral domains with `DELETE /domains/{domain_id}` verified by `GET -> 404`.

```json
{
  "learning": {
    "enabled": true,
    "sequential": {
      "enabled": true,
      "episode_path": "metadata.episode_id",
      "step_path": "metadata.step",
      "next_context_path": "values.next_state",
      "reward_path": "values.step_reward",
      "terminal_path": "values.terminal",
      "discount": 0.95,
      "bound_transition": {
        "enabled": true,
        "state_selector_feature": "svc_slot",
        "action_selector_feature": "rb_slot",
        "belief_planning_mode": "exact",
        "belief_max_worlds": 256,
        "belief_information_weight": 0.85,
        "topology_model": "factorized_conjunctive",
        "effect_assignment": "bijective",
        "reversible_effects": true
      }
    }
  }
}
```

- **For 78,480-world 3-axis scale (Arena 93B):** set `"belief_planning_mode": "qmdp"`, `"belief_max_worlds": 64`, `"belief_qmdp_max_worlds": 64`, `"belief_planning_horizon": 3`.
- **For topology-free DAG signature learning (Arena 93C):** set `"topology_model": "none"`, `"effect_assignment": "independent"`, `"reversible_effects": false`, `"empirical_topology_inference": true`.

---

## 5. Planning Modes and Evaluation Results

1. **Slot-symmetry ceiling:** `bound_transition.terminal_reward_map` is shared across entity slots and cannot natively represent per-service asymmetric SLA tiers (`$10k–$250k`) or heterogeneous probe latencies (`1.5–5.0 min`), giving SLA-aware offline Exact VoI (`+3.926`), POMCP (`+3.777`), and 20k-episode PPO (`+3.657`) a structural ceiling advantage in 93A.
2. **Tie-break premature commits:** Under 100% pure native API execution (zero client-side guards), `finite_world_belief_search` still emits occasional premature commits on `selection:core_exploration_tie_break` (`50` premature commits across 30 episodes in 93A).
3. **QMDP probe undervaluation:** `finite_world_qmdp` solves the 40 s exact timeout wall with `0` HTTP 504s, but its 1-step full-observability approximation undervalues pure diagnostic probes (`-0.631` clean / `-0.255` flaky vs. `+1.420` exact VoI).
4. **Flaky-probe determinism:** Under `15%` transient probe timeouts (`93B-Flaky15`), Adapt-1 avoids zero-world crashes via graceful fallback, but lacks native probabilistic observation-noise likelihoods compared to `NoiseAware_Bayes_Oracle` (`+0.945`).
5. **Paired statistical comparisons:** `scorecard.json` contains all 4 arena contrast tables (`m = 19–20` Holm-Bonferroni family comparisons per arena), stratifying contrasts against online/bandit/RL peers separately from full-simulator upper-bound oracles (`20` contrasts: Exact VoI, POMCP, Bayes oracles), high-budget offline PPO (`3` contrasts at 5k–20k episodes vs. episode-zero Adapt-1), hand-coded SRE heuristic (`1`), and the `93B-Clean` QMDP probe-approximation regime (`11`), alongside `21` statistical ties.

---

## 6. Files & Independent Recomputation

| File | Rows | Description |
|---|---:|---|
| `exp93a_episodes.csv` | 630 | Episode returns, simulated SLA economics (`1:25/1:50/1:100`), health & downtime across 21 systems (3 live + 18 offline) × 30 seeds |
| `exp93a_live_steps.csv` | 1,304 | Step-level live trace for 3 Arena 93A Adapt-1 arms (world counts, latencies, strategies) |
| `exp93b_episodes.csv` | 1,230 | Episode results for `93B_clean` (20 systems) and `93B_flaky15` (21 systems) × 30 seeds |
| `exp93b_qmdp_live_steps.csv` | 1,581 | Step-level live trace for 4 Arena 93B Adapt-1 arms (78,480-world collapse, latencies, flake flags) |
| `exp93c_campaigns.csv` | 630 | Campaign-level 2-wave DAG results across 21 systems × 30 campaigns (`effect_learned`, `bap_planned`, Wave 1/2 health) |
| `exp93c_live_steps.csv` | 1,473 | Step-level live trace for 3 Arena 93C Adapt-1 arms (`effect_status`, `assigned_signatures`, wave transitions) |
| `exp93_prod_parity_steps.csv` | 204 | Paired step trace (140 `93B_clean` + 64 `93A`) verifying 204/204 bit-exact actions and prod p50 latencies |
| `exp93_prod_parity_summary.csv` | 15 | Per-seed paired parity summary (10 `93B_clean` + 5 `93A` seeds, 0.0 return/dollar diff) |
| `scorecard.json` | — | Audited metrics, PPO curves, Holm-Bonferroni contrasts, operating notes, and verification targets |

### Quick Local Recomputation (Python 3 Stdlib Only)

```python
import csv, statistics

def mean_col(path, pred, col):
    with open(path, encoding="utf-8") as f:
        vals = [float(r[col]) for r in csv.DictReader(f) if pred(r)]
    return sum(vals) / len(vals)

# Exp 93A headline metrics & PPO curve
print("93A Native return:", round(mean_col("exp93a_episodes.csv", lambda r: r["system"]=="Live_Adapt1_FullTriad_Native", "cum_reward"), 3))
print("93A Native net $:", round(mean_col("exp93a_episodes.csv", lambda r: r["system"]=="Live_Adapt1_FullTriad_Native", "net_value_usd_1_50"), 0))
print("93A PPO 0/1k/5k/20k:", [round(mean_col("exp93a_episodes.csv", lambda r, k=k: r["system"]==f"PPO_Factored_{k}ep", "cum_reward"), 3) for k in (0, 1000, 5000, 20000)])

# Exp 93B 78,480 -> 95.0 (median 61.5) world collapse at step 6
with open("exp93b_qmdp_live_steps.csv", encoding="utf-8") as f:
    w6 = [int(r["full_world_count"]) for r in csv.DictReader(f) if r["arena"]=="93B_clean" and r["arm"]=="Live_Adapt1_QMDP_Native" and r["step"]=="6" and r["full_world_count"]]
print("93B step-6 worlds mean/median:", round(statistics.mean(w6), 1), statistics.median(w6))

# Exp 93-PROD 204/204 bit-exact parity & p50 latencies (upper-median q_sorted[n//2])
with open("exp93_prod_parity_steps.csv", encoding="utf-8") as f:
    pr = list(csv.DictReader(f))
print("Parity agreement:", sum(r["action_match"]=="True" for r in pr), "/", len(pr))
for aid in ("93B_clean", "93A"):
    qs = sorted(float(r["prod_query_ms"]) for r in pr if r["arena"]==aid)
    print(f"PROD {aid} p50 ms:", qs[len(qs)//2])
```
