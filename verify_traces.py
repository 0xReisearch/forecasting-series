#!/usr/bin/env python3
"""Rei Labs — Adapt-1 Research Series (forecasting-series) Verifier.

Standalone, zero-dependency Python 3 (stdlib-only) verification suite.
Enforces five independent verification gates:
  1. SHA-256 cryptographic checksum verification against SHA256SUMS.txt
     (plus MANIFEST.json file inventory consistency).
  2. Strict RFC-8259 JSON/JSONL/CSV parsing (zero NaN/Infinity tokens) and
     PNG header / non-empty figure asset validation.
  3. Zero-leak clean-room hygiene audit (zero local paths, secrets/keys,
     internal hostnames, prohibited hardware claims, or internal engine leaks).
  4. Direct raw-trace metric reconciliation across all 6 Part 2 sections.
  5. GitHub file-size sanity check (no single file exceeding 25 MB).

Usage:
    python3 verify_traces.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

IGNORE_NAMES = {".DS_Store", "Thumbs.db"}
IGNORE_DIRS = {"__pycache__", ".git", "__MACOSX"}
TEXT_EXTS = {".md", ".txt", ".json", ".jsonl", ".csv", ".py", ".html", ".svg", ""}
BAD_CSV_TOKENS = {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MIN_PNG_BYTES = 5000
EXPECTED_FIGURES = (
    "figures/fig01_joint_tails_and_drift.png",
    "figures/fig02_online_probability_models.png",
    "figures/fig03_plasticity_and_recurrence.png",
    "figures/fig04_conditioning_across_time.png",
    "figures/fig05_belief_collapse_and_qmdp.png",
    "figures/fig06_autonomous_discovery.png",
)
MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB GitHub size guardrail

# Non-self-matching regexes so verify_traces.py itself scans clean.
# Public blog/API terms (linear_residual, numeric_ridge, episode_hash,
# warm_start, mlp_v2, log_loss, qmdp, omitted_probability) are explicitly allowed.
HYGIENE_PATTERNS = {
    "local_filesystem_path": re.compile(r"/[U]sers/|/[h]ome/", re.I),
    "raw_secret_or_key": re.compile(
        r"(?:s[k]-[A-Za-z0-9]{16,}|"
        r"B[e]arer\s+(?=[A-Za-z0-9._\-]*[0-9_\-.])[A-Za-z0-9._\-]{10,}|"
        r"B[e]arer\s+[A-Za-z0-9._\-]{20,}|"
        r"P[R]OD_KEY_[0-9]|A[K]IA[0-9A-Z]{16}|g[h]p_[A-Za-z0-9]{20,}|"
        r"\b[0-9a-f]{24}:[0-9a-f]{32}\b)",
        re.I,
    ),
    # Allow the public production host, but keep other deployment hosts private.
    "internal_api_hostname": re.compile(
        r"r[e]i-neuroadapt(?!-api[.]reilabs[.]org(?![A-Za-z0-9.-]))", re.I
    ),
    "prohibited_hardware_claim": re.compile(r"(?:zero|0|no)[- ]?[g]pu|c[p]u[- ]only", re.I),
    "internal_atom_or_recipe_leak": re.compile(
        r"r[e]ceipt\.atoms|has_r[e]ceipt_atoms|used_r[e]ceipt_atoms|"
        r"b[r]anch_wire_pts|b[r]anch_probs|"
        r"local[_ ]g[a]ussian|adapt1[_ ]t[w]in|t[w]in_es|"
        r"d[e]claration\.json|\bB(?:1[4-9]|2[0-2])\b|"
        r"r[e]verse.engineer|l[o]cal_50l|l[o]cal_25l|r[i]dgenw|g[a]ussiannw|"
        r"r[m]s-?l2|f[i]xed-u|h[a]t.?matrix|1 - h[_]\{?i|l[i]nspace\(0, nh|\b[t]win\b",
        re.I,
    ),
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reject_bad_const(tok: str) -> None:
    raise ValueError(f"Non-RFC-8259 numeric constant: {tok}")


def iter_files(root: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d not in IGNORE_DIRS)
        for fn in sorted(fns):
            if fn in IGNORE_NAMES:
                continue
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            out.append((rel, full))
    out.sort()
    return out


# ---------------------------------------------------------------------------
# Gate 1: SHA-256 Checksums + MANIFEST.json consistency
# ---------------------------------------------------------------------------

def check_sha256_and_manifest(root: str, files: Sequence[Tuple[str, str]]) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    sums_path = os.path.join(root, "SHA256SUMS.txt")
    if not os.path.isfile(sums_path):
        return False, ["Missing SHA256SUMS.txt"]

    expected: Dict[str, str] = {}
    with open(sums_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("  ", 1)
            if len(parts) != 2 or len(parts[0]) != 64:
                errs.append(f"Malformed SHA256SUMS.txt line {i}")
                continue
            expected[parts[1]] = parts[0]

    actual_map = {rel: full for rel, full in files if rel != "SHA256SUMS.txt"}
    if set(expected) != set(actual_map):
        missing = sorted(set(expected) - set(actual_map))
        extra = sorted(set(actual_map) - set(expected))
        if missing:
            errs.append(f"Missing files listed in SHA256SUMS.txt: {missing[:5]}")
        if extra:
            errs.append(f"Unindexed files absent from SHA256SUMS.txt: {extra[:5]}")

    computed_hashes: Dict[str, str] = {}
    for rel, full in actual_map.items():
        digest = sha256_file(full)
        computed_hashes[rel] = digest
        if rel in expected and digest != expected[rel]:
            errs.append(f"SHA-256 mismatch: {rel}")

    manifest_path = os.path.join(root, "MANIFEST.json")
    if not os.path.isfile(manifest_path):
        errs.append("Missing MANIFEST.json")
    else:
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f, parse_constant=reject_bad_const)
            entries = manifest.get("files", [])
            manifest_map = {e["path"]: e for e in entries}
            expected_manifest_paths = {rel for rel in actual_map if rel != "MANIFEST.json"}
            if set(manifest_map) != expected_manifest_paths:
                m_missing = sorted(expected_manifest_paths - set(manifest_map))
                m_extra = sorted(set(manifest_map) - expected_manifest_paths)
                if m_missing:
                    errs.append(f"MANIFEST.json missing files: {m_missing[:5]}")
                if m_extra:
                    errs.append(f"MANIFEST.json lists absent files: {m_extra[:5]}")
            for rel, entry in manifest_map.items():
                if rel in actual_map:
                    sz = os.path.getsize(actual_map[rel])
                    if entry.get("bytes") != sz:
                        errs.append(f"MANIFEST.json byte size mismatch for {rel}: {entry.get('bytes')} != {sz}")
                    if entry.get("sha256") != computed_hashes.get(rel):
                        errs.append(f"MANIFEST.json sha256 mismatch for {rel}")
        except Exception as exc:
            errs.append(f"MANIFEST.json validation error: {exc}")

    return len(errs) == 0, errs


# ---------------------------------------------------------------------------
# Gate 2: Strict RFC-8259 JSON/JSONL/CSV Syntax & PNG Asset Validation
# ---------------------------------------------------------------------------

def check_syntax(files: Sequence[Tuple[str, str]]) -> Tuple[bool, List[str], Tuple[int, int, int, int]]:
    errs: List[str] = []
    n_json = n_jsonl = n_csv = n_png = 0
    seen_pngs = set()
    for rel, full in files:
        ext = os.path.splitext(rel)[1].lower()
        try:
            if ext == ".json":
                with open(full, encoding="utf-8") as f:
                    json.load(f, parse_constant=reject_bad_const)
                n_json += 1
            elif ext == ".jsonl":
                with open(full, encoding="utf-8") as f:
                    for ln in f:
                        if ln.strip():
                            json.loads(ln, parse_constant=reject_bad_const)
                n_jsonl += 1
            elif ext == ".csv":
                with open(full, encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    hdr = next(reader)
                    if len(hdr) < 2:
                        errs.append(f"CSV header too narrow ({len(hdr)} cols): {rel}")
                        continue
                    ncols = len(hdr)
                    bad_cell = None
                    for row_idx, row in enumerate(reader, 2):
                        if len(row) != ncols:
                            errs.append(f"CSV column count mismatch in {rel} line {row_idx}: {len(row)} != {ncols}")
                            break
                        for cell in row:
                            if cell.strip().lower() in BAD_CSV_TOKENS:
                                bad_cell = (row_idx, cell)
                                break
                        if bad_cell is not None:
                            errs.append(f"Non-finite token {bad_cell[1]!r} in {rel} line {bad_cell[0]}")
                            break
                n_csv += 1
            elif ext == ".png":
                sz = os.path.getsize(full)
                with open(full, "rb") as f:
                    hdr = f.read(16)
                if not hdr.startswith(PNG_MAGIC) or hdr[12:16] != b"IHDR" or sz < MIN_PNG_BYTES:
                    errs.append(f"Invalid or empty PNG asset ({sz:,} bytes): {rel}")
                seen_pngs.add(rel)
                n_png += 1
        except Exception as exc:
            errs.append(f"Parse error in {rel}: {exc}")
    for exp_fig in EXPECTED_FIGURES:
        if exp_fig not in seen_pngs:
            errs.append(f"Missing expected diagnostic figure: {exp_fig}")
    return len(errs) == 0, errs, (n_json, n_jsonl, n_csv, n_png)


# ---------------------------------------------------------------------------
# Gate 3: Clean-Room Hygiene Audit
# ---------------------------------------------------------------------------

def check_hygiene(files: Sequence[Tuple[str, str]]) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    for rel, full in files:
        ext = os.path.splitext(rel)[1].lower()
        if ext not in TEXT_EXTS and not os.path.basename(rel).startswith("."):
            continue
        with open(full, encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        for name, rx in HYGIENE_PATTERNS.items():
            m = rx.search(txt)
            if m:
                errs.append(f"Hygiene leak [{name}] in {rel}: {m.group(0)!r}")
    return len(errs) == 0, errs


# ---------------------------------------------------------------------------
# Gate 4: Direct Raw-Trace Metric Reconciliation (Sections 01..06)
# ---------------------------------------------------------------------------

def _read_csv_rows(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        hdr = list(reader.fieldnames or [])
        rows = list(reader)
    return hdr, rows


def _mean(vals: Sequence[float]) -> float:
    return sum(vals) / len(vals)


def _clip_prob(p: float) -> float:
    return min(1.0 - 1e-6, max(1e-6, p))


def _brier(probs: Sequence[float], ys: Sequence[float]) -> float:
    return sum((_clip_prob(p) - y) ** 2 for p, y in zip(probs, ys)) / len(ys)


def _logloss(probs: Sequence[float], ys: Sequence[float]) -> float:
    tot = 0.0
    for p, y in zip(probs, ys):
        pc = _clip_prob(p)
        tot += -(y * math.log(pc) + (1.0 - y) * math.log(1.0 - pc))
    return tot / len(ys)


def _ece_10bin(probs: Sequence[float], ys: Sequence[float]) -> float:
    bins_p: List[List[float]] = [[] for _ in range(10)]
    bins_y: List[List[float]] = [[] for _ in range(10)]
    for p, y in zip(probs, ys):
        pc = _clip_prob(p)
        b = min(9, int(pc * 10.0))
        bins_p[b].append(pc)
        bins_y[b].append(y)
    n = len(ys)
    ece = 0.0
    for bp, by in zip(bins_p, bins_y):
        if bp:
            ece += (len(bp) / n) * abs(_mean(bp) - _mean(by))
    return ece


def _auroc(probs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = sorted((_clip_prob(p), int(round(y))) for p, y in zip(probs, ys))
    n_pos = sum(y for _, y in pairs)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = 0.0
    i = 0
    n = len(pairs)
    while i < n:
        j = i + 1
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        pos_in_tie = sum(pairs[k][1] for k in range(i, j))
        rank_sum_pos += pos_in_tie * avg_rank
        i = j
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _prob_arrays(rows: Sequence[Dict[str, str]], p_col: str, y_col: str) -> Tuple[List[float], List[float]]:
    return [float(r[p_col]) for r in rows], [float(r[y_col]) for r in rows]


def check_trace_reconciliation(root: str) -> Tuple[bool, List[str], List[str]]:
    errs: List[str] = []
    checks: List[str] = []
    part2 = os.path.join(root, "part-2-forecasting-through-drift")

    def assert_close(label: str, got: float, exp: float, tol: float = 1e-4) -> None:
        if not math.isfinite(got) or abs(got - exp) > tol:
            errs.append(f"Reconciliation mismatch [{label}]: got={got}, exp={exp} (tol={tol})")
        else:
            checks.append(f"{label}: {got:.6f} == {exp:.6f}")

    def assert_eq(label: str, got, exp) -> None:
        if got != exp:
            errs.append(f"Reconciliation mismatch [{label}]: got={got!r}, exp={exp!r}")
        else:
            checks.append(f"{label}: {got!r} == {exp!r}")

    def assert_true(label: str, cond: bool, detail: str = "") -> None:
        if not cond:
            errs.append(f"Reconciliation failed [{label}]: {detail}")
        else:
            checks.append(f"{label}: OK ({detail})" if detail else f"{label}: OK")

    # =========================================================================
    # SECTION 01: Joint tails and drift
    # =========================================================================
    s1 = os.path.join(part2, "01_joint_tails_and_drift")

    hdr98a, r98a = _read_csv_rows(os.path.join(s1, "exp98a_smard_power_oos_trace.csv"))
    assert_eq("S01_Exp98A_rows", len(r98a), 450)
    es_linres = _mean([float(r["es_adapt1_linres"]) for r in r98a])
    assert_close("S01_Exp98A_Adapt1_LinRes_ES4D", es_linres, 58.762184, tol=5e-3)
    assert_close("S01_Exp98A_BestGJR_ES4D", _mean([float(r["es_ar1_gjr_fhs_r128_r21"]) for r in r98a]), 63.0849, tol=5e-3)
    gjr_es_cols = [c for c in hdr98a if c.startswith("es_") and "gjr" in c]
    assert_eq("S01_Exp98A_GJR_count", len(gjr_es_cols), 12)
    gjr_es_means = [_mean([float(r[c]) for r in r98a]) for c in gjr_es_cols]
    assert_true("S01_Exp98A_beats_12of12_GJR", all(es_linres < m for m in gjr_es_means), f"58.76 < min_GJR={min(gjr_es_means):.2f}")
    assert_close("S01_Exp98A_Adapt1_Kernel_ES4D", _mean([float(r["es_adapt1_kernel"]) for r in r98a]), 65.2555, tol=5e-3)
    assert_close("S01_Exp98A_QHGBR_ES4D", _mean([float(r["es_q_hgbr"]) for r in r98a]), 65.9960, tol=5e-3)
    assert_close("S01_Exp98A_LocalLinRes_ES4D", _mean([float(r["es_local_linres_nw"]) for r in r98a]), 57.7976, tol=5e-3)
    assert_close("S01_Exp98A_Desk1_Adapt1_LinRes", _mean([float(r["desk1_adapt1_linres"]) for r in r98a]), 71.8259, tol=2e-2)
    assert_close("S01_Exp98A_Desk1_LocalLinRes", _mean([float(r["desk1_local_linres_nw"]) for r in r98a]), 71.98, tol=2e-2)
    gjr_desk1_cols = [c for c in hdr98a if c.startswith("desk1_") and "gjr" in c]
    gjr_desk1_means = [_mean([float(r[c]) for r in r98a]) for c in gjr_desk1_cols]
    assert_close("S01_Exp98A_Desk1_GJR_min", min(gjr_desk1_means), 63.20, tol=2e-2)
    assert_close("S01_Exp98A_Desk1_GJR_max", max(gjr_desk1_means), 67.44, tol=2e-2)
    assert_eq("S01_Exp98A_VaR99_Adapt1_LinRes", sum(int(float(r["var99_br_adapt1_linres"])) for r in r98a), 11)
    assert_eq("S01_Exp98A_VaR99_LocalLinRes", sum(int(float(r["var99_br_local_linres_nw"])) for r in r98a), 15)
    assert_eq("S01_Exp98A_VaR99_QHGBR", sum(int(float(r["var99_br_q_hgbr"])) for r in r98a), 30)
    assert_eq("S01_Exp98A_VaR99_GJR_R512", sum(int(float(r["var99_br_gjr_fhs_r512_r21"])) for r in r98a), 6)
    gjr_br_cols = [c for c in hdr98a if c.startswith("var99_br_") and "gjr" in c]
    gjr_br_sorted = sorted(sum(int(float(r[c])) for r in r98a) for c in gjr_br_cols)
    assert_eq("S01_Exp98A_VaR99_12GJR_profile", gjr_br_sorted, [6, 10, 10, 10, 10, 10, 10, 10, 10, 14, 18, 21])

    hdr79b, r79b = _read_csv_rows(os.path.join(s1, "exp79b_crypto_cascade_oos_trace.csv"))
    assert_eq("S01_Exp79B_rows", len(r79b), 1149)
    es5d_cols = [c for c in hdr79b if c.startswith("es5d_")]
    assert_eq("S01_Exp79B_ES5D_model_count", len(es5d_cols), 12)
    es5d_means = {c: _mean([float(r[c]) for r in r79b]) for c in es5d_cols}
    assert_close("S01_Exp79B_Adapt1_ES5D", es5d_means["es5d_adapt1"], 0.696984, tol=5e-4)
    assert_true("S01_Exp79B_Adapt1_rank1_of_12", es5d_means["es5d_adapt1"] == min(es5d_means.values()), f"min={min(es5d_means.values()):.6f}")
    assert_close("S01_Exp79B_Adapt1_FinalEq", float(r79b[-1]["eq5_adapt1"]), 0.08808, tol=5e-4)
    assert_close("S01_Exp79B_Spot1x_FinalEq", float(r79b[-1]["eq5_static_1x_spot"]), 0.43466, tol=5e-4)
    assert_close("S01_Exp79B_EWMA_FinalEq", float(r79b[-1]["eq5_ewma"]), 0.08969, tol=5e-4)
    assert_close("S01_Exp79B_DCCGARCH_FinalEq", float(r79b[-1]["eq5_dcc_garch"]), 0.08217, tol=5e-4)
    assert_eq("S01_Exp79B_QHGBR_Ruined", float(r79b[-1]["eq5_q_hgbr"]), 0.0)
    oct_crash_rows = [r for r in r79b if r["is_oct2025_crash_step"] == "1"]
    assert_eq("S01_Exp79B_OctCrash_step_count", len(oct_crash_rows), 1)
    assert_close("S01_Exp79B_OctCrash_Adapt1_Lev", float(oct_crash_rows[0]["lev_adapt1"]), 0.0792, tol=5e-4)
    eq_series = [float(r["eq5_adapt1"]) for r in r79b]
    win_eqs = [float(r["eq5_adapt1"]) for r in r79b if r["is_oct2025_crash_window"] == "1"]
    oct_dd_pct = 100.0 * (min(win_eqs) / max([1.0] + eq_series[:91]) - 1.0)
    assert_close("S01_Exp79B_Oct2025_DD_pct", oct_dd_pct, -28.372, tol=5e-2)

    _, r90c = _read_csv_rows(os.path.join(s1, "exp90c_longctx_tsfm_survival_summary.csv"))
    assert_eq("S01_Exp90C_rows", len(r90c), 45)
    raw_tsfm = [r for r in r90c if r["is_raw_tsfm"] == "1"]
    assert_eq("S01_Exp90C_RawTSFM_total", len(raw_tsfm), 32)
    assert_eq("S01_Exp90C_RawTSFM_ruined_32of32", sum(1 for r in raw_tsfm if r["survived_5bps"] == "0"), 32)
    raw_2048 = [r for r in raw_tsfm if r["context_length"] == "2048"]
    assert_eq("S01_Exp90C_RawTSFM_ctx2048_ruined_8of8", sum(1 for r in raw_2048 if r["survived_5bps"] == "0"), 8)

    hdr89b, r89b = _read_csv_rows(os.path.join(s1, "exp89b_eu_dunkelflaute_oos_trace.csv"))
    assert_eq("S01_Exp89B_rows", len(r89b), 323)
    es89_cols = [c for c in hdr89b if c.startswith("es_")]
    assert_eq("S01_Exp89B_model_count", len(es89_cols), 25)
    es89_means = {c: _mean([float(r[c]) for r in r89b]) for c in es89_cols}
    assert_close("S01_Exp89B_AirbagC2_ES", es89_means["es_adapt1_airbag_c2"], 0.771524, tol=2e-4)
    assert_close("S01_Exp89B_AirbagDiscC2_ES", es89_means["es_adapt1_airbag_disc_chronos2"], 0.769808, tol=2e-4)
    assert_true(
        "S01_Exp89B_AirbagDiscC2_rank1_of_25",
        es89_means["es_adapt1_airbag_disc_chronos2"] == min(es89_means.values()),
        f"min={min(es89_means.values()):.6f}",
    )
    es89_ranked = sorted(es89_means.items(), key=lambda kv: kv[1])
    assert_eq("S01_Exp89B_AirbagC2_rank2_of_25", es89_ranked[1][0], "es_adapt1_airbag_c2")
    assert_close(
        "S01_Exp89B_AirbagDiscC2_first150_ES",
        _mean([float(r["es_adapt1_airbag_disc_chronos2"]) for r in r89b[:150]]),
        0.786943,
        tol=2e-4,
    )
    assert_close("S01_Exp89B_VoLL_AirbagDiscC2_MEUR", _mean([float(r["cost20x_adapt1_airbag_disc_chronos2"]) for r in r89b]) * 365.0 / 1e6, 991.8725, tol=0.05)
    assert_eq("S01_Exp89B_Outages_AirbagDiscC2", sum(int(float(r["outage20x_adapt1_airbag_disc_chronos2"])) for r in r89b), 7)
    assert_eq("S01_Exp89B_VaR95_AirbagDiscC2", sum(int(float(r["var95_br_adapt1_airbag_disc_chronos2"])) for r in r89b), 19)
    assert_close("S01_Exp89B_VaR95_pct_AirbagDiscC2", 100.0 * _mean([float(r["var95_br_adapt1_airbag_disc_chronos2"]) for r in r89b]), 5.88, tol=0.02)
    assert_close("S01_Exp89B_VoLL_AirbagC2_MEUR", _mean([float(r["cost20x_adapt1_airbag_c2_eur"]) for r in r89b]) * 365.0 / 1e6, 1000.7915, tol=0.05)
    assert_close("S01_Exp89B_VoLL_RawT3_MEUR", _mean([float(r["cost20x_t3_ctx1024_cov_gc_eur"]) for r in r89b]) * 365.0 / 1e6, 1811.3850, tol=0.05)
    assert_close("S01_Exp89B_VoLL_AirbagT3_MEUR", _mean([float(r["cost20x_adapt1_airbag_t3_eur"]) for r in r89b]) * 365.0 / 1e6, 1111.6362, tol=0.05)
    assert_eq("S01_Exp89B_Outages_RawT3", sum(int(float(r["outage20x_t3_ctx1024_cov_gc"])) for r in r89b), 47)
    assert_eq("S01_Exp89B_Outages_AirbagT3", sum(int(float(r["outage20x_adapt1_airbag_t3"])) for r in r89b), 9)
    assert_eq("S01_Exp89B_Outages_RawC2", sum(int(float(r["outage20x_c2_ctx1024_cov_gc"])) for r in r89b), 14)
    assert_eq("S01_Exp89B_Outages_AirbagC2", sum(int(float(r["outage20x_adapt1_airbag_c2"])) for r in r89b), 7)

    # =========================================================================
    # SECTION 02: Online probability models
    # =========================================================================
    s2 = os.path.join(part2, "02_online_probability_models")

    _, r103a = _read_csv_rows(os.path.join(s2, "exp103a_crypto_funding_oos_trace.csv"))
    assert_eq("S02_Exp103A_rows", len(r103a), 16077)
    p, y = _prob_arrays(r103a, "Adapt1_CUDA_MLPv2_FixEp_EpHash_W4608", "y_neg_funding")
    assert_close("S02_Exp103A_EpHash_Full_Brier", _brier(p, y), 0.12108, tol=5e-5)
    assert_close("S02_Exp103A_EpHash_Full_ECE", _ece_10bin(p, y), 0.01091, tol=5e-5)
    r103a_2026 = [r for r in r103a if r["ts_utc"] >= "2026-01-01"]
    p26, y26 = _prob_arrays(r103a_2026, "Adapt1_CUDA_MLPv2_FixEp_EpHash_W4608", "y_neg_funding")
    assert_close("S02_Exp103A_EpHash_Y2026_Brier", _brier(p26, y26), 0.14814, tol=5e-5)
    assert_close("S02_Exp103A_EpHash_Y2026_LogLoss", _logloss(p26, y26), 0.45387, tol=5e-5)
    assert_close("S02_Exp103A_EpHash_Y2026_AUROC", _auroc(p26, y26), 0.8453, tol=1e-4)
    r103a_sep = [r for r in r103a if r["ts_utc"] >= "2026-09-01"]
    psep, ysep = _prob_arrays(r103a_sep, "Adapt1_CUDA_MLPv2_FixEp_EpHash_W4608", "y_neg_funding")
    assert_close("S02_Exp103A_EpHash_PostSep2026_Brier", _brier(psep, ysep), 0.13721, tol=5e-5)
    assert_close("S02_Exp103A_HGBR_W1536_Brier", _brier(*_prob_arrays(r103a, "Online_HGBR_Cls_Pooled_W1536", "y_neg_funding")), 0.12049, tol=5e-5)
    assert_close("S02_Exp103A_ExtraTrees_W1536_ECE", _ece_10bin(*_prob_arrays(r103a, "ExtraTrees_Cls_Pooled_W1536", "y_neg_funding")), 0.02634, tol=5e-5)
    assert_close("S02_Exp103A_LogReg_W1536_ECE", _ece_10bin(*_prob_arrays(r103a, "LogisticRegression_L2_Pooled_W1536", "y_neg_funding")), 0.02773, tol=5e-5)
    assert_close("S02_Exp103A_AR1GJR_W1024_Brier", _brier(*_prob_arrays(r103a, "AR1_GJR_SignProb_W1024", "y_neg_funding")), 0.12633, tol=5e-5)
    assert_close("S02_Exp103A_Chron_W4608_Brier", _brier(*_prob_arrays(r103a, "Adapt1_CUDA_MLPv2_FixEp_Chron_W4608", "y_neg_funding")), 0.12862, tol=5e-5)
    assert_close("S02_Exp103A_Chron_W3072_Brier", _brier(*_prob_arrays(r103a, "Adapt1_CUDA_MLPv2_FixEp_Chron_W3072", "y_neg_funding")), 0.13266, tol=5e-5)

    hdr103b_lt, r103b_lt = _read_csv_rows(os.path.join(s2, "exp103b_macro_left_tail_oos_trace.csv"))
    assert_eq("S02_Exp103B_LT_rows", len(r103b_lt), 13856)
    lt_meta = {"date", "iso_week", "vix_inversion", "asset", "y_left_tail"}
    lt_models = [c for c in hdr103b_lt if c not in lt_meta]
    assert_eq("S02_Exp103B_LT_model_count", len(lt_models), 10)
    plt_c3, ylt = _prob_arrays(r103b_lt, "Adapt1_103B_C3_MLPv2_Compact_LogLoss", "y_left_tail")
    assert_close("S02_Exp103B_LT_Compact_Brier", _brier(plt_c3, ylt), 0.06445, tol=5e-5)
    assert_close("S02_Exp103B_LT_Compact_ECE", _ece_10bin(plt_c3, ylt), 0.00081, tol=5e-5)
    assert_close("S02_Exp103B_LT_Compact_AUROC", _auroc(plt_c3, ylt), 0.5688, tol=1e-4)
    assert_close("S02_Exp103B_LT_GJR_Brier", _brier(*_prob_arrays(r103b_lt, "Real_MLE_GJR_GARCH_TailProb", "y_left_tail")), 0.06430, tol=5e-5)
    plt_base, _ = _prob_arrays(r103b_lt, "Adapt1_CUDA_MLPv2_LeftTail", "y_left_tail")
    assert_close("S02_Exp103B_LT_Baseline_ECE", _ece_10bin(plt_base, ylt), 0.01510, tol=5e-5)
    assert_close("S02_Exp103B_LT_Baseline_AUROC", _auroc(plt_base, ylt), 0.5871, tol=1e-4)
    lt_eces = {m: _ece_10bin(*_prob_arrays(r103b_lt, m, "y_left_tail")) for m in lt_models}
    assert_true(
        "S02_Exp103B_LT_Compact_ECE_rank1_of_10",
        lt_eces["Adapt1_103B_C3_MLPv2_Compact_LogLoss"] == min(lt_eces.values()),
        f"min={min(lt_eces.values()):.6f}",
    )

    hdr103b_em, r103b_em = _read_csv_rows(os.path.join(s2, "exp103b_macro_extreme_move_oos_trace.csv"))
    assert_eq("S02_Exp103B_EM_rows", len(r103b_em), 13856)
    em_meta = {"date", "iso_week", "vix_inversion", "asset", "y_extreme_move"}
    em_models = [c for c in hdr103b_em if c not in em_meta]
    assert_eq("S02_Exp103B_EM_model_count", len(em_models), 10)
    pem, yem = _prob_arrays(r103b_em, "Adapt1_103B_C3_MLPv2_Compact_LogLoss", "y_extreme_move")
    assert_close("S02_Exp103B_EM_Compact_Brier", _brier(pem, yem), 0.06045, tol=5e-5)
    assert_close("S02_Exp103B_EM_Compact_ECE", _ece_10bin(pem, yem), 0.00402, tol=5e-5)
    assert_close("S02_Exp103B_EM_Compact_AUROC", _auroc(pem, yem), 0.6266, tol=1e-4)
    r103b_vix = [r for r in r103b_em if r["vix_inversion"] == "1"]
    pvix, yvix = _prob_arrays(r103b_vix, "Adapt1_103B_C3_MLPv2_Compact_LogLoss", "y_extreme_move")
    assert_close("S02_Exp103B_EM_VIXInv_Brier", _brier(pvix, yvix), 0.12048, tol=5e-5)
    assert_close("S02_Exp103B_EM_VIXInv_LogLoss", _logloss(pvix, yvix), 0.40562, tol=5e-5)
    r103b_26 = [r for r in r103b_em if r["date"] >= "2026-01-01"]
    pem26, yem26 = _prob_arrays(r103b_26, "Adapt1_103B_C3_MLPv2_Compact_LogLoss", "y_extreme_move")
    assert_close("S02_Exp103B_EM_Y2026_Brier", _brier(pem26, yem26), 0.06808, tol=5e-5)
    assert_close("S02_Exp103B_EM_Y2026_ECE", _ece_10bin(pem26, yem26), 0.00619, tol=5e-5)

    hdr102c, r102c = _read_csv_rows(os.path.join(s2, "exp102c_german_power_oos_trace.csv"))
    assert_eq("S02_Exp102C_rows", len(r102c), 1800)
    m102_meta = {"delivery_date", "iso_week", "block", "y_extreme"}
    m102_models = [c for c in hdr102c if c not in m102_meta]
    assert_eq("S02_Exp102C_model_count", len(m102_models), 15)
    assert_close("S02_Exp102C_Linear_Full_Brier", _brier(*_prob_arrays(r102c, "Adapt1_CUDA_Linear_PowerPool2048", "y_extreme")), 0.16078, tol=5e-5)
    r102c_eve = [r for r in r102c if r["block"] == "P_EVE"]
    assert_close("S02_Exp102C_Linear_PEVE_Brier", _brier(*_prob_arrays(r102c_eve, "Adapt1_CUDA_Linear_PowerPool2048", "y_extreme")), 0.14004, tol=5e-5)
    brier_102c_warm = _brier(*_prob_arrays(r102c, "Adapt1_CUDA_MLPv2_PowerPool2048", "y_extreme"))
    assert_close("S02_Exp102C_MLPv2_Full_Brier", brier_102c_warm, 0.18012, tol=5e-5)
    assert_close("S02_Exp102C_HGBR_Full_Brier", _brier(*_prob_arrays(r102c, "Online_HGBR_Cls_Pooled", "y_extreme")), 0.15627, tol=5e-5)
    p102_d8, y102 = _prob_arrays(r102c, "Adapt1_CUDA_MLPv2_Disc8_PowerPool2048", "y_extreme")
    brier_102c_disc8 = _brier(p102_d8, y102)
    auroc_102c_disc8 = _auroc(p102_d8, y102)
    assert_close("S02_Exp102C_Disc8_Full_Brier", brier_102c_disc8, 0.14933, tol=5e-5)
    assert_close("S02_Exp102C_Disc8_Full_LogLoss", _logloss(p102_d8, y102), 0.46342, tol=5e-5)
    assert_close("S02_Exp102C_Disc8_Full_AUROC", auroc_102c_disc8, 0.8440, tol=1e-4)
    assert_close("S02_Exp102C_Disc8_Full_ECE", _ece_10bin(p102_d8, y102), 0.05297, tol=5e-5)
    p102_d8_eve, y102_eve = _prob_arrays(r102c_eve, "Adapt1_CUDA_MLPv2_Disc8_PowerPool2048", "y_extreme")
    assert_close("S02_Exp102C_Disc8_PEVE_Brier", _brier(p102_d8_eve, y102_eve), 0.12937, tol=5e-5)
    assert_close("S02_Exp102C_Disc8_PEVE_LogLoss", _logloss(p102_d8_eve, y102_eve), 0.40908, tol=5e-5)
    assert_close("S02_Exp102C_Disc8_PEVE_ECE", _ece_10bin(p102_d8_eve, y102_eve), 0.04908, tol=5e-5)
    m102_briers = {m: _brier(*_prob_arrays(r102c, m, "y_extreme")) for m in m102_models}
    m102_lls = {m: _logloss(*_prob_arrays(r102c, m, "y_extreme")) for m in m102_models}
    m102_aurocs = {m: _auroc(*_prob_arrays(r102c, m, "y_extreme")) for m in m102_models}
    m102_eve_briers = {m: _brier(*_prob_arrays(r102c_eve, m, "y_extreme")) for m in m102_models}
    assert_true(
        "S02_Exp102C_Disc8_Full_Brier_rank1_of_15",
        m102_briers["Adapt1_CUDA_MLPv2_Disc8_PowerPool2048"] == min(m102_briers.values()),
        f"min={min(m102_briers.values()):.5f}",
    )
    assert_true(
        "S02_Exp102C_Disc8_Full_LogLoss_rank1_of_15",
        m102_lls["Adapt1_CUDA_MLPv2_Disc8_PowerPool2048"] == min(m102_lls.values()),
        f"min={min(m102_lls.values()):.5f}",
    )
    assert_true(
        "S02_Exp102C_Disc8_Full_AUROC_rank1_of_15",
        m102_aurocs["Adapt1_CUDA_MLPv2_Disc8_PowerPool2048"] == max(m102_aurocs.values()),
        f"max={max(m102_aurocs.values()):.4f}",
    )
    assert_true(
        "S02_Exp102C_Disc8_PEVE_Brier_rank1_of_15",
        m102_eve_briers["Adapt1_CUDA_MLPv2_Disc8_PowerPool2048"] == min(m102_eve_briers.values()),
        f"min={min(m102_eve_briers.values()):.5f}",
    )
    assert_close("S02_Exp102C_HGBR_Disc8_Full_Brier", _brier(*_prob_arrays(r102c, "Online_HGBR_Cls_Pooled_Disc8", "y_extreme")), 0.15431, tol=5e-5)
    assert_close("S02_Exp102C_HGBR_Disc8_PEVE_Brier", _brier(*_prob_arrays(r102c_eve, "Online_HGBR_Cls_Pooled_Disc8", "y_extreme")), 0.13916, tol=5e-5)
    assert_close("S02_Exp102C_ExtraTrees_Disc8_Full_Brier", _brier(*_prob_arrays(r102c, "ExtraTrees_Cls_Pooled_Disc8", "y_extreme")), 0.17588, tol=5e-5)
    assert_close("S02_Exp102C_LogReg_Disc8_Full_Brier", _brier(*_prob_arrays(r102c, "LogisticRegression_L2_Pooled_Disc8", "y_extreme")), 0.18996, tol=5e-5)

    _, r100a = _read_csv_rows(os.path.join(s2, "exp100a_install_lockout_blocks.csv"))
    inst_rows = [r for r in r100a if r["update_policy"] == "expanding_install_true"]
    for tgt, exp_cnt in [
        ("abs_r", 36),
        ("rv5_fwd", 32),
        ("ev_peak_spike_200", 14),
        ("p_peak", 16),
        ("ev_neg_midday", 17),
    ]:
        got_cnt = sum(1 for r in inst_rows if r["target"] == tgt and r["train_status"] == "retained")
        assert_eq(f"S02_Exp100A_retained_{tgt}", got_cnt, exp_cnt)

    # =========================================================================
    # SECTION 03: Plasticity and recurrence
    # =========================================================================
    s3 = os.path.join(part2, "03_plasticity_and_recurrence")

    _, r96_n15 = _read_csv_rows(os.path.join(s3, "exp96b_n15_step_trace.csv"))
    assert_eq("S03_Exp96B_n15_rows", len(r96_n15), 4500)
    assert_eq("S03_Exp96B_t180_opt_actions", sum(int(r["adapt1_native_correct"]) for r in r96_n15 if int(r["step_t"]) == 180), 14)
    assert_eq("S03_Exp96B_t181_A1_reactivation", sum(int(r["adapt1_native_mlr_in_a1_pool"]) for r in r96_n15 if int(r["step_t"]) == 181), 15)
    a2_f10 = [r for r in r96_n15 if 180 <= int(r["step_t"]) < 190]
    assert_eq("S03_Exp96B_A2_first10_n", len(a2_f10), 150)
    assert_close("S03_Exp96B_A2_first10_Native", _mean([int(r["adapt1_native_correct"]) for r in a2_f10]), 0.733333, tol=1e-4)
    assert_close("S03_Exp96B_A2_first10_ADWIN_HGBR", _mean([int(r["adwin_hgbr_correct"]) for r in a2_f10]), 0.526667, tol=1e-4)
    assert_close("S03_Exp96B_A2_first10_NeuralUCB", _mean([int(r["online_neuralucb_correct"]) for r in a2_f10]), 0.460000, tol=1e-4)
    a2_full = [r for r in r96_n15 if r["phase"] == "A2"]
    assert_eq("S03_Exp96B_A2_full_n", len(a2_full), 900)
    assert_close("S03_Exp96B_A2_full_Native", _mean([int(r["adapt1_native_correct"]) for r in a2_full]), 0.786667, tol=1e-4)
    b2_last20 = [r for r in r96_n15 if 280 <= int(r["step_t"]) < 300]
    assert_eq("S03_Exp96B_B2star_last20_n", len(b2_last20), 300)
    assert_close("S03_Exp96B_B2star_last20_Native", _mean([int(r["adapt1_native_correct"]) for r in b2_last20]), 0.010000, tol=1e-4)
    lin_cb_cols = [
        "ewma_eps_greedy_correct",
        "sw_linucb_correct",
        "adwin_ridge_correct",
        "stationary_linucb_correct",
        "discounted_ts_correct",
        "exp3_s_correct",
    ]
    best_lincb = max(_mean([int(r[c]) for r in r96_n15]) for c in lin_cb_cols)
    assert_close("S03_Exp96B_BestLinearCB_overall", best_lincb, 0.292667, tol=1e-4)

    _, r96_abl = _read_csv_rows(os.path.join(s3, "exp96b_live_ablations_step_trace.csv"))
    assert_eq("S03_Exp96B_ablations_rows", len(r96_abl), 9000)
    for arm, exp_val in [
        ("adapt1_lb_native", 0.7875),
        ("adapt1_lb_off", 0.5750),
        ("adapt1_lb_policy_reset", 0.2375),
        ("adapt1_lb_delayed_d4", 0.6500),
    ]:
        sub = [int(r["is_correct"]) for r in r96_abl if r["arm"] == arm and 180 <= int(r["step_t"]) < 190]
        assert_close(f"S03_Exp96B_n8_A2f10_{arm}", _mean(sub), exp_val, tol=1e-4)
    b2_rst = [int(r["is_correct"]) for r in r96_abl if r["arm"] == "adapt1_lb_policy_reset" and 280 <= int(r["step_t"]) < 300]
    assert_close("S03_Exp96B_n8_B2star_policy_reset", _mean(b2_rst), 0.0125, tol=1e-4)

    _, r96_sum = _read_csv_rows(os.path.join(s3, "exp96b_per_seed_summary.csv"))
    assert_eq("S03_Exp96B_summary_rows", len(r96_sum), 234)
    n15_nat_sum = [float(r["a2_first10_acc"]) for r in r96_sum if r["cohort"] == "n15_primary" and r["model"] == "Adapt1_LatentBelief_Native"]
    assert_eq("S03_Exp96B_summary_n15_seeds", len(n15_nat_sum), 15)
    assert_close("S03_Exp96B_summary_n15_A2f10", _mean(n15_nat_sum), 0.733333, tol=1e-4)

    _, r79a = _read_csv_rows(os.path.join(s3, "exp79a_spy_tlt_flip_trace.csv"))
    assert_eq("S03_Exp79A_rows", len(r79a), 439)
    assert_close("S03_Exp79A_Smooth_ES4D", _mean([float(r["adapt1_smooth_es4d"]) for r in r79a]), 1.345254, tol=2e-4)
    assert_close("S03_Exp79A_Smooth_MaxDD_pct", min(float(r["adapt1_smooth_dd5_pct"]) for r in r79a), -19.6134, tol=0.05)
    assert_close("S03_Exp79A_Smooth_NetRet_pct", (float(r79a[-1]["adapt1_smooth_eq5"]) - 1.0) * 100.0, -2.2489, tol=0.05)
    assert_eq("S03_Exp79A_Smooth_VaR99_misses", sum(int(r["adapt1_smooth_var99_miss"]) for r in r79a), 5)
    assert_close("S03_Exp79A_Flush21d_ES4D", _mean([float(r["adapt1_flush21d_es4d"]) for r in r79a]), 1.367069, tol=2e-4)
    assert_close("S03_Exp79A_Flush21d_NetRet_pct", (float(r79a[-1]["adapt1_flush21d_eq5"]) - 1.0) * 100.0, -5.9468, tol=0.05)
    assert_eq("S03_Exp79A_Flush21d_rotations", sum(int(r["adapt1_flush21d_rotated_after_step"]) for r in r79a), 8)
    assert_close("S03_Exp79A_Static6040_MaxDD_pct", min(float(r["static_60_40_dd5_pct"]) for r in r79a), -28.7363, tol=0.05)
    assert_close("S03_Exp79A_QHGBR_ES4D", _mean([float(r["q_hgbr_online_es4d"]) for r in r79a]), 1.361788, tol=2e-4)
    assert_close("S03_Exp79A_QHGBR_MaxDD_pct", min(float(r["q_hgbr_online_dd5_pct"]) for r in r79a), -22.8593, tol=0.05)
    assert_close("S03_Exp79A_Chronos2_ES4D", _mean([float(r["chronos2_es4d"]) for r in r79a]), 1.362511, tol=2e-4)
    assert_close("S03_Exp79A_Chronos2_MaxDD_pct", min(float(r["chronos2_dd5_pct"]) for r in r79a), -21.5788, tol=0.05)

    # =========================================================================
    # SECTION 04: Conditioning across time
    # =========================================================================
    s4 = os.path.join(part2, "04_conditioning_across_time")

    _, r16_1024 = _read_csv_rows(os.path.join(s4, "exp16_macro_w1024_bw020_trace.csv"))
    e16_1024 = [r for r in r16_1024 if r["is_eval_890"] == "1"]
    assert_eq("S04_Exp16_w1024_eval_n", len(e16_1024), 890)
    assert_close("S04_Exp16_w1024_bw020_ES5D", _mean([float(r["energy_score"]) for r in e16_1024]), 2.322632, tol=1e-4)
    r2017 = [r for r in e16_1024 if r["date"].startswith("2017")]
    assert_close(
        "S04_Exp16_GFC_ratio_2017",
        sum(float(r["gfc_mass_cond"]) for r in r2017) / sum(float(r["gfc_mass_prior"]) for r in r2017),
        0.6671,
        tol=1e-3,
    )
    by_date_16 = {r["date"]: r for r in e16_1024}
    assert_close("S04_Exp16_GFC_ratio_2026_08_13", float(by_date_16["2026-08-13"]["gfc_ratio"]), 0.4749, tol=1e-3)
    assert_close("S04_Exp16_GFC_ratio_2018_02_06", float(by_date_16["2018-02-06"]["gfc_ratio"]), 2.7034, tol=1e-3)
    assert_close("S04_Exp16_GFC_ratio_2020_03_03", float(by_date_16["2020-03-03"]["gfc_ratio"]), 2.3354, tol=1e-3)
    covid_13w = [r for r in e16_1024 if 598 <= int(r["idx"]) <= 610]
    assert_close(
        "S04_Exp16_GFC_ratio_COVID13w",
        sum(float(r["gfc_mass_cond"]) for r in covid_13w) / sum(float(r["gfc_mass_prior"]) for r in covid_13w),
        1.6185,
        tol=1e-3,
    )
    assert_close("S04_Exp16_GFC_ratio_2025_04_04", float(by_date_16["2025-04-04"]["gfc_ratio"]), 2.6212, tol=1e-3)

    _, r16_512 = _read_csv_rows(os.path.join(s4, "exp16_macro_w0512_bw030_trace.csv"))
    e16_512 = [r for r in r16_512 if r["is_eval_890"] == "1"]
    assert_eq("S04_Exp16_w0512_eval_n", len(e16_512), 890)
    assert_close("S04_Exp16_w0512_bw030_ES5D", _mean([float(r["energy_score"]) for r in e16_512]), 2.320846, tol=1e-4)

    _, r16_sw = _read_csv_rows(os.path.join(s4, "exp16_sweep_summary.csv"))
    prim9 = [r for r in r16_sw if r["in_primary_9_sweep"] == "1"]
    assert_eq("S04_Exp16_primary9_count", len(prim9), 9)
    assert_eq("S04_Exp16_primary9_FRTB_pass", sum(int(r["frtb_3of3_pass"]) for r in prim9), 9)
    assert_eq("S04_Exp16_primary9_sig_beat_DCC", sum(int(r["sig_beat_dcc_fhs_05"]) for r in prim9), 5)
    sw_by_id = {r["model_id"]: r for r in r16_sw}
    assert_eq("S04_Exp16_bw010_ES99_fail", int(sw_by_id["w1024_bw010_sup0512_kern"]["es99_pass"]), 0)
    assert_close("S04_Exp16_DCC_FHS_ES5D", float(sw_by_id["DCC-GARCH(1,1)-FHS"]["es_5d_full19y"]), 2.333647, tol=1e-4)

    _, r_nyc = _read_csv_rows(os.path.join(s4, "nyc_311_lower_manhattan_revisions_trace.csv"))
    _, r_chi = _read_csv_rows(os.path.join(s4, "chicago_311_post2026_09_revisions_trace.csv"))
    _, r_sf = _read_csv_rows(os.path.join(s4, "sf_311_post2026_09_revisions_trace.csv"))
    assert_eq("S04_311_NYC_rows", len(r_nyc), 4330)
    assert_eq("S04_311_CHI_rows", len(r_chi), 2621)
    assert_eq("S04_311_SF_rows", len(r_sf), 4375)
    assert_eq("S04_311_total_checkpoints", len(r_nyc) + len(r_chi) + len(r_sf), 11326)

    nyc_oos = [r for r in r_nyc if r["is_oos"] == "1"]
    nyc_by_tkt: Dict[str, Dict[int, Dict[str, str]]] = defaultdict(dict)
    for r in nyc_oos:
        nyc_by_tkt[r["ticket_hash"]][int(r["rev"])] = r
    ident_tkts = [d for d in nyc_by_tkt.values() if {0, 1, 2}.issubset(d.keys())]
    assert_eq("S04_NYC_identical_cohort_n", len(ident_tkts), 826)
    for rev, exp_es in [(0, 0.618733), (1, 0.553428), (2, 0.530845)]:
        assert_close(f"S04_NYC_ident_rev{rev}_ES", _mean([float(d[rev]["es_mixed"]) for d in ident_tkts]), exp_es, tol=2e-4)
    for rev, exp_n, exp_p, exp_y in [
        (0, 1947, 16.65, 17.15),
        (1, 1100, 30.27, 30.36),
        (2, 826, 40.72, 40.44),
    ]:
        sub = [r for r in nyc_oos if int(r["rev"]) == rev]
        assert_eq(f"S04_NYC_oos_rev{rev}_n", len(sub), exp_n)
        assert_close(f"S04_NYC_oos_rev{rev}_pred7d_pct", 100.0 * _mean([float(r["p_overrun_7d"]) for r in sub]), exp_p, tol=0.05)
        assert_close(f"S04_NYC_oos_rev{rev}_actual7d_pct", 100.0 * _mean([float(r["y_overrun_7d"]) for r in sub]), exp_y, tol=0.05)

    for rev, exp_n, exp_es in [(0, 1106, 0.471546), (1, 855, 0.400815), (2, 660, 0.257926)]:
        sub = [r for r in r_chi if int(r["rev"]) == rev]
        assert_eq(f"S04_CHI_rev{rev}_n", len(sub), exp_n)
        assert_close(f"S04_CHI_rev{rev}_ES", _mean([float(r["es_mixed"]) for r in sub]), exp_es, tol=2e-4)

    for rev, exp_n, exp_es in [(0, 1722, 0.645932), (1, 1545, 0.629789), (2, 1108, 0.560150)]:
        sub = [r for r in r_sf if int(r["rev"]) == rev]
        assert_eq(f"S04_SF_rev{rev}_n", len(sub), exp_n)
        assert_close(f"S04_SF_rev{rev}_ES", _mean([float(r["es_mixed"]) for r in sub]), exp_es, tol=2e-4)

    # =========================================================================
    # SECTION 05: Belief collapse
    # =========================================================================
    s5 = os.path.join(part2, "05_belief_collapse")

    _, r93a_ep = _read_csv_rows(os.path.join(s5, "exp93a_episodes.csv"))
    def _s5_ep_mean(rows, sys_name, col, arena=None):
        sub = [float(r[col]) for r in rows if r["system"] == sys_name and (arena is None or r["arena"] == arena)]
        return _mean(sub)

    assert_close("S05_Exp93A_FullTriad_reward", _s5_ep_mean(r93a_ep, "Live_Adapt1_FullTriad_Native", "cum_reward"), 2.253127, tol=1e-3)
    assert_close("S05_Exp93A_FullTriad_USD", _s5_ep_mean(r93a_ep, "Live_Adapt1_FullTriad_Native", "net_value_usd_1_50"), 236578.33, tol=2.0)
    assert_close("S05_Exp93A_PPO_0ep_reward", _s5_ep_mean(r93a_ep, "PPO_Factored_0ep", "cum_reward"), -5.091381, tol=1e-3)
    assert_close("S05_Exp93A_PPO_1000ep_reward", _s5_ep_mean(r93a_ep, "PPO_Factored_1000ep", "cum_reward"), 0.232381, tol=1e-3)
    assert_close("S05_Exp93A_PPO_5000ep_USD", _s5_ep_mean(r93a_ep, "PPO_Factored_5000ep", "net_value_usd_1_50"), 67255.0, tol=2.0)
    assert_close("S05_Exp93A_PPO_20000ep_reward", _s5_ep_mean(r93a_ep, "PPO_Factored_20000ep", "cum_reward"), 3.656562, tol=1e-3)
    assert_close("S05_Exp93A_ExactVoI_reward", _s5_ep_mean(r93a_ep, "Exact_2Step_VoI", "cum_reward"), 3.926419, tol=1e-3)
    assert_close("S05_Exp93A_UnauthoredBT_reward", _s5_ep_mean(r93a_ep, "Live_Adapt1_UnauthoredBT", "cum_reward"), -0.758143, tol=1e-3)

    _, r93a_st = _read_csv_rows(os.path.join(s5, "exp93a_live_steps.csv"))
    w93a_s6 = [float(r["full_world_count"]) for r in r93a_st if r["arm"] == "Live_Adapt1_FullTriad_Native" and r["step"] == "6" and r["full_world_count"] != ""]
    assert_close("S05_Exp93A_Step6_worlds_mean", _mean(w93a_s6), 3.333333, tol=1e-3)
    assert_close("S05_Exp93A_Step6_worlds_median", statistics.median(w93a_s6), 3.0, tol=1e-6)

    _, r93b_ep = _read_csv_rows(os.path.join(s5, "exp93b_episodes.csv"))
    assert_close("S05_Exp93B_QMDP_Clean_reward", _s5_ep_mean(r93b_ep, "Live_Adapt1_QMDP_Native", "cum_reward", arena="93B_clean"), -0.630783, tol=1e-3)
    assert_close("S05_Exp93B_QMDP_Flaky15_reward", _s5_ep_mean(r93b_ep, "Live_Adapt1_QMDP_Flaky15", "cum_reward", arena="93B_flaky15"), -0.254842, tol=1e-3)
    assert_close("S05_Exp93B_Exact1StepVoI_reward", _s5_ep_mean(r93b_ep, "Exact_1Step_VoI", "cum_reward", arena="93B_clean"), 1.420454, tol=1e-3)

    _, r93b_st = _read_csv_rows(os.path.join(s5, "exp93b_qmdp_live_steps.csv"))
    w93b_s6 = [float(r["full_world_count"]) for r in r93b_st if r["arena"] == "93B_clean" and r["arm"] == "Live_Adapt1_QMDP_Native" and r["step"] == "6" and r["full_world_count"] != ""]
    assert_close("S05_Exp93B_Step6_worlds_mean", _mean(w93b_s6), 95.033333, tol=1e-2)
    assert_close("S05_Exp93B_Step6_worlds_median", statistics.median(w93b_s6), 61.5, tol=1e-6)

    _, r93c_cp = _read_csv_rows(os.path.join(s5, "exp93c_campaigns.csv"))
    r93c_eb = [r for r in r93c_cp if r["system"] == "Live_Adapt1_FactorizedEB_Native"]
    assert_close("S05_Exp93C_FactorizedEB_reward", _mean([float(r["cum_reward"]) for r in r93c_eb]), 3.618613, tol=1e-3)
    assert_close("S05_Exp93C_FactorizedEB_USD", _mean([float(r["net_value_usd_1_50"]) for r in r93c_eb]), 385985.33, tol=2.0)
    assert_eq("S05_Exp93C_FactorizedEB_learned", sum(1 for r in r93c_eb if r["effect_learned"] == "True"), 30)

    _, r93_par = _read_csv_rows(os.path.join(s5, "exp93_prod_parity_steps.csv"))
    assert_eq("S05_Exp93_Parity_total", len(r93_par), 204)
    assert_eq("S05_Exp93_Parity_matches", sum(1 for r in r93_par if r["action_match"] == "True"), 204)
    ms_93b = sorted(float(r["prod_query_ms"]) for r in r93_par if r["arena"] == "93B_clean")
    ms_93a = sorted(float(r["prod_query_ms"]) for r in r93_par if r["arena"] == "93A")
    assert_close("S05_Exp93B_QMDP_p50_ms", ms_93b[len(ms_93b) // 2], 562.72, tol=0.1)
    assert_close("S05_Exp93A_Exact_p50_ms", ms_93a[len(ms_93a) // 2], 442.50, tol=0.1)

    # =========================================================================
    # SECTION 06 (Companion Audit): Autonomous Discovery
    # =========================================================================
    s6 = os.path.join(part2, "06_autonomous_discovery")
    _, r105 = _read_csv_rows(os.path.join(s6, "exp105_discovery_benchmark_summary.csv"))
    assert_eq("S06_Discovery_rows", len(r105), 15)
    by_bid = {r["benchmark_id"]: r for r in r105}
    for bid, exp_val, tol in [
        ("exp98a_td", 60.8357, 1e-3),
        ("exp98a_sd_peak", 0.8065, 1e-3),
        ("exp79b_td", 0.73399, 1e-4),
        ("exp89b_td", 0.769808, 1e-4),
        ("exp102c_td_val", 0.14933, 1e-4),
        ("exp102c_td_val_auroc", 0.8440, 1e-3),
        ("exp16_td", 2.56320, 1e-3),
        ("exp53_td_rev0_to_rev2", 0.28310, 1e-3),
        ("exp93c_emp_topo", 3.0924, 1e-3),
    ]:
        assert_close(f"S06_{bid}", float(by_bid[bid]["discovery_value"]), exp_val, tol=tol)
    assert_eq("S06_exp89b_td_slice_n", int(by_bid["exp89b_td"]["slice_n"]), 323)
    assert_close(
        "S06_exp89b_td_reconciles_S01_csv",
        float(by_bid["exp89b_td"]["discovery_value"]),
        es89_means["es_adapt1_airbag_disc_chronos2"],
        tol=1e-5,
    )
    assert_close(
        "S06_exp89b_authored_reconciles_S01_csv",
        float(by_bid["exp89b_td"]["authored_value"]),
        es89_means["es_adapt1_airbag_c2"],
        tol=1e-5,
    )
    assert_close(
        "S06_exp102c_brier_reconciles_S02_csv",
        float(by_bid["exp102c_td_val"]["discovery_value"]),
        brier_102c_disc8,
        tol=1e-5,
    )
    assert_close(
        "S06_exp102c_authored_reconciles_S02_csv",
        float(by_bid["exp102c_td_val"]["authored_value"]),
        brier_102c_warm,
        tol=1e-5,
    )
    assert_close(
        "S06_exp102c_auroc_reconciles_S02_csv",
        float(by_bid["exp102c_td_val_auroc"]["discovery_value"]),
        auroc_102c_disc8,
        tol=1e-4,
    )

    return len(errs) == 0, errs, checks


# ---------------------------------------------------------------------------
# Gate 5: GitHub File-Size Sanity Check
# ---------------------------------------------------------------------------

def check_file_sizes(files: Sequence[Tuple[str, str]]) -> Tuple[bool, List[str], int, Tuple[str, int]]:
    errs: List[str] = []
    total_bytes = 0
    largest = ("", 0)
    for rel, full in files:
        sz = os.path.getsize(full)
        total_bytes += sz
        if sz > largest[1]:
            largest = (rel, sz)
        if sz > MAX_FILE_BYTES:
            errs.append(f"File exceeds 25 MB GitHub guardrail ({sz:,} bytes): {rel}")
    return len(errs) == 0, errs, total_bytes, largest


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify Adapt-1 Research Series replication traces")
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--verbose", action="store_true", help="Print every individual reconciliation assertion.")
    args = ap.parse_args(list(argv) if argv is not None else None)
    root = os.path.abspath(args.root)
    files = iter_files(root)

    ok_sha, e_sha = check_sha256_and_manifest(root, files)
    ok_syn, e_syn, counts = check_syntax(files)
    ok_hyg, e_hyg = check_hygiene(files)
    ok_rec, e_rec, checks = check_trace_reconciliation(root)
    ok_sz, e_sz, total_bytes, largest = check_file_sizes(files)

    print(f"=== REI LABS FORECASTING-SERIES VERIFICATION ({len(files)} files, {total_bytes / (1024 * 1024):.2f} MB) ===")
    print(f"[{'PASS' if ok_sha else 'FAIL'}] 1. SHA-256 Checksums & MANIFEST.json ({len(files) - 1} hashed files)")
    print(f"[{'PASS' if ok_syn else 'FAIL'}] 2. Strict RFC-8259 Syntax & PNG Assets (json={counts[0]}, jsonl={counts[1]}, csv={counts[2]}, png={counts[3]})")
    print(f"[{'PASS' if ok_hyg else 'FAIL'}] 3. Zero-Leak Clean-Room Hygiene (0 paths/secrets/hosts/hardware/internal leaks)")
    print(f"[{'PASS' if ok_rec else 'FAIL'}] 4. Raw-Trace Metric Reconciliation ({len(checks)}/{len(checks) + len(e_rec)} assertions across Sections 01-06)")
    print(f"[{'PASS' if ok_sz else 'FAIL'}] 5. GitHub File-Size Guardrail (max={largest[1] / (1024 * 1024):.2f} MB < 25.00 MB; {largest[0]})")

    if args.verbose:
        for c in checks:
            print(f"       OK  {c}")

    all_errs = e_sha + e_syn + e_hyg + e_rec + e_sz
    if all_errs:
        print(f"\nFAILURES ({len(all_errs)} total):")
        for e in all_errs[:40]:
            print("  -", e)
        return 1

    print("\nALL 5 VERIFICATION GATES PASSED (0 DEFECTS).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
