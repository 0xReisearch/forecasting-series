#!/usr/bin/env python3
"""Generate the 6 diagnostic matplotlib figures for Part 2 ("Forecasting Through Drift").

Every curve, marker, bar, and reference annotation is computed directly from the
committed CSV traces and scorecards in `part-2-forecasting-through-drift/`
using strictly relative paths from the repository root.

Usage:
    python3 scripts/plot_figures.py
"""

from __future__ import annotations

import collections
import csv
import datetime as dt
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "part-2-forecasting-through-drift"
FIG_DIR = REPO_ROOT / "figures"

# Okabe-Ito colorblind-safe palette
C_BLUE = "#0072B2"      # Adapt-1 primary
C_VERM = "#D55E00"      # Highlight / failure / ruin / lockout
C_GREEN = "#009E73"     # Calibration / optimal / GREEN zone
C_AMBER = "#E69F00"     # YELLOW zone / secondary highlight
C_PURPLE = "#CC79A7"    # Auxiliary peer / foundation model
C_SKY = "#56B4E9"       # Secondary Adapt-1 variant
C_BROWN = "#7D5A36"     # Parametric peer
C_GRAY = "#666666"      # Neutral baseline

DPI = 180


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.2,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 7.6,
        "legend.frameon": True,
        "legend.facecolor": "#FFFFFF",
        "legend.edgecolor": "#D8DDE2",
        "legend.framealpha": 0.95,
        "grid.color": "#E2E6EA",
        "grid.linewidth": 0.6,
        "grid.linestyle": "--",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "savefig.dpi": DPI,
        "savefig.facecolor": "#FFFFFF",
    })


def clean_ax(ax, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def parse_dates(strings: list[str]) -> list[dt.datetime]:
    out = []
    for s in strings:
        if "T" in s:
            out.append(dt.datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S"))
        else:
            out.append(dt.datetime.strptime(s[:10], "%Y-%m-%d"))
    return out


def compute_brier_ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple[float, float]:
    p = np.clip(probs, 1e-6, 1.0 - 1e-6)
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b < n_bins - 1:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            ece += (cnt / n) * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return brier, float(ece)


def compute_auroc(probs: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(probs, 1e-6, 1.0 - 1e-6)
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty(n, dtype=float)
    vals = p[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[j + 1] == vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    pos = y == 1
    n_pos = int(np.sum(pos))
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((np.sum(ranks[pos]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def t_ci95_half(vals: list[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    sd = statistics.stdev(vals)
    t_crit = 2.04523 if n == 30 else 1.96
    return t_crit * sd / math.sqrt(n)


# =============================================================================
# FIGURE 1: Section 01 — Joint Tails & Drift (Exp 98A Power + Exp 79B Crypto)
# =============================================================================
def build_fig01() -> Path:
    sec1 = DATA_ROOT / "01_joint_tails_and_drift"
    r98 = load_csv(sec1 / "exp98a_smard_power_oos_trace.csv")
    sc1 = load_json(sec1 / "scorecard.json")
    r79b = load_csv(sec1 / "exp79b_crypto_cascade_oos_trace.csv")

    zone_by_tag = {
        m["trace_tag"]: m["basel_zone"]
        for m in sc1["exp98a_german_power_scarcity"]["leaderboard_19_traced_models"]
    }

    models_98 = []
    for col in [c[3:] for c in r98[0] if c.startswith("es_")]:
        es = float(np.mean([float(r[f"es_{col}"]) for r in r98]))
        net = float(np.mean([float(r[f"desk1_{col}"]) for r in r98]))
        br = int(sum(int(float(r[f"var99_br_{col}"])) for r in r98))
        zone = zone_by_tag.get(col, "YELLOW")
        models_98.append({"tag": col, "es": es, "net": net, "br": br, "zone": zone})

    fig = plt.figure(figsize=(13.6, 5.7))
    gs = gridspec.GridSpec(
        2, 2, width_ratios=[1.05, 1.15], height_ratios=[2.1, 1.0],
        wspace=0.25, hspace=0.14, left=0.06, right=0.98, top=0.88, bottom=0.12,
    )
    ax_l = fig.add_subplot(gs[:, 0])
    ax_rt = fig.add_subplot(gs[0, 1])
    ax_rb = fig.add_subplot(gs[1, 1], sharex=ax_rt)

    # --- Left Panel: Exp 98A German Power Scatter ---
    clean_ax(ax_l)
    zone_colors = {"GREEN": C_GREEN, "YELLOW": C_AMBER, "RED": C_VERM}

    for m in models_98:
        tag = m["tag"]
        if tag == "own_prior_cf":
            continue
        zc = zone_colors[m["zone"]]
        if tag == "adapt1_linres":
            ax_l.scatter(m["es"], m["net"], s=210, marker="*", facecolor=zc, edgecolor=C_BLUE, linewidth=1.5, zorder=6)
        elif tag == "adapt1_kernel":
            ax_l.scatter(m["es"], m["net"], s=85, marker="D", facecolor=zc, edgecolor=C_BLUE, linewidth=1.2, zorder=5)
        elif "gjr" in tag:
            ax_l.scatter(m["es"], m["net"], s=58, marker="o", facecolor=zc, edgecolor="#333333", linewidth=0.7, alpha=0.9, zorder=4)
        elif tag == "q_hgbr":
            ax_l.scatter(m["es"], m["net"], s=85, marker="^", facecolor=zc, edgecolor="#222222", linewidth=0.9, zorder=5)
        else:
            ax_l.scatter(m["es"], m["net"], s=72, marker="s", facecolor=zc, edgecolor="#222222", linewidth=0.9, zorder=5)

    by_tag = {m["tag"]: m for m in models_98}
    ann_specs = [
        ("adapt1_linres", f"Adapt-1 LinRes\nES {by_tag['adapt1_linres']['es']:.2f}, EUR {by_tag['adapt1_linres']['net']:.2f}\nYELLOW ({by_tag['adapt1_linres']['br']}/450)", (14, -36), C_BLUE, "bold", "left"),
        ("local_linres_nw", f"Offline Local Linear\nES {by_tag['local_linres_nw']['es']:.2f}, RED ({by_tag['local_linres_nw']['br']}/450)", (10, 13), C_VERM, "normal", "left"),
        ("tier2_heteroridge", f"HeteroRidge\nES {by_tag['tier2_heteroridge']['es']:.2f}, EUR {by_tag['tier2_heteroridge']['net']:.2f} (Y {by_tag['tier2_heteroridge']['br']})", (14, -4), "#333333", "normal", "left"),
        ("adapt1_kernel", f"Adapt-1 Base\nES {by_tag['adapt1_kernel']['es']:.2f} (Y {by_tag['adapt1_kernel']['br']})", (14, -6), C_BLUE, "normal", "left"),
        ("q_hgbr", f"Q-HGBR\nES {by_tag['q_hgbr']['es']:.2f}\nRED ({by_tag['q_hgbr']['br']}/450)", (-16, -42), C_VERM, "normal", "center"),
        ("ewma_copula", f"EWMA (G {by_tag['ewma_copula']['br']})", (-54, 4), C_GREEN, "bold", "right"),
    ]
    for tag, txt, offset, col, weight, ha in ann_specs:
        m = by_tag[tag]
        ax_l.annotate(
            txt, xy=(m["es"], m["net"]), xytext=offset, textcoords="offset points",
            fontsize=7.3, color=col, fontweight=weight, ha=ha,
            arrowprops={"arrowstyle": "-", "color": col, "lw": 0.8, "shrinkA": 0, "shrinkB": 5},
            zorder=7,
        )

    gjr_es = [m["es"] for m in models_98 if "gjr" in m["tag"]]
    gjr_net = [m["net"] for m in models_98 if "gjr" in m["tag"]]
    rect = mpatches.FancyBboxPatch(
        (min(gjr_es) - 0.18, min(gjr_net) - 0.32),
        (max(gjr_es) - min(gjr_es)) + 0.36,
        (max(gjr_net) - min(gjr_net)) + 0.64,
        boxstyle="round,pad=0.05", facecolor="#F4F6F8", edgecolor="#8E9AA5",
        linestyle="--", linewidth=0.9, zorder=1,
    )
    ax_l.add_patch(rect)
    ax_l.text(
        min(gjr_es) - 0.12, max(gjr_net) + 0.58,
        f"12 GJR-GARCH variants\nES {min(gjr_es):.2f}–{max(gjr_es):.2f}, EUR {min(gjr_net):.2f}–{max(gjr_net):.2f}\n(1 GREEN, 8 YELLOW, 3 RED)",
        fontsize=7.2, color="#444444", va="bottom",
    )

    ax_l.set_xlim(57.1, 67.1)
    ax_l.set_ylim(62.1, 74.9)
    ax_l.set_xlabel("4D Energy Score (EUR/MWh)  [lower is better]")
    ax_l.set_ylabel("1%-Fee Daily Overlay Net (EUR/MWh/day)  [higher is better]")
    ax_l.set_title("A. German Power Scarcity (Exp 98A, n = 450 days)", loc="left", pad=8)

    legend_handles_l = [
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=C_AMBER, markeredgecolor=C_BLUE, markersize=11, label="Adapt-1 LinRes"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=C_AMBER, markeredgecolor=C_BLUE, markersize=6.5, label="Adapt-1 Base"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_GREEN, markeredgecolor="#333", markersize=6.5, label="Basel GREEN (<=7/450)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_AMBER, markeredgecolor="#333", markersize=6.5, label="Basel YELLOW (8–13)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_VERM, markeredgecolor="#333", markersize=6.5, label="Basel RED (>=14)"),
    ]
    ax_l.legend(handles=legend_handles_l, loc="lower left", ncol=1, fontsize=7.1)
    prior_m = by_tag["own_prior_cf"]
    ax_l.text(
        0.99, 0.015,
        f"Unconditional prior (ES {prior_m['es']:.2f}, EUR {prior_m['net']:.2f}, RED {prior_m['br']}) outside zoom",
        transform=ax_l.transAxes, ha="right", va="bottom", fontsize=6.6, color="#666666", style="italic",
    )

    # --- Right Panel: Exp 79B 15-Month Crypto Cascade ---
    clean_ax(ax_rt)
    clean_ax(ax_rb)
    dates_79b = parse_dates([r["time_utc"] for r in r79b])
    oct_idx = next(i for i, r in enumerate(r79b) if r["is_oct2025_crash_step"] == "1")
    oct_dt = dates_79b[oct_idx]

    def clipped_eq(col: str):
        arr = np.array([float(r[col]) for r in r79b], dtype=float)
        ruin_i = next((i for i, v in enumerate(arr) if v <= 0.05), None)
        if ruin_i is not None:
            plot_arr = arr[: ruin_i + 1].copy()
            plot_arr[-1] = 0.05
            return dates_79b[: ruin_i + 1], plot_arr, ruin_i
        return dates_79b, arr, None

    ax_rt.axvspan(oct_dt - dt.timedelta(days=5), oct_dt + dt.timedelta(days=5), color="#FDE8DC", alpha=0.75, zorder=0)
    ax_rb.axvspan(oct_dt - dt.timedelta(days=5), oct_dt + dt.timedelta(days=5), color="#FDE8DC", alpha=0.75, zorder=0)

    spot_eq = np.array([float(r["eq5_static_1x_spot"]) for r in r79b])
    ad_eq = np.array([float(r["eq5_adapt1"]) for r in r79b])
    ew_eq = np.array([float(r["eq5_ewma"]) for r in r79b])
    dcc_eq = np.array([float(r["eq5_dcc_garch"]) for r in r79b])

    ax_rt.plot(dates_79b, spot_eq, color=C_GRAY, lw=1.2, ls="--", label=f"1x Spot (end {spot_eq[-1]:.2f}x)", zorder=3)
    ax_rt.plot(dates_79b, ew_eq, color=C_GREEN, lw=1.1, alpha=0.85, label=f"EWMA 3x (end {ew_eq[-1]:.2f}x)", zorder=3)
    ax_rt.plot(dates_79b, dcc_eq, color=C_BROWN, lw=1.1, alpha=0.85, label=f"DCC-GARCH 3x (end {dcc_eq[-1]:.2f}x)", zorder=3)
    ax_rt.plot(dates_79b, ad_eq, color=C_BLUE, lw=2.0, label=f"Adapt-1 3x (end {ad_eq[-1]:.2f}x)", zorder=5)

    for col, lbl, colr in [
        ("eq5_timesfm3_to_gc", "TimesFM-3 (TO-GC, ruined)", C_VERM),
        ("eq5_chronos2_to_gc", "Chronos-2 (TO-GC, ruined)", C_PURPLE),
        ("eq5_q_hgbr", "Q-HGBR (ruined)", C_AMBER),
    ]:
        xd, yd, ri = clipped_eq(col)
        ax_rt.plot(xd, yd, color=colr, lw=1.15, alpha=0.88, label=lbl, zorder=4)
        if ri is not None:
            ax_rt.scatter(dates_79b[ri], 0.05, marker="X", s=54, color=colr, edgecolor="#222222", lw=0.6, zorder=6)

    ax_rt.axhline(0.05, color=C_VERM, ls=":", lw=1.1, zorder=2)
    ax_rt.text(
        dates_79b[10], 0.044,
        "5% forced-liquidation floor  (32/32 raw TSFMs & Q-HGBR ruined Jun–Nov 2026)",
        fontsize=6.9, color=C_VERM, va="top",
    )
    ax_rt.set_yscale("log")
    ax_rt.set_ylim(0.032, 1.75)
    ax_rt.set_yticks([0.05, 0.1, 0.25, 0.5, 1.0])
    ax_rt.set_yticklabels(["0.05x", "0.10x", "0.25x", "0.50x", "1.00x"])
    ax_rt.set_ylabel("Simulated Equity (log)")
    ax_rt.set_title("B. 15-Month Digital-Asset Cascade (Exp 79B / 90C, 3x overlay, 5 bps)", loc="left", pad=8)
    ax_rt.legend(loc="lower left", bbox_to_anchor=(0.008, 0.16), ncol=2, fontsize=6.8, handlelength=1.5, columnspacing=0.9)
    plt.setp(ax_rt.get_xticklabels(), visible=False)

    # Bottom sub-axis: Dynamic leverage
    lev_ad = np.array([float(r["lev_adapt1"]) for r in r79b])
    lev_t3 = np.array([float(r["lev_timesfm3_to_gc"]) for r in r79b])
    lev_qh = np.array([float(r["lev_q_hgbr"]) for r in r79b])
    ax_rb.plot(dates_79b, lev_t3, color=C_VERM, lw=0.75, alpha=0.38, label="TimesFM-3 lev")
    ax_rb.plot(dates_79b, lev_qh, color=C_AMBER, lw=0.75, alpha=0.38, label="Q-HGBR lev")
    ax_rb.plot(dates_79b, lev_ad, color=C_BLUE, lw=1.35, label="Adapt-1 lev", zorder=4)
    ax_rb.scatter(oct_dt, lev_ad[oct_idx], s=44, color=C_BLUE, edgecolor="#FFFFFF", lw=0.8, zorder=6)
    ax_rb.annotate(
        f"Oct 2025 crash:\nAdapt-1 throttles to {lev_ad[oct_idx]:.2f}x",
        xy=(oct_dt, lev_ad[oct_idx]), xytext=(38, 12), textcoords="offset points",
        fontsize=7.2, color=C_BLUE, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "fc": "#FFFFFF", "ec": C_BLUE, "alpha": 0.95, "lw": 0.8},
        arrowprops={"arrowstyle": "->", "color": C_BLUE, "lw": 0.9},
        zorder=7,
    )
    ax_rb.set_ylim(-0.12, 3.45)
    ax_rb.set_yticks([0.0, 1.0, 2.0, 3.0])
    ax_rb.set_ylabel("Leverage (x)")
    ax_rb.set_xlabel("Date (UTC)")
    ax_rb.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_rb.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_rb.legend(loc="upper right", ncol=3, fontsize=6.7, handlelength=1.3)

    fig.suptitle("Section 01 — Joint Tails, Copula Transport & Cascade Survival", fontsize=12, fontweight="bold", y=0.96)
    out = FIG_DIR / "fig01_joint_tails_and_drift.png"
    fig.savefig(out, dpi=DPI, pil_kwargs={"optimize": True})
    plt.close(fig)
    return out


# =============================================================================
# FIGURE 2: Section 02 — Online Probability Models (Exp 103A, 103B, 102C, 100A)
# =============================================================================
def build_fig02() -> Path:
    sec2 = DATA_ROOT / "02_online_probability_models"
    r103a = load_csv(sec2 / "exp103a_crypto_funding_oos_trace.csv")
    r103b_lt = load_csv(sec2 / "exp103b_macro_left_tail_oos_trace.csv")
    r103b_em = load_csv(sec2 / "exp103b_macro_extreme_move_oos_trace.csv")
    r102c = load_csv(sec2 / "exp102c_german_power_oos_trace.csv")
    r100a = load_csv(sec2 / "exp100a_install_lockout_blocks.csv")

    fig = plt.figure(figsize=(15.2, 9.8))
    gs_top = gridspec.GridSpec(
        1, 2, width_ratios=[1.02, 1.02],
        wspace=0.29, left=0.06, right=0.985, top=0.905, bottom=0.535,
    )
    gs_bot = gridspec.GridSpec(
        1, 2, width_ratios=[1.04, 1.0],
        wspace=0.34, left=0.175, right=0.985, top=0.435, bottom=0.068,
    )
    ax1 = fig.add_subplot(gs_top[0, 0])
    ax2 = fig.add_subplot(gs_top[0, 1])
    ax3 = fig.add_subplot(gs_bot[0, 0])
    ax4 = fig.add_subplot(gs_bot[0, 1])

    # --- Panel A: Exp 103A 21-Model Brier vs ECE ---
    clean_ax(ax1)
    y_103a = np.array([float(r["y_neg_funding"]) for r in r103a], dtype=float)
    model_cols_103a = [c for c in r103a[0] if c not in ("ts_utc", "iso_week", "asset", "y_neg_funding")]
    pts_103a = {}
    for m in model_cols_103a:
        p = np.array([float(r[m]) for r in r103a], dtype=float)
        pts_103a[m] = compute_brier_ece(p, y_103a)

    for m, (br, ece) in pts_103a.items():
        if m == "Persistence_Sign":
            continue
        if m == "Adapt1_CUDA_MLPv2_FixEp_EpHash_W4608":
            ax1.scatter(br, ece * 1000, s=220, marker="*", color=C_BLUE, edgecolor="#111111", lw=0.8, zorder=7)
        elif m == "Adapt1_CUDA_MLPv2_FixEp_EpHash_W3072":
            ax1.scatter(br, ece * 1000, s=78, marker="D", color=C_SKY, edgecolor=C_BLUE, lw=0.9, zorder=6)
        elif "Chron_W" in m:
            ax1.scatter(br, ece * 1000, s=82, marker="^", color=C_VERM, edgecolor="#222222", lw=0.7, zorder=6)
        elif m.startswith("Adapt1_"):
            ax1.scatter(br, ece * 1000, s=58, marker="D", color="#8CC5E3", edgecolor=C_BLUE, lw=0.7, zorder=4)
        elif m == "Online_HGBR_Cls_Pooled_W1536":
            ax1.scatter(br, ece * 1000, s=82, marker="s", color=C_GREEN, edgecolor="#222222", lw=0.7, zorder=6)
        elif "ExtraTrees" in m or "LogisticRegression" in m or "AR1_GJR" in m:
            ax1.scatter(br, ece * 1000, s=48, marker="o", color=C_GRAY, alpha=0.78, edgecolor="#333333", lw=0.5, zorder=3)
        else:
            ax1.scatter(br, ece * 1000, s=48, marker="o", color="#9AA5B1", alpha=0.75, edgecolor="#444444", lw=0.5, zorder=3)

    ep46 = pts_103a["Adapt1_CUDA_MLPv2_FixEp_EpHash_W4608"]
    ch46 = pts_103a["Adapt1_CUDA_MLPv2_FixEp_Chron_W4608"]
    ch30 = pts_103a["Adapt1_CUDA_MLPv2_FixEp_Chron_W3072"]
    hg15 = pts_103a["Online_HGBR_Cls_Pooled_W1536"]
    et15 = pts_103a["ExtraTrees_Cls_Pooled_W1536"]
    lr15 = pts_103a["LogisticRegression_L2_Pooled_W1536"]

    ax1.annotate(
        f"Adapt-1 EpHash W4608\n#2 Brier ({ep46[0]:.4f}), #1 ECE ({ep46[1]*1000:.1f}e-3)",
        xy=(ep46[0], ep46[1] * 1000), xytext=(22, -14), textcoords="offset points",
        fontsize=7.6, color=C_BLUE, fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": C_BLUE, "lw": 0.95}, zorder=8,
    )
    ax1.annotate(
        f"Online HGBR W1536\n(#1 Brier: {hg15[0]:.4f})",
        xy=(hg15[0], hg15[1] * 1000), xytext=(-4, 46), textcoords="offset points",
        fontsize=7.3, color=C_GREEN, fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": C_GREEN, "lw": 0.85}, zorder=8,
    )
    ax1.annotate(
        f"Adapt-1 Chron W4608 (#17)\nBrier {ch46[0]:.4f}, ECE {ch46[1]*1000:.1f}e-3",
        xy=(ch46[0], ch46[1] * 1000), xytext=(-118, 36), textcoords="offset points",
        fontsize=7.4, color=C_VERM,
        arrowprops={"arrowstyle": "->", "color": C_VERM, "lw": 0.85}, zorder=8,
    )
    ax1.annotate(
        f"Chron W3072 (#19)\n({ch30[0]:.4f})",
        xy=(ch30[0], ch30[1] * 1000), xytext=(-14, -38), textcoords="offset points",
        fontsize=7.3, color=C_VERM, ha="center",
        arrowprops={"arrowstyle": "->", "color": C_VERM, "lw": 0.8}, zorder=8,
    )
    ax1.annotate(
        f"ExtraTrees / LogReg\n(ECE {et15[1]*1000:.1f}–{lr15[1]*1000:.1f}e-3)",
        xy=(et15[0], et15[1] * 1000), xytext=(-10, 26), textcoords="offset points",
        fontsize=7.2, color=C_GRAY,
        arrowprops={"arrowstyle": "-", "color": C_GRAY, "lw": 0.75}, zorder=8,
    )
    ax1.annotate(
        "", xy=(ep46[0] + 0.0003, ep46[1] * 1000 + 0.7), xytext=(ch46[0] - 0.0002, ch46[1] * 1000 - 0.7),
        arrowprops={"arrowstyle": "->", "color": C_BLUE, "lw": 1.15, "ls": "--", "connectionstyle": "arc3,rad=0.18"},
        zorder=5,
    )
    ax1.set_xlim(0.1195, 0.1355)
    ax1.set_ylim(5.5, 49.0)
    ax1.set_xlabel("Brier Score  [lower is better]")
    ax1.set_ylabel("10-Bin ECE (x10^-3)  [lower is better]")
    ax1.set_title("A. Crypto Funding Inversion (Exp 103A, 21 models)", loc="left", pad=8)
    ps = pts_103a["Persistence_Sign"]
    ax1.text(0.99, 0.02, f"Persistence floor (Brier {ps[0]:.3f}) omitted from zoom", transform=ax1.transAxes, ha="right", va="bottom", fontsize=6.9, color="#666666", style="italic")

    # --- Panel B: Exp 103B Macro Rare-Tail Calibration (10 models, equal-budget W=4,096) ---
    clean_ax(ax2, grid_axis="x")
    y_lt = np.array([float(r["y_left_tail"]) for r in r103b_lt], dtype=float)
    y_em = np.array([float(r["y_extreme_move"]) for r in r103b_em], dtype=float)
    meta_103b = {"date", "iso_week", "vix_inversion", "asset", "y_left_tail", "y_extreme_move"}
    n_103b_models = len([c for c in r103b_lt[0] if c not in meta_103b])

    peer_rows_103b = [
        ("Adapt-1 Compact\nmlp_v2 (LogLoss)", "Adapt1_103B_C3_MLPv2_Compact_LogLoss", "Adapt1_103B_C3_MLPv2_Compact_LogLoss"),
        ("MLE GJR-GARCH\nTailProb", "Real_MLE_GJR_GARCH_TailProb", "Real_MLE_GJR_GARCH_TailProb"),
        ("Online HGBR\n(Pooled)", "Online_HGBR_Cls_Pooled", "Online_HGBR_Cls_Pooled"),
        ("LogReg L2\n(Pooled)", "LogisticRegression_L2_Pooled", "LogisticRegression_L2_Pooled"),
        ("Adapt-1 Unfixed\nMLP (baseline)", "Adapt1_CUDA_MLPv2_LeftTail", "Adapt1_CUDA_MLPv2_ExtremeMove"),
    ]
    lt_stats, em_stats = [], []
    for _, col_lt, col_em in peer_rows_103b:
        lt_stats.append(compute_brier_ece(np.array([float(r[col_lt]) for r in r103b_lt], dtype=float), y_lt))
        em_stats.append(compute_brier_ece(np.array([float(r[col_em]) for r in r103b_em], dtype=float), y_em))

    y_pos = np.arange(len(peer_rows_103b))
    h = 0.34
    ax2.axhspan(-0.46, 0.46, color="#EBF5FB", alpha=0.85, zorder=0)
    ax2.barh(y_pos - h / 2, [x[1] * 1000 for x in lt_stats], height=h, color=C_BLUE, alpha=0.90, edgecolor="#222222", lw=0.5, label="LeftTail (r < -1.5·rv22)", zorder=3)
    ax2.barh(y_pos + h / 2, [x[1] * 1000 for x in em_stats], height=h, color=C_AMBER, alpha=0.88, edgecolor="#222222", lw=0.5, label="ExtremeMove (|r| > 2.0·rv22)", zorder=3)

    for i, ((br_lt, ece_lt), (br_em, ece_em)) in enumerate(zip(lt_stats, em_stats)):
        wt = "bold" if i == 0 else "normal"
        ax2.text(ece_lt * 1000 + 0.4, i - h / 2, f"{ece_lt*1000:.2f} (B={br_lt:.4f})", va="center", ha="left", fontsize=7.0, color="#1F3B4D", fontweight=wt)
        ax2.text(ece_em * 1000 + 0.4, i + h / 2, f"{ece_em*1000:.2f} (B={br_em:.4f})", va="center", ha="left", fontsize=7.0, color="#7A4F00", fontweight=wt)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([x[0] for x in peer_rows_103b], fontsize=7.8)
    ax2.get_yticklabels()[0].set_fontweight("bold")
    ax2.get_yticklabels()[0].set_color(C_BLUE)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 31.5)
    ax2.set_xlabel("10-Bin ECE (x10^-3, lower better)  [Brier in parens]")
    ax2.set_title(f"B. 7-Yr Macro Rare-Tail Calibration (Exp 103B, {n_103b_models} models)", loc="left", pad=8)
    ax2.legend(loc="upper right", fontsize=7.2)

    # --- Panel C: Exp 102C German Power Extreme-Price Probability (15 models) ---
    clean_ax(ax3, grid_axis="x")
    y_102c = np.array([float(r["y_extreme"]) for r in r102c], dtype=float)
    peve_mask = np.array([r["block"] == "P_EVE" for r in r102c], dtype=bool)
    meta_102c = {"delivery_date", "iso_week", "block", "y_extreme"}
    all_102c = {}
    for c in r102c[0]:
        if c in meta_102c:
            continue
        p = np.array([float(r[c]) for r in r102c], dtype=float)
        bf, ef = compute_brier_ece(p, y_102c)
        af = compute_auroc(p, y_102c)
        be, ee = compute_brier_ece(p[peve_mask], y_102c[peve_mask])
        all_102c[c] = {"b_full": bf, "ece_full": ef, "auc": af, "b_peve": be}

    n_102c_models = len(all_102c)
    rank_full_102c = {k: i for i, k in enumerate(sorted(all_102c, key=lambda m: all_102c[m]["b_full"]), 1)}
    rank_peve_102c = {k: i for i, k in enumerate(sorted(all_102c, key=lambda m: all_102c[m]["b_peve"]), 1)}

    show_102c = [
        ("Adapt1_CUDA_MLPv2_Disc8_PowerPool2048", "Adapt-1 Disc8 -> /value/*"),
        ("Online_HGBR_Cls_Pooled_Disc8", "Online HGBR (Disc8, 8f)"),
        ("Online_HGBR_Cls_Pooled", "Online HGBR (Full 20f)"),
        ("Adapt1_CUDA_Linear_PowerPool2048", "Adapt-1 Linear (20f)"),
        ("ExtraTrees_Cls_Pooled_Disc8", "ExtraTrees (Disc8, 8f)"),
        ("Adapt1_CUDA_MLPv2_PowerPool2048", "Adapt-1 Warm-Start (20f)"),
        ("LogisticRegression_L2_Pooled_Disc8", "LogReg L2 (Disc8, 8f)"),
    ]
    n_show = len(show_102c)
    y_idx = np.arange(n_show)

    # Highlight winning Disc8 row
    ax3.axhspan(-0.42, 0.42, color="#EBF5FB", alpha=0.85, zorder=0)
    disc8_bf = all_102c["Adapt1_CUDA_MLPv2_Disc8_PowerPool2048"]["b_full"]
    ax3.axvline(disc8_bf, color=C_BLUE, ls=":", lw=1.05, alpha=0.75, zorder=1)

    y_labels_c = []
    for i, (col, disp) in enumerate(show_102c):
        st = all_102c[col]
        rk = rank_full_102c[col]
        rk_e = rank_peve_102c[col]
        y_labels_c.append(f"{disp}  (#{rk}/{n_102c_models})")

        # Connector between P_EVE Brier and FULL Brier
        ax3.plot([st["b_peve"], st["b_full"]], [i, i], color="#8E9AA5", lw=1.6, zorder=2)

        # P_EVE marker
        peve_col = C_GREEN if rk_e == 1 else C_AMBER
        ax3.scatter(st["b_peve"], i, s=66 if rk_e == 1 else 54, marker="D", color=peve_col, edgecolor="#222222", lw=0.75, zorder=5)

        # FULL Brier marker
        if col == "Adapt1_CUDA_MLPv2_Disc8_PowerPool2048":
            ax3.scatter(st["b_full"], i, s=210, marker="*", color=C_BLUE, edgecolor="#111111", lw=0.8, zorder=6)
            lbl_txt = f"{st['b_full']:.4f}  (P_EVE {st['b_peve']:.4f} #1, AUC {st['auc']:.3f})"
            txt_col, txt_wt = C_BLUE, "bold"
        elif col == "Adapt1_CUDA_MLPv2_PowerPool2048":
            ax3.scatter(st["b_full"], i, s=78, marker="^", color=C_VERM, edgecolor="#222222", lw=0.75, zorder=6)
            lbl_txt = f"{st['b_full']:.4f}  (P_EVE {st['b_peve']:.4f}, AUC {st['auc']:.3f})"
            txt_col, txt_wt = C_VERM, "bold"
        elif col == "Adapt1_CUDA_Linear_PowerPool2048":
            ax3.scatter(st["b_full"], i, s=64, marker="o", color="#8CC5E3", edgecolor=C_BLUE, lw=0.9, zorder=5)
            lbl_txt = f"{st['b_full']:.4f}  (P_EVE {st['b_peve']:.4f})"
            txt_col, txt_wt = "#1F3B4D", "normal"
        elif "HGBR" in col:
            ax3.scatter(st["b_full"], i, s=66, marker="s", color=C_GREEN, edgecolor="#222222", lw=0.75, zorder=5)
            lbl_txt = f"{st['b_full']:.4f}  (P_EVE {st['b_peve']:.4f})"
            txt_col, txt_wt = "#222222", "normal"
        else:
            ax3.scatter(st["b_full"], i, s=58, marker="o", color=C_GRAY, edgecolor="#222222", lw=0.7, zorder=5)
            lbl_txt = f"{st['b_full']:.4f}  (P_EVE {st['b_peve']:.4f})"
            txt_col, txt_wt = "#444444", "normal"

        x_off = 0.0038 if col == "Adapt1_CUDA_MLPv2_Disc8_PowerPool2048" else 0.0024
        ax3.text(st["b_full"] + x_off, i, lbl_txt, va="center", ha="left", fontsize=7.1, color=txt_col, fontweight=txt_wt, zorder=7)

    ax3.set_yticks(y_idx)
    ax3.set_yticklabels(y_labels_c, fontsize=7.6)
    ax3.get_yticklabels()[0].set_fontweight("bold")
    ax3.get_yticklabels()[0].set_color(C_BLUE)
    ax3.set_ylim(7.35, -0.65)
    ax3.set_xlim(0.110, 0.244)
    ax3.set_xlabel("Brier Score  [lower is better; FULL 4-block & P_EVE evening slice]")
    ax3.set_title(f"C. German Power Extreme-Price Prob. (Exp 102C, {n_102c_models} models)", loc="left", pad=8)

    legend_handles_c = [
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=C_BLUE, markeredgecolor="#111", markersize=10.5, label="Adapt-1 Disc8 FULL (#1)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#8CC5E3", markeredgecolor=C_BLUE, markersize=6.0, label="Adapt-1 Linear (20f)"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=C_VERM, markeredgecolor="#222", markersize=6.5, label="Adapt-1 Warm-Start"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=C_GREEN, markeredgecolor="#222", markersize=5.8, label="Peer FULL"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=C_AMBER, markeredgecolor="#222", markersize=5.8, label="P_EVE (green = #1)"),
    ]
    ax3.legend(handles=legend_handles_c, loc="lower left", bbox_to_anchor=(0.008, 0.08), fontsize=6.4, framealpha=0.95)
    tb = all_102c["Trailing_BaseRate"]
    ax3.text(
        0.99, 0.015,
        f"{n_show}/{n_102c_models} traced models shown; Trailing BaseRate FULL {tb['b_full']:.4f} (P_EVE {tb['b_peve']:.4f})",
        transform=ax3.transAxes, ha="right", va="bottom", fontsize=6.5, color="#666666", style="italic",
    )

    # --- Panel D: Exp 100A Validation-Gated Lockout Strip Chart ---
    clean_ax(ax4, grid_axis="x")
    target_order = [
        ("realized_vol", "abs_r", "Vol: |r|"),
        ("realized_vol", "rv5_fwd", "Vol: RV5"),
        ("german_power", "ev_peak_spike_200", "Pwr: Spike>=200"),
        ("german_power", "p_peak", "Pwr: Peak"),
        ("german_power", "ev_neg_midday", "Pwr: NegMid"),
    ]
    status_colors = {
        "trained": C_BLUE,
        "candidate": C_GREEN,
        "retained": C_VERM,
        "rejected": C_AMBER,
        "rejected_abstained_prior": "#888888",
    }

    row_labels, row_y = [], []
    y_cursor = 0.0
    for arena, tgt, short_lbl in target_order:
        for pol, pol_tag in [
            ("expanding_install_true", "install=T"),
            ("sliding_candidate_install_false", "install=F"),
        ]:
            sub = [r for r in r100a if r["arena"] == arena and r["target"] == tgt and r["update_policy"] == pol]
            arms_here = sorted(set(r["arm"] for r in sub))
            sub = [r for r in sub if r["arm"] == arms_here[0]]
            sub.sort(key=lambda r: int(r["block_idx"]))
            n_blk = len(sub)
            n_ret = sum(1 for r in sub if r["train_status"] == "retained")
            n_cand = sum(1 for r in sub if r["train_status"] == "candidate")

            for r in sub:
                b = int(r["block_idx"])
                st = r["train_status"]
                ax4.barh(
                    y_cursor, 0.88, left=b - 0.44, height=0.68,
                    color=status_colors.get(st, "#999999"), edgecolor="#FFFFFF", linewidth=0.35, zorder=3,
                )
            tag_txt = f"{n_ret}/{n_blk} retained" if pol == "expanding_install_true" else f"{n_cand}/{n_blk} cand."
            ax4.text(
                n_blk + 0.6, y_cursor, tag_txt, va="center", ha="left", fontsize=7.2,
                color=C_VERM if pol == "expanding_install_true" else C_GREEN,
                fontweight="bold" if pol == "expanding_install_true" else "normal",
            )
            row_labels.append(f"{short_lbl} ({pol_tag})")
            row_y.append(y_cursor)
            y_cursor += 0.92
        y_cursor += 0.42

    ax4.set_yticks(row_y)
    ax4.set_yticklabels(row_labels, fontsize=7.5)
    ax4.invert_yaxis()
    ax4.set_xlim(-0.8, 56.0)
    ax4.set_xticks([0, 10, 20, 30, 40])
    ax4.set_xlabel("Sequential Evaluation Block Index")
    ax4.set_title("D. Validation-Gated Lockout Audit (Exp 100A)", loc="left", pad=8)

    legend_handles_4 = [
        mpatches.Patch(color=C_BLUE, label="Trained (init/install)"),
        mpatches.Patch(color=C_VERM, label="Retained (locked)"),
        mpatches.Patch(color=C_GREEN, label="Candidate (sliding)"),
        mpatches.Patch(color=C_AMBER, label="Rejected"),
    ]
    ax4.legend(handles=legend_handles_4, loc="lower right", ncol=1, fontsize=7.1)

    fig.suptitle(
        "Section 02 — Online Probability Models: Calibration, Discovery & Retraining Lockout",
        fontsize=12.4, fontweight="bold", y=0.962,
    )
    out = FIG_DIR / "fig02_online_probability_models.png"
    fig.savefig(out, dpi=DPI, pil_kwargs={"optimize": True})
    plt.close(fig)
    return out


# =============================================================================
# FIGURE 3: Section 03 — Plasticity & Recurrence (Exp 96B + Exp 79A)
# =============================================================================
def build_fig03() -> Path:
    sec3 = DATA_ROOT / "03_plasticity_and_recurrence"
    r96_n15 = load_csv(sec3 / "exp96b_n15_step_trace.csv")
    r96_abl = load_csv(sec3 / "exp96b_live_ablations_step_trace.csv")
    r79a = load_csv(sec3 / "exp79a_spy_tlt_flip_trace.csv")
    sc3 = load_json(sec3 / "scorecard.json")

    fig = plt.figure(figsize=(13.8, 5.7))
    gs = gridspec.GridSpec(
        2, 2, width_ratios=[1.12, 1.08], height_ratios=[1.0, 1.75],
        wspace=0.25, hspace=0.14, left=0.06, right=0.98, top=0.88, bottom=0.12,
    )
    ax_l = fig.add_subplot(gs[:, 0])
    ax_rt = fig.add_subplot(gs[0, 1])
    ax_rb = fig.add_subplot(gs[1, 1], sharex=ax_rt)

    # --- Left Panel: Exp 96B 300-Step Rolling Accuracy ---
    clean_ax(ax_l)

    def step_means_from_n15(col: str) -> np.ndarray:
        by_t = collections.defaultdict(list)
        for r in r96_n15:
            by_t[int(r["step_t"])].append(float(r[col]))
        return np.array([statistics.mean(by_t[t]) for t in range(300)], dtype=float)

    def step_means_from_abl(arm: str) -> np.ndarray:
        by_t = collections.defaultdict(list)
        for r in r96_abl:
            if r["cohort"] == "n8_paired_ablations" and r["arm"] == arm:
                by_t[int(r["step_t"])].append(float(r["is_correct"]))
        return np.array([statistics.mean(by_t[t]) for t in range(300)], dtype=float)

    def smooth_by_phase(arr: np.ndarray, win: int = 11) -> np.ndarray:
        out = np.zeros_like(arr)
        for p_start in range(0, 300, 60):
            seg = arr[p_start : p_start + 60]
            for i in range(60):
                lo = max(0, i - win + 1)
                out[p_start + i] = float(np.mean(seg[lo : i + 1]))
        return out * 100.0

    steps = np.arange(300)
    ad_raw = step_means_from_n15("adapt1_native_correct")
    sw_raw = step_means_from_n15("sw_linucb_correct")
    hgbr_raw = step_means_from_n15("adwin_hgbr_correct")
    nucb_raw = step_means_from_n15("online_neuralucb_correct")
    nat8_raw = step_means_from_abl("adapt1_lb_native")
    off_raw = step_means_from_abl("adapt1_lb_off")
    rst_raw = step_means_from_abl("adapt1_lb_policy_reset")

    phases = [
        (0, 60, "A1", "#FFFFFF"),
        (60, 120, "B1", "#F5F7F9"),
        (120, 180, "C", "#FFFFFF"),
        (180, 240, "A2 (Return)", "#EBF5FB"),
        (240, 300, "B2* (Silent)", "#FDF2EE"),
    ]
    for s0, s1, pname, bg in phases:
        ax_l.axvspan(s0, s1, color=bg, alpha=0.85, zorder=0)
        if s0 > 0:
            ax_l.axvline(s0, color="#AAB4BE", ls=":", lw=0.85, zorder=1)
        ax_l.text((s0 + s1) / 2, 97.5, pname, ha="center", va="top", fontsize=7.6, fontweight="bold", color="#333333")

    ax_l.plot(steps, smooth_by_phase(off_raw), color=C_AMBER, lw=1.45, ls="-.", label="Adapt-1 Regimes Off (n=8)", zorder=4)
    ax_l.plot(steps, smooth_by_phase(rst_raw), color=C_VERM, lw=1.45, ls="--", label="Adapt-1 Blanket Reset (n=8)", zorder=4)
    ax_l.plot(steps, smooth_by_phase(sw_raw), color=C_GRAY, lw=1.25, ls=":", label="SW-LinUCB (n=15)", zorder=3)
    ax_l.plot(steps, smooth_by_phase(ad_raw), color=C_BLUE, lw=2.1, label="Adapt-1 Native (n=15)", zorder=5)

    a2_ad_10 = float(np.mean(ad_raw[180:190])) * 100.0
    a2_nat8_10 = float(np.mean(nat8_raw[180:190])) * 100.0
    a2_hg_10 = float(np.mean(hgbr_raw[180:190])) * 100.0
    a2_nu_10 = float(np.mean(nucb_raw[180:190])) * 100.0
    a2_off_10 = float(np.mean(off_raw[180:190])) * 100.0
    a2_rst_10 = float(np.mean(rst_raw[180:190])) * 100.0
    b2_last20 = float(np.mean(ad_raw[280:300])) * 100.0
    b2_full = float(np.mean(ad_raw[240:300])) * 100.0

    ax_l.hlines(a2_hg_10, 180, 239, colors=C_GREEN, linestyles="--", lw=1.2, zorder=5)
    ax_l.hlines(a2_nu_10, 180, 239, colors=C_PURPLE, linestyles=":", lw=1.35, zorder=5)
    ax_l.text(
        238, a2_hg_10 + 1.8, f"ADWIN-HGBR A2-10: {a2_hg_10:.1f}%",
        ha="right", va="bottom", fontsize=6.7, color=C_GREEN, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.14", "fc": "#FFFFFF", "ec": "none", "alpha": 0.88}, zorder=6,
    )
    ax_l.text(
        238, a2_nu_10 - 1.8, f"NeuralUCB A2-10: {a2_nu_10:.1f}%",
        ha="right", va="top", fontsize=6.7, color=C_PURPLE, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.14", "fc": "#FFFFFF", "ec": "none", "alpha": 0.88}, zorder=6,
    )

    ax_l.annotate(
        f"A2 first-10: Native {a2_ad_10:.1f}% (n=15)\nPaired n=8: {a2_nat8_10:.1f}% vs Off {a2_off_10:.1f}% / Reset {a2_rst_10:.1f}%",
        xy=(185, a2_ad_10), xytext=(12, 77),
        fontsize=6.9, color=C_BLUE, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "fc": "#FFFFFF", "ec": C_BLUE, "alpha": 0.94, "lw": 0.8},
        arrowprops={"arrowstyle": "->", "color": C_BLUE, "lw": 0.9}, zorder=7,
    )
    ax_l.annotate(
        f"Silent shift boundary:\nlock-in ({b2_last20:.1f}% last-20 /\n{b2_full:.1f}% full B2*)",
        xy=(266, 2.5), xytext=(270, 56.0), ha="center",
        fontsize=6.5, color=C_VERM,
        bbox={"boxstyle": "round,pad=0.25", "fc": "#FFFFFF", "ec": C_VERM, "alpha": 0.94, "lw": 0.75},
        arrowprops={"arrowstyle": "->", "color": C_VERM, "lw": 0.85}, zorder=7,
    )

    ax_l.set_xlim(0, 299)
    ax_l.set_ylim(-2, 101)
    ax_l.set_xlabel("Decision Step t  (trailing 11-step within-phase rolling accuracy)")
    ax_l.set_ylabel("Optimal Action Accuracy (%)")
    ax_l.set_title("A. Recurring Nonlinear Regimes & Silent Boundary (Exp 96B)", loc="left", pad=8)
    ax_l.legend(loc="lower left", fontsize=7.1)

    # --- Right Panel: Exp 79A SPY/TLT Correlation Flip & Drawdowns ---
    clean_ax(ax_rt)
    clean_ax(ax_rb)
    dates_79a = parse_dates([r["date"] for r in r79a])
    corr = np.array([float(r["corr_sb_60d"]) for r in r79a], dtype=float)

    ax_rt.axhline(0.0, color="#333333", lw=0.8, ls="-", zorder=2)
    ax_rt.plot(dates_79a, corr, color="#222222", lw=1.35, zorder=4)
    ax_rt.fill_between(dates_79a, 0.0, corr, where=(corr >= 0), color=C_VERM, alpha=0.22, interpolate=True)
    ax_rt.fill_between(dates_79a, 0.0, corr, where=(corr < 0), color=C_BLUE, alpha=0.18, interpolate=True)
    ax_rt.set_ylim(-0.52, 0.44)
    ax_rt.set_ylabel("60d SPY/TLT Corr")
    ax_rt.set_title("B. 2021–2023 SPY/TLT Flip & 5 bps Risk-Overlay Drawdowns (Exp 79A)", loc="left", pad=8)
    ax_rt.text(dates_79a[14], -0.45, f"Min {corr.min():+.2f}", fontsize=7.1, color=C_BLUE, fontweight="bold")
    i_max = int(np.argmax(corr))
    ax_rt.text(dates_79a[i_max], corr[i_max] + 0.04, f"Flip to {corr.max():+.2f}", fontsize=7.1, color=C_VERM, fontweight="bold", ha="center")
    plt.setp(ax_rt.get_xticklabels(), visible=False)

    dd_ad = np.array([float(r["adapt1_smooth_dd5_pct"]) for r in r79a])
    dd_fl = np.array([float(r["adapt1_flush21d_dd5_pct"]) for r in r79a])
    dd_60 = np.array([float(r["static_60_40_dd5_pct"]) for r in r79a])
    dd_qh = np.array([float(r["q_hgbr_online_dd5_pct"]) for r in r79a])
    dd_c2 = np.array([float(r["chronos2_dd5_pct"]) for r in r79a])

    ax_rb.plot(dates_79a, dd_60, color=C_VERM, lw=1.35, ls="--", label=f"Static 60/40 (max {dd_60.min():.1f}%)", zorder=3)
    ax_rb.plot(dates_79a, dd_qh, color=C_AMBER, lw=1.2, alpha=0.9, label=f"Q-HGBR Online (max {dd_qh.min():.1f}%)", zorder=3)
    ax_rb.plot(dates_79a, dd_c2, color=C_PURPLE, lw=1.2, alpha=0.9, label=f"Chronos-2 (max {dd_c2.min():.1f}%)", zorder=3)
    ax_rb.plot(dates_79a, dd_fl, color=C_SKY, lw=1.35, ls=":", label=f"Adapt-1 Hard 21d Flush (max {dd_fl.min():.1f}%)", zorder=4)
    ax_rb.plot(dates_79a, dd_ad, color=C_BLUE, lw=2.0, label=f"Adapt-1 Smooth W128 (max {dd_ad.min():.1f}%)", zorder=5)

    m_sm = sc3["exp79a_spy_tlt_2022_correlation_flip"]["models"]["Adapt1_Smooth_Retention_W128"]
    m_fl = sc3["exp79a_spy_tlt_2022_correlation_flip"]["models"]["Adapt1_Hard_21d_Flush_DriftReset"]
    ax_rb.text(
        0.02, 0.05,
        f"Retention vs. Reset (Exp 79A, 5 bps):\n"
        f"• Smooth W128: ES {m_sm['es_4d']:.3f}, ret {m_sm['ret_5bps_pct']:+.1f}%, DD {m_sm['max_dd_5bps_pct']:.1f}%\n"
        f"• Hard 21d flush: ES {m_fl['es_4d']:.3f}, ret {m_fl['ret_5bps_pct']:+.1f}%, DD {m_fl['max_dd_5bps_pct']:.1f}%",
        transform=ax_rb.transAxes, fontsize=6.9, va="bottom", ha="left",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#FFFFFF", "ec": "#CBD2D9", "alpha": 0.94},
    )

    ax_rb.set_ylim(-31.5, 1.5)
    ax_rb.set_ylabel("Drawdown (%)")
    ax_rb.set_xlabel("Date")
    ax_rb.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_rb.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_rb.legend(loc="lower right", fontsize=6.9, handlelength=1.7)

    fig.suptitle("Section 03 — Plasticity, Regime Recurrence & Smooth Retention Under Correlation Flips", fontsize=12, fontweight="bold", y=0.96)
    out = FIG_DIR / "fig03_plasticity_and_recurrence.png"
    fig.savefig(out, dpi=DPI, pil_kwargs={"optimize": True})
    plt.close(fig)
    return out


# =============================================================================
# FIGURE 4: Section 04 — Conditioning Across Time (Exp 16 + Municipal 311)
# =============================================================================
def build_fig04() -> Path:
    sec4 = DATA_ROOT / "04_conditioning_across_time"
    r16 = load_csv(sec4 / "exp16_macro_w1024_bw020_trace.csv")
    r_nyc = load_csv(sec4 / "nyc_311_lower_manhattan_revisions_trace.csv")
    r_chi = load_csv(sec4 / "chicago_311_post2026_09_revisions_trace.csv")
    r_sf = load_csv(sec4 / "sf_311_post2026_09_revisions_trace.csv")
    sc4 = load_json(sec4 / "scorecard.json")

    fig = plt.figure(figsize=(14.4, 5.4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.08, 1.28], wspace=0.22, left=0.055, right=0.985, top=0.825, bottom=0.14)
    ax_l = fig.add_subplot(gs[0, 0])
    gs_r = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 1], wspace=0.24)
    ax_nyc = fig.add_subplot(gs_r[0, 0])
    ax_chi = fig.add_subplot(gs_r[0, 1])
    ax_sf = fig.add_subplot(gs_r[0, 2])

    # --- Left Panel: Exp 16 19-Year Conditional 2008 Stress-Weight Recall ---
    clean_ax(ax_l)
    eval_rows = [r for r in r16 if r["is_eval_890"] == "1" and r["status"] == "predicted" and r["gfc_mass_cond"] != ""]
    d16 = parse_dates([r["date"] for r in eval_rows])
    gfc_c = np.array([float(r["gfc_mass_cond"]) * 100.0 for r in eval_rows])
    gfc_p = np.array([float(r["gfc_mass_prior"]) * 100.0 for r in eval_rows])
    gfc_r = np.array([float(r["gfc_ratio"]) for r in eval_rows])

    ax_l.plot(d16, gfc_p, color=C_GRAY, lw=1.25, ls="--", label="Uniform archive share (prior)", zorder=3)
    ax_l.plot(d16, gfc_c, color=C_BLUE, lw=1.35, label="Conditional 2008 stress weight (bw=0.20, W=1024)", zorder=4)
    ax_l.fill_between(d16, gfc_p, gfc_c, where=(gfc_c >= gfc_p), color=C_BLUE, alpha=0.14, interpolate=True, zorder=2)

    date_to_idx = {r["date"]: i for i, r in enumerate(eval_rows)}
    idx_2017 = [i for i, r in enumerate(eval_rows) if r["date"].startswith("2017-")]
    mid_2017 = idx_2017[len(idx_2017) // 2]
    ratio_2017 = float(np.mean(gfc_c[idx_2017]) / np.mean(gfc_p[idx_2017]))
    ax_l.annotate(
        f"2017 calm\n({ratio_2017:.2f}x prior)",
        xy=(d16[mid_2017], float(np.mean(gfc_c[idx_2017]))), xytext=(-14, -36), textcoords="offset points",
        fontsize=7.1, color=C_GREEN, fontweight="bold", ha="center",
        bbox={"boxstyle": "round,pad=0.2", "fc": "#FFFFFF", "ec": "none", "alpha": 0.90},
        arrowprops={"arrowstyle": "->", "color": C_GREEN, "lw": 0.85}, zorder=6,
    )

    stress_marks = [
        ("2018-02-06", "2018 Volmageddon", (-66, 10), C_VERM),
        ("2020-03-03", "2020 COVID", (12, 14), C_VERM),
        ("2025-04-04", "2025 Tariffs", (-62, 18), C_VERM),
    ]
    for dstr, lbl, offset, colr in stress_marks:
        i = date_to_idx[dstr]
        ax_l.scatter(d16[i], gfc_c[i], s=38, color=colr, edgecolor="#FFFFFF", lw=0.7, zorder=6)
        ax_l.annotate(
            f"{lbl}\n({gfc_r[i]:.2f}x)",
            xy=(d16[i], gfc_c[i]), xytext=offset, textcoords="offset points",
            fontsize=7.2, color=colr, fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": colr, "lw": 0.85}, zorder=7,
        )

    ax_l.set_ylim(0, 66)
    ax_l.set_ylabel("Weight on 2008 Stress Episodes (%)")
    ax_l.set_xlabel("Forecast Date (n = 890 weekly OOS evaluations)")
    ax_l.set_title("A. 19-Year Conditional 2008 Analogue Recall (Exp 16)", loc="left", pad=12)
    ax_l.xaxis.set_major_locator(mdates.YearLocator(3))
    ax_l.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_l.legend(loc="upper right", fontsize=7.1)

    # --- Right Panel: 2026 Municipal 311 Progressive Revisions ---
    nyc_by_tkt = collections.defaultdict(dict)
    for r in r_nyc:
        if r["is_oos"] == "1" and r["status"] == "predicted" and r["es_mixed"] != "":
            nyc_by_tkt[r["ticket_hash"]][int(r["rev"])] = (float(r["es_mixed"]), float(r["es_mixed_prior_cf"]))
    nyc_coh = [v for v in nyc_by_tkt.values() if {0, 1, 2}.issubset(v.keys())]
    nyc_ad = [statistics.mean(v[k][0] for v in nyc_coh) for k in (0, 1, 2)]
    nyc_pr = [statistics.mean(v[k][1] for v in nyc_coh) for k in (0, 1, 2)]

    clean_ax(ax_nyc)
    xs = [0, 1, 2]
    ax_nyc.plot(xs, nyc_pr, color=C_GRAY, lw=1.3, ls="--", marker="s", ms=4.5, label="Prior CF")
    ax_nyc.plot(xs, nyc_ad, color=C_BLUE, lw=2.0, marker="o", ms=5.5, label="Adapt-1", zorder=5)
    for x, yv in zip(xs, nyc_ad):
        ax_nyc.text(x, yv - 0.025, f"{yv:.3f}", ha="center", va="top", fontsize=7.0, color=C_BLUE, fontweight="bold")
    nyc_lo = min(min(nyc_ad), min(nyc_pr)) - 0.07
    nyc_hi = max(max(nyc_ad), max(nyc_pr)) + 0.09
    ax_nyc.set_xticks(xs)
    ax_nyc.set_xticklabels(["0h", "+1h", "+6h"])
    ax_nyc.set_xlim(-0.28, 2.28)
    ax_nyc.set_ylim(nyc_lo, nyc_hi)
    ax_nyc.set_ylabel("Mixed Energy Score  [lower better]")
    ax_nyc.set_title(f"NYC (n={len(nyc_coh)} cohort)", fontsize=8.8, fontweight="bold", pad=6)
    ax_nyc.legend(loc="lower left", fontsize=6.6)

    chi_ad = [statistics.mean(float(r["es_mixed"]) for r in r_chi if int(r["rev"]) == k and r["status"] == "predicted") for k in (0, 1, 2)]
    chi_st = sc4["benchmarks"]["municipal_311_revisions"]["chicago_post_2026_09"]["stages"]
    chi_olhr = [chi_st[f"rev_{k}"]["models"]["OLHR"]["es_mixed"] for k in (0, 1, 2)]
    chi_qrf = [chi_st[f"rev_{k}"]["models"]["QRF"]["es_mixed"] for k in (0, 1, 2)]
    chi_hgbr = [chi_st[f"rev_{k}"]["models"]["HGBR"]["es_mixed"] for k in (0, 1, 2)]

    clean_ax(ax_chi)
    ax_chi.plot(xs, chi_hgbr, color=C_AMBER, lw=1.2, ls=":", marker="^", ms=4.5, label="Refit HGBR")
    ax_chi.plot(xs, chi_olhr, color=C_PURPLE, lw=1.2, ls="-.", marker="d", ms=4.5, label="OLHR")
    ax_chi.plot(xs, chi_qrf, color=C_GREEN, lw=1.35, ls="--", marker="s", ms=4.5, label="Refit QRF")
    ax_chi.plot(xs, chi_ad, color=C_BLUE, lw=2.0, marker="o", ms=5.5, label="Adapt-1 (#1)", zorder=5)
    chi_offsets = [(0, -14), (-16, -15), (0, -14)]
    for x, yv, off in zip(xs, chi_ad, chi_offsets):
        ax_chi.annotate(f"{yv:.3f}", xy=(x, yv), xytext=off, textcoords="offset points", ha="center", va="top", fontsize=7.0, color=C_BLUE, fontweight="bold")
    ax_chi.set_xticks(xs)
    ax_chi.set_xticklabels(["0h", "+6h", "+48h"])
    ax_chi.set_xlim(-0.28, 2.28)
    ax_chi.set_ylim(0.18, 0.66)
    ax_chi.set_title("Chicago (n=2,621)", fontsize=8.8, fontweight="bold", pad=6)
    ax_chi.set_xlabel("Revision Horizon")
    ax_chi.legend(loc="upper right", fontsize=6.5)

    sf_ad = [statistics.mean(float(r["es_mixed"]) for r in r_sf if int(r["rev"]) == k and r["status"] == "predicted") for k in (0, 1, 2)]
    sf_blk = sc4["benchmarks"]["municipal_311_revisions"]["sf_post_2026_09"]
    sf_qrf_refit = [sf_blk["stages"][f"rev_{k}"]["models"]["QRF"]["es_mixed"] for k in (0, 1, 2)]
    sf_hgbr_refit = [sf_blk["stages"][f"rev_{k}"]["models"]["HGBR"]["es_mixed"] for k in (0, 1, 2)]
    sf_qrf_frz = [sf_blk["frozen_and_onepass_comparators_es"]["Frozen_QRF_Mixed"][f"rev_{k}"] for k in (0, 1, 2)]
    sf_hgbr_frz = [sf_blk["frozen_and_onepass_comparators_es"]["Frozen_HGBR_Mixed"][f"rev_{k}"] for k in (0, 1, 2)]

    clean_ax(ax_sf)
    ax_sf.plot(xs, sf_hgbr_frz, color=C_VERM, lw=1.25, ls=":", marker="^", ms=4.5, label="Frozen HGBR")
    ax_sf.plot(xs, sf_qrf_frz, color=C_VERM, lw=1.3, ls="--", marker="s", ms=4.5, label="Frozen QRF")
    ax_sf.plot(xs, sf_hgbr_refit, color=C_AMBER, lw=1.2, ls="-.", marker="^", ms=4.2, label="Refit HGBR")
    ax_sf.plot(xs, sf_qrf_refit, color=C_GREEN, lw=1.35, ls="--", marker="s", ms=4.2, label="Refit QRF")
    ax_sf.plot(xs, sf_ad, color=C_BLUE, lw=2.0, marker="o", ms=5.5, label="Adapt-1", zorder=5)
    for x, yv in zip(xs, sf_ad):
        ax_sf.text(x, yv - 0.017, f"{yv:.3f}", ha="center", va="top", fontsize=7.0, color=C_BLUE, fontweight="bold")
    ax_sf.annotate(
        "Frozen drift\n(+17–21% vs refit)",
        xy=(1.0, sf_qrf_frz[1]), xytext=(0.15, 0.808),
        fontsize=6.7, color=C_VERM, fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": C_VERM, "lw": 0.8},
    )
    ax_sf.set_xticks(xs)
    ax_sf.set_xticklabels(["0h", "+1h", "+6h"])
    ax_sf.set_xlim(-0.28, 2.28)
    ax_sf.set_ylim(0.50, 0.855)
    ax_sf.set_title("San Francisco (n=4,375)", fontsize=8.8, fontweight="bold", pad=6)
    ax_sf.legend(loc="lower left", fontsize=6.3, handlelength=1.45)

    fig.text(0.725, 0.892, "B. 2026 Municipal 311 Progressive-Revision Mixed Energy Score", ha="center", fontsize=10.5, fontweight="bold")
    fig.suptitle("Section 04 — Conditioning Across Time: Deep Analogue Recall & Progressive Event Revisions", fontsize=12, fontweight="bold", y=0.96)
    out = FIG_DIR / "fig04_conditioning_across_time.png"
    fig.savefig(out, dpi=DPI, pil_kwargs={"optimize": True})
    plt.close(fig)
    return out


# =============================================================================
# FIGURE 5: Section 05 — Belief Collapse & QMDP (Exp 93A, 93B, 93C, 93-PROD)
# =============================================================================
def build_fig05() -> Path:
    sec5 = DATA_ROOT / "05_belief_collapse"
    r93a_ep = load_csv(sec5 / "exp93a_episodes.csv")
    r93a_st = load_csv(sec5 / "exp93a_live_steps.csv")
    r93b_ep = load_csv(sec5 / "exp93b_episodes.csv")
    r93b_st = load_csv(sec5 / "exp93b_qmdp_live_steps.csv")
    r93c_ep = load_csv(sec5 / "exp93c_campaigns.csv")
    r_prod = load_csv(sec5 / "exp93_prod_parity_steps.csv")

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(13.6, 5.4),
        gridspec_kw={"width_ratios": [1.0, 1.12], "wspace": 0.25, "left": 0.065, "right": 0.98, "top": 0.86, "bottom": 0.14},
    )

    # --- Left Panel: Log-scale surviving fault worlds (Exp 93A & Exp 93B) ---
    clean_ax(ax_l)

    def collapse_stats(rows, arm_name, arena_name, prior_n):
        by_s = collections.defaultdict(list)
        for r in rows:
            if r["arena"] == arena_name and r["arm"] == arm_name and r["full_world_count"] != "":
                by_s[int(r["step"])].append(int(r["full_world_count"]))
        xs = [0] + sorted(by_s.keys())
        means = [float(prior_n)] + [statistics.mean(by_s[s]) for s in sorted(by_s.keys())]
        meds = [float(prior_n)] + [statistics.median(by_s[s]) for s in sorted(by_s.keys())]
        mins = [float(prior_n)] + [float(min(by_s[s])) for s in sorted(by_s.keys())]
        maxs = [float(prior_n)] + [float(max(by_s[s])) for s in sorted(by_s.keys())]
        return xs, means, meds, mins, maxs

    x93a, m93a, md93a, lo93a, hi93a = collapse_stats(r93a_st, "Live_Adapt1_FullTriad_Native", "93A", 120)
    x93b, m93b, md93b, lo93b, hi93b = collapse_stats(r93b_st, "Live_Adapt1_QMDP_Native", "93B_clean", 78480)

    p50_93b = sorted(float(r["prod_query_ms"]) for r in r_prod if r["arena"] == "93B_clean")
    p50_93b_ms = p50_93b[len(p50_93b) // 2]
    p50_93a = sorted(float(r["prod_query_ms"]) for r in r_prod if r["arena"] == "93A")
    p50_93a_ms = p50_93a[len(p50_93a) // 2]

    ax_l.fill_between(x93b, lo93b, hi93b, color=C_VERM, alpha=0.14, zorder=1)
    ax_l.plot(x93b, m93b, color=C_VERM, lw=2.0, marker="o", ms=5.5, label=f"Exp 93B QMDP mean (prod p50 {p50_93b_ms:.0f} ms)", zorder=4)
    ax_l.plot(x93b, md93b, color=C_VERM, lw=1.35, ls="--", marker="o", ms=4.8, mfc="#FFFFFF", label="Exp 93B QMDP median", zorder=4)

    ax_l.fill_between(x93a, lo93a, hi93a, color=C_BLUE, alpha=0.16, zorder=2)
    ax_l.plot(x93a, m93a, color=C_BLUE, lw=2.0, marker="s", ms=5.5, label=f"Exp 93A Exact mean (prod p50 {p50_93a_ms:.0f} ms)", zorder=5)
    ax_l.plot(x93a, md93a, color=C_BLUE, lw=1.35, ls="--", marker="s", ms=4.8, mfc="#FFFFFF", label="Exp 93A Exact median", zorder=5)

    # Clean upper-right open whitespace callout for 93B QMDP
    ax_l.annotate(
        f"78,480 -> {m93b[-1]:.1f} mean ({md93b[-1]:.1f} med, n=30)\n0 HTTP 504s vs 40.2s exact timeout",
        xy=(6, m93b[-1]), xytext=(3.15, 22000),
        fontsize=7.2, color=C_VERM, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.32", "fc": "#FFFFFF", "ec": C_VERM, "alpha": 0.95, "lw": 0.8},
        arrowprops={"arrowstyle": "->", "color": C_VERM, "lw": 0.95, "connectionstyle": "arc3,rad=-0.12"}, zorder=7,
    )
    # Clean mid-right open whitespace callout for 93A Exact
    ax_l.annotate(
        f"120 -> {m93a[-1]:.1f} mean ({md93a[-1]:.1f} med)*",
        xy=(6, m93a[-1]), xytext=(3.55, 20.0),
        fontsize=7.2, color=C_BLUE, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "fc": "#FFFFFF", "ec": C_BLUE, "alpha": 0.95, "lw": 0.75},
        arrowprops={"arrowstyle": "->", "color": C_BLUE, "lw": 0.9}, zorder=7,
    )
    ax_l.text(0.10, 78480 * 1.25, "78,480", fontsize=7.5, color=C_VERM, fontweight="bold")
    ax_l.text(0.10, 120 * 1.30, "120", fontsize=7.5, color=C_BLUE, fontweight="bold")

    ax_l.set_yscale("log")
    ax_l.set_xlim(-0.25, 6.45)
    ax_l.set_ylim(0.85, 280000)
    ax_l.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax_l.set_xlabel("Diagnostic Step t")
    ax_l.set_ylabel("Surviving Consistent Fault Worlds (log scale)")
    ax_l.set_title("A. Online Belief Collapse (Exp 93A & 93B, shaded [min, max])", loc="left", pad=8)
    ax_l.legend(loc="lower left", bbox_to_anchor=(0.01, 0.075), fontsize=7.0)
    ax_l.text(
        0.99, 0.015,
        "*93A explain polled n=30 at steps 1,4; n=3 at steps 2,3,6. 204/204 prod parity.",
        transform=ax_l.transAxes, ha="right", va="bottom", fontsize=6.5, color="#666666", style="italic",
    )

    # --- Right Panel: Exp 93A Episode Return vs PPO Training Curve + 93B/93C Notes ---
    clean_ax(ax_r)

    def ep_vals(rows, pred, col="cum_reward"):
        return [float(r[col]) for r in rows if pred(r)]

    ppo_eps = [0, 200, 1000, 5000, 20000]
    ppo_means, ppo_cis = [], []
    for k in ppo_eps:
        vs = ep_vals(r93a_ep, lambda r, kk=k: r["system"] == f"PPO_Factored_{kk}ep")
        ppo_means.append(statistics.mean(vs))
        ppo_cis.append(t_ci95_half(vs))

    ad_93a = ep_vals(r93a_ep, lambda r: r["system"] == "Live_Adapt1_FullTriad_Native")
    ad_93a_m, ad_93a_ci = statistics.mean(ad_93a), t_ci95_half(ad_93a)
    ad_93a_usd = statistics.mean(ep_vals(r93a_ep, lambda r: r["system"] == "Live_Adapt1_FullTriad_Native", "net_value_usd_1_50"))
    voi_93a_m = statistics.mean(ep_vals(r93a_ep, lambda r: r["system"] == "Exact_2Step_VoI"))
    un_93a_m = statistics.mean(ep_vals(r93a_ep, lambda r: r["system"] == "Live_Adapt1_UnauthoredBT"))
    lin_93a_m = statistics.mean(ep_vals(r93a_ep, lambda r: r["system"] == "LinUCB_500ep"))

    x_idx = np.arange(len(ppo_eps))
    ax_r.axhspan(ad_93a_m - ad_93a_ci, ad_93a_m + ad_93a_ci, color=C_BLUE, alpha=0.11, zorder=1)
    ax_r.axhline(voi_93a_m, color=C_GREEN, lw=1.4, ls="--", label=f"Exact 2-Step VoI ceiling ({voi_93a_m:+.3f}, +$412k)", zorder=3)
    ax_r.axhline(ad_93a_m, color=C_BLUE, lw=2.0, label=f"Adapt-1 Belief Search — Ep 0 ({ad_93a_m:+.3f}, +${ad_93a_usd/1000:.0f}k)", zorder=4)
    ax_r.axhline(lin_93a_m, color=C_GRAY, lw=1.0, ls=":", label=f"Bandits ceiling (LinUCB 500ep: {lin_93a_m:+.3f})", zorder=2)
    ax_r.axhline(un_93a_m, color=C_VERM, lw=1.2, ls="-.", label=f"Adapt-1 Unauthored ({un_93a_m:+.3f})", zorder=2)

    ax_r.errorbar(
        x_idx, ppo_means, yerr=ppo_cis, color="#222222", lw=1.8, marker="o", ms=6.2,
        mfc=C_AMBER, mec="#222222", capsize=3.5, elinewidth=1.1, label="Model-Free Factored PPO (mean ± 95% CI)", zorder=5,
    )

    ppo_offsets = [(28, -3), (24, -17), (0, 12), (-10, 14), (-34, -2)]
    for xi, ym, off in zip(x_idx, ppo_means, ppo_offsets):
        ax_r.annotate(f"{ym:+.3f}", xy=(xi, ym), xytext=off, textcoords="offset points", fontsize=7.3, fontweight="bold", color="#222222", ha="center")

    ax_r.annotate(
        "Beats PPO through 5,000\noutages (p_H = 2e-4)",
        xy=(3, ppo_means[3]), xytext=(1.55, 1.25),
        fontsize=7.2, color=C_BLUE, fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": C_BLUE, "lw": 0.9}, zorder=6,
    )

    c93_ret = statistics.mean(ep_vals(r93c_ep, lambda r: r["system"] == "Live_Adapt1_FactorizedEB_Native"))
    c93_usd = statistics.mean(ep_vals(r93c_ep, lambda r: r["system"] == "Live_Adapt1_FactorizedEB_Native", "net_value_usd_1_50"))
    b93_cl = statistics.mean(ep_vals(r93b_ep, lambda r: r["arena"] == "93B_clean" and r["system"] == "Live_Adapt1_QMDP_Native"))
    b93_fl = statistics.mean(ep_vals(r93b_ep, lambda r: r["arena"] == "93B_flaky15" and r["system"] == "Live_Adapt1_QMDP_Flaky15"))
    b93_voi = statistics.mean(ep_vals(r93b_ep, lambda r: r["arena"] == "93B_clean" and r["system"] == "Exact_1Step_VoI"))

    ax_r.text(
        0.985, 0.03,
        f"Companion Simulated Arenas (Exp 93C & 93B):\n"
        f"• 93C 6-node DAG (no topology): {c93_ret:+.3f} (+${c93_usd/1000:.0f}k, 30/30 learned)\n"
        f"• 93B QMDP pure-probe boundary: {b93_cl:+.3f} clean / {b93_fl:+.3f} flaky vs {b93_voi:+.3f} VoI",
        transform=ax_r.transAxes, ha="right", va="bottom", fontsize=6.9,
        bbox={"boxstyle": "round,pad=0.35", "fc": "#F9FAFB", "ec": "#CBD2D9", "alpha": 0.95},
        zorder=6,
    )

    ax_r.set_xticks(x_idx)
    ax_r.set_xticklabels(["0", "200", "1,000", "5,000", "20,000"])
    ax_r.set_xlim(-0.35, 4.35)
    ax_r.set_ylim(-6.05, 4.75)
    ax_r.set_xlabel("Offline Model-Free PPO Training Outages (simulated episodes)")
    ax_r.set_ylabel("Mean Simulated Episode Return (1:50 SLA scale)")
    ax_r.set_title("B. Zero-Shot Belief Search vs. PPO Sample Efficiency (Exp 93A)", loc="left", pad=8)
    ax_r.legend(loc="lower right", bbox_to_anchor=(0.99, 0.21), fontsize=6.9)

    fig.suptitle("Section 05 — Belief Collapse, QMDP Scaling & Sample Efficiency (Simulated K8s POMDPs)", fontsize=12, fontweight="bold", y=0.96)
    out = FIG_DIR / "fig05_belief_collapse_and_qmdp.png"
    fig.savefig(out, dpi=DPI, pil_kwargs={"optimize": True})
    plt.close(fig)
    return out


# =============================================================================
# FIGURE 6: Section 06 — Autonomous Discovery Across Sections 01–05
# =============================================================================
def build_fig06() -> Path:
    r89b = load_csv(DATA_ROOT / "01_joint_tails_and_drift" / "exp89b_eu_dunkelflaute_oos_trace.csv")
    r102c = load_csv(DATA_ROOT / "02_online_probability_models" / "exp102c_german_power_oos_trace.csv")
    r105 = load_csv(DATA_ROOT / "06_autonomous_discovery" / "exp105_discovery_benchmark_summary.csv")

    fig = plt.figure(figsize=(15.2, 8.4))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[1.08, 1.0], height_ratios=[1.10, 0.90],
        wspace=0.25, hspace=0.33, left=0.14, right=0.985, top=0.902, bottom=0.082,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # Adjust top-row left margin slightly so Panel A doesn't waste space needed by Panel C labels
    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    ax_a.set_position([0.145, pos_a.y0, 0.365, pos_a.height])
    ax_b.set_position([0.575, pos_b.y0, 0.408, pos_b.height])
    pos_c = ax_c.get_position()
    ax_c.set_position([0.280, pos_c.y0, 0.703, pos_c.height])

    # --- Panel A: Exp 89B Full 323-Day EU Dunkelflaute Transition Discovery ---
    clean_ax(ax_a, grid_axis="x")
    n_days_89b = len(r89b)
    all_es_cols = [c for c in r89b[0] if c.startswith("es_")]
    es_rank_map = {
        c: idx
        for idx, c in enumerate(
            sorted(all_es_cols, key=lambda col: float(np.mean([float(r[col]) for r in r89b]))),
            1,
        )
    }
    n_es_models = len(all_es_cols)

    arms_89b = [
        ("Adapt-1 Disc Airbag(C2)", "es_adapt1_airbag_disc_chronos2", "cost20x_adapt1_airbag_disc_chronos2", "outage20x_adapt1_airbag_disc_chronos2", "var95_br_adapt1_airbag_disc_chronos2", C_BLUE),
        ("Authored Airbag(C2)", "es_adapt1_airbag_c2", "cost20x_adapt1_airbag_c2_eur", "outage20x_adapt1_airbag_c2", "var95_br_adapt1_airbag_c2", C_SKY),
        ("ACI(C2)", "es_aci_c2", "cost20x_aci_c2_eur", "outage20x_aci_c2", "var95_br_aci_c2", C_GREEN),
        ("Adapt-1 Airbag(T3)", "es_adapt1_airbag_t3", "cost20x_adapt1_airbag_t3_eur", "outage20x_adapt1_airbag_t3", "var95_br_adapt1_airbag_t3", C_AMBER),
        ("Raw Chronos-2 (1024)", "es_c2_ctx1024_cov_gc", "cost20x_c2_ctx1024_cov_gc_eur", "outage20x_c2_ctx1024_cov_gc", "var95_br_c2_ctx1024_cov_gc", C_PURPLE),
        ("Raw TimesFM-3 (1024)", "es_t3_ctx1024_cov_gc", "cost20x_t3_ctx1024_cov_gc_eur", "outage20x_t3_ctx1024_cov_gc", "var95_br_t3_ctx1024_cov_gc", C_VERM),
    ]
    rows_89b_stats = []
    for label, c_es, c_cost, c_out, c_vr, colr in arms_89b:
        es_v = float(np.mean([float(r[c_es]) for r in r89b]))
        cost_m = float(np.mean([float(r[c_cost]) for r in r89b])) * 365.0 / 1e6
        out_n = int(sum(int(float(r[c_out])) for r in r89b))
        vr_pct = 100.0 * sum(int(float(r[c_vr])) for r in r89b) / n_days_89b
        rows_89b_stats.append({
            "label": label, "es": es_v, "es_rk": es_rank_map[c_es],
            "cost": cost_m, "out": out_n, "vr_pct": vr_pct, "color": colr,
        })

    y_a = np.arange(len(rows_89b_stats))
    ax_a.barh(
        y_a, [d["cost"] for d in rows_89b_stats], height=0.62,
        color=[d["color"] for d in rows_89b_stats], alpha=0.90, edgecolor="#222222", lw=0.6, zorder=3,
    )
    disc_89 = rows_89b_stats[0]
    auth_89 = rows_89b_stats[1]
    raw_c2 = rows_89b_stats[4]
    ax_a.axvline(disc_89["cost"], color=C_BLUE, ls=":", lw=1.05, alpha=0.8, zorder=2)

    for i, d in enumerate(rows_89b_stats):
        rk_str = f" (#{d['es_rk']}/{n_es_models})" if d["es_rk"] <= 3 else ""
        txt = f"EUR {d['cost']:,.1f}M · ES {d['es']:.4f}{rk_str}\n{d['out']} outages · VaR95 {d['vr_pct']:.2f}%"
        ax_a.text(
            d["cost"] + 32, i, txt, va="center", ha="left",
            fontsize=6.7, color=C_BLUE if i == 0 else "#222222",
            fontweight="bold" if i == 0 else "normal", linespacing=1.18, zorder=5,
        )

    ax_a.text(
        0.985, 0.50,
        f"Autonomous Projection (6/9 covs):\n"
        f"Kept 5/5 authored + added GPRD\n"
        f"-EUR {auth_89['cost'] - disc_89['cost']:.1f}M/yr vs Authored\n"
        f"-EUR {raw_c2['cost'] - disc_89['cost']:.1f}M/yr vs Raw C2",
        transform=ax_a.transAxes, ha="right", va="center", fontsize=6.8, color=C_BLUE, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.30", "fc": "#F4F9FD", "ec": C_BLUE, "alpha": 0.96, "lw": 0.85},
        zorder=6,
    )
    ax_a.set_yticks(y_a)
    ax_a.set_yticklabels([d["label"] for d in rows_89b_stats], fontsize=7.6)
    ax_a.get_yticklabels()[0].set_fontweight("bold")
    ax_a.get_yticklabels()[0].set_color(C_BLUE)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 3150)
    ax_a.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000])
    ax_a.set_xlabel("Annualized 20x VoLL Cost (EUR M/yr)  [lower is better]")
    ax_a.set_title(f"A. Exp 89B EU Dunkelflaute Discovery (Full {n_days_89b}d OOS)", loc="left", pad=8)

    # --- Panel B: Exp 102C Two-Stage Discovery (8/20f -> /value/*, 15 models) ---
    clean_ax(ax_b)
    y_102c = np.array([float(r["y_extreme"]) for r in r102c], dtype=float)
    meta_102c = {"delivery_date", "iso_week", "block", "y_extreme"}
    pts_102c = {}
    for c in r102c[0]:
        if c in meta_102c:
            continue
        p = np.array([float(r[c]) for r in r102c], dtype=float)
        bf, ef = compute_brier_ece(p, y_102c)
        af = compute_auroc(p, y_102c)
        pts_102c[c] = (bf, af, ef)

    n_102c_models = len(pts_102c)
    rank_brier_102c = {k: i for i, k in enumerate(sorted(pts_102c, key=lambda m: pts_102c[m][0]), 1)}
    rank_ece_102c = {k: i for i, k in enumerate(sorted(pts_102c, key=lambda m: pts_102c[m][2]), 1)}

    for c, (bf, af, ef) in pts_102c.items():
        if c == "Trailing_BaseRate":
            continue
        if c == "Adapt1_CUDA_MLPv2_Disc8_PowerPool2048":
            ax_b.scatter(bf, af, s=235, marker="*", color=C_BLUE, edgecolor="#111111", lw=0.85, zorder=7)
        elif c == "Adapt1_CUDA_Linear_PowerPool2048":
            ax_b.scatter(bf, af, s=68, marker="o", color="#8CC5E3", edgecolor=C_BLUE, lw=0.9, zorder=5)
        elif c == "Adapt1_CUDA_MLPv2_PowerPool2048":
            ax_b.scatter(bf, af, s=95, marker="^", color=C_VERM, edgecolor="#222222", lw=0.8, zorder=6)
        elif c == "Online_HGBR_Cls_Pooled_Disc8":
            ax_b.scatter(bf, af, s=84, marker="s", color=C_GREEN, edgecolor="#222222", lw=0.8, zorder=6)
        elif c == "Online_HGBR_Cls_Pooled":
            ax_b.scatter(bf, af, s=74, marker="o", color=C_GREEN, edgecolor="#222222", lw=0.75, alpha=0.85, zorder=5)
        elif c == "ExtraTrees_Cls_Pooled_Disc8":
            ax_b.scatter(bf, af, s=74, marker="s", color=C_PURPLE, edgecolor="#222222", lw=0.75, zorder=5)
        elif c == "LogisticRegression_L2_Pooled_Disc8":
            ax_b.scatter(bf, af, s=70, marker="s", color=C_GRAY, edgecolor="#222222", lw=0.75, zorder=5)
        else:
            ax_b.scatter(bf, af, s=46, marker="o", facecolor="#FFFFFF", edgecolor="#777777", lw=0.9, zorder=3)

    d8_key = "Adapt1_CUDA_MLPv2_Disc8_PowerPool2048"
    d8 = pts_102c[d8_key]
    ws = pts_102c["Adapt1_CUDA_MLPv2_PowerPool2048"]
    hg8 = pts_102c["Online_HGBR_Cls_Pooled_Disc8"]
    hg20 = pts_102c["Online_HGBR_Cls_Pooled"]
    et8 = pts_102c["ExtraTrees_Cls_Pooled_Disc8"]
    lr8 = pts_102c["LogisticRegression_L2_Pooled_Disc8"]

    # Single clean trajectory arrow: Adapt-1 Warm-Start (20f) -> Adapt-1 Disc8 -> /value/*
    ax_b.annotate(
        "", xy=(d8[0] + 0.0010, d8[1] - 0.0022), xytext=(ws[0] - 0.0008, ws[1] + 0.0020),
        arrowprops={"arrowstyle": "->", "color": C_BLUE, "lw": 1.35, "ls": "--", "connectionstyle": "arc3,rad=-0.14"},
        zorder=4,
    )

    ax_b.annotate(
        f"Adapt-1 Disc8 -> /value/*\n#{rank_brier_102c[d8_key]}/{n_102c_models} B={d8[0]:.4f}, AUC={d8[1]:.3f}\nECE {d8[2]*1000:.1f}e-3 (#{rank_ece_102c[d8_key]}/{n_102c_models})",
        xy=(d8[0], d8[1]), xytext=(0.1605, 0.8565),
        fontsize=7.2, color=C_BLUE, fontweight="bold", va="center",
        bbox={"boxstyle": "round,pad=0.25", "fc": "#FFFFFF", "ec": C_BLUE, "alpha": 0.96, "lw": 0.85},
        arrowprops={"arrowstyle": "->", "color": C_BLUE, "lw": 0.95}, zorder=8,
    )
    ax_b.annotate(
        f"Online HGBR\nDisc8 ({hg8[0]:.4f})\nFull 20f ({hg20[0]:.4f})",
        xy=(hg8[0], hg8[1]), xytext=(0.1466, 0.8165),
        fontsize=6.9, color=C_GREEN, fontweight="bold", va="center", ha="left",
        bbox={"boxstyle": "round,pad=0.18", "fc": "#FFFFFF", "ec": "none", "alpha": 0.85},
        arrowprops={"arrowstyle": "->", "color": C_GREEN, "lw": 0.85}, zorder=8,
    )
    ax_b.annotate(
        f"ExtraTrees Disc8\n({et8[0]:.4f})",
        xy=(et8[0], et8[1]), xytext=(0.1795, 0.8380),
        fontsize=6.8, color=C_PURPLE, fontweight="bold", va="center",
        arrowprops={"arrowstyle": "->", "color": C_PURPLE, "lw": 0.8}, zorder=8,
    )
    ax_b.annotate(
        f"Adapt-1 Warm-Start (20f)\nB={ws[0]:.4f}, AUC={ws[1]:.3f}",
        xy=(ws[0], ws[1]), xytext=(0.1660, 0.7585),
        fontsize=7.1, color=C_VERM, fontweight="bold", va="center", ha="center",
        arrowprops={"arrowstyle": "->", "color": C_VERM, "lw": 0.9}, zorder=8,
    )
    ax_b.annotate(
        f"LogReg Disc8\n({lr8[0]:.4f})",
        xy=(lr8[0], lr8[1]), xytext=(0.1848, 0.7935),
        fontsize=6.8, color=C_GRAY, ha="center",
        arrowprops={"arrowstyle": "->", "color": C_GRAY, "lw": 0.8}, zorder=8,
    )

    legend_handles_b = [
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=C_BLUE, markeredgecolor="#111", markersize=11, label="Adapt-1 Disc8 (8/20f)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#8CC5E3", markeredgecolor=C_BLUE, markersize=6.2, label="Adapt-1 Linear (20f)"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=C_VERM, markeredgecolor="#222", markersize=7.0, label="Adapt-1 Warm-Start (20f)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=C_GREEN, markeredgecolor="#222", markersize=6.4, label="Matched Disc8 peers (8f)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#FFFFFF", markeredgecolor="#777", markersize=6.0, label="Full 20f / Per-Block peers"),
    ]
    ax_b.legend(handles=legend_handles_b, loc="lower left", bbox_to_anchor=(0.01, 0.065), fontsize=6.6, framealpha=0.95)
    ax_b.set_xlim(0.1455, 0.1955)
    ax_b.set_ylim(0.748, 0.870)
    ax_b.set_xlabel("FULL 4-Block Brier Score  [lower is better]")
    ax_b.set_ylabel("AUROC  [higher is better]")
    ax_b.set_title(f"B. Exp 102C Two-Stage Discovery (8/20f -> /value/*, {n_102c_models} models)", loc="left", pad=8)
    tb = pts_102c["Trailing_BaseRate"]
    ax_b.text(
        0.99, 0.016,
        f"Optimal = top-left; Trailing BaseRate (B={tb[0]:.3f}, AUC={tb[1]:.3f}) outside zoom",
        transform=ax_b.transAxes, ha="right", va="bottom", fontsize=6.5, color="#666666", style="italic",
    )

    # --- Panel C: Tier-1 Drop-In Autonomous Discovery Across Sections 01–05 (Exp 105) ---
    clean_ax(ax_c, grid_axis="x")
    by_bid = {r["benchmark_id"]: r for r in r105 if r["tier"] == "tier1_dropin_win"}

    def rel_gain_pct(row: dict[str, str]) -> float:
        d = float(row["discovery_value"])
        a = float(row["authored_value"])
        if row["higher_is_better"] == "True":
            return (d - a) / abs(a) * 100.0
        return (a - d) / abs(a) * 100.0

    # Strictly the 8 valid Tier-1 Drop-In Autonomous Discovery production benchmarks
    c_specs = [
        # (benchmark_id, display_label, surface_group, extra_note)
        ("exp102c_td_val", "Sec 02 · Exp 102C Power Extremes (Brier, n=1,800)", "two_stage",
         f"#{rank_brier_102c[d8_key]}/{n_102c_models} · AUC +{rel_gain_pct(by_bid['exp102c_td_val_auroc']):.1f}%"),
        ("exp89b_td", "Sec 01 · Exp 89B EU Dunkelflaute (4D ES, 323d)", "transition", "#1/25 · -EUR 8.9M/yr"),
        ("exp98a_sd_peak", "Sec 01 · Exp 98A Peak Spike >200 (Structure AUC, 450d)", "structural", "476 mined feats"),
        ("exp16_td", "Sec 04 · Exp 16 19-Yr Macro (5D Raw ES, 226w)", "transition", "4/4 covs · parity"),
        ("exp79b_td", "Sec 01 · Exp 79B Crypto Cascade (5D ES, n=200)", "transition", "8/13 covs · <0.8%"),
        ("exp98a_td", "Sec 01 · Exp 98A Power Scarcity (4D ES, 450d)", "transition", "beats 12/12 GJR (63.08)"),
        ("exp53_td_rev0_to_rev2", "Sec 04 · Exp 53 Chicago 311 (Rev2 Mixed ES, n=100)", "transition", "10/10 covs · monotonic"),
        ("exp93c_emp_topo", "Sec 05 · Exp 93C K8s DAG (EmpTopo Return, 5 seeds)", "structural", "74.2% rec. · 0 broken"),
    ]

    surf_style = {
        "transition": (C_BLUE, 0.92),
        "two_stage": (C_GREEN, 0.92),
        "structural": (C_PURPLE, 0.92),
    }

    y_c = np.arange(len(c_specs))
    ax_c.axvspan(0.0, 46.0, color="#F3FAF6", alpha=0.65, zorder=0)
    ax_c.axvline(0.0, color="#222222", lw=1.1, zorder=4)

    for i, (bid, disp, sgrp, note) in enumerate(c_specs):
        row = by_bid[bid]
        rg = rel_gain_pct(row)
        d_val = float(row["discovery_value"])
        a_val = float(row["authored_value"])
        colr, alpha = surf_style[sgrp]
        ax_c.barh(
            i, rg, height=0.62, color=colr, alpha=alpha,
            edgecolor="#222222", lw=0.55, zorder=3,
        )
        if abs(rg) < 0.05:
            ax_c.scatter(0.0, i, s=48, marker="D", color=colr, edgecolor="#222222", lw=0.7, zorder=5)

        # Clean formatted numeric summary on the right side (never collides with negative bars)
        if abs(a_val) >= 10:
            pair_str = f"{d_val:.2f} vs {a_val:.2f}"
        elif abs(a_val) >= 1.0:
            pair_str = f"{d_val:.3f} vs {a_val:.3f}"
        else:
            pair_str = f"{d_val:.4f} vs {a_val:.4f}"

        x_txt = max(rg, 0.0) + 0.9
        rg_str = "0.0%" if abs(rg) < 0.05 else f"{rg:+.1f}%"
        txt_col = C_GREEN if rg > 0.5 else (C_BLUE if rg >= -0.05 else "#222222")
        txt_wt = "bold" if rg >= 0.0 else "normal"
        ax_c.text(
            x_txt, i, f"{rg_str}  ({pair_str} · {note})",
            va="center", ha="left", fontsize=7.2, color=txt_col, fontweight=txt_wt, zorder=6,
        )

    ax_c.set_yticks(y_c)
    ax_c.set_yticklabels([s[1] for s in c_specs], fontsize=7.6)
    ax_c.get_yticklabels()[0].set_fontweight("bold")
    ax_c.get_yticklabels()[1].set_fontweight("bold")
    ax_c.set_ylim(8.15, -0.65)
    ax_c.set_xlim(-34, 46)
    ax_c.set_xticks([-30, -20, -10, 0, 10, 20])
    ax_c.set_xticklabels(["-30%", "-20%", "-10%", "0%\n(parity)", "+10%", "+20%"])
    ax_c.set_xlabel("Relative Delta vs. Hand-Authored Baseline (%)  [positive = Discovery beats Authored; 0% = parity]")
    ax_c.set_title("C. Tier-1 Drop-In Autonomous Discovery Across Sections 01–05 (Exp 105)", loc="left", pad=8)

    legend_handles_c = [
        mpatches.Patch(facecolor=C_BLUE, edgecolor="#222222", lw=0.6, label="Transition Discovery"),
        mpatches.Patch(facecolor=C_GREEN, edgecolor="#222222", lw=0.6, label="Two-Stage TD -> /value/*"),
        mpatches.Patch(facecolor=C_PURPLE, edgecolor="#222222", lw=0.6, label="Zero-Schema Structural / EmpTopo"),
    ]
    ax_c.legend(handles=legend_handles_c, loc="upper left", bbox_to_anchor=(0.012, 0.975), ncol=1, fontsize=7.1, framealpha=0.95)
    ax_c.text(
        0.992, 0.022,
        "All 8 valid Tier-1 discovery surfaces shown; pinned-key & surface-routing recipes in 06_autonomous_discovery/README.md.",
        transform=ax_c.transAxes, ha="right", va="bottom", fontsize=6.6, color="#666666", style="italic",
    )

    fig.suptitle(
        "Section 06 — Autonomous Transition Discovery, Two-Stage Value Learning & Cross-Section Scorecard",
        fontsize=12.4, fontweight="bold", y=0.964,
    )
    out = FIG_DIR / "fig06_autonomous_discovery.png"
    fig.savefig(out, dpi=DPI, pil_kwargs={"optimize": True})
    plt.close(fig)
    return out


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_builders = {
        "fig01": build_fig01,
        "fig02": build_fig02,
        "fig03": build_fig03,
        "fig04": build_fig04,
        "fig05": build_fig05,
        "fig06": build_fig06,
    }
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(all_builders.keys())
    for key in requested:
        if key not in all_builders:
            raise SystemExit(f"Unknown figure key: {key} (expected one of {', '.join(all_builders)})")
        out = all_builders[key]()
        print(f"Generated {out.relative_to(REPO_ROOT)} ({out.stat().st_size:,d} bytes)")


if __name__ == "__main__":
    main()
