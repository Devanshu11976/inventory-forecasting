"""
open_days_comparison.py
=======================
Addresses reviewer concern: does XGBoost's 70% MAE advantage hold up when
store-closure days (units_sold = 0, Open = 0) are excluded from evaluation?

Runs the same walk-forward evaluation, then splits metrics into:
  - ALL DAYS   (including closures, same as paper_results_collector.py)
  - OPEN DAYS  (Open = 1 only, dropping closure days from metric calc)

Reports side-by-side so the paper can honestly show both.

Usage:
    $env:PYTHONPATH="inventory-forecasting"
    python inventory-forecasting/open_days_comparison.py
"""

import os
import sqlite3
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.real_data_loader import load_rossmann_to_sqlite
from evaluation.walk_forward import run_walk_forward_evaluation
from evaluation.metrics import mae, rmse, mape, smape

# ── config ─────────────────────────────────────────────────────────────────────
STORE_IDS   = [1, 4, 9]
LIMIT_DAYS  = 730
TEST_DAYS   = 60

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(SCRIPT_DIR, "data", "open_days.db")
SCHEMA_PATH = os.path.join(SCRIPT_DIR, "data", "schema.sql")
TRAIN_CSV   = os.path.abspath(os.path.join(SCRIPT_DIR, "dataset", "train.csv"))
STORE_CSV   = os.path.abspath(os.path.join(SCRIPT_DIR, "dataset", "store.csv"))
PLOT_PATH   = os.path.join(SCRIPT_DIR, "fig_open_days_store1.png")


def init_db(db_path, schema_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        with open(schema_path) as f:
            conn.executescript(f.read())


def load_open_flag(train_csv, store_ids, limit_days=730):
    """
    Returns a dict: store_id (int) -> pd.DataFrame with columns
    [date, units_sold, open_flag], sorted by date, tail(limit_days).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_csv(train_csv, dtype={"StateHoliday": str})

    df = df[df["Store"].isin(store_ids)].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store", "Date"])

    result = {}
    for sid in store_ids:
        sub = df[df["Store"] == sid][["Date", "Sales", "Open"]].copy()
        sub = sub.rename(columns={"Date": "date", "Sales": "units_sold", "Open": "open_flag"})
        sub = sub.tail(limit_days).reset_index(drop=True)
        result[sid] = sub
    return result


def metrics_dict(act, pred, label=""):
    return {
        "label": label,
        "mae":   mae(act, pred),
        "rmse":  rmse(act, pred),
        "mape":  mape(act, pred),
        "smape": smape(act, pred),
        "n":     len(act),
    }


def pct_change(new_val, old_val):
    return (old_val - new_val) / old_val * 100.0 if old_val != 0 else float("nan")


def main():
    print("=" * 72)
    print("OPEN-DAYS-ONLY COMPARISON  (Rossmann Stores 1, 4, 9)")
    print("=" * 72)

    # 1. Init DB + load data
    init_db(DB_PATH, SCHEMA_PATH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        load_rossmann_to_sqlite(TRAIN_CSV, STORE_CSV, DB_PATH,
                                store_ids=STORE_IDS, limit_days=LIMIT_DAYS)

    # 2. Load Open flag alongside sales
    open_data = load_open_flag(TRAIN_CSV, STORE_IDS, LIMIT_DAYS)

    # 3. Per-store evaluation
    all_rows  = []   # for aggregate table

    for sid in STORE_IDS:
        df_open = open_data[sid]

        # Build sales df for walk-forward (must include product_id column)
        df_sales = df_open[["date", "units_sold"]].copy()
        df_sales["product_id"] = str(sid)
        df_sales["date"] = pd.to_datetime(df_sales["date"])
        df_sales = df_sales.sort_values("date").reset_index(drop=True)

        # open_flag for the test window (last TEST_DAYS rows)
        test_open = df_open["open_flag"].values[-TEST_DAYS:]
        open_mask = test_open == 1   # True on trading days

        n_closed  = int((test_open == 0).sum())
        n_open    = int(open_mask.sum())

        print(f"\n{'='*60}")
        print(f"  Store {sid}  |  test window: {TEST_DAYS} days  "
              f"({n_open} open, {n_closed} closed)")
        print(f"{'='*60}")
        print(f"  {'Model':<16} | {'ALL DAYS':^35} | {'OPEN DAYS ONLY':^35}")
        print(f"  {'':<16} | {'MAE':>7} {'RMSE':>8} {'MAPE%':>7} {'sMAPE%':>8} | "
              f"{'MAE':>7} {'RMSE':>8} {'MAPE%':>7} {'sMAPE%':>8}")
        print(f"  {'-'*16}-+-{'-'*35}-+-{'-'*35}")

        for model_name in ["baseline_ma", "arima", "xgboost"]:
            res = run_walk_forward_evaluation(df_sales, model_name, tune=False)
            act_all  = res["actual_units"].values
            pred_all = res["predicted_units"].values

            # All-days metrics
            m_all = metrics_dict(act_all, pred_all, "all")

            # Open-days-only metrics (drop closure days)
            act_open  = act_all[open_mask]
            pred_open = pred_all[open_mask]
            m_open = metrics_dict(act_open, pred_open, "open")

            print(f"  {model_name:<16} | "
                  f"{m_all['mae']:>7.1f} {m_all['rmse']:>8.1f} "
                  f"{m_all['mape']:>7.2f} {m_all['smape']:>8.2f} | "
                  f"{m_open['mae']:>7.1f} {m_open['rmse']:>8.1f} "
                  f"{m_open['mape']:>7.2f} {m_open['smape']:>8.2f}")

            all_rows.append({
                "store": sid,
                "model": model_name,
                "n_open": n_open, "n_closed": n_closed,
                # all days
                "mae_all":   m_all["mae"],   "rmse_all":   m_all["rmse"],
                "mape_all":  m_all["mape"],  "smape_all":  m_all["smape"],
                # open days only
                "mae_open":  m_open["mae"],  "rmse_open":  m_open["rmse"],
                "mape_open": m_open["mape"], "smape_open": m_open["smape"],
                # predictions/actuals for further analysis
                "_act_all": act_all, "_pred_all": pred_all,
                "_act_open": act_open, "_pred_open": pred_open,
            })

        # --- how much of XGBoost's gain is closure-driven? ---
        base_all  = next(r for r in all_rows if r["store"]==sid and r["model"]=="baseline_ma")
        xgb_all   = next(r for r in all_rows if r["store"]==sid and r["model"]=="xgboost")
        base_open = base_all["mae_open"]
        xgb_open  = xgb_all["mae_open"]
        base_a    = base_all["mae_all"]
        xgb_a     = xgb_all["mae_all"]

        gain_all  = pct_change(xgb_a,    base_a)
        gain_open = pct_change(xgb_open, base_open)

        print(f"\n  XGBoost vs Baseline:")
        print(f"    MAE improvement (ALL days) : +{gain_all:.1f}%")
        print(f"    MAE improvement (OPEN only): +{gain_open:.1f}%")
        closure_share = gain_all - gain_open
        print(f"    Closure-day contribution   : ~{closure_share:.1f} ppt of the {gain_all:.1f}% total")

    # ── aggregate tables ───────────────────────────────────────────────────────
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in all_rows])

    print("\n" + "=" * 72)
    print("AGGREGATE: AVERAGE ACROSS ALL 3 STORES")
    print("=" * 72)

    agg_all  = df.groupby("model")[["mae_all",  "rmse_all",  "mape_all",  "smape_all"]].mean()
    agg_open = df.groupby("model")[["mae_open", "rmse_open", "mape_open", "smape_open"]].mean()

    print(f"\n  {'Model':<16}  {'--- ALL DAYS ---':^36}  {'--- OPEN DAYS ONLY ---':^36}")
    print(f"  {'':16}  {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'sMAPE%':>8}  "
          f"{'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'sMAPE%':>8}")
    for model in ["baseline_ma", "arima", "xgboost"]:
        ra = agg_all.loc[model]
        ro = agg_open.loc[model]
        print(f"  {model:<16}  "
              f"{ra['mae_all']:>8.1f} {ra['rmse_all']:>8.1f} "
              f"{ra['mape_all']:>8.2f} {ra['smape_all']:>8.2f}  "
              f"{ro['mae_open']:>8.1f} {ro['rmse_open']:>8.1f} "
              f"{ro['mape_open']:>8.2f} {ro['smape_open']:>8.2f}")

    # headline improvement on open days
    mae_gain_all_avg  = pct_change(agg_all.loc["xgboost","mae_all"],
                                   agg_all.loc["baseline_ma","mae_all"])
    mae_gain_open_avg = pct_change(agg_open.loc["xgboost","mae_open"],
                                   agg_open.loc["baseline_ma","mae_open"])
    mape_gain_open    = pct_change(agg_open.loc["xgboost","mape_open"],
                                   agg_open.loc["baseline_ma","mape_open"])

    print(f"\n  XGBoost vs Baseline (3-store average):")
    print(f"    MAE improvement, ALL days   : +{mae_gain_all_avg:.1f}%")
    print(f"    MAE improvement, OPEN only  : +{mae_gain_open_avg:.1f}%")
    print(f"    MAPE improvement, OPEN only : +{mape_gain_open:.1f}%")
    print(f"    Closure-day contribution    : ~{mae_gain_all_avg - mae_gain_open_avg:.1f} ppt")

    # ── RMSE parity explanation ─────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("RMSE PARITY EXPLANATION (Baseline vs ARIMA)")
    print("=" * 72)
    for sid in STORE_IDS:
        base = next(r for r in all_rows if r["store"]==sid and r["model"]=="baseline_ma")
        arm  = next(r for r in all_rows if r["store"]==sid and r["model"]=="arima")
        n_closed = base["n_closed"]
        print(f"  Store {sid}: Baseline RMSE={base['rmse_all']:.1f}  "
              f"ARIMA RMSE={arm['rmse_all']:.1f}  "
              f"(closed days={n_closed} — squared errors on closure days dominate both)")

    print("\n  Both Baseline MA and ARIMA predict a non-zero value on closed days")
    print("  (they have no Open signal). The resulting large squared errors swamp")
    print("  the RMSE calculation equally for both models, making RMSE appear")
    print("  almost identical even when MAE diverges. RMSE on Open days only:")
    for sid in STORE_IDS:
        base = next(r for r in all_rows if r["store"]==sid and r["model"]=="baseline_ma")
        arm  = next(r for r in all_rows if r["store"]==sid and r["model"]=="arima")
        print(f"    Store {sid}: Baseline RMSE(open)={base['rmse_open']:.1f}  "
              f"ARIMA RMSE(open)={arm['rmse_open']:.1f}")

    # ── Figure: open-day comparison for Store 1 ─────────────────────────────
    s1_base = next(r for r in all_rows if r["store"]==1 and r["model"]=="baseline_ma")
    s1_xgb  = next(r for r in all_rows if r["store"]==1 and r["model"]=="xgboost")
    open_flag_s1 = open_data[1]["open_flag"].values[-TEST_DAYS:]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    x_all  = list(range(TEST_DAYS))
    x_open = [i for i, o in enumerate(open_flag_s1) if o == 1]

    for ax, (act, pred_base, pred_xgb, title) in [
        (axes[0], (s1_base["_act_all"], s1_base["_pred_all"],
                   s1_xgb["_pred_all"], "All days (incl. closures)")),
        (axes[1], (s1_base["_act_open"], s1_base["_pred_open"],
                   s1_xgb["_pred_open"], "Open days only")),
    ]:
        x = list(range(len(act)))
        ax.plot(x, act,       label="Actual",      color="#1a3a5c", linewidth=2)
        ax.plot(x, pred_xgb,  label="XGBoost",     color="#e85d04",
                linewidth=1.8, linestyle="--")
        ax.plot(x, pred_base, label="Baseline MA",  color="#2dc653",
                linewidth=1.4, linestyle=":", alpha=0.85)
        ax.set_title(f"Store 1 (Type C) - {title}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Test Day Index", fontsize=10)
        ax.set_ylabel("Daily Sales (units)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)

    plt.tight_layout(pad=2.5)
    plt.savefig(PLOT_PATH, dpi=300)
    plt.close()
    print(f"\n  Figure saved -> {PLOT_PATH}")

    print("\n" + "=" * 72)
    print("PAPER-READY SUMMARY")
    print("=" * 72)
    print(f"  All-days  MAE improvement (XGBoost vs Baseline): +{mae_gain_all_avg:.1f}%")
    print(f"  Open-only MAE improvement (XGBoost vs Baseline): +{mae_gain_open_avg:.1f}%")
    ppt_closure = mae_gain_all_avg - mae_gain_open_avg
    share = ppt_closure / mae_gain_all_avg * 100
    print(f"  Closure-day effect: {ppt_closure:.1f} ppt ({share:.0f}% of total advantage)")
    print(f"  Open-only MAPE improvement: +{mape_gain_open:.1f}%")
    print()
    print("  Interpretation for paper:")
    if mae_gain_open_avg > 40:
        print("    XGBoost retains a large advantage even on open trading days alone.")
        print("    Calendar features explain part of the closure improvement but the")
        print("    core forecasting advantage on normal days is robust.")
    else:
        print("    A significant portion of XGBoost's advantage is closure-driven.")
        print("    Recommend reporting open-days-only as the primary metric in paper.")
    print()
    print("  DONE.\n")


if __name__ == "__main__":
    main()
