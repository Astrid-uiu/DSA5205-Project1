"""DSA5205 Project 1 - Task 2: Lasso on the same simulations as Task 1.

Run after task1_simulation.py (or directly):
    python task2_lasso.py

This script imports the DGP function and all fixed seeds from Task 1. Therefore
every Lasso model is evaluated on exactly the same simulated data, observed
feature blocks, and train/test splits as the ridge benchmark.
"""

from pathlib import Path
import os

OUTPUT_DIR = Path(__file__).resolve().parent / "task2_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from task1_simulation import (
    T_train, base_seed, calculate_metrics, cq_grid, generate_simulation_data,
    n_simulations,
)


# =============================================================================
# 1. New-model choice
# =============================================================================
# sklearn's Lasso solves:
# (1 / (2*T_train)) * ||y - X beta||^2 + alpha * ||beta||_1.
# Signals are already generated with unit variance, so no additional scaling is
# needed. This fixed alpha is stated in the report; it is not tuned on test data.
alpha = 0.05


def fit_lasso(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Fit a no-intercept Lasso and return the estimated coefficient vector."""
    model = Lasso(
        alpha=alpha,
        fit_intercept=False,  # Task 1 DGP and ridge benchmark have no intercept.
        max_iter=20_000,
        tol=1e-6,
        selection="cyclic",
    )
    model.fit(X_train, y_train)
    return model.coef_


def run_one_lasso_simulation(seed: int) -> list[dict[str, float]]:
    """Repeat every cq experiment using exactly Task 1's draw and permutation."""
    data = generate_simulation_data(seed)
    X_train_full = data["X_train_full"]
    X_test_full = data["X_test_full"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_permutation = data["feature_permutation"]
    rows: list[dict[str, float]] = []

    for cq_requested in cq_grid:
        P1 = int(round(cq_requested * T_train))
        observed_columns = feature_permutation[:P1]
        X_train = X_train_full[:, observed_columns]
        X_test = X_test_full[:, observed_columns]

        beta_hat = fit_lasso(X_train, y_train)
        y_hat = X_test @ beta_hat
        metrics = calculate_metrics(y_test, y_hat)
        rows.append({
            "seed": seed,
            "cq": P1 / T_train,
            "P1": P1,
            "alpha": alpha,
            "beta_norm_sq": float(beta_hat @ beta_hat),
            "n_nonzero": int(np.count_nonzero(beta_hat)),
            **metrics,
        })

    return rows


def make_figure(summary: pd.DataFrame, metric: str, ylabel: str, filename: str) -> None:
    """Task 2 has one fixed alpha, so each figure is one interpretable line."""
    fig, ax = plt.subplots(figsize=(8, 5))
    summary = summary.sort_values("cq")
    ax.plot(summary["cq"], summary[metric], color="#D55E00", marker="o",
            linewidth=2, label=f"Lasso, alpha={alpha:g}")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1,
               label="interpolation cq=1")
    ax.set_xscale("log")
    ax.set_xticks(cq_grid)
    ax.set_xticklabels([f"{value:g}" for value in cq_grid], rotation=45)
    ax.set_xlabel(r"Observed complexity $cq=P_1/T_{train}$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Task 2 Lasso: {ylabel} versus observed complexity")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("Running Task 2 Lasso on the same Task 1 simulations...")
    print(f"Fixed alpha={alpha}; {len(cq_grid)} cq values x {n_simulations} seeds")
    all_rows: list[dict[str, float]] = []

    for simulation_number in range(n_simulations):
        seed = base_seed + simulation_number
        all_rows.extend(run_one_lasso_simulation(seed))
        print(f"  completed simulation {simulation_number + 1}/{n_simulations}")

    raw = pd.DataFrame(all_rows)
    summary = (raw.groupby(["cq", "P1", "alpha"], as_index=False)
               .mean(numeric_only=True)
               .sort_values("cq"))
    raw.to_csv(OUTPUT_DIR / "task2_lasso_raw_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "task2_lasso_summary_results.csv", index=False)

    make_figure(summary, "r2_paper", r"Out-of-sample $R^2_{paper}$", "task2_r2_paper.png")
    make_figure(summary, "expected_timing_return", r"$E[R^{\pi}_{t+1}]$", "task2_expected_timing_return.png")
    make_figure(summary, "sharpe_paper", "Paper Sharpe ratio", "task2_sharpe_paper.png")
    make_figure(summary, "beta_norm_sq", r"$||\hat{\beta}||^2$", "task2_beta_norm_sq.png")
    print(f"\nFinished. Files written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
