"""DSA5205 Project 1 - Task 1: misspecified simulation with ridge.

Run from this folder:
    python task1_simulation.py

Outputs are written to task1_outputs/:
    task1_raw_results.csv       one row for each seed, cq, and z
    task1_summary_results.csv   averages across seeds
    task1_*.png                 four required figures
"""

from pathlib import Path
import os

OUTPUT_DIR = Path(__file__).resolve().parent / "task1_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# 1. Experimental design - report these choices in Task 1
# =============================================================================
T_train = 100
T_test = 2_000
c_true = 10                    # P / T_train in the true DGP
P = c_true * T_train
b_star = 1.0                   # ||beta_star||^2
sigma_epsilon = 1.0
n_simulations = 10             # independent finite-sample repetitions
base_seed = 20260902

# The handout's suggested grids. cq is denser near interpolation (cq = 1).
cq_grid = np.array([
    0.50, 0.75, 0.90, 0.95, 0.98,
    1.02, 1.05, 1.10, 1.25, 1.50,
    2.00, 3.00, 5.00, 7.00, 10.00,
])
z_grid = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0])


# =============================================================================
# 2. Estimation and evaluation functions
# =============================================================================
def ridge_path_svd(X: np.ndarray, y: np.ndarray, z_values: np.ndarray) -> list[np.ndarray]:
    """Fit all ridge models using the handout scaling and one stable SVD.

    beta_hat(z) = (X'X + z*T_train*I)^(-1) X'y.
    At z=0, the SVD denominator implements the Moore-Penrose minimum-norm
    solution. We never directly invert X'X.
    """
    n_train = X.shape[0]
    U, singular_values, Vt = np.linalg.svd(X, full_matrices=False)
    Uy = U.T @ y
    coefficients = []

    for z in z_values:
        denominator = singular_values ** 2 + z * n_train
        weights = np.divide(
            singular_values * Uy, denominator,
            out=np.zeros_like(singular_values), where=denominator > 1e-12,
        )
        coefficients.append(Vt.T @ weights)

    return coefficients


def calculate_metrics(y: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    """Calculate the requested out-of-sample Task 1 quantities."""
    timing_return = y_hat * y
    r2_paper = 1.0 - np.mean((y - y_hat) ** 2) / np.mean(y ** 2)
    expected_timing_return = np.mean(timing_return)

    # Eq. (5): this paper Sharpe uses an uncentered second moment.
    sharpe_paper = expected_timing_return / np.sqrt(np.mean(timing_return ** 2))
    variance = np.var(timing_return)
    sharpe_var = expected_timing_return / np.sqrt(variance) if variance > 0 else np.nan

    return {
        "r2_paper": r2_paper,
        "expected_timing_return": expected_timing_return,
        "sharpe_paper": sharpe_paper,
        "sharpe_var": sharpe_var,
    }


# =============================================================================
# 3. One complete misspecified simulation run
# =============================================================================
def generate_simulation_data(seed: int) -> dict[str, np.ndarray]:
    """Generate one DGP draw, shared by Tasks 1 and 2 for a fair comparison."""
    rng = np.random.default_rng(seed)

    # Dense, isotropic beta*: all P signals enter the DGP, ||beta*||^2=b*.
    beta_star = rng.normal(size=P)
    beta_star *= np.sqrt(b_star) / np.linalg.norm(beta_star)

    # Psi=I: every signal row is N(0, I_P).
    S_all = rng.normal(size=(T_train + T_test, P))
    epsilon = rng.normal(scale=sigma_epsilon, size=T_train + T_test)
    y_all = S_all @ beta_star + epsilon

    X_train_full, X_test_full = S_all[:T_train], S_all[T_train:]
    y_train, y_test = y_all[:T_train], y_all[T_train:]

    # Draw ONE permutation and reuse it for every cq: observed sets are nested.
    feature_permutation = rng.permutation(P)

    return {
        "X_train_full": X_train_full,
        "X_test_full": X_test_full,
        "y_train": y_train,
        "y_test": y_test,
        "feature_permutation": feature_permutation,
    }


def run_one_simulation(seed: int) -> list[dict[str, float]]:
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

        # One SVD gives every ridge estimator at this cq.
        beta_path = ridge_path_svd(X_train, y_train, z_grid)
        for z, beta_hat in zip(z_grid, beta_path):
            y_hat = X_test @ beta_hat
            metrics = calculate_metrics(y_test, y_hat)
            rows.append({
                "seed": seed,
                "cq": P1 / T_train,       # actual cq after integer rounding
                "P1": P1,
                "z": z,
                "beta_norm_sq": float(beta_hat @ beta_hat),
                **metrics,
            })

    return rows


# =============================================================================
# 4. Figure creation
# =============================================================================
def make_figure(summary: pd.DataFrame, metric: str, ylabel: str, filename: str) -> None:
    """Plot one requested metric against cq, with one line per z."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(z_grid)))

    for color, z in zip(colors, z_grid):
        subset = summary.loc[summary["z"] == z].sort_values("cq")
        ax.plot(subset["cq"], subset[metric], marker="o", markersize=3,
                linewidth=1.5, color=color, label=f"z={z:g}")

    ax.axvline(1.0, color="black", linestyle="--", linewidth=1,
               label="interpolation cq=1")
    ax.set_xscale("log")
    ax.set_xticks(cq_grid)
    ax.set_xticklabels([f"{value:g}" for value in cq_grid], rotation=45)
    ax.set_xlabel(r"Observed complexity $cq=P_1/T_{train}$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Task 1: {ylabel} versus observed complexity")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 5. Run, save results, and make all figures
# =============================================================================
def main() -> None:
    print("Running Task 1 simulation...")
    print(f"DGP: T_train={T_train}, T_test={T_test}, P={P}, c={c_true}, b*={b_star}")
    print(f"Grid: {len(cq_grid)} cq values x {len(z_grid)} z values x {n_simulations} seeds")

    all_rows: list[dict[str, float]] = []
    for simulation_number in range(n_simulations):
        seed = base_seed + simulation_number
        all_rows.extend(run_one_simulation(seed))
        print(f"  completed simulation {simulation_number + 1}/{n_simulations}")

    raw = pd.DataFrame(all_rows)
    summary = (raw.groupby(["cq", "P1", "z"], as_index=False)
               .mean(numeric_only=True)
               .sort_values(["z", "cq"]))
    raw.to_csv(OUTPUT_DIR / "task1_raw_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "task1_summary_results.csv", index=False)

    make_figure(summary, "r2_paper", r"Out-of-sample $R^2_{paper}$", "task1_r2_paper.png")
    make_figure(summary, "expected_timing_return", r"$E[R^{\pi}_{t+1}]$", "task1_expected_timing_return.png")
    make_figure(summary, "sharpe_paper", "Paper Sharpe ratio", "task1_sharpe_paper.png")
    make_figure(summary, "beta_norm_sq", r"$||\hat{\beta}(z)||^2$", "task1_beta_norm_sq.png")

    print("\nFinished. Files written to:")
    print(OUTPUT_DIR)
    print("\nReport check: ridgeless z=0 should be unstable near cq=1;")
    print("positive ridge z should make coefficient norms and OOS metrics smoother.")


if __name__ == "__main__":
    main()
