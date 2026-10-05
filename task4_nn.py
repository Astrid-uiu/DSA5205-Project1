"""
DSA5205 Project 1 - Task 4
Early stopping, generalization, and overfitting
in a noisy return-prediction neural network.

This is a simple simulation motivated by:
Montanari & Urbani (2025),
"Dynamical Decoupling of Generalization and
Overfitting in Large Two-Layer Networks."

IMPORTANT:
This experiment is NOT a direct replication of
their DMFT theory. It is a small financial-ML
illustration of the paper's qualitative insight:
training fit and out-of-sample generalization can
decouple as training continues.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# Configuration
# ============================================================

SEED = 5205

N_TRAIN = 600
N_TEST = 2000

P = 20
HIDDEN_WIDTH = 128

NOISE_STD = 2.0

EPOCHS = 1000
LEARNING_RATE = 0.01

BATCH_SIZE = 64

EVALUATE_EVERY = 5

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "task4_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # More reproducible CPU/GPU behavior.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Data-generating process
# ============================================================

def true_signal(X):
    """
    Nonlinear conditional expected return.

    Only the first few features contain signal.
    Remaining features are irrelevant noise predictors.

    The signal is deliberately nonlinear so that a
    neural network has a meaningful feature-learning task.
    """

    signal = (
        0.8 * np.sin(X[:, 0])
        + 0.6 * X[:, 1] * X[:, 2]
        + 0.5 * (X[:, 3] ** 2 - 1.0)
        - 0.4 * np.tanh(X[:, 4])
    )

    return signal


def generate_data(seed):

    rng = np.random.default_rng(seed)

    n_total = N_TRAIN + N_TEST

    X = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_total, P),
    )

    mu = true_signal(X)

    epsilon = rng.normal(
        loc=0.0,
        scale=NOISE_STD,
        size=n_total,
    )

    y = mu + epsilon

    # Chronological split.
    X_train = X[:N_TRAIN]
    y_train = y[:N_TRAIN]
    mu_train = mu[:N_TRAIN]

    X_test = X[N_TRAIN:]
    y_test = y[N_TRAIN:]
    mu_test = mu[N_TRAIN:]

    return (
        X_train,
        y_train,
        mu_train,
        X_test,
        y_test,
        mu_test,
    )


# ============================================================
# Standardization
# ============================================================

def standardize_train_test(X_train, X_test):

    mean = X_train.mean(axis=0)

    std = X_train.std(axis=0)

    std[std < 1e-12] = 1.0

    X_train_std = (
        X_train - mean
    ) / std

    X_test_std = (
        X_test - mean
    ) / std

    return X_train_std, X_test_std


# ============================================================
# Two-layer neural network
# ============================================================

class TwoLayerNet(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_width,
    ):

        super().__init__()

        self.hidden = nn.Linear(
            input_dim,
            hidden_width,
        )

        self.activation = nn.ReLU()

        self.output = nn.Linear(
            hidden_width,
            1,
        )

    def forward(self, x):

        x = self.hidden(x)

        x = self.activation(x)

        x = self.output(x)

        return x.squeeze(-1)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    y,
    yhat,
):

    mse = float(
        np.mean(
            (y - yhat) ** 2
        )
    )

    denominator = np.mean(
        y**2
    )

    r2_paper = float(
        1.0
        - mse / denominator
    )

    timing_return = (
        yhat * y
    )

    expected_timing_return = float(
        np.mean(
            timing_return
        )
    )

    second_moment = np.mean(
        timing_return**2
    )

    if second_moment > 0:

        sharpe_paper = float(
            expected_timing_return
            / np.sqrt(
                second_moment
            )
        )

    else:

        sharpe_paper = np.nan

    timing_variance = np.var(
        timing_return
    )

    if timing_variance > 0:

        sharpe_var = float(
            expected_timing_return
            / np.sqrt(
                timing_variance
            )
        )

    else:

        sharpe_var = np.nan

    return {
        "mse": mse,
        "r2_paper": r2_paper,
        "expected_timing_return":
            expected_timing_return,
        "sharpe_paper":
            sharpe_paper,
        "sharpe_var":
            sharpe_var,
    }


# ============================================================
# Evaluation
# ============================================================

def predict(
    model,
    X,
    device,
):

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():

        yhat = model(
            X_tensor
        ).cpu().numpy()

    return yhat


# ============================================================
# Main experiment
# ============================================================

def run_experiment():

    set_seed(SEED)

    (
        X_train,
        y_train,
        mu_train,
        X_test,
        y_test,
        mu_test,
    ) = generate_data(SEED)

    (
        X_train,
        X_test,
    ) = standardize_train_test(
        X_train,
        X_test,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(
        f"Using device: {device}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    X_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y_train,
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = TwoLayerNet(
        input_dim=P,
        hidden_width=HIDDEN_WIDTH,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.MSELoss()

    history = []

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        for xb, yb in loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            prediction = model(xb)

            loss = criterion(
                prediction,
                yb,
            )

            loss.backward()

            optimizer.step()

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        if (
            epoch == 1
            or epoch % EVALUATE_EVERY == 0
        ):

            train_hat = predict(
                model,
                X_train,
                device,
            )

            test_hat = predict(
                model,
                X_test,
                device,
            )

            train_metrics = (
                compute_metrics(
                    y_train,
                    train_hat,
                )
            )

            test_metrics = (
                compute_metrics(
                    y_test,
                    test_hat,
                )
            )

            # Accuracy relative to the TRUE conditional
            # expected return is observable here because
            # this is simulated data.
            signal_mse = float(
                np.mean(
                    (
                        test_hat
                        - mu_test
                    ) ** 2
                )
            )

            # Parameter norm as a simple diagnostic of
            # effective model size / weight growth.
            parameter_norm_sq = 0.0

            with torch.no_grad():

                for parameter in (
                    model.parameters()
                ):

                    parameter_norm_sq += (
                        parameter
                        .pow(2)
                        .sum()
                        .item()
                    )

            history.append({
                "epoch":
                    epoch,

                "train_mse":
                    train_metrics["mse"],

                "test_mse":
                    test_metrics["mse"],

                "test_r2_paper":
                    test_metrics[
                        "r2_paper"
                    ],

                "test_expected_timing_return":
                    test_metrics[
                        "expected_timing_return"
                    ],

                "test_sharpe_paper":
                    test_metrics[
                        "sharpe_paper"
                    ],

                "test_sharpe_var":
                    test_metrics[
                        "sharpe_var"
                    ],

                "test_signal_mse":
                    signal_mse,

                "parameter_norm_sq":
                    parameter_norm_sq,
            })

            if epoch % 100 == 0:

                print(
                    f"Epoch {epoch:4d} | "
                    f"Train MSE "
                    f"{train_metrics['mse']:.4f} | "
                    f"Test MSE "
                    f"{test_metrics['mse']:.4f} | "
                    f"R2 "
                    f"{test_metrics['r2_paper']:.4f} | "
                    f"Sharpe "
                    f"{test_metrics['sharpe_paper']:.4f}"
                )

    return pd.DataFrame(history)


# ============================================================
# Plotting
# ============================================================

def make_plots(results):

    # --------------------------------------------------------
    # Figure 1:
    # Train vs test error
    # --------------------------------------------------------

    best_mse_idx = (
        results["test_mse"].idxmin()
    )

    best_mse_epoch = int(
        results.loc[
            best_mse_idx,
            "epoch",
        ]
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        results["epoch"],
        results["train_mse"],
        label="Training MSE",
    )

    plt.plot(
        results["epoch"],
        results["test_mse"],
        label="Test MSE",
    )

    plt.axvline(
        best_mse_epoch,
        linestyle="--",
        label=(
            "Minimum test MSE "
            f"(epoch {best_mse_epoch})"
        ),
    )

    plt.xlabel(
        "Training epoch"
    )

    plt.ylabel(
        "Mean squared error"
    )

    plt.title(
        "Task 4: Training Fit and "
        "Out-of-Sample Generalization"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "task4_train_test_mse.png",
        dpi=300,
    )

    plt.close()

    # --------------------------------------------------------
    # Figure 2:
    # OOS paper R2
    # --------------------------------------------------------

    best_r2_idx = (
        results[
            "test_r2_paper"
        ].idxmax()
    )

    best_r2_epoch = int(
        results.loc[
            best_r2_idx,
            "epoch",
        ]
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        results["epoch"],
        results["test_r2_paper"],
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.axvline(
        best_r2_epoch,
        linestyle="--",
        label=(
            "Best OOS R² "
            f"(epoch {best_r2_epoch})"
        ),
    )

    plt.xlabel(
        "Training epoch"
    )

    plt.ylabel(
        r"Out-of-sample $R^2_{paper}$"
    )

    plt.title(
        "Task 4: OOS Prediction "
        "Performance over Training"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "task4_oos_r2.png",
        dpi=300,
    )

    plt.close()

    # --------------------------------------------------------
    # Figure 3:
    # OOS Sharpe
    # --------------------------------------------------------

    best_sr_idx = (
        results[
            "test_sharpe_paper"
        ].idxmax()
    )

    best_sr_epoch = int(
        results.loc[
            best_sr_idx,
            "epoch",
        ]
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        results["epoch"],
        results[
            "test_sharpe_paper"
        ],
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.axvline(
        best_sr_epoch,
        linestyle="--",
        label=(
            "Best OOS Sharpe "
            f"(epoch {best_sr_epoch})"
        ),
    )

    plt.xlabel(
        "Training epoch"
    )

    plt.ylabel(
        "Paper Sharpe ratio"
    )

    plt.title(
        "Task 4: Economic Value "
        "over Training"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "task4_oos_sharpe.png",
        dpi=300,
    )

    plt.close()

    # --------------------------------------------------------
    # Figure 4:
    # Fit to true signal
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        results["epoch"],
        results[
            "test_signal_mse"
        ],
    )

    plt.xlabel(
        "Training epoch"
    )

    plt.ylabel(
        "MSE relative to true conditional mean"
    )

    plt.title(
        "Task 4: Learning and "
        "Potential Unlearning of Signal"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "task4_signal_mse.png",
        dpi=300,
    )

    plt.close()

    return (
        best_mse_epoch,
        best_r2_epoch,
        best_sr_epoch,
    )


# ============================================================
# Summary
# ============================================================

def create_summary(
    results,
    best_mse_epoch,
    best_r2_epoch,
    best_sr_epoch,
):

    best_mse_row = results[
        results["epoch"]
        == best_mse_epoch
    ].iloc[0]

    best_r2_row = results[
        results["epoch"]
        == best_r2_epoch
    ].iloc[0]

    best_sr_row = results[
        results["epoch"]
        == best_sr_epoch
    ].iloc[0]

    final_row = (
        results.iloc[-1]
    )

    summary = pd.DataFrame([
        {
            "criterion":
                "minimum_test_mse",

            "epoch":
                best_mse_epoch,

            "train_mse":
                best_mse_row[
                    "train_mse"
                ],

            "test_mse":
                best_mse_row[
                    "test_mse"
                ],

            "test_r2_paper":
                best_mse_row[
                    "test_r2_paper"
                ],

            "test_sharpe_paper":
                best_mse_row[
                    "test_sharpe_paper"
                ],
        },

        {
            "criterion":
                "maximum_test_r2",

            "epoch":
                best_r2_epoch,

            "train_mse":
                best_r2_row[
                    "train_mse"
                ],

            "test_mse":
                best_r2_row[
                    "test_mse"
                ],

            "test_r2_paper":
                best_r2_row[
                    "test_r2_paper"
                ],

            "test_sharpe_paper":
                best_r2_row[
                    "test_sharpe_paper"
                ],
        },

        {
            "criterion":
                "maximum_test_sharpe",

            "epoch":
                best_sr_epoch,

            "train_mse":
                best_sr_row[
                    "train_mse"
                ],

            "test_mse":
                best_sr_row[
                    "test_mse"
                ],

            "test_r2_paper":
                best_sr_row[
                    "test_r2_paper"
                ],

            "test_sharpe_paper":
                best_sr_row[
                    "test_sharpe_paper"
                ],
        },

        {
            "criterion":
                "final_epoch",

            "epoch":
                int(
                    final_row["epoch"]
                ),

            "train_mse":
                final_row[
                    "train_mse"
                ],

            "test_mse":
                final_row[
                    "test_mse"
                ],

            "test_r2_paper":
                final_row[
                    "test_r2_paper"
                ],

            "test_sharpe_paper":
                final_row[
                    "test_sharpe_paper"
                ],
        },
    ])

    summary.to_csv(
        OUTPUT_DIR
        / "task4_summary.csv",
        index=False,
    )

    return summary


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 65
    )

    print(
        "DSA5205 PROJECT 1 - TASK 4"
    )

    print(
        "=" * 65
    )

    print()

    print(
        f"Train observations : {N_TRAIN}"
    )

    print(
        f"Test observations  : {N_TEST}"
    )

    print(
        f"Features           : {P}"
    )

    print(
        f"Hidden neurons     : {HIDDEN_WIDTH}"
    )

    print(
        f"Noise std          : {NOISE_STD}"
    )

    print(
        f"Training epochs    : {EPOCHS}"
    )

    print()

    results = run_experiment()

    results.to_csv(
        OUTPUT_DIR
        / "task4_training_history.csv",
        index=False,
    )

    (
        best_mse_epoch,
        best_r2_epoch,
        best_sr_epoch,
    ) = make_plots(
        results
    )

    summary = create_summary(
        results,
        best_mse_epoch,
        best_r2_epoch,
        best_sr_epoch,
    )

    print()
    print(
        "=" * 65
    )

    print(
        "TASK 4 SUMMARY"
    )

    print(
        "=" * 65
    )

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Figures and CSV files saved to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()