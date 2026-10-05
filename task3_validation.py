"""
DSA5205 Project 1 - Task 3
Model selection and final public-test prediction.

Run:
    python task3_validation.py

Workflow for each pair A/B/C:
1. Read and validate training and public-test CSV files.
2. Use a chronological 80/20 split within the training file.
3. Compare Ridge and Lasso on the validation block.
4. Select the model primarily by the paper Sharpe ratio.
5. Refit the selected model on the FULL training sample.
6. Predict the official public test features.
7. Save the required CSV with exactly two columns: t,yhat.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "task3_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

VALIDATION_FRACTION = 0.20

# Ridge uses the same scaling as Task 1:
# beta_hat = (X'X + z*T*I)^(-1) X'y
RIDGE_Z_GRID = np.array([
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    25.0,
    50.0,
])

LASSO_ALPHA_GRID = np.array([
    0.01,
    0.03,
    0.05,
    0.10,
    0.20,
])


# ============================================================
# Input validation
# ============================================================

def get_feature_columns(df):
    """Return feature columns in their original order."""
    return [c for c in df.columns if c.startswith("feature")]


def validate_training_file(df, pair):
    """Validate the official Task 3 training CSV."""

    if "t" not in df.columns:
        raise ValueError(f"Pair {pair}: training file has no t column.")

    if "return" not in df.columns:
        raise ValueError(
            f"Pair {pair}: training file has no return column."
        )

    features = get_feature_columns(df)

    if len(features) == 0:
        raise ValueError(
            f"Pair {pair}: no feature columns found."
        )

    expected_columns = ["t", *features, "return"]

    if list(df.columns) != expected_columns:
        raise ValueError(
            f"Pair {pair}: training columns must be "
            "t, feature1..featureP, return."
        )

    if df.isna().any().any():
        raise ValueError(
            f"Pair {pair}: training file contains NaNs."
        )

    if df["t"].duplicated().any():
        raise ValueError(
            f"Pair {pair}: duplicated t values in training file."
        )

    if not df["t"].is_monotonic_increasing:
        raise ValueError(
            f"Pair {pair}: training t must be strictly increasing."
        )

    return features


def validate_test_file(df, pair, training_features):
    """Validate public test file and enforce exact feature alignment."""

    if "t" not in df.columns:
        raise ValueError(
            f"Pair {pair}: test file has no t column."
        )

    if "return" in df.columns:
        raise ValueError(
            f"Pair {pair}: public test file must not contain return."
        )

    test_features = get_feature_columns(df)

    if test_features != training_features:
        missing = set(training_features) - set(test_features)
        extra = set(test_features) - set(training_features)

        raise ValueError(
            f"Pair {pair}: train/test feature mismatch.\n"
            f"Missing: {sorted(missing)}\n"
            f"Extra: {sorted(extra)}"
        )

    expected_columns = ["t", *training_features]

    if list(df.columns) != expected_columns:
        raise ValueError(
            f"Pair {pair}: test feature order does not match training."
        )

    if df.isna().any().any():
        raise ValueError(
            f"Pair {pair}: test file contains NaNs."
        )

    if df["t"].duplicated().any():
        raise ValueError(
            f"Pair {pair}: duplicated t values in test file."
        )

    if not df["t"].is_monotonic_increasing:
        raise ValueError(
            f"Pair {pair}: test t must be strictly increasing."
        )


# ============================================================
# Standardization
# ============================================================

def fit_standardizer(X):
    """
    Fit feature standardization.

    Mean and standard deviation are estimated ONLY from
    the fitting sample to avoid validation/test leakage.
    """

    mean = X.mean(axis=0)
    std = X.std(axis=0)

    # Avoid division by zero.
    std[std < 1e-12] = 1.0

    return mean, std


def standardize(X, mean, std):
    return (X - mean) / std


# ============================================================
# Ridge
# ============================================================

def ridge_fit_svd(X, y, z):
    """
    Stable Ridge / ridgeless estimator.

    beta_hat =
        (X'X + z*T*I)^(-1) X'y

    z=0 automatically gives the Moore-Penrose
    minimum-norm solution through SVD.
    """

    n = X.shape[0]

    U, s, Vt = np.linalg.svd(
        X,
        full_matrices=False,
    )

    Uy = U.T @ y

    denominator = s**2 + z * n

    weights = np.divide(
        s * Uy,
        denominator,
        out=np.zeros_like(s),
        where=denominator > 1e-12,
    )

    beta = Vt.T @ weights

    return beta


def ridge_path_svd(X, y, z_grid):
    """Efficiently calculate the entire Ridge path."""

    n = X.shape[0]

    U, s, Vt = np.linalg.svd(
        X,
        full_matrices=False,
    )

    Uy = U.T @ y

    betas = []

    for z in z_grid:

        denominator = s**2 + z * n

        weights = np.divide(
            s * Uy,
            denominator,
            out=np.zeros_like(s),
            where=denominator > 1e-12,
        )

        beta = Vt.T @ weights

        betas.append(beta)

    return betas


# ============================================================
# Evaluation metrics
# ============================================================

def paper_metrics(y, yhat):
    """
    Task 3 evaluation metrics.

    R2_paper
    Expected timing return
    Paper Sharpe ratio
    Variance-based Sharpe ratio
    """

    timing_return = yhat * y

    expected_timing_return = float(
        np.mean(timing_return)
    )

    mse = np.mean((y - yhat) ** 2)

    return_second_moment = np.mean(y**2)

    r2_paper = float(
        1.0 - mse / return_second_moment
    )

    timing_second_moment = np.mean(
        timing_return**2
    )

    if timing_second_moment > 0:

        sharpe_paper = float(
            expected_timing_return
            / np.sqrt(timing_second_moment)
        )

    else:
        sharpe_paper = 0.0

    timing_variance = np.var(
        timing_return
    )

    if timing_variance > 0:

        sharpe_var = float(
            expected_timing_return
            / np.sqrt(timing_variance)
        )

    else:
        sharpe_var = np.nan

    return {
        "validation_r2_paper": r2_paper,
        "validation_expected_timing_return":
            expected_timing_return,
        "validation_sharpe_paper":
            sharpe_paper,
        "validation_sharpe_var":
            sharpe_var,
    }


# ============================================================
# Validation / model selection
# ============================================================

def select_model(pair, train_df, features):

    n = len(train_df)

    cutoff = int(
        np.floor(
            (1.0 - VALIDATION_FRACTION) * n
        )
    )

    development = train_df.iloc[
        :cutoff
    ].copy()

    validation = train_df.iloc[
        cutoff:
    ].copy()

    # Save splits for reproducibility.
    development.to_csv(
        OUTPUT_DIR
        / f"pair{pair}_development.csv",
        index=False,
    )

    validation.to_csv(
        OUTPUT_DIR
        / f"pair{pair}_validation.csv",
        index=False,
    )

    X_dev = development[
        features
    ].to_numpy(dtype=float)

    y_dev = development[
        "return"
    ].to_numpy(dtype=float)

    X_val = validation[
        features
    ].to_numpy(dtype=float)

    y_val = validation[
        "return"
    ].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    mean, std = fit_standardizer(X_dev)

    X_dev_std = standardize(
        X_dev,
        mean,
        std,
    )

    X_val_std = standardize(
        X_val,
        mean,
        std,
    )

    results = []

    # --------------------------------------------------------
    # Ridge
    # --------------------------------------------------------

    ridge_betas = ridge_path_svd(
        X_dev_std,
        y_dev,
        RIDGE_Z_GRID,
    )

    for z, beta in zip(
        RIDGE_Z_GRID,
        ridge_betas,
    ):

        yhat = X_val_std @ beta

        metrics = paper_metrics(
            y_val,
            yhat,
        )

        results.append({
            "pair": pair,
            "model": "ridge",
            "regularization": float(z),
            "beta_norm_sq":
                float(beta @ beta),
            "n_nonzero":
                int(np.count_nonzero(beta)),
            **metrics,
        })

    # --------------------------------------------------------
    # Lasso
    # --------------------------------------------------------

    for alpha in LASSO_ALPHA_GRID:

        model = Lasso(
            alpha=float(alpha),
            fit_intercept=False,
            max_iter=50_000,
            tol=1e-6,
            selection="cyclic",
        )

        model.fit(
            X_dev_std,
            y_dev,
        )

        beta = model.coef_

        yhat = model.predict(
            X_val_std
        )

        metrics = paper_metrics(
            y_val,
            yhat,
        )

        results.append({
            "pair": pair,
            "model": "lasso",
            "regularization":
                float(alpha),
            "beta_norm_sq":
                float(beta @ beta),
            "n_nonzero":
                int(np.count_nonzero(beta)),
            **metrics,
        })

    ranking = pd.DataFrame(results)

    # --------------------------------------------------------
    # Model-selection rule
    # --------------------------------------------------------
    #
    # Primary criterion:
    #     paper Sharpe
    #
    # Tie breakers:
    #     expected timing return
    #     R2_paper
    #
    # This directly aligns selection with the economic
    # performance emphasized by the project.
    # --------------------------------------------------------

    ranking = ranking.sort_values(
        [
            "validation_sharpe_paper",
            "validation_expected_timing_return",
            "validation_r2_paper",
        ],
        ascending=False,
    ).reset_index(drop=True)

    ranking.insert(
        0,
        "rank",
        np.arange(
            1,
            len(ranking) + 1,
        ),
    )

    ranking.to_csv(
        OUTPUT_DIR
        / f"pair{pair}_model_ranking.csv",
        index=False,
    )

    best = ranking.iloc[0]

    print()
    print("=" * 65)
    print(f"PAIR {pair}")
    print("=" * 65)

    print(
        f"Training observations : {n}"
    )

    print(
        f"Development           : {len(development)}"
    )

    print(
        f"Validation            : {len(validation)}"
    )

    print(
        f"Number of features    : {len(features)}"
    )

    print()
    print("Selected model")
    print(
        f"  model          = {best['model']}"
    )

    print(
        f"  regularization = "
        f"{best['regularization']}"
    )

    print(
        f"  validation SR  = "
        f"{best['validation_sharpe_paper']:.6f}"
    )

    print(
        f"  validation R2  = "
        f"{best['validation_r2_paper']:.6f}"
    )

    return {
        "model":
            str(best["model"]),
        "regularization":
            float(best["regularization"]),
        "ranking":
            ranking,
    }


# ============================================================
# Refit on full training data
# ============================================================

def fit_final_model(
    train_df,
    test_df,
    features,
    selected_model,
    regularization,
):

    X_train = train_df[
        features
    ].to_numpy(dtype=float)

    y_train = train_df[
        "return"
    ].to_numpy(dtype=float)

    X_test = test_df[
        features
    ].to_numpy(dtype=float)

    # IMPORTANT:
    # Refit standardization using the ENTIRE training file.
    #
    # Public test information is never used to estimate
    # preprocessing parameters.

    mean, std = fit_standardizer(
        X_train
    )

    X_train_std = standardize(
        X_train,
        mean,
        std,
    )

    X_test_std = standardize(
        X_test,
        mean,
        std,
    )

    # --------------------------------------------------------
    # Ridge
    # --------------------------------------------------------

    if selected_model == "ridge":

        beta = ridge_fit_svd(
            X_train_std,
            y_train,
            regularization,
        )

        yhat = X_test_std @ beta

        n_nonzero = int(
            np.count_nonzero(beta)
        )

        beta_norm_sq = float(
            beta @ beta
        )

    # --------------------------------------------------------
    # Lasso
    # --------------------------------------------------------

    elif selected_model == "lasso":

        model = Lasso(
            alpha=regularization,
            fit_intercept=False,
            max_iter=50_000,
            tol=1e-6,
            selection="cyclic",
        )

        model.fit(
            X_train_std,
            y_train,
        )

        beta = model.coef_

        yhat = model.predict(
            X_test_std
        )

        n_nonzero = int(
            np.count_nonzero(beta)
        )

        beta_norm_sq = float(
            beta @ beta
        )

    else:

        raise ValueError(
            f"Unknown model: "
            f"{selected_model}"
        )

    return (
        yhat,
        beta_norm_sq,
        n_nonzero,
    )


# ============================================================
# Process one dataset
# ============================================================

def process_pair(pair):

    train_path = (
        BASE_DIR
        / f"pair{pair}_train.csv"
    )

    test_path = (
        BASE_DIR
        / f"pair{pair}_test_features.csv"
    )

    if not train_path.exists():

        raise FileNotFoundError(
            f"Missing {train_path.name}"
        )

    if not test_path.exists():

        raise FileNotFoundError(
            f"Missing {test_path.name}"
        )

    # --------------------------------------------------------
    # Read files
    # --------------------------------------------------------

    train_df = pd.read_csv(
        train_path
    )

    test_df = pd.read_csv(
        test_path
    )

    # --------------------------------------------------------
    # Strict validation
    # --------------------------------------------------------

    features = validate_training_file(
        train_df,
        pair,
    )

    validate_test_file(
        test_df,
        pair,
        features,
    )

    # --------------------------------------------------------
    # Internal validation
    # --------------------------------------------------------

    selection = select_model(
        pair,
        train_df,
        features,
    )

    selected_model = (
        selection["model"]
    )

    regularization = (
        selection["regularization"]
    )

    # --------------------------------------------------------
    # Refit using ALL training observations
    # --------------------------------------------------------

    (
        yhat,
        beta_norm_sq,
        n_nonzero,
    ) = fit_final_model(
        train_df=train_df,
        test_df=test_df,
        features=features,
        selected_model=selected_model,
        regularization=regularization,
    )

    # --------------------------------------------------------
    # Required prediction format
    # --------------------------------------------------------

    predictions = pd.DataFrame({
        "t":
            test_df["t"].to_numpy(),
        "yhat":
            yhat,
    })

    # Safety checks.

    if list(predictions.columns) != [
        "t",
        "yhat",
    ]:

        raise RuntimeError(
            "Prediction file has incorrect columns."
        )

    if len(predictions) != len(test_df):

        raise RuntimeError(
            "Prediction row count does not match test."
        )

    if not np.array_equal(
        predictions["t"].to_numpy(),
        test_df["t"].to_numpy(),
    ):

        raise RuntimeError(
            "Prediction t order does not match test."
        )

    if predictions["yhat"].isna().any():

        raise RuntimeError(
            "Predictions contain NaN."
        )

    if not np.isfinite(
        predictions["yhat"]
    ).all():

        raise RuntimeError(
            "Predictions contain non-finite values."
        )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------
    #
    # Replace YOUR_STUDENT_ID later with your actual ID.
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / f"A0356693M_predictions_{pair}.csv"
    )

    predictions.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    # Save final model information for report / reproducibility.

    final_info = pd.DataFrame([{
        "pair":
            pair,
        "selected_model":
            selected_model,
        "regularization":
            regularization,
        "n_train":
            len(train_df),
        "n_test":
            len(test_df),
        "n_features":
            len(features),
        "beta_norm_sq":
            beta_norm_sq,
        "n_nonzero":
            n_nonzero,
    }])

    final_info.to_csv(
        OUTPUT_DIR
        / f"pair{pair}_final_model.csv",
        index=False,
    )

    print()
    print("Final model refitted on full training sample")

    print(
        f"  model       : "
        f"{selected_model}"
    )

    print(
        f"  regularizer : "
        f"{regularization}"
    )

    print(
        f"  beta norm²  : "
        f"{beta_norm_sq:.6f}"
    )

    print(
        f"  nonzero beta: "
        f"{n_nonzero}"
    )

    print(
        f"  predictions : "
        f"{len(predictions)}"
    )

    print(
        f"  saved to    : "
        f"{output_path.name}"
    )

    return final_info


# ============================================================
# Main
# ============================================================

def main():

    all_final_info = []

    for pair in (
        "A",
        "B",
        "C",
    ):

        info = process_pair(
            pair
        )

        all_final_info.append(
            info
        )

    # Combine final model summary.

    summary = pd.concat(
        all_final_info,
        ignore_index=True,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "task3_final_model_summary.csv",
        index=False,
    )

    print()
    print("=" * 65)
    print("TASK 3 COMPLETE")
    print("=" * 65)

    print()
    print(
        summary[
            [
                "pair",
                "selected_model",
                "regularization",
                "n_train",
                "n_test",
                "n_features",
                "n_nonzero",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "All final predictions saved in:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    


if __name__ == "__main__":
    main()

