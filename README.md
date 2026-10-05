# DSA5205 Project 1 — The Virtue of Complexity in Return Prediction

**Student ID:** A0356693M  
**Email:** gusiyu@u.nus.edu

This repository contains the reproducible code and outputs for DSA5205 Project 1. The project studies complexity, regularization, return prediction, and neural-network generalization through four tasks.

## Repository structure

```text
DSA5205-Project1/
├── task1_simulation.py
├── task2_lasso.py
├── task3_validation.py
├── task4_nn.py
├── run_all.py
├── requirements.txt
├── task1_outputs/
├── task2_outputs/
├── task3_outputs/
├── task4_outputs/
└── A0356693M_report.pdf
```

The course-provided Task 3 data files are intentionally not stored in this GitHub repository because some are too large for standard GitHub storage. To reproduce Task 3, place the following six files in the repository root:

```text
pairA_train.csv
pairA_test_features.csv
pairB_train.csv
pairB_test_features.csv
pairC_train.csv
pairC_test_features.csv
```

## Environment

Recommended: Python 3.11 or 3.12.

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```text
.venv\Scripts\activate
```

## Reproduction order

Run all four tasks from the repository root with:

```bash
python run_all.py
```

Alternatively, run them individually in this order:

```bash
python task1_simulation.py
python task2_lasso.py
python task3_validation.py
python task4_nn.py
```

Task 2 imports the Task 1 data-generating process and seeds, so keeping `task1_simulation.py` in the same directory is required. Task 3 additionally requires the six course-provided CSV files listed above.

## Task 1 — Ridge under misspecification

`task1_simulation.py` implements the misspecified linear return-prediction experiment. The main settings are:

- `T_train = 100`
- `T_test = 2000`
- true complexity `c = P/T_train = 10`, hence `P = 1000`
- dense isotropic signal with `||beta*||^2 = 1`
- `Psi = I`
- Gaussian noise with standard deviation 1
- 10 independent repetitions
- base seed `20260902`; simulation `j` uses seed `20260902 + j`
- one feature permutation per simulation, reused across all observed-complexity values so the observed feature sets are nested

The observed-complexity grid is

```text
0.50, 0.75, 0.90, 0.95, 0.98, 1.02, 1.05, 1.10,
1.25, 1.50, 2.00, 3.00, 5.00, 7.00, 10.00
```

and the ridge grid is

```text
0, 0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100
```

Ridge uses the project scaling

```text
beta_hat(z) = (X'X + z*T_train*I)^(-1) X'y.
```

At `z = 0`, the estimator is computed through SVD as the Moore–Penrose minimum-norm solution rather than by directly inverting `X'X`.

Outputs are written to `task1_outputs/`, including raw and averaged CSV results and the four requested figures for OOS R-squared, expected timing return, paper Sharpe ratio, and coefficient norm.

## Task 2 — Lasso comparison

`task2_lasso.py` applies Lasso to exactly the same simulated draws, feature permutations, train/test splits, and observed-complexity grid as Task 1.

The estimator uses scikit-learn's objective

```text
(1 / (2*T_train)) * ||y - X beta||^2 + alpha * ||beta||_1
```

with fixed `alpha = 0.05`. The fixed penalty is used to isolate the effect of changing observed complexity and is not tuned using test data.

The script uses the same Task 1 seeds by importing the Task 1 DGP directly. Outputs are written to `task2_outputs/`.

## Task 3 — Model selection and public-test predictions

`task3_validation.py` processes pairs A, B, and C independently.

For each pair it:

1. validates the official training and public-test files;
2. uses a chronological 80/20 split within the training sample;
3. estimates preprocessing parameters only from the development block;
4. compares Ridge and Lasso using the internal validation block;
5. ranks candidates primarily by the paper Sharpe ratio, with expected timing return and OOS R-squared as tie-breakers;
6. refits the selected specification using the full training file;
7. predicts the public test features; and
8. writes a two-column `t,yhat` prediction file while preserving the original test `t` order.

Candidate grids are:

```text
Ridge z:     0, 0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50
Lasso alpha: 0.01, 0.03, 0.05, 0.10, 0.20
```

The selected specifications are:

- Pair A: Ridge, `z = 2.0`
- Pair B: Lasso, `alpha = 0.10`
- Pair C: Lasso, `alpha = 0.20`

Final prediction files are stored in `task3_outputs/` as:

```text
A0356693M_predictions_A.csv
A0356693M_predictions_B.csv
A0356693M_predictions_C.csv
```

No hidden test labels are used anywhere in model selection or preprocessing.

## Task 4 — Neural-network training dynamics

`task4_nn.py` implements a small simulated financial-ML experiment motivated by Montanari and Urbani (2025). It is an illustration of their qualitative insight, not a direct replication of their dynamical mean-field theory.

The simulation uses:

- seed `5205`
- 600 training observations
- 2000 test observations
- 20 predictors
- a nonlinear conditional expected-return function driven by the first five predictors
- Gaussian return noise with standard deviation 2
- a two-layer ReLU network with 128 hidden units
- Adam optimizer with learning rate 0.01
- batch size 64
- 1000 epochs
- evaluation every 5 epochs

The script records training MSE, test MSE, OOS R-squared, timing return, paper Sharpe, prediction error relative to the known conditional mean, and parameter norm. Outputs are written to `task4_outputs/`.

PyTorch automatically uses CUDA when available, Apple MPS when available, and otherwise CPU. Small numerical differences across hardware or PyTorch versions may occur even with fixed seeds.

## Random seeds and reproducibility

- Task 1: base seed `20260902`, with 10 consecutive seeds
- Task 2: exactly the same seeds and simulated datasets as Task 1
- Task 3: deterministic estimators and chronological data split; no randomized train/test split is used
- Task 4: seed `5205` for Python, NumPy, PyTorch, and DataLoader shuffling

All scripts use paths relative to their own location and create their output directories automatically.

## Runtime notes

Task 1 performs repeated SVDs over the complexity grid and can take several minutes depending on the machine. Task 2 is typically faster. Task 3 can require substantial memory because the official B and C public-test feature files are large. Task 4 can run on CPU, CUDA, or Apple MPS; GPU/MPS generally reduces runtime.

Existing output folders are included for inspection, but all reported outputs can be regenerated by rerunning the scripts.

## References

Kelly, B. T., Malamud, S., & Zhou, K. (2024). *The Virtue of Complexity in Return Prediction*. Journal of Finance, 79(1), 459–503.

Montanari, A., & Urbani, P. (2025). *Dynamical Decoupling of Generalization and Overfitting in Large Two-Layer Networks*. Advances in Neural Information Processing Systems (NeurIPS 2025).
