import numpy as np
import click
import os
import pickle
from typing import Any
from sklearn.preprocessing import StandardScaler
from quantile_forest import RandomForestQuantileRegressor
from experiments.helpers.cp_methods import (
    cqr_QuantileForest,
    cqr_QuantileNN,
    QuantileNN,
    quantile_cross_validation,
)
from xtrapolation.xtrapolation import Xtrapolation
from xtrapolation.helpers import DerivativeConfig, ExtrapolationConfig
from ucimlrepo import fetch_ucirepo


# =============================================================================
# Helper Functions
# =============================================================================


def tuning_wrapper_fun(
    X: np.ndarray,
    qmat: np.ndarray,
    y: np.ndarray,
    quantiles: list[float],
    no_xtra_features: list[int] | None,
    parameters: dict[str, Any],
    rf_pars_list: list[dict[str, int]],
    pen_list: list[float],
    tol: float,
) -> tuple[list[dict[str, int]], list[float]]:
    """Tune hyperparameters for each quantile."""
    rf_pars: list[dict[str, int]] = []
    pens: list[float] = []
    xtra = Xtrapolation(**parameters)
    for i, qq in enumerate(quantiles):
        rf_par, pen = xtra.parameter_tuning(
            X,
            (qmat[:, 0] + qmat[:, 1]) / 2,
            y,
            no_xtra_features,
            rf_pars_list,
            pen_list,
            loss="quantile",
            q=qq,
            tol=tol,
        )
        rf_pars.append(rf_par)
        pens.append(pen)
        print(f"quantile {qq}:")
        print(rf_pars[i])
        print(pens[i])
    return rf_pars, pens


def train_test_split_abalone(
    X: np.ndarray,
    y: np.ndarray,
    train_ind: np.ndarray,
    method: str,
    quantiles: list[float],
    no_xtra_features: list[int] | None,
    parameters: dict[str, Any],
    rf_pars_list: list[dict[str, Any]],
    pen_list: list[float],
    tol: float,
) -> dict[str, Any]:
    """Train test split for abalone data with xtrapolation bounds."""
    Xtrain = X[train_ind, :]
    ytrain = y[train_ind]

    # Scaling (mean and variance)
    std_scale = StandardScaler()
    std_scale.fit(Xtrain)
    Xtrain: np.ndarray = std_scale.transform(Xtrain)  # type: ignore
    Xscaled: np.ndarray = std_scale.transform(X, copy=True)  # type: ignore

    if method == "qrf":
        # Run quantile regression (with forests)
        opt_params = quantile_cross_validation(
            Xtrain, ytrain, quantiles, "qrf", n_folds=5, tol=1
        )
        opt_params["n_estimators"] = 500
        opt_params["n_jobs"] = -1
        print(opt_params)
        qrf = RandomForestQuantileRegressor(**opt_params)  # type: ignore
        qrf.fit(Xtrain, ytrain)
        qmat = qrf.predict(Xscaled, quantiles=quantiles)
    elif method == "conf-qrf":
        # Run conformalized quantile regression (with forests)
        idx = np.random.permutation(Xtrain.shape[0])
        n_half = int(np.floor(Xtrain.shape[0] / 2))
        idx_train, idx_cal = (idx[:n_half], idx[n_half : 2 * n_half])
        opt_params = quantile_cross_validation(
            Xtrain[idx_train, :],
            ytrain[idx_train],
            quantiles,
            "conf-qrf",
            n_folds=5,
            tol=tol,
        )
        opt_params["n_estimators"] = 500
        opt_params["n_jobs"] = -1
        print(opt_params)
        qmat = cqr_QuantileForest(
            Xtrain, ytrain, Xscaled, quantiles, opt_params, (idx_train, idx_cal)
        )
    elif method == "qnn":
        # Run quantile regression (with neural nets)
        opt_params = quantile_cross_validation(
            Xtrain, ytrain, quantiles, "qnn", n_folds=5, tol=1
        )
        opt_params["batch_size"] = 64
        opt_params["epochs"] = 1000
        print(opt_params)
        qmat = QuantileNN(Xtrain, ytrain, Xscaled, quantiles, opt_params)
    elif method == "conf-qnn":
        # Run conformalized quantile regression (with neural nets)
        idx = np.random.permutation(Xtrain.shape[0])
        n_half = int(np.floor(Xtrain.shape[0] / 2))
        idx_train, idx_cal = (idx[:n_half], idx[n_half : 2 * n_half])
        opt_params = quantile_cross_validation(
            Xtrain[idx_train, :],
            ytrain[idx_train],
            quantiles,
            "conf-qnn",
            n_folds=5,
            tol=1,
        )
        opt_params["batch_size"] = 64
        opt_params["epochs"] = 1000
        print(opt_params)
        qmat = cqr_QuantileNN(
            Xtrain, ytrain, Xscaled, quantiles, opt_params, (idx_train, idx_cal)
        )
    else:
        raise ValueError(f"method {method} does not exist")

    # Run parameter tuning
    rf_pars, pens = tuning_wrapper_fun(
        Xtrain,
        qmat[train_ind, :],
        ytrain,
        quantiles,
        no_xtra_features,
        parameters,
        rf_pars_list,
        pen_list,
        tol,
    )

    # Xtrapolation
    bounds_list: list[np.ndarray] = []
    for i, qq in enumerate(quantiles):
        # Run xtrapolation on quantile
        parameters["deriv_config"].rf_pars = rf_pars[i]
        parameters["deriv_config"].pen = pens[i]
        xtra = Xtrapolation(**parameters)
        print(f"Working on quantile {qq}")
        bounds = xtra.prediction_bounds(
            Xtrain, qmat[train_ind, i], Xscaled, no_xtra_features, refit=True
        )
        bounds_list.append(bounds)

    # Return results
    return {
        "train_ind": train_ind,
        "quantiles": quantiles,
        "pars": (rf_pars, pens),
        "qmat": qmat,
        "bounds_list": bounds_list,
    }


# =============================================================================
# Main Function
# =============================================================================


@click.command()
@click.option("--split", type=int, required=True, help="Number of split to run")
@click.option(
    "--method",
    type=click.Choice(["qrf", "conf-qrf", "qnn", "conf-qnn"]),
    required=True,
    help="Quantile regression method",
)
@click.option("--runname", type=str, required=True, help="Name for experiment")
@click.option(
    "--output-dir",
    type=str,
    default="experiments/results/abalone_results/",
    help="Output directory for results",
)
@click.option(
    "--no-locpol",
    type=bool,
    default=True,
    help="Whether to disable local polynomial fitting",
)
def main(
    split: int, method: str, runname: str, output_dir: str, no_locpol: bool
) -> None:
    """Run abalone xtrapolation experiment."""
    print("Working directory: " + os.getcwd())
    print(f"Running split {split} with method {method} (runname: {runname})")

    # Set output path
    os.makedirs(output_dir, exist_ok=True)

    ##
    # Load data and process
    ##

    # Fetch dataset
    abalone = fetch_ucirepo(id=1)

    # Data (as pandas dataframes)
    Xraw = abalone.data.features  # type: ignore
    yraw = abalone.data.targets  # type: ignore

    # Subsample and convert
    np.random.seed(42)
    n = Xraw.shape[0]
    nmax = n
    indices = np.random.choice(n, size=nmax, replace=False)
    X = Xraw.iloc[indices, :]
    X.iloc[:, 0] = X["Sex"].factorize()[0]
    X = X.astype(float).to_numpy()
    y = yraw.iloc[indices].to_numpy().flatten()

    ##
    # Setup parameters
    ##

    # Set quantiles
    quantiles = [0.1, 0.9]
    # Set xtrapolation parameters
    no_xtra_features = [0]
    parameters: dict[str, Any] = {
        "orders": [1],
        "deriv_config": DerivativeConfig(
            num_rotate=0,
            num_trees=50,
            smoothed_predictions=False,
            penalize_intercept=False,
            no_locpol=no_locpol,
        ),
        "extra_config": ExtrapolationConfig(
            nn=200,
            dist="rfnn",
            aggregation="intersection-corrected",
            alpha=0.0,
        ),
        "verbose": 2,
    }
    # CV parameter grids
    tol = 1.0
    if no_locpol:
        rf_pars_list = [
            {"min_samples_leaf": 180, "n_jobs": 4},
            {"min_samples_leaf": 120, "n_jobs": 4},
            {"min_samples_leaf": 60, "n_jobs": 4},
        ]
        pen_list = [0.0]
    else:
        rf_pars_list = [
            {"min_samples_leaf": 5, "n_jobs": 4},
        ]
        pen_list = [8 / nmax * 10, 8 / nmax, 0.0]

    ##
    # Create splits
    ##

    # Set seed
    np.random.seed(1)

    num_splits = 8
    split_len = int(len(y) / num_splits)
    thresholds = [k * split_len for k in range(num_splits)] + [len(y)]

    # Extrapolating train-test splits (depending on X)
    xvec = X[:, 1]
    sortX = np.argsort(xvec)
    train_inds = [np.repeat(True, len(y)) for _ in range(num_splits)]
    for k in range(num_splits):
        train_inds[k][sortX[thresholds[k] : thresholds[k + 1]]] = False

    # Interpolating train-test splits (fully random)
    randX = np.random.permutation(len(y))
    train_inds += [
        (randX <= thresholds[k]) | (thresholds[k + 1] < randX)
        for k in range(num_splits)
    ]

    ##
    # Run experiment
    ##

    res = train_test_split_abalone(
        X,
        y,
        train_inds[split],
        method,
        quantiles,
        no_xtra_features,
        parameters,
        rf_pars_list,
        pen_list,
        tol,
    )

    # Save results
    results_dict = {"res": res, "parameters": parameters}

    output_file = output_dir + f"abalone_{runname}_{method}_{split}.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(results_dict, f)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
