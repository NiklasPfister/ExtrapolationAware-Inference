import numpy as np
import pandas as pd
import click
import os
import json
from typing import Any
from scipy.spatial.distance import cdist
from xtrapolation.xtrapolation import Xtrapolation
from xtrapolation.helpers import (
    DerivativeConfig,
    ExtrapolationConfig,
)
import experiments.helpers.regression_methods as reg
import experiments.helpers.examples as ex


# =============================================================================
# Main Function
# =============================================================================


@click.command()
@click.option("--seed", type=int, required=True, help="Random seed for reproducibility")
@click.option("--dimension", type=int, required=True, help="Dimension of predictors")
@click.option("--samplesize", type=int, required=True, help="Sample size of data")
@click.option("--runname", type=str, required=True, help="Name for experiment")
@click.option(
    "--output-dir",
    type=str,
    default="experiments/results/simulation_results/",
    help="Output directory for results",
)
@click.option(
    "--no-locpol",
    type=bool,
    default=False,
    help="Whether to disable local polynomial fitting",
)
@click.option("--debug-plot", is_flag=True, help="Show a debug plot of RF xtrapolation fit")
def main(
    seed: int,
    dimension: int,
    samplesize: int,
    runname: str,
    output_dir: str,
    no_locpol: bool,
    debug_plot: bool,
) -> None:
    """Run xtrapolation simulation experiment."""
    print("Working directory: " + os.getcwd())
    print(
        f"Running experiment: seed={seed}, dimension={dimension}, samplesize={samplesize}, runname={runname}"
    )

    n = samplesize
    d = dimension

    # Set seed
    np.random.seed(seed)

    # Set output path
    os.makedirs(output_dir, exist_ok=True)

    ##
    # Sample random function
    ##

    # Sample training domain
    bds = np.random.choice([0, 1, 2, 3], d, replace=True)

    # Sample slope_vec (ensuring that slopes appear in observed domain)
    grid = np.linspace(-5, 5, 11)
    slopes = np.random.uniform(-10, 10, 3)
    slope_vec = np.random.choice(slopes, 10, replace=True)
    ind = np.delete(list(range(3, 7)), bds[0])
    slope_vec[ind] = slopes

    # Adjust slopes to control signal to noise ratio
    Xtmp = np.random.uniform(-2, 2, 1000 * d).reshape(-1, d)
    slope_vec = slope_vec / np.sqrt(np.var(ex.piecewise_linear(Xtmp, slope_vec, grid)))

    def ff(x: np.ndarray) -> np.ndarray:
        return ex.piecewise_linear(x, slope_vec, grid)  # type: ignore

    ##
    # Generate data
    ##

    # Noise is scaled according to dimension
    X, _ = ex.sample_X(n, bds)
    Y = np.array(ff(X)).flatten() + np.random.normal(0, 1 / 10, n)

    ntest = 200
    X_Din, X_Dout = ex.sample_X(ntest, bds)
    X_full = np.random.uniform(-2, 2, ntest * d).reshape(-1, d)
    regions: list[np.ndarray] = [X_Din, X_Dout, X_full]

    ##
    # Fit regression models
    ##

    print("Fitting regressions...")

    reg_rf, fval_rf, mspe_rf = reg.rf_regression(X, Y, rf_screen=True)
    reg_svr, fval_svr, mspe_svr = reg.svr_regression(X, Y, rf_screen=True)
    reg_mlp, fval_mlp, mspe_mlp = reg.mlp_regression(X, Y, rf_screen=True)
    reg_ols, fval_ols, mspe_ols = reg.ols_regression(X, Y)

    cv_mspe: dict[str, float] = {
        "rf_mspe": mspe_rf,
        "svr_mspe": mspe_svr,
        "mlp_mspe": mspe_mlp,
        "ols_mspe": mspe_ols,
    }
    print(cv_mspe)

    # Save MSPEs
    with open(output_dir + f"{runname}_mspe_dim{d}_size{n}_seed{seed}.json", "w") as f:
        json.dump(cv_mspe, f, indent=2)

    ##
    # Tuning of Xtrapolation/ parameters
    ##

    tol: float = 1.0
    if no_locpol:
        rf_pars_list: list[dict[str, float]] = [
            {"min_samples_leaf": int(np.sqrt(n) * 2), "n_jobs": 4},
        ]
        pen_list: list[float] = [0]
    else:
        rf_pars_list: list[dict[str, float]] = [
            {"min_samples_leaf": int(n * 0.1), "n_jobs": 4},
        ]
        pen_list: list[float] = [d / n * 100]

    parameters: dict[str, Any] = {
        "orders": [1],
        "deriv_config": DerivativeConfig(
            num_rotate=0,
            num_trees=100,
            smoothed_predictions=False,
            penalize_intercept=False,
            no_locpol=no_locpol,
        ),
        "extra_config": ExtrapolationConfig(
            nn=int(n * 0.5),
            dist="euclidean",
            alpha=0.0,
            aggregation="intersection-corrected",
        ),
        "verbose": 2,
    }

    xtra = Xtrapolation(**parameters)

    # RF parameters
    rf_pars_rf, pen_rf = xtra.parameter_tuning(
        X, fval_rf, Y, None, rf_pars_list, pen_list, loss="mse", tol=tol
    )
    print(rf_pars_rf)
    print(pen_rf)

    if debug_plot:
        import matplotlib.pyplot as plt

        parameters["deriv_config"].rf_pars = rf_pars_rf
        parameters["pen"].pen = pen_rf
        xtra_test = Xtrapolation(**parameters)
        bounds_rf_region2 = xtra_test.prediction_bounds(
            X,
            fval_rf,
            regions[2],
        )
        lower_bounds = np.max(bounds_rf_region2[:, :, 0], axis=1)
        upper_bounds = np.min(bounds_rf_region2[:, :, 1], axis=1)

        ncols = min(3, d)
        nrows = (d + ncols - 1) // ncols
        _, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.atleast_1d(axes).flatten()

        for dim in range(d):
            ax = axes[dim]
            ax.scatter(X[:, dim], Y, alpha=0.5, label="Training data", s=20)

            sorted_idx = np.argsort(regions[2][:, dim])
            x_sorted = regions[2][sorted_idx, dim]
            lower_sorted = lower_bounds[sorted_idx]
            upper_sorted = upper_bounds[sorted_idx]

            ax.plot(x_sorted, lower_sorted, "r-", label="Xtra Lower (RF)", linewidth=2)
            ax.plot(x_sorted, upper_sorted, "b-", label="Xtra Upper (RF)", linewidth=2)
            ax.plot(
                x_sorted,
                np.array(ff(regions[2][sorted_idx, :])).flatten(),
                "g-",
                label="ground truth",
                linewidth=2,
            )
            sorted_idxx = np.argsort(X[:, dim])
            ax.plot(
                X[sorted_idxx, dim],
                np.array(fval_rf[sorted_idxx]).flatten(),
                "k-",
                label="RF",
                linewidth=2,
            )

            ax.set_xlabel(f"X{dim+1}")
            ax.set_ylabel("Y")
            ax.set_title(f"Dimension {dim+1}")
            ax.legend()

        for dim in range(d, len(axes)):
            axes[dim].axis("off")

        plt.suptitle(
            "Debug: Xtrapolation Bounds (RF) - Region 2 (All Dimensions)", fontsize=14
        )
        plt.tight_layout()
        plt.show()

    # SVR parameters
    rf_pars_svr, pen_svr = xtra.parameter_tuning(
        X, fval_svr, Y, None, rf_pars_list, pen_list, loss="mse", tol=tol
    )
    print(rf_pars_svr)
    print(pen_svr)

    # MLP parameters
    rf_pars_mlp, pen_mlp = xtra.parameter_tuning(
        X, fval_mlp, Y, None, rf_pars_list, pen_list, loss="mse", tol=tol
    )
    print(rf_pars_mlp)
    print(pen_mlp)

    # OLS parameters
    rf_pars_ols, pen_ols = xtra.parameter_tuning(
        X, fval_ols, Y, None, rf_pars_list, pen_list, loss="mse", tol=tol
    )
    print(rf_pars_ols)
    print(pen_ols)

    ##
    # Apply Xtrapolation
    ##

    # Baseline Xtrapolation (for oracle)
    parameters_oracle: dict[str, Any] = {
        "orders": [1],
        "extra_config": ExtrapolationConfig(
            nn=n,
            dist="euclidean",
            alpha=0,
            aggregation="intersection-corrected",
        ),
        "verbose": 2,
    }

    # Xtrapolation for RandomForest
    print("Fitting Xtrapolation derivatives for RF...")
    parameters["deriv_config"].rf_pars = rf_pars_rf
    parameters["deriv_config"].pen = pen_rf
    xtra_rf = Xtrapolation(**parameters)
    xtra_rf.fit_derivatives(X, fval_rf)

    # Xtrapolation for SVR
    print("Fitting Xtrapolation derivatives for SVR...")
    parameters["deriv_config"].rf_pars = rf_pars_svr
    parameters["deriv_config"].pen = pen_svr
    xtra_svr = Xtrapolation(**parameters)
    xtra_svr.fit_derivatives(X, fval_svr)

    # Xtrapolation for MLP
    print("Fitting Xtrapolation derivatives for MLP...")
    parameters["deriv_config"].rf_pars = rf_pars_mlp
    parameters["deriv_config"].pen = pen_mlp
    xtra_mlp = Xtrapolation(**parameters)
    xtra_mlp.fit_derivatives(X, fval_mlp)

    # Xtrapolation for OLS
    print("Fitting Xtrapolation derivatives for OLS...")
    parameters["deriv_config"].rf_pars = rf_pars_ols
    parameters["deriv_config"].pen = pen_ols
    xtra_ols = Xtrapolation(**parameters)
    xtra_ols.fit_derivatives(X, fval_ols)

    # Xtrapolation for oracle
    print("Fitting Xtrapolation derivatives for oracle...")
    xtra_oracle = Xtrapolation(**parameters_oracle)
    xtra_oracle.fit_derivatives_oracle(X, ff)

    ##
    # Evaluate different methods
    ##

    # Regression results
    prediction_results = pd.DataFrame({})
    for k in range(len(regions)):
        prediction_results[f"truth_{k}"] = ff(regions[k])
        prediction_results[f"RF_{k}"] = reg_rf.predict(regions[k])
        prediction_results[f"SVR_{k}"] = reg_svr.predict(regions[k])
        prediction_results[f"MLP_{k}"] = reg_mlp.predict(regions[k])
        prediction_results[f"OLS_{k}"] = reg_ols.predict(regions[k])

    # Xtrapolation results
    for k in range(len(regions)):
        print(f"Xtrapolation prediction on region: {k}")
        # RF
        bounds_rf = xtra_rf.prediction_bounds(X, fval_rf, regions[k])
        prediction_results[f"XtraLo-RF_{k}"] = np.max(bounds_rf[:, :, 0], axis=1)
        prediction_results[f"XtraUp-RF_{k}"] = np.min(bounds_rf[:, :, 1], axis=1)
        # SVR
        bounds_svr = xtra_svr.prediction_bounds(X, fval_svr, regions[k])
        prediction_results[f"XtraLo-SVR_{k}"] = np.max(bounds_svr[:, :, 0], axis=1)
        prediction_results[f"XtraUp-SVR_{k}"] = np.min(bounds_svr[:, :, 1], axis=1)
        # MLP
        bounds_mlp = xtra_mlp.prediction_bounds(X, fval_mlp, regions[k])
        prediction_results[f"XtraLo-MLP_{k}"] = np.max(bounds_mlp[:, :, 0], axis=1)
        prediction_results[f"XtraUp-MLP_{k}"] = np.min(bounds_mlp[:, :, 1], axis=1)
        # OLS
        bounds_ols = xtra_ols.prediction_bounds(X, fval_ols, regions[k])
        prediction_results[f"XtraLo-OLS_{k}"] = np.max(bounds_ols[:, :, 0], axis=1)
        prediction_results[f"XtraUp-OLS_{k}"] = np.min(bounds_ols[:, :, 1], axis=1)
        # Oracle
        bounds_oracle = xtra_oracle.prediction_bounds_oracle(X, ff, regions[k])
        prediction_results[f"XtraLo-oracle_{k}"] = np.max(
            bounds_oracle[:, :, 0], axis=1
        )
        prediction_results[f"XtraUp-oracle_{k}"] = np.min(
            bounds_oracle[:, :, 1], axis=1
        )

    # Nearest neighbor
    for k in range(len(regions)):
        prediction_results[f"euclidean_nn_{k}"] = np.min(cdist(X, regions[k]), axis=0)

    # Save first coordinate of regions
    for k in range(len(regions)):
        prediction_results[f"X1_region_{k}"] = regions[k][:, 0]

    ##
    # Save all results
    ##
    prediction_results.to_csv(
        output_dir + f"{runname}_res_dim{d}_size{n}_seed{seed}.csv"
    )

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
