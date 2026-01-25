"""
Linear vs Xtrapolation Experiment
==================================
"""

import numpy as np
import click
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from xtrapolation.xtrapolation import Xtrapolation
from xtrapolation.helpers import DerivativeConfig, ExtrapolationConfig
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib
import joblib


# Set plotting parameters
def setup_plotting():
    """Configure matplotlib for publication-quality plots."""
    params = {
        "axes.labelsize": 8,
        "font.size": 8,
        "legend.fontsize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "lines.linewidth": 0.7,
        "text.usetex": True,
        "axes.unicode_minus": True,
    }
    matplotlib.rcParams.update(params)
    plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})


def nonlinear_model(x):
    return 1 / (1 + np.exp(-3.25 * x))


def linear_model(x):
    return 0.5 * x + 0.35


class ExperimentConfig(BaseModel):
    """Configuration for the linear vs xtrapolation experiment."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Data parameters
    n_samples: int = 250
    seed: int = 42

    # Model parameters
    n_estimators: int = 200
    min_samples_leaf: int = 10

    # Xtrapolation parameters
    deriv_config: DerivativeConfig = Field(
        default_factory=lambda: DerivativeConfig(
            num_trees=200,
            no_locpol=False,
        )
    )
    extra_config: ExtrapolationConfig = Field(
        default_factory=lambda: ExtrapolationConfig(
            nn=20,
            aggregation="intersection-corrected",
        )
    )

    # Tuning parameters
    tol: float = 1.0
    rf_pars_list: list = Field(
        default_factory=lambda: [
            {"min_samples_leaf": 80},
            {"min_samples_leaf": 10},
        ]
    )
    pen_list: list = Field(default_factory=lambda: [1 / 250 * 10])

    # Bootstrap parameters
    n_bootstrap: int = 100
    n_experiments: int = 100

    # Evaluation points
    xeval_min: float = -3.0
    xeval_max: float = 3.0
    xeval_points: int = 100

    # Output parameters
    output_dir: Path = Field(default=Path("experiments/results"))
    plot_steps: bool = True
    verbose: int = 1


def generate_data(
    n: int = 200, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate linear and nonlinear data samples."""
    if seed is not None:
        np.random.seed(seed)

    X = np.random.uniform(-1, 1, n).reshape(-1, 1)
    Y1 = linear_model(X.reshape(-1)) + np.random.normal(0, 0.1, n)
    Y2 = nonlinear_model(X.reshape(-1)) + np.random.normal(0, 0.1, n)
    return X, Y1, Y2


def fit_models(
    X: np.ndarray,
    Y1: np.ndarray,
    Y2: np.ndarray,
    n_estimators: int = 200,
    min_samples_leaf: int = 5,
    verbose: int = 1,
) -> dict:
    """Fit RF and OLS models to both datasets."""
    models = {}

    # RandomForest on linear
    reg_rflinear = RandomForestRegressor(
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf
    )
    reg_rflinear.fit(X, Y1)
    fval_rflinear = reg_rflinear.predict(X).reshape(-1, 1)
    mrss_rflinear = np.mean((Y1 - fval_rflinear.flatten()) ** 2)
    models["rflinear"] = {
        "reg": reg_rflinear,
        "fval": fval_rflinear,
        "rmse": np.sqrt(mrss_rflinear),
        "Y": Y1,
        "true_fn": linear_model,
    }
    # OLS on linear
    reg_olslinear = LinearRegression()
    reg_olslinear.fit(X, Y1)
    fval_olslinear = reg_olslinear.predict(X).reshape(-1, 1)
    mrss_olslinear = np.mean((Y1 - fval_olslinear.flatten()) ** 2)
    models["olslinear"] = {
        "reg": reg_olslinear,
        "fval": fval_olslinear,
        "rmse": np.sqrt(mrss_olslinear),
        "Y": Y1,
        "true_fn": linear_model,
    }
    # RandomForest on nonlinear
    reg_rfnonlin = RandomForestRegressor(
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf
    )
    reg_rfnonlin.fit(X, Y2)
    fval_rfnonlin = reg_rfnonlin.predict(X).reshape(-1, 1)
    mrss_rfnonlin = np.mean((Y2 - fval_rfnonlin.flatten()) ** 2)
    models["rfnonlin"] = {
        "reg": reg_rfnonlin,
        "fval": fval_rfnonlin,
        "rmse": np.sqrt(mrss_rfnonlin),
        "Y": Y2,
        "true_fn": nonlinear_model,
    }
    # OLS on nonlinear
    reg_olsnonlin = LinearRegression()
    reg_olsnonlin.fit(X, Y2)
    fval_olsnonlin = reg_olsnonlin.predict(X).reshape(-1, 1)
    mrss_olsnonlin = np.mean((Y2 - fval_olsnonlin.flatten()) ** 2)
    models["olsnonlin"] = {
        "reg": reg_olsnonlin,
        "fval": fval_olsnonlin,
        "rmse": np.sqrt(mrss_olsnonlin),
        "Y": Y2,
        "true_fn": nonlinear_model,
    }
    if verbose > 0:
        print(
            f"RMSE - OLS linear: {models['olslinear']['rmse']:.4f}, OLS nonlinear: {models['olsnonlin']['rmse']:.4f}"
        )
        print(
            f"RMSE - RF linear: {models['rflinear']['rmse']:.4f}, RF nonlinear: {models['rfnonlin']['rmse']:.4f}"
        )
    return models


def apply_xtrapolation(
    X: np.ndarray, models: dict, xeval: np.ndarray, config: ExperimentConfig
) -> tuple[dict, dict]:
    """Apply Xtrapolation to all models and return bounds + tuned parameters."""
    results = {}
    tuned_params = {}

    for model_name in ["rflinear", "olslinear", "rfnonlin", "olsnonlin"]:
        if config.verbose > 1:
            print(f"Applying Xtrapolati0.0806on to {model_name}...")
        model_data = models[model_name]
        # Create Xtrapolation instance
        xtra = Xtrapolation(
            orders=[1],
            deriv_config=config.deriv_config.model_copy(),
            extra_config=config.extra_config,
            verbose=config.verbose,
        )
        # Parameter tuning
        xtra.parameter_tuning(
            X,
            model_data["fval"],
            model_data["Y"],
            None,
            config.rf_pars_list,
            config.pen_list,
            loss="mse",
            tol=config.tol,
        )
        if config.verbose > 1:
            print(
                f"Selected parameters: {xtra.deriv_config.rf_pars}, pen={xtra.deriv_config.pen}"
            )
        tuned_params[f"{model_name}_rf_pars"] = xtra.deriv_config.rf_pars
        tuned_params[f"{model_name}_pen"] = xtra.deriv_config.pen
        # Get prediction bounds
        bounds = xtra.prediction_bounds(X, model_data["fval"], xeval.reshape(-1, 1))
        # Extract bounds (first order only)
        ybdd = np.empty((len(xeval), 2))
        ybdd[:, 0] = bounds[:, 0, 0]
        ybdd[:, 1] = bounds[:, 0, 1]
        yhat = (ybdd[:, 0] + ybdd[:, 1]) / 2
        results[model_name] = {
            "xtra": xtra,
            "bounds": bounds,
            "ybdd": ybdd,
            "yhat": yhat,
        }
    return results, tuned_params


def bootstrap_evaluation(
    X: np.ndarray,
    Y: np.ndarray,
    xeval: np.ndarray,
    config: ExperimentConfig,
    deriv_config_bootstrap: DerivativeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Run single bootstrap iteration."""
    n = X.shape[0]
    boot = np.random.choice(range(n), n, replace=True)

    # Fit new RF on bootstrap sample
    reg = RandomForestRegressor(
        n_estimators=config.n_estimators, min_samples_leaf=config.min_samples_leaf
    )
    reg.fit(X[boot, :], Y[boot])
    fval = reg.predict(X[boot, :]).reshape(-1, 1)
    # Create xtrapolation with fitted parameters
    xtra = Xtrapolation(
        orders=[1],
        deriv_config=deriv_config_bootstrap,
        extra_config=config.extra_config,
        verbose=0,
    )
    # Get bounds
    bounds = xtra.prediction_bounds(X[boot, :], fval, xeval.reshape(-1, 1), refit=True)
    lower = bounds[:, 0, 0]
    upper = bounds[:, 0, 1]
    return lower, upper


def compute_bootstrap_cis(
    X: np.ndarray,
    Y1: np.ndarray,
    Y2: np.ndarray,
    xeval: np.ndarray,
    config: ExperimentConfig,
    tuned_params: dict,
) -> dict:
    """Compute bootstrap CIs for both models."""
    B = config.n_bootstrap
    n_eval = len(xeval)
    ybdd1_lo = np.zeros((n_eval, B))
    ybdd1_up = np.zeros((n_eval, B))
    ybdd2_lo = np.zeros((n_eval, B))
    ybdd2_up = np.zeros((n_eval, B))

    # Create derivative config once to reuse in all bootstrap iterations
    deriv_config_linear = DerivativeConfig(
        num_trees=config.deriv_config.num_trees,
        no_locpol=config.deriv_config.no_locpol,
        rf_pars=tuned_params["rflinear_rf_pars"],
        pen=tuned_params["rflinear_pen"],
    )
    deriv_config_nonlinear = DerivativeConfig(
        num_trees=config.deriv_config.num_trees,
        no_locpol=config.deriv_config.no_locpol,
        rf_pars=tuned_params["rfnonlin_rf_pars"],
        pen=tuned_params["rfnonlin_pen"],
    )

    for k in range(B):
        if config.verbose > 0 and k % 20 == 0:
            print(f"Bootstrap {k+1}/{B}")
        # Linear model
        lower, upper = bootstrap_evaluation(X, Y1, xeval, config, deriv_config_linear)
        ybdd1_lo[:, k] = lower
        ybdd1_up[:, k] = upper
        # Nonlinear model
        lower, upper = bootstrap_evaluation(
            X, Y2, xeval, config, deriv_config_nonlinear
        )
        ybdd2_lo[:, k] = lower
        ybdd2_up[:, k] = upper
    return {
        "linear": {
            "lower": np.quantile(ybdd1_lo, 0.05, axis=1),
            "upper": np.quantile(ybdd1_up, 0.95, axis=1),
        },
        "nonlinear": {
            "lower": np.quantile(ybdd2_lo, 0.05, axis=1),
            "upper": np.quantile(ybdd2_up, 0.95, axis=1),
        },
        "raw": {
            "ybdd1_lo": ybdd1_lo,
            "ybdd1_up": ybdd1_up,
            "ybdd2_lo": ybdd2_lo,
            "ybdd2_up": ybdd2_up,
        },
    }


def plot_xtrapolation_bounds(
    X: np.ndarray,
    Y1: np.ndarray,
    Y2: np.ndarray,
    models: dict,
    xtra_results: dict,
    xeval: np.ndarray,
    config: ExperimentConfig,
):
    """Plot Xtrapolation bounds without bootstrap."""
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    width = 5.876
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(width, width * 0.35)
    ax[0].set_xlabel("$X$")
    ax[0].set_ylabel("$Y$")
    ax[1].set_xlabel("$X$")
    ax[1].set_ylabel("$Y$")
    ax[0].set_xlim([-1.5, 1.5])
    ax[0].set_ylim([-0.5, 1])
    ax[1].set_xlim([-1.5, 1.5])
    ax[1].set_ylim([-0.4, 1.3])

    # Plot data
    ax[0].scatter(X[:, 0], Y1, alpha=0.5, s=4, label="Data")
    ax[1].scatter(X[:, 0], Y2, alpha=0.5, s=4, label="Data")

    # Plot RF fits
    ax[0].plot(
        xeval,
        models["rflinear"]["reg"].predict(xeval.reshape(-1, 1)),
        color="brown",
        label="RF",
    )
    ax[1].plot(
        xeval,
        models["rfnonlin"]["reg"].predict(xeval.reshape(-1, 1)),
        color="brown",
        label="RF",
    )

    # Plot Xtrapolation bounds (direct from xtrapolation, no bootstrap)
    ybdd_rflinear = xtra_results["rflinear"]["ybdd"]
    ybdd_rfnonlin = xtra_results["rfnonlin"]["ybdd"]

    ax[0].plot(
        xeval, linear_model(xeval), color="black", linestyle="dashed", label="$\\Psi$"
    )
    ax[0].plot(
        xeval, ybdd_rflinear[:, 0], color="mediumblue", label="Xtrapolation bounds"
    )
    ax[0].plot(xeval, ybdd_rflinear[:, 1], color="mediumblue")
    ax[0].fill_between(
        xeval, ybdd_rflinear[:, 0], ybdd_rflinear[:, 1], color="mediumblue", alpha=0.2
    )

    ax[1].plot(xeval, nonlinear_model(xeval), color="black", linestyle="dashed")
    ax[1].plot(xeval, ybdd_rfnonlin[:, 0], color="mediumblue")
    ax[1].plot(xeval, ybdd_rfnonlin[:, 1], color="mediumblue")
    ax[1].fill_between(
        xeval, ybdd_rfnonlin[:, 0], ybdd_rfnonlin[:, 1], color="mediumblue", alpha=0.2
    )

    fig.tight_layout()
    fig.legend(loc="lower center", ncols=4, bbox_to_anchor=(0.5, -0.1))
    plt.savefig(
        output_dir / "linear_vs_xtrapolation_xtrapolation_bounds.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_results(
    X: np.ndarray,
    Y1: np.ndarray,
    Y2: np.ndarray,
    models: dict,
    bootstrap_cis: dict,
    xeval: np.ndarray,
    config: ExperimentConfig,
):
    """Generate main experiment plots with bootstrap CIs."""
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    width = 5.876
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(width, width * 0.35)
    ax[0].set_xlabel("$X$")
    ax[0].set_ylabel("$Y$")
    ax[1].set_xlabel("$X$")
    ax[1].set_ylabel("$Y$")
    ax[0].set_xlim([-1.5, 1.5])
    ax[0].set_ylim([-0.5, 1])
    ax[1].set_xlim([-1.5, 1.5])
    ax[1].set_ylim([-0.4, 1.3])

    # Plot data
    ax[0].scatter(X[:, 0], Y1, alpha=0.5, s=4)
    ax[1].scatter(X[:, 0], Y2, alpha=0.5, s=4)
    fig.tight_layout()

    if config.plot_steps:
        plt.savefig(
            output_dir / "linear_vs_xtrapolation_part1.pdf", bbox_inches="tight"
        )
    # Plot OLS fits
    ax[0].plot(
        xeval,
        models["olslinear"]["reg"].predict(xeval.reshape(-1, 1)),
        color="darkgreen",
        linestyle="dashdot",
        label="OLS",
    )
    ax[1].plot(
        xeval,
        models["olsnonlin"]["reg"].predict(xeval.reshape(-1, 1)),
        color="darkgreen",
        linestyle="dashdot",
    )
    fig.tight_layout()
    if config.plot_steps:
        plt.savefig(
            output_dir / "linear_vs_xtrapolation_part2.pdf", bbox_inches="tight"
        )
    # Plot RF fits
    ax[0].plot(
        xeval,
        models["rflinear"]["reg"].predict(xeval.reshape(-1, 1)),
        color="brown",
        label="RF",
    )
    ax[1].plot(
        xeval, models["rfnonlin"]["reg"].predict(xeval.reshape(-1, 1)), color="brown"
    )
    fig.tight_layout()
    if config.plot_steps:
        plt.savefig(
            output_dir / "linear_vs_xtrapolation_part3.pdf", bbox_inches="tight"
        )
    # Plot Xtrapolation
    bounds1_lo = bootstrap_cis["linear"]["lower"]
    bounds1_up = bootstrap_cis["linear"]["upper"]
    bounds2_lo = bootstrap_cis["nonlinear"]["lower"]
    bounds2_up = bootstrap_cis["nonlinear"]["upper"]

    ax[0].plot(
        xeval, linear_model(xeval), color="black", linestyle="dashed", label="$\\Psi$"
    )
    ax[0].plot(xeval, bounds1_lo, color="mediumblue", label="extrapolation-aware CI")
    ax[0].plot(xeval, bounds1_up, color="mediumblue")
    ax[0].fill_between(xeval, bounds1_lo, bounds1_up, color="mediumblue", alpha=0.2)
    ax[1].plot(xeval, nonlinear_model(xeval), color="black", linestyle="dashed")
    ax[1].plot(xeval, bounds2_lo, color="mediumblue")
    ax[1].plot(xeval, bounds2_up, color="mediumblue")
    ax[1].fill_between(xeval, bounds2_lo, bounds2_up, color="mediumblue", alpha=0.2)
    fig.tight_layout()
    fig.legend(loc="lower center", ncols=4, bbox_to_anchor=(0.5, -0.1))
    plt.savefig(output_dir / "linear_vs_xtrapolation.pdf", bbox_inches="tight")
    plt.close()


def run_coverage_experiment(config: ExperimentConfig, tuned_params: dict) -> dict:
    """Run coverage verification experiment."""
    if config.verbose > 0:
        print("\n" + "=" * 50)
        print("Coverage verification experiment")
        print("=" * 50)

    # Create RNG for reproducible but different data in each experiment
    rng = np.random.RandomState(config.seed)

    xeval = np.linspace(config.xeval_min, config.xeval_max, config.xeval_points)
    coverage_linear = np.zeros(len(xeval))
    coverage_nonlinear = np.zeros(len(xeval))

    # Create output directory for incremental plots
    coverage_dir = config.output_dir / "coverage_progress"
    coverage_dir.mkdir(parents=True, exist_ok=True)

    # Setup figure for incremental plotting
    width = 5.876
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(width, width * 0.35)

    # Setup axes
    ax[0].set_xlabel("$X$")
    ax[0].set_ylabel("$Y$")
    ax[0].set_title("Linear Model - Coverage CIs")
    ax[0].set_xlim([config.xeval_min, config.xeval_max])
    ax[0].set_ylim([-0.5, 1])
    ax[0].grid(alpha=0.3)

    ax[1].set_xlabel("$X$")
    ax[1].set_ylabel("$Y$")
    ax[1].set_title("Nonlinear Model - Coverage CIs")
    ax[1].set_xlim([config.xeval_min, config.xeval_max])
    ax[1].set_ylim([-0.4, 1.3])
    ax[1].grid(alpha=0.3)

    # Plot true functions (these stay on top)
    true_linear = linear_model(xeval)
    true_nonlinear = nonlinear_model(xeval)

    # Create derivative config once to reuse in all bootstrap iterations
    deriv_config_linear = DerivativeConfig(
        num_trees=config.deriv_config.num_trees,
        no_locpol=config.deriv_config.no_locpol,
        rf_pars=tuned_params["rflinear_rf_pars"],
        pen=tuned_params["rflinear_pen"],
    )
    deriv_config_nonlinear = DerivativeConfig(
        num_trees=config.deriv_config.num_trees,
        no_locpol=config.deriv_config.no_locpol,
        rf_pars=tuned_params["rfnonlin_rf_pars"],
        pen=tuned_params["rfnonlin_pen"],
    )

    # Track coverage counts for efficient running average computation
    coverage_count_linear = 0
    coverage_count_nonlinear = 0
    n_eval = len(xeval)

    for exp in range(config.n_experiments):
        if config.verbose > 0 and exp % 10 == 0:
            print(f"Coverage exp {exp+1}/{config.n_experiments}")
        # Generate new data for each experiment with a different seed
        exp_seed = rng.randint(0, 2**31 - 1)
        X_exp, Y1_exp, Y2_exp = generate_data(config.n_samples, seed=exp_seed)
        # Run bootstrap for both models (parameter tuning done within bootstrap)
        boot_lower_linear = np.zeros((len(xeval), config.n_bootstrap))
        boot_upper_linear = np.zeros((len(xeval), config.n_bootstrap))
        boot_lower_nonlinear = np.zeros((len(xeval), config.n_bootstrap))
        boot_upper_nonlinear = np.zeros((len(xeval), config.n_bootstrap))
        for b in range(config.n_bootstrap):
            # Linear
            lower, upper = bootstrap_evaluation(
                X_exp, Y1_exp, xeval, config, deriv_config_linear
            )
            boot_lower_linear[:, b] = lower
            boot_upper_linear[:, b] = upper
            # Nonlinear
            lower, upper = bootstrap_evaluation(
                X_exp, Y2_exp, xeval, config, deriv_config_nonlinear
            )
            boot_lower_nonlinear[:, b] = lower
            boot_upper_nonlinear[:, b] = upper
        # Calculate bootstrap CIs
        ci_lower_linear = np.quantile(boot_lower_linear, 0.05, axis=1)
        ci_upper_linear = np.quantile(boot_upper_linear, 0.95, axis=1)
        ci_lower_nonlinear = np.quantile(boot_lower_nonlinear, 0.05, axis=1)
        ci_upper_nonlinear = np.quantile(boot_upper_nonlinear, 0.95, axis=1)

        # Check if true model is within CI and accumulate counts
        coverage_cond_linear = (true_linear >= ci_lower_linear) & (
            true_linear <= ci_upper_linear
        )
        coverage_cond_nonlinear = (true_nonlinear >= ci_lower_nonlinear) & (
            true_nonlinear <= ci_upper_nonlinear
        )
        coverage_linear += coverage_cond_linear
        coverage_nonlinear += coverage_cond_nonlinear
        coverage_count_linear += coverage_cond_linear.sum()
        coverage_count_nonlinear += coverage_cond_nonlinear.sum()

        # Determine if this CI covers the true function (for coloring)
        covers_linear = np.all(
            (true_linear >= ci_lower_linear) & (true_linear <= ci_upper_linear)
        )
        covers_nonlinear = np.all(
            (true_nonlinear >= ci_lower_nonlinear)
            & (true_nonlinear <= ci_upper_nonlinear)
        )

        # Add CI bound lines to plot with color based on coverage
        alpha = 0.4
        linewidth = 0.5
        color_linear = "green" if covers_linear else "red"
        color_nonlinear = "green" if covers_nonlinear else "red"

        ax[0].plot(
            xeval, ci_lower_linear, color=color_linear, alpha=alpha, linewidth=linewidth
        )
        ax[0].plot(
            xeval, ci_upper_linear, color=color_linear, alpha=alpha, linewidth=linewidth
        )
        ax[1].plot(
            xeval,
            ci_lower_nonlinear,
            color=color_nonlinear,
            alpha=alpha,
            linewidth=linewidth,
        )
        ax[1].plot(
            xeval,
            ci_upper_nonlinear,
            color=color_nonlinear,
            alpha=alpha,
            linewidth=linewidth,
        )

        # Redraw true functions on top
        ax[0].plot(
            xeval,
            true_linear,
            color="black",
            linestyle="dashed",
            linewidth=1.5,
            zorder=100,
        )
        ax[1].plot(
            xeval,
            true_nonlinear,
            color="black",
            linestyle="dashed",
            linewidth=1.5,
            zorder=100,
        )

        # Update current coverage in titles (using pre-computed counts)
        current_coverage_linear = coverage_count_linear / ((exp + 1) * n_eval)
        current_coverage_nonlinear = coverage_count_nonlinear / ((exp + 1) * n_eval)
        ax[0].set_title(
            f"Linear Model ({exp+1}/{config.n_experiments})\n"
            f"Avg Coverage: {current_coverage_linear:.2%}"
        )
        ax[1].set_title(
            f"Nonlinear Model ({exp+1}/{config.n_experiments})\n"
            f"Avg Coverage: {current_coverage_nonlinear:.2%}"
        )

        # Save incremental plot after every experiment
        fig.tight_layout()
        plt.savefig(
            coverage_dir / f"coverage_progress_{exp+1:03d}.pdf", bbox_inches="tight"
        )
        if config.verbose > 1 and (exp + 1) % 10 == 0:
            print(f"  Saved progress plot at experiment {exp+1}")

    plt.close(fig)

    # Calculate coverage probabilities
    coverage_prob_linear = coverage_linear / config.n_experiments
    coverage_prob_nonlinear = coverage_nonlinear / config.n_experiments

    # Create results dict for plotting
    coverage_results = {
        "linear": coverage_prob_linear,
        "nonlinear": coverage_prob_nonlinear,
        "xeval": xeval,
    }

    # Plot coverage results using extracted function
    plot_coverage_results(coverage_results, config)

    if config.verbose > 0:
        print(
            f"\nCoverage: Linear {coverage_prob_linear.mean():.2%}, Nonlinear {coverage_prob_nonlinear.mean():.2%}"
        )
        print(f"Progress plots: {coverage_dir}/")

    return coverage_results


def print_summary(
    models: dict, bootstrap_cis: dict, coverage_results: dict | None = None
):
    """Print experiment summary."""
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(
        f"RMSE - Linear: OLS={models['olslinear']['rmse']:.4f}, RF={models['rflinear']['rmse']:.4f}"
    )
    print(
        f"RMSE - Nonlinear: OLS={models['olsnonlin']['rmse']:.4f}, RF={models['rfnonlin']['rmse']:.4f}"
    )
    lin_width = np.mean(
        bootstrap_cis["linear"]["upper"] - bootstrap_cis["linear"]["lower"]
    )
    nonlin_width = np.mean(
        bootstrap_cis["nonlinear"]["upper"] - bootstrap_cis["nonlinear"]["lower"]
    )
    print(f"CI Width - Linear: {lin_width:.4f}, Nonlinear: {nonlin_width:.4f}")
    if coverage_results is not None:
        print(
            f"Coverage (target 90%) - Linear: {coverage_results['linear'].mean():.2%}, Nonlinear: {coverage_results['nonlinear'].mean():.2%}"
        )
    print("=" * 50)


def save_experiment_results(results, X, Y1, Y2, xeval, config, filepath):
    """Save experiment results to disk."""
    data = {
        "X": X,
        "Y1": Y1,
        "Y2": Y2,
        "xeval": xeval,
        "models": results["models"],
        "xtra_results": results["xtra_results"],
        "bootstrap_cis": results["bootstrap_cis"],
        "coverage_results": results["coverage_results"],
        "config": config,
    }
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, output_path)
    print(f"\nResults saved to {output_path}")


def load_experiment_results(filepath):
    """Load experiment results from disk."""
    return joblib.load(filepath)


def plot_coverage_results(coverage_results, config):
    """Plot coverage verification results."""
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    xeval = coverage_results["xeval"]
    coverage_prob_linear = coverage_results["linear"]
    coverage_prob_nonlinear = coverage_results["nonlinear"]

    width = 5.876
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(width, width * 0.35)
    ax[0].set_xlabel("$X$")
    ax[0].set_ylabel("coverage")
    ax[0].set_title("linear")
    ax[1].set_xlabel("$X$")
    ax[1].set_ylabel("coverage")
    ax[1].set_title("nonlinear")
    ax[0].set_xlim([-3, 3])
    ax[0].set_ylim([0, 1.1])
    ax[1].set_xlim([-3, 3])
    ax[1].set_ylim([0, 1.1])

    # Coverage linear model
    ax[0].axhline(y=0.90, color="r", linestyle="--", label="Target Coverage")
    ax[0].plot(xeval, coverage_prob_linear, label="Empirical Coverage")
    ax[0].grid(alpha=0.3)
    ax[0].fill_between(
        [-1, 1],
        [-10, -10],
        [10, 10],
        color="black",
        alpha=0.1,
        label="$\\mathcal{D}$",
    )

    # Coverage nonlinear model
    ax[1].axhline(y=0.90, color="r", linestyle="--")
    ax[1].plot(xeval, coverage_prob_nonlinear)
    ax[1].grid(alpha=0.3)
    ax[1].fill_between(
        [-1, 1],
        [-10, -10],
        [10, 10],
        color="black",
        alpha=0.1,
    )

    fig.legend(loc="lower center", ncols=4, bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()
    plt.savefig(output_dir / "linear_vs_xtrapolation_coverage.pdf", bbox_inches="tight")
    plt.close()


def plot_from_results(data, output_dir=None):
    """Generate all plots from saved results."""
    # Extract data
    X = data["X"]
    Y1 = data["Y1"]
    Y2 = data["Y2"]
    xeval = data["xeval"]
    models = data["models"]
    xtra_results = data["xtra_results"]
    bootstrap_cis = data["bootstrap_cis"]
    coverage_results = data["coverage_results"]
    config = data["config"]

    # Override output directory if specified
    if output_dir is not None:
        config.output_dir = Path(output_dir)

    setup_plotting()
    if config.verbose > 0:
        print("Generating plots from saved results...")

    # Generate plots
    plot_results(X, Y1, Y2, models, bootstrap_cis, xeval, config)

    if coverage_results is not None:
        plot_coverage_results(coverage_results, config)

    if config.verbose > 0:
        print_summary(models, bootstrap_cis, coverage_results)


def run_experiment(config: ExperimentConfig, skip_coverage: bool = False):
    """Run complete experiment."""
    setup_plotting()
    xeval = np.linspace(config.xeval_min, config.xeval_max, config.xeval_points)
    if config.verbose > 0:
        print("=" * 50)
        print(
            f"Linear vs Xtrapolation (n={config.n_samples}, bootstrap={config.n_bootstrap})"
        )
        print("=" * 50)

    X, Y1, Y2 = generate_data(config.n_samples, config.seed)
    models = fit_models(
        X, Y1, Y2, config.n_estimators, config.min_samples_leaf, config.verbose
    )
    xtra_results, tuned_params = apply_xtrapolation(X, models, xeval, config)
    plot_xtrapolation_bounds(X, Y1, Y2, models, xtra_results, xeval, config)
    bootstrap_cis = compute_bootstrap_cis(X, Y1, Y2, xeval, config, tuned_params)
    plot_results(X, Y1, Y2, models, bootstrap_cis, xeval, config)

    coverage_results = (
        run_coverage_experiment(config, tuned_params) if not skip_coverage else None
    )

    if config.verbose > 0:
        print_summary(models, bootstrap_cis, coverage_results)
        print(f"\nPlots saved to {config.output_dir}/")

    results = {
        "models": models,
        "xtra_results": xtra_results,
        "bootstrap_cis": bootstrap_cis,
        "coverage_results": coverage_results,
    }
    return results, X, Y1, Y2, xeval


@click.command()
@click.option(
    "--skip-coverage", is_flag=True, help="Skip coverage verification experiment"
)
@click.option(
    "--verbose", default=1, help="Verbosity level (0=silent, 1=normal, 2=debug)"
)
@click.option("--no-plot-steps", is_flag=True, help="Skip intermediate plotting steps")
@click.option(
    "--save-results",
    type=click.Path(),
    default=None,
    help="Save results to file for later plotting",
)
@click.option(
    "--load-results",
    type=click.Path(exists=True),
    default=None,
    help="Load results from file and generate plots only",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Override output directory for plots (when loading results)",
)
def main(skip_coverage, verbose, no_plot_steps, save_results, load_results, output_dir):
    """
    Run the linear vs xtrapolation experiment.

    This script compares linear and nonlinear models, applies xtrapolation,
    computes bootstrap confidence intervals, and optionally runs a coverage
    verification experiment.

    Use --save-results to save raw results for later plotting.
    Use --load-results to skip computation and only regenerate plots.
    """
    # Load and plot mode
    if load_results is not None:
        print("Loading results from file...")
        data = load_experiment_results(load_results)
        plot_from_results(data, output_dir=output_dir)
        return

    # Normal experiment mode
    config = ExperimentConfig(
        seed=42,
        n_samples=250,
        n_estimators=200,
        min_samples_leaf=5,
        deriv_config=DerivativeConfig(
            num_trees=100,
            no_locpol=False,
        ),
        extra_config=ExtrapolationConfig(
            nn=20,
            aggregation="intersection-corrected",
        ),
        tol=1.0,
        rf_pars_list=[
            {"min_samples_leaf": 100},
            {"min_samples_leaf": 10},
        ],
        pen_list=[5.0, 0.05],
        n_bootstrap=50,
        n_experiments=200,
        xeval_min=-3.0,
        xeval_max=3.0,
        xeval_points=100,
        output_dir=(
            Path(output_dir)
            if output_dir
            else Path("experiments/results/visualizations")
        ),
        plot_steps=not no_plot_steps,
        verbose=verbose,
    )

    # Run experiment
    results, X, Y1, Y2, xeval = run_experiment(config, skip_coverage=skip_coverage)

    # Save results if requested
    if save_results is not None:
        save_experiment_results(results, X, Y1, Y2, xeval, config, save_results)

    return results


if __name__ == "__main__":
    main()
