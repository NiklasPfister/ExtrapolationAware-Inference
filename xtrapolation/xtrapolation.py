"""
Xtrapolation: Extrapolation-Aware Nonparametric Statistical Inference

Based on Pfister & Bühlmann (2024) - "Extrapolation-Aware Nonparametric Statistical Inference"

This module provides tools for computing extrapolation bounds on conditional functions
using Taylor's theorem and random forest-based derivative estimation.
"""

import math
from typing import Callable, Literal

import numpy as np
from sklearn.model_selection import KFold
import time

from xtrapolation.helpers import (
    DerivativeConfig,
    ExtrapolationConfig,
    get_feature_indices,
    aggregate_bounds,
    compute_anchor_weights,
    transform_features_by_gradient,
    apply_feature_transform,
    rf_weights_adaXT,
    fit_derivatives_adaXT,
    penalized_locpol,
    locpol_predict,
    Dv,
)


class Xtrapolation:
    """
    Xtrapolation: Extrapolation-Aware Nonparametric Statistical Inference.

    Computes extrapolation bounds on conditional functions using Taylor's theorem
    and random forest-based derivative estimation.

    Args:
        orders: Array of derivative orders to compute bounds for (default: [1])
        deriv_config: Configuration for derivative estimation
        extra_config: Configuration for extrapolation bounds
        verbose: Verbosity level (0=silent, 1=progress, 2+=debug)
    """

    def __init__(
        self,
        orders: list[int] | None = None,
        deriv_config: DerivativeConfig | None = None,
        extra_config: ExtrapolationConfig | None = None,
        verbose: int = 0,
    ):
        self.orders = np.array([1]) if orders is None else np.asarray(orders)
        self.deriv_config = deriv_config or DerivativeConfig()
        self.extra_config = extra_config or ExtrapolationConfig()
        self.verbose = verbose

        # Cache computed values
        self._max_order = int(np.max(self.orders))
        self._factorials = np.array(
            [math.factorial(k) for k in range(self._max_order + 1)]
        )

        # Fitted attributes (set during fit/predict)
        self.weights_: list[np.ndarray] | None = None
        self.predict_weights_: np.ndarray | None = None
        self.derivatives_: np.ndarray | list[np.ndarray] | None = None

    # -------------------------------------------------------------------------
    # Random Forest Weight Fitting
    # -------------------------------------------------------------------------

    def fit_rfweights(
        self,
        X: np.ndarray,
        fval: np.ndarray,
        no_xtra_features: list | None = None,
        Xeval: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """
        Fit random forest weights for derivative estimation.

        Args:
            X: Feature matrix of shape (n, d)
            fval: Function values of shape (n,)
            no_xtra_features: Features to exclude from extrapolation

        Returns:
            List of weight matrices, one per extrapolation feature
        """
        _, d = X.shape
        fval = fval.flatten()
        xtra_features, _ = get_feature_indices(d, no_xtra_features)

        weights = []
        for var in xtra_features:
            # Reorder features to put current variable first
            var_order = np.array([var] + [i for i in range(d) if i != var])

            if self.verbose > 1:
                print(f"Fitting RF weights for coordinate {var}")

            weights.append(
                rf_weights_adaXT(
                    X[:, var_order],
                    fval,
                    Xeval=Xeval,
                    rf_pars=self.deriv_config.rf_pars,
                    num_trees=self.deriv_config.num_trees,
                    criteria=self.deriv_config.criteria,
                )
            )

        return weights

    # -------------------------------------------------------------------------
    # Derivative Estimation
    # -------------------------------------------------------------------------

    def fit_derivatives(
        self,
        X: np.ndarray,
        fval: np.ndarray,
        no_xtra_features: list | None = None,
        refit: bool = False,
    ) -> np.ndarray:
        """
        Estimate derivatives using local polynomials or adaXT.

        Args:
            X: Feature matrix of shape (n, d)
            fval: Function values of shape (n,)
            no_xtra_features: Features to exclude
            refit: Whether to refit even if already fitted

        Returns:
            Derivatives array of shape (max_order+1, n, d_xtra)
        """
        n, d = X.shape
        fval = fval.flatten()
        xtra_features, no_xtra_list = get_feature_indices(d, no_xtra_features)
        if self.verbose > 2:
            print("DEBUG:")
            print(f"Extrapolation features: {xtra_features}")
            print(f"Excluded features: {no_xtra_list}")
        d_xtra = len(xtra_features)

        # Higher-order derivatives only supported for 1D
        if self._max_order > 1 and d > 1:
            raise ValueError(
                "Higher-order derivatives (order > 1) only supported for 1D data"
            )

        # Fit weights if needed (for no-locpol method)
        if not self.deriv_config.no_locpol:
            if self.weights_ is None or refit:
                if self.verbose > 0:
                    print("Fitting weights for local polynomial...")
                self.weights_ = self.fit_rfweights(X, fval, no_xtra_features)

        # Initialize derivatives array
        derivatives = np.zeros((self._max_order + 1, n, d_xtra))
        X_work = X[:, xtra_features]

        # Optionally rotate features based on gradient directions
        Vt = None
        for rotation in range(self.deriv_config.num_rotate + 1):
            if rotation > 0:
                mu = derivatives[1].mean(axis=0)
                Vt = np.linalg.svd(derivatives[1] - mu[None, :])[2]
                X_work = X[:, xtra_features] @ Vt.T

                if not self.deriv_config.no_locpol:
                    X_full = np.c_[X_work, X[:, no_xtra_list]]
                    self.weights_ = self.fit_rfweights(
                        X_full, fval, no_xtra_features=list(range(d_xtra, d))
                    )

            # Fit derivatives for each direction
            for j in range(d_xtra):
                direction = np.zeros((d_xtra, 1))
                direction[j] = 1

                if self.deriv_config.no_locpol:
                    dfeature = xtra_features[j]
                    xtra_order = [dfeature] + [
                        k for k in xtra_features if k != dfeature
                    ]

                    # Minimize memory copies: use column_stack only when necessary
                    xtra_cols = X[:, xtra_order]
                    if no_xtra_list:
                        X_full = np.column_stack([xtra_cols, X[:, no_xtra_list]])
                    else:
                        X_full = xtra_cols

                    if self.verbose > 1:
                        print(f"Coordinate {dfeature}")

                    deriv_estimates = fit_derivatives_adaXT(
                        X=X_full,
                        Y=fval,
                        rf_pars=self.deriv_config.rf_pars,
                        num_trees=self.deriv_config.num_trees,
                        criteria=self.deriv_config.criteria,
                    )
                else:
                    if self.weights_ is None:
                        raise ValueError("Weights cannot be None...")
                    t = time.time()
                    deriv_estimates = penalized_locpol(
                        fval,
                        direction,
                        X_work,
                        self.weights_[j],
                        degree=self._max_order + 1,
                        pen=self.deriv_config.pen,
                        penalize_intercept=self.deriv_config.penalize_intercept,
                    )
                    print(time.time() - t)
                # Store results
                for k in range(self._max_order + 1):
                    if k == 0 and not self.deriv_config.smoothed_predictions:
                        derivatives[k, :, j] = fval
                    else:
                        derivatives[k, :, j] = deriv_estimates[:, k]

            # Rotate derivatives back
            if rotation > 0:
                for k in range(1, self._max_order + 1):
                    derivatives[k] = derivatives[k] @ Vt

        return derivatives

    def fit_derivatives_oracle(
        self, X: np.ndarray, ff: Callable, no_xtra_features: list | None = None
    ) -> list[np.ndarray]:
        """
        Compute oracle (true) derivatives using automatic differentiation.

        Args:
            X: Feature matrix
            ff: True function (must be JAX-differentiable)
            no_xtra_features: Features to exclude

        Returns:
            List of [gradient_matrix, hessian_matrix]
        """
        n, d = X.shape
        xtra_features, _ = get_feature_indices(d, no_xtra_features)
        d_xtra = len(xtra_features)

        if self._max_order > 2:
            raise ValueError("Oracle bounds only supported up to order 2")

        grad_mat = np.zeros((d_xtra, n))
        hessian_mat = np.zeros((d_xtra, d_xtra, n))

        if self._max_order > 0:
            for j, var in enumerate(xtra_features):
                if self.verbose > 0:
                    print(f"Computing oracle derivative for variable {j}")

                direction = np.zeros((d, 1))
                direction[var] = 1
                grad_mat[j, :] = Dv(ff, direction)(X)

                if self._max_order > 1:
                    for k, var2 in enumerate(xtra_features[j:]):
                        direction2 = np.zeros((d, 1))
                        direction2[var2] = 1
                        hessian_mat[j, k, :] = Dv(Dv(ff, direction), direction2)(X)
                        hessian_mat[k, j, :] = hessian_mat[j, k, :]

        return [grad_mat, hessian_mat]

    # -------------------------------------------------------------------------
    # Extrapolation Bounds Computation
    # -------------------------------------------------------------------------

    def prediction_bounds(
        self,
        X: np.ndarray,
        fval: np.ndarray,
        x0: np.ndarray,
        no_xtra_features: list | None = None,
        refit: bool = False,
    ) -> np.ndarray:
        """
        Compute extrapolation bounds for test points.

        Args:
            X: Training features of shape (n, d)
            fval: Function values of shape (n,)
            x0: Test points of shape (n0, d)
            no_xtra_features: Features to exclude from extrapolation
            refit: Whether to refit derivatives

        Returns:
            Bounds array of shape (n0, n_orders, 3) where last dim is [lower, upper, middle]
        """
        assert x0.ndim == 2, "x0 is expected to be 2-dimensional"
        n, d = X.shape
        n0 = x0.shape[0]
        fval = fval.flatten()

        xtra_features, no_xtra_list = get_feature_indices(d, no_xtra_features)

        # Fit derivatives if needed
        if self.derivatives_ is None or refit:
            if self.verbose > 0:
                print("Fitting derivatives...")
            self.derivatives_ = self.fit_derivatives(
                X, fval, no_xtra_features=no_xtra_features, refit=refit
            )

        # Transform features for distance computation
        TT = transform_features_by_gradient(self.derivatives_[1])
        X_transformed = apply_feature_transform(X, TT, xtra_features, no_xtra_list)
        x0_transformed = apply_feature_transform(x0, TT, xtra_features, no_xtra_list)

        if self.verbose > 1:
            print("Computing anchor weights...")

        # Compute anchor weights
        anchor_weights = compute_anchor_weights(
            X_transformed, x0_transformed, fval, self.extra_config
        )
        self.predict_weights_ = anchor_weights

        # Pre-compute values used in inner loop
        deriv_mean_0 = self.derivatives_[0].mean(axis=1)
        factorials = self._factorials
        max_order = self._max_order
        orders = self.orders
        n_orders = len(orders)
        alpha = self.extra_config.alpha
        quantile_levels = np.array([alpha, 1 - alpha])

        # Pre-allocate arrays reused in inner loop
        deriv_mat = np.zeros((n, max_order + 1))

        # Compute bounds for each test point
        bounds = np.zeros((n0, n_orders, 3))

        for i, xpt in enumerate(x0):
            anchor_indices = np.where(anchor_weights[i, :] != 0)[0]
            n_anchors = len(anchor_indices)

            if self.verbose > 2:
                print(f"Test point {i}: {n_anchors} anchor points")

            f_lower = np.zeros((n_anchors, n_orders))
            f_upper = np.zeros((n_anchors, n_orders))

            for j, anchor_idx in enumerate(anchor_indices):
                anchor_point = X[anchor_idx, :]

                # Compute direction and distance to anchor
                direction_vec = xpt[xtra_features] - anchor_point[xtra_features]
                distance = np.sqrt(np.sum(direction_vec**2))

                # Compute directional derivatives (reuse pre-allocated array)
                deriv_mat[:, 0] = deriv_mean_0

                if distance > np.finfo(float).eps:
                    unit_direction = (direction_vec / distance).reshape(-1, 1)
                    for k in range(1, max_order + 1):
                        deriv_mat[:, k] = (
                            self.derivatives_[k] @ (unit_direction**k)
                        ).flatten()
                else:
                    deriv_mat[:, 1:] = 0

                # Compute quantile bounds on derivatives (single call for all 3)
                quantiles = np.quantile(deriv_mat, quantile_levels, axis=0)
                deriv_min, deriv_max = (
                    quantiles[0],
                    quantiles[1],
                )

                # Pre-compute powers of distance
                dist_powers = distance ** np.arange(max_order + 1)

                # Compute Taylor expansion bounds
                cumulative_term = 0.0
                order_idx = 0

                for oo in range(max_order + 1):
                    if oo in orders:
                        factor = dist_powers[oo] / factorials[oo]
                        f_lower[j, order_idx] = cumulative_term + deriv_min[oo] * factor
                        f_upper[j, order_idx] = cumulative_term + deriv_max[oo] * factor
                        order_idx += 1

                    cumulative_term += (
                        deriv_mat[anchor_idx, oo] * dist_powers[oo] / factorials[oo]
                    )

            # Aggregate bounds
            weights = anchor_weights[i, anchor_indices] / anchor_weights[i, :].sum()
            weights = weights[:, None]

            # Lower and upper depend on aggregation method
            bounds[i, :, 0], bounds[i, :, 1], bounds[i, :, 2] = aggregate_bounds(
                f_lower,
                f_upper,
                method=self.extra_config.aggregation,
                weights=weights,
                beta=self.extra_config.beta,
            )

        return bounds

    def prediction_bounds_oracle(
        self,
        X: np.ndarray,
        ff: Callable,
        x0: np.ndarray,
        no_xtra_features: list | None = None,
        refit: bool = False,
    ) -> np.ndarray:
        """
        Compute oracle extrapolation bounds using true derivatives.

        Args:
            X: Training features
            ff: True function (JAX-differentiable)
            x0: Test points
            no_xtra_features: Features to exclude
            refit: Whether to refit

        Returns:
            Bounds array of shape (n0, n_orders, 2) with [lower, upper]
        """
        x0 = np.atleast_2d(x0)
        n, d = X.shape
        n0 = x0.shape[0]

        xtra_features, no_xtra_list = get_feature_indices(d, no_xtra_features)

        if self._max_order > 2:
            raise ValueError("Oracle bounds only supported up to order 2")

        # Compute oracle derivatives
        if self.derivatives_ is None or refit:
            if self.verbose > 0:
                print("Fitting oracle derivatives...")
            self.derivatives_ = self.fit_derivatives_oracle(X, ff, no_xtra_features)

        grad_mat = self.derivatives_[0]
        hessian_mat = self.derivatives_[1]
        fval = ff(X).flatten()

        # Transform features
        mu = grad_mat.mean(axis=1)
        _, D, Vt = np.linalg.svd(grad_mat.T - mu[None, :])
        TT = Vt.T * D[None, :]

        X_transformed = apply_feature_transform(X, TT, xtra_features, no_xtra_list)
        x0_transformed = apply_feature_transform(x0, TT, xtra_features, no_xtra_list)

        # Compute anchor weights
        anchor_weights = compute_anchor_weights(
            X_transformed, x0_transformed, fval, self.extra_config
        )
        self.predict_weights_ = anchor_weights

        # Pre-compute values
        factorials = self._factorials
        max_order = self._max_order
        orders = self.orders
        n_orders = len(orders)
        alpha = self.extra_config.alpha

        # Compute bounds
        bounds = np.zeros((n0, n_orders, 2))

        for i, xpt in enumerate(x0):
            anchor_indices = np.where(anchor_weights[i, :] != 0)[0]
            f_lower = np.zeros((len(anchor_indices), n_orders))
            f_upper = np.zeros((len(anchor_indices), n_orders))

            for j, anchor_idx in enumerate(anchor_indices):
                anchor_point = X[anchor_idx, :]
                direction_vec = xpt[xtra_features] - anchor_point[xtra_features]
                distance = np.sqrt(np.sum(direction_vec**2))

                # Compute directional derivatives
                deriv_mat = np.zeros((n, max_order + 1))
                deriv_mat[:, 0] = fval

                if distance > np.finfo(float).eps:
                    unit_dir = (direction_vec / distance).reshape(1, -1)
                    if max_order >= 1:
                        deriv_mat[:, 1] = (unit_dir @ grad_mat).flatten()
                    if max_order >= 2:
                        deriv_mat[:, 2] = (
                            unit_dir @ (unit_dir @ hessian_mat)[0]
                        ).flatten()

                # Compute bounds (single quantile call)
                quantiles = np.quantile(deriv_mat, [alpha, 1 - alpha], axis=0)
                deriv_min, deriv_max = quantiles[0], quantiles[1]

                # Pre-compute powers
                dist_powers = distance ** np.arange(max_order + 1)

                cumulative_term = 0.0
                order_idx = 0

                for order in range(max_order + 1):
                    if order in orders:
                        factor = dist_powers[order] / factorials[order]
                        f_lower[j, order_idx] = (
                            cumulative_term + deriv_min[order] * factor
                        )
                        f_upper[j, order_idx] = (
                            cumulative_term + deriv_max[order] * factor
                        )
                        order_idx += 1

                    cumulative_term += (
                        deriv_mat[anchor_idx, order]
                        * dist_powers[order]
                        / factorials[order]
                    )

            # Aggregate bounds
            weights = anchor_weights[i, anchor_indices] / anchor_weights[i, :].sum()
            bounds[i, :, 0], bounds[i, :, 1], _ = aggregate_bounds(
                f_lower,
                f_upper,
                method=self.extra_config.aggregation,
                weights=weights[:, None],
                beta=self.extra_config.beta,
            )

        return bounds

    # -------------------------------------------------------------------------
    # Parameter Tuning
    # -------------------------------------------------------------------------

    def parameter_tuning(
        self,
        X: np.ndarray,
        fval: np.ndarray,
        Y: np.ndarray,
        no_xtra_features: list | None = None,
        rf_pars_list: list | None = None,
        pen_list: list | None = None,
        loss: Literal["mse", "quantile"] = "mse",
        q: float = 0.9,
        tol: float = 3.0,
    ) -> tuple[dict, float]:
        """
        Tune hyperparameters using cross-validation.

        Args:
            X: Features
            fval: Estimated function values
            Y: True response values
            no_xtra_features: Features to exclude
            rf_pars_list: List of RF parameter dicts to try
            pen_list: List of penalty values to try
            loss: Loss function ("mse" or "quantile")
            q: Quantile for quantile loss
            tol: Tolerance for one-standard-error rule

        Returns:
            Tuple of (optimal_rf_pars, optimal_penalty)
        """
        n, d = X.shape
        eps = np.finfo(float).eps

        rf_pars_list = rf_pars_list or [self.deriv_config.rf_pars]
        if self.deriv_config.no_locpol:
            pen_list = [0]
        else:
            pen_list = pen_list or [self.deriv_config.pen]

        xtra_features, no_xtra_list = get_feature_indices(d, no_xtra_features)
        d_xtra = len(xtra_features)

        if len(rf_pars_list) == 1 and len(pen_list) == 1:
            self.deriv_config.rf_pars = rf_pars_list[0]
            self.deriv_config.pen = pen_list[0]
            return rf_pars_list[0], pen_list[0]

        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True)
        err_mat = np.zeros((n, len(rf_pars_list), len(pen_list)))

        # Accumulate predictions across folds
        fhat_mat = np.zeros((n, len(rf_pars_list), len(pen_list)))
        degree = self._max_order + 1
        X_work = X[:, xtra_features]

        for kk, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f"CV split {kk}...")
            for i, rf_pars in enumerate(rf_pars_list):
                self.deriv_config.rf_pars = rf_pars

                if self.deriv_config.no_locpol:
                    for jj in range(d_xtra):
                        new_order = [jj] + [k for k in range(d_xtra) if k != jj]
                        X_full = np.c_[X_work[:, new_order], X[:, no_xtra_list]]
                        fhat_mat[test_idx, i, 0] += (
                            fit_derivatives_adaXT(
                                X_full[train_idx, :],
                                fval[train_idx],
                                X_full[test_idx, :],
                                rf_pars=rf_pars,
                                num_trees=self.deriv_config.num_trees,
                                criteria=self.deriv_config.criteria,
                            )[:, 0]
                            / d_xtra
                        )
                else:
                    # Process one feature at a time to reduce peak memory
                    # Pre-extract training data slices to avoid repeated indexing
                    X_train = X[train_idx, :]
                    fval_train = fval[train_idx]
                    X_test = X[test_idx, :]

                    for var in xtra_features:
                        # Fit weights for single feature
                        var_order = np.array([var] + [k for k in range(d) if k != var])

                        # Pass list of [train, test] to avoid creating full (n, n_train) matrix
                        weights = rf_weights_adaXT(
                            X_train[:, var_order],
                            fval_train,
                            Xeval=X[:, var_order],
                            rf_pars=rf_pars,
                            num_trees=self.deriv_config.num_trees,
                            criteria=self.deriv_config.criteria,
                            batch_size=2000,
                        )

                        # Extract weight matrices from list
                        weights_train = weights[train_idx, :]
                        weights_test = weights[test_idx, :]

                        direction = np.zeros((d, 1))
                        direction[var] = 1

                        for j, pen in enumerate(pen_list):
                            print("Fitting locpol")
                            deriv_mat = penalized_locpol(
                                fval_train,
                                direction,
                                X_train,
                                weights_train,
                                degree=degree,
                                pen=pen,
                                penalize_intercept=self.deriv_config.penalize_intercept,
                            )
                            coefs = deriv_mat / np.array(
                                [math.factorial(k) for k in range(degree + 1)]
                            )
                            fhat_mat[test_idx, i, j] += (
                                locpol_predict(
                                    coefs,
                                    weights_test,
                                    X_train,
                                    X_test,
                                    direction,
                                    degree,
                                )
                                / d_xtra
                            )

        # Compute errors for all parameter combinations
        for i in range(len(rf_pars_list)):
            for j in range(len(pen_list)):
                fhat = fhat_mat[:, i, j]
                if loss == "quantile":
                    err_mat[:, i, j] = q * np.abs(fhat - Y) * (Y > fhat) + (
                        1 - q
                    ) * np.abs(fhat - Y) * (Y <= fhat)
                else:
                    err_mat[:, i, j] = (fhat - Y) ** 2

        # Select optimal parameters using one-standard-error rule
        mean_mat = err_mat.mean(axis=0)
        min_idx = np.unravel_index(mean_mat.argmin(), mean_mat.shape)

        var_mat = np.var(
            err_mat - err_mat[:, min_idx[0], min_idx[1]][:, None, None], axis=0
        )
        se_mat = np.sqrt(var_mat) / np.sqrt(n)

        # Find most regularized params within tolerance
        valid = mean_mat <= mean_mat[min_idx] + tol * se_mat + eps
        valid_idx = np.where(valid)

        opt_rf_idx = valid_idx[0].min()
        opt_pen_idx = valid_idx[1][valid_idx[0] == opt_rf_idx].min()

        opt_rf_pars = rf_pars_list[opt_rf_idx]
        opt_pen = pen_list[opt_pen_idx]

        if self.verbose > 0:
            print(f"Mean errors:\n{mean_mat}")
            print(f"Standard errors:\n{se_mat}")
            print(f"Selected: rf_pars={opt_rf_pars}, pen={opt_pen}")

        self.deriv_config.rf_pars = opt_rf_pars
        self.deriv_config.pen = opt_pen

        return opt_rf_pars, opt_pen
