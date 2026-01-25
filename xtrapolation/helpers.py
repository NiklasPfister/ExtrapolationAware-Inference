"""
Helper functions and configuration for Xtrapolation.

This module provides:
- Configuration models (Pydantic)
- Automatic differentiation utilities (using JAX)
- Random forest weight computation (using adaXT)
- Local polynomial fitting for derivative estimation
- Bound aggregation utilities
"""

import math
from typing import Literal
from scipy.linalg import solve

import jax
import numpy as np
from adaXT.criteria import PartialLinear, PartialQuadratic, SquaredError
from adaXT.random_forest import RandomForest
from pydantic import BaseModel, Field


# =============================================================================
# Configuration Models
# =============================================================================


class DerivativeConfig(BaseModel):
    """Configuration for derivative estimation."""

    rf_pars: dict = Field(default_factory=dict)
    num_trees: int = 200
    num_rotate: int = 0
    pen: float = 0.0
    penalize_intercept: bool = False
    smoothed_predictions: bool = False
    no_locpol: bool = True
    criteria: Literal["SquaredError", "PartialLinear", "PartialQuadratic"] = (
        "PartialQuadratic"
    )

    class Config:
        arbitrary_types_allowed = True


class ExtrapolationConfig(BaseModel):
    """Configuration for extrapolation bound computation."""

    nn: int = 1  # Number of nearest neighbors for anchor points
    dist: Literal["rf", "rfnn", "euclidean"] = "rf"
    num_trees: int = 200
    alpha: float = 0.0  # Quantile for derivative bounds
    beta: float = 0.0  # Parameter for quantile-intersection aggregation
    aggregation: Literal[
        "average",
        "intersection",
        "tightest",
        "quantile-intersection",
        "sorted",
        "intersection-corrected",
    ] = "intersection-corrected"
    rf_pars: dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# Feature Handling
# =============================================================================


def get_feature_indices(
    d: int, no_xtra_features: list | None = None
) -> tuple[list, list]:
    """
    Compute which features to use for extrapolation.

    Args:
        d: Total number of features
        no_xtra_features: Features to exclude from extrapolation

    Returns:
        Tuple of (xtra_features, no_xtra_features)
    """
    if no_xtra_features is None:
        no_xtra_features = []
    xtra_features = [k for k in range(d) if k not in no_xtra_features]
    return xtra_features, no_xtra_features


# =============================================================================
# Bound Aggregation
# =============================================================================


def aggregate_bounds(
    f_lower: np.ndarray,
    f_upper: np.ndarray,
    method: str,
    weights: np.ndarray | None = None,
    beta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate bounds from multiple anchor points.

    Args:
        f_lower: Lower bounds array of shape (num_anchors, num_orders)
        f_upper: Upper bounds array of shape (num_anchors, num_orders)
        method: Aggregation method
        weights: Optional weights for averaging
        beta: Parameter for quantile-intersection

    Returns:
        Aggregated (lower, upper) bounds
    """

    if method == "average":
        if weights is None:
            weights = np.ones((f_lower.shape[0], 1)) / f_lower.shape[0]
        middle = f_lower + f_upper / 2.0
        lower = np.sum(f_lower * weights, axis=0)
        upper = np.sum(f_upper * weights, axis=0)

    elif method == "intersection":
        lower = np.max(f_lower, axis=0)
        upper = np.min(f_upper, axis=0)

    elif method == "tightest":
        ind = np.argmin(f_upper - f_lower, axis=0)
        lower = f_lower[ind, np.arange(len(ind))]
        upper = f_upper[ind, np.arange(len(ind))]

    elif method == "quantile-intersection":
        lower = np.quantile(f_lower, 1 - beta, axis=0)
        upper = np.quantile(f_upper, beta, axis=0)

    elif method == "sorted":
        f_upper_sorted = np.sort(f_upper, axis=0)
        f_lower_sorted = np.sort(f_lower, axis=0)[::-1]
        ind = np.argmax((f_upper_sorted - f_lower_sorted) >= 0, axis=0)
        lower = f_lower_sorted[ind, np.arange(len(ind))]
        upper = f_upper_sorted[ind, np.arange(len(ind))]

    elif method == "intersection-corrected":
        lower = np.max(f_lower, axis=0)
        upper = np.min(f_upper, axis=0)
        # Fix cases where bounds cross
        invalid = upper < lower
        middle = (upper + lower) / 2
        lower[invalid] = middle[invalid]
        upper[invalid] = middle[invalid]

    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    middle = (lower + upper) / 2
    return lower, upper, middle

# =============================================================================
# Anchor Weight Computation
# =============================================================================


def compute_anchor_weights(
    X_transformed: np.ndarray,
    x0_transformed: np.ndarray,
    fval: np.ndarray,
    config: ExtrapolationConfig,
) -> np.ndarray:
    """
    Compute weights for anchor points based on distance method.

    Args:
        X_transformed: Transformed training features
        x0_transformed: Transformed test points
        fval: Function values
        config: Extrapolation configuration

    Returns:
        Weight matrix of shape (n_test, n_train)
    """
    n0, n = x0_transformed.shape[0], X_transformed.shape[0]

    if config.dist in ["rf", "rfnn"]:
        weights = rf_weights_adaXT(
            X_transformed,
            fval,
            Xeval=x0_transformed,
            rf_pars=config.rf_pars,
            num_trees=config.num_trees,
            criteria="SquaredError",
        )

        if config.dist == "rfnn":
            # Keep only top nn neighbors
            for i in range(n0):
                nn_max = min(int(np.sum(weights[i, :] > 0)), config.nn)
                top_indices = np.argsort(weights[i, :])[-nn_max:]
                weights[i, :] = 0
                weights[i, top_indices] = 1 / config.nn

    elif config.dist == "euclidean":
        weights = np.zeros((n0, n))
        for i in range(n0):
            distances = np.sum((X_transformed - x0_transformed[i]) ** 2, axis=1)
            top_indices = np.argsort(distances)[: config.nn]
            weights[i, top_indices] = 1 / config.nn
    else:
        raise ValueError(f"Unknown distance method: {config.dist}")

    return weights


# =============================================================================
# Feature Transformation
# =============================================================================


def transform_features_by_gradient(
    derivatives: np.ndarray,
) -> np.ndarray:
    """
    Compute transformation matrix using SVD of gradient matrix.

    Args:
        derivatives: First-order derivatives of shape (n, d_xtra)

    Returns:
        Transformation matrix TT
    """
    mu = derivatives.mean(axis=0)
    _, D, Vt = np.linalg.svd(derivatives - mu[None, :])
    TT = Vt.T * D[None, :]
    return TT


def apply_feature_transform(
    X: np.ndarray,
    TT: np.ndarray,
    xtra_features: list,
    no_xtra_features: list,
) -> np.ndarray:
    """Apply gradient-based transformation to features."""
    X_transformed = X[:, xtra_features] @ TT
    if no_xtra_features:
        X_transformed = np.c_[X_transformed, X[:, no_xtra_features]]
    return X_transformed


# =============================================================================
# Automatic Differentiation (JAX-based)
# =============================================================================


def Dv(f, v: np.ndarray):
    """
    Compute the first-order directional derivative of f in direction v.

    Args:
        f: Function to differentiate. Should accept input of shape (1, d).
        v: Direction vector of shape (d, 1).

    Returns:
        Vectorized function that computes D_v f(x) for each row in x.
    """

    def f_along_v(t: float, x: np.ndarray) -> float:
        return f((x + t * v).reshape(1, -1)).reshape(())

    grad_f = jax.grad(f_along_v)

    def directional_derivative(x: np.ndarray) -> float:
        return grad_f(0.0, x.reshape(-1, 1))

    return jax.vmap(directional_derivative, in_axes=0)


def DDv(f, v: np.ndarray):
    """
    Compute the second-order directional derivative of f in direction v.

    Args:
        f: Function to differentiate.
        v: Direction vector of shape (d, 1).

    Returns:
        Vectorized function that computes D_v^2 f(x) for each row in x.
    """

    def Dv_along_v(t: float, x: np.ndarray) -> float:
        return Dv(f, v)((x + t * v).reshape(1, -1)).reshape(())

    grad_Dv = jax.grad(Dv_along_v)

    def second_derivative(x: np.ndarray) -> float:
        return grad_Dv(0.0, x.reshape(-1, 1))

    return jax.vmap(second_derivative, in_axes=0)


# =============================================================================
# Random Forest Weight Functions (adaXT-based)
# =============================================================================


def rf_weights_adaXT(
    X: np.ndarray,
    Y: np.ndarray,
    Xeval: np.ndarray | None = None,
    rf_pars: dict | None = None,
    num_trees: int = 1000,
    criteria: Literal[
        "SquaredError", "PartialLinear", "PartialQuadratic"
    ] = "PartialQuadratic",
    batch_size: int | None = None,
) -> np.ndarray:
    """
    Compute random forest weights using adaXT.

    Args:
        X: Training features of shape (n, d).
        Y: Training responses of shape (n,).
        Xeval: Optional evaluation points of shape (m, d).
        rf_pars: Additional random forest parameters.
        num_trees: Number of trees in the forest.
        criteria: Splitting criteria.
        batch_size: If provided, process predictions in batches to reduce memory.
                    Recommended: 500-1000 for large datasets.

    Returns:
        Weight matrix of shape (n_eval, n_train) where each row sums to 1.
    """
    rf_pars = rf_pars or {}

    criteria_map = {
        "SquaredError": SquaredError,
        "PartialLinear": PartialLinear,
        "PartialQuadratic": PartialQuadratic,
    }

    if criteria not in criteria_map:
        raise ValueError(
            f"Invalid criteria '{criteria}'. "
            f"Choose from: {list(criteria_map.keys())}"
        )

    rf = RandomForest(
        "Gradient",
        sampling="honest_tree",
        n_estimators=num_trees,
        criteria=criteria_map[criteria],
        **rf_pars,
    )
    rf.fit(X, Y.flatten())

    def predict_weights_batched(X_eval: np.ndarray | None) -> np.ndarray:
        """Predict weights in batches to reduce memory usage."""
        if X_eval is None:
            return rf.predict_weights(scale=True)
        if batch_size is None or X_eval.shape[0] <= batch_size:
            return rf.predict_weights(X_eval, scale=True)

        n_eval = X_eval.shape[0]
        n_train = X.shape[0]
        weight_mat = np.zeros((n_eval, n_train))

        for start_idx in range(0, n_eval, batch_size):
            end_idx = min(start_idx + batch_size, n_eval)
            weight_mat[start_idx:end_idx, :] = rf.predict_weights(
                X_eval[start_idx:end_idx, :], scale=True
            )

        return weight_mat

    return predict_weights_batched(Xeval)


def fit_derivatives_adaXT(
    X: np.ndarray,
    Y: np.ndarray,
    Xeval: np.ndarray | None = None,
    rf_pars: dict | None = None,
    num_trees: int = 1000,
    criteria: Literal[
        "SquaredError", "PartialLinear", "PartialQuadratic"
    ] = "PartialQuadratic",
) -> np.ndarray:
    """
    Estimate derivatives using adaXT's gradient forest.

    Args:
        X: Features of shape (n, d).
        Y: Responses of shape (n,).
        rf_pars: Additional random forest parameters.
        num_trees: Number of trees.
        criteria: Splitting criteria.

    Returns:
        Derivative estimates at each training point.
    """
    rf_pars = rf_pars or {}

    criteria_map = {
        "SquaredError": SquaredError,
        "PartialLinear": PartialLinear,
        "PartialQuadratic": PartialQuadratic,
    }

    rf = RandomForest(
        "Gradient",
        sampling="honest_tree",
        n_estimators=num_trees,
        criteria=criteria_map[criteria],
        **rf_pars,
    )
    rf.fit(X, Y.flatten())

    return rf.predict(X) if Xeval is None else rf.predict(Xeval)


# =============================================================================
# Local Polynomial Fitting
# =============================================================================


def penalized_locpol(
    fval: np.ndarray,
    v: np.ndarray,
    X: np.ndarray,
    weights: np.ndarray,
    degree: int,
    pen: float = 0.0,
    penalize_intercept: bool = False,
    approximate: bool = False,
) -> np.ndarray:
    """
    Fit local polynomials with optional smoothness penalty.
    Args:
        fval: Function values of shape (n,).
        v: Direction vector of shape (d, 1) for directional derivatives.
        X: Features of shape (n, d).
        weights: Weight matrix of shape (n, n).
        degree: Polynomial degree.
        pen: Penalty strength for smoothness regularization.
        penalize_intercept: Whether to include intercept in the penalty.
    Returns:
        Derivative matrix of shape (n, degree+1).
    """
    v = v.reshape(-1, 1)
    n = X.shape[0]
    dd = degree + 1

    if approximate and n > 1000:
        # Split into groups of size < 1000
        num_groups = (n // 1000) + 1
        indices = np.random.permutation(n)
        groups = np.array_split(indices, num_groups)

        # Accumulate predictions and counts
        deriv_sum = np.zeros((n, dd))

        for g, group_idx in enumerate(groups):
            # Fit on this group
            X_train = X[group_idx, :]
            fval_train = fval[group_idx]
            weights_train = weights[np.ix_(group_idx, group_idx)]

            coefs = _fit_locpol(
                fval_train, v, X_train, weights_train, degree, pen, penalize_intercept
            )
            deriv_sum[group_idx] += coefs * np.array(
                [math.factorial(k) for k in range(dd)]
            )

            # Predict on all other groups
            for h, other_idx in enumerate(groups):
                if g == h:
                    continue
                X_test = X[other_idx, :]
                weights_pred = weights[np.ix_(other_idx, group_idx)]

                deriv_pred = locpol_predict_derivmat(
                    coefs, weights_pred, X_train, X_test, v, degree
                )
                deriv_sum[other_idx, :] += deriv_pred

        return deriv_sum / len(groups)

    coefs = _fit_locpol(fval, v, X, weights, degree, pen, penalize_intercept)
    deriv_mat = coefs * np.array([math.factorial(k) for k in range(dd)])
    return deriv_mat


def _fit_locpol(
    fval: np.ndarray,
    v: np.ndarray,
    X: np.ndarray,
    weights: np.ndarray,
    degree: int,
    pen: float,
    penalize_intercept: bool,
) -> np.ndarray:
    """Fit local polynomial and return raw coefficients (not scaled by factorial)."""
    n = X.shape[0]
    dd = degree + 1
    if penalize_intercept:
        pen_list = list(range(0, dd))
    else:
        pen_list = list(range(1, dd))

    # Precompute X.dot(v) once instead of n times in loop
    Xv = X.dot(v)

    # Precompute these outside the loop
    powers = np.arange(dd)
    sqrt_weights = np.sqrt(weights)
    fval_col = fval.reshape(-1, 1)

    DDmat = np.zeros((n * dd, n * dd))
    DYmat = np.zeros((n * dd, 1))

    for i in range(n):
        Wi = sqrt_weights[i, :].reshape(-1, 1)
        diff = (Xv - Xv[i]).reshape(-1, 1)
        Di = (diff ** powers) * Wi

        start = i * dd
        end = (i + 1) * dd
        DDmat[start:end, start:end] = Di.T @ Di
        DYmat[start:end, :] = Di.T @ (fval_col * Wi)

    # Build Z^2 directly since we need kron(A.T @ A, Z @ Z)
    Z_sq_diag = np.zeros(dd)
    for kk in pen_list:
        fact_k = math.factorial(kk)
        Z_sq_diag[kk] = fact_k * fact_k
    Z_sq = np.diag(Z_sq_diag)

    # Use kron(A.T @ A, Z @ Z) instead of kron(A, Z).T @ kron(A, Z)
    # This avoids computing and multiplying two large matrices
    weight_diff = np.diag(np.sum(weights, axis=1)) - weights
    ATA = weight_diff.T @ weight_diff
    penmat = pen * np.kron(ATA, Z_sq)

    B = solve(DDmat + penmat, DYmat, assume_a="pos")
    coefs = B.reshape(n, -1)
    return coefs


def locpol_predict(
    coefs: np.ndarray,
    weights: np.ndarray,
    Xtrain: np.ndarray,
    Xtest: np.ndarray,
    v: np.ndarray,
    degree: int,
) -> np.ndarray:
    """
    Predict using fitted local polynomials.

    Args:
        coefs: Coefficients from penalized_locpol, shape (n_train, degree+1).
        weights: Weight matrix of shape (n_test, n_train).
        Xtrain: Training features of shape (n_train, d).
        Xtest: Test features of shape (n_test, d).
        v: Direction vector of shape (d, 1).
        degree: Polynomial degree.

    Returns:
        Predictions of shape (n_test,).
    """
    dd = degree + 1
    XXv = Xtest.dot(v) - (Xtrain.dot(v)).T  # (n_test, n_train)
    D = (XXv[..., None] ** np.arange(dd) * coefs).sum(axis=2)
    return (D * weights).sum(axis=1)


def locpol_predict_derivmat(
    coefs: np.ndarray,
    weights: np.ndarray,
    Xtrain: np.ndarray,
    Xtest: np.ndarray,
    v: np.ndarray,
    degree: int,
) -> np.ndarray:
    """
    Predict using fitted local polynomials.

    Args:
        coefs: Coefficients from penalized_locpol, shape (n_train, degree+1).
        weights: Weight matrix of shape (n_test, n_train).
        Xtrain: Training features of shape (n_train, d).
        Xtest: Test features of shape (n_test, d).
        v: Direction vector of shape (d, 1).
        degree: Polynomial degree.

    Returns:
        Predicted derivatives of shape (n_test, degree+1).
    """
    dd = degree + 1
    ntest = Xtest.shape[0]
    deriv_mat = np.zeros((ntest, dd))
    XXv = Xtest.dot(v) - (Xtrain.dot(v)).T  # (n_test, n_train)

    for k in range(dd):
        fact_coefs = np.array(
            [math.factorial(j) // math.factorial(j - k) for j in range(k, dd)]
        )
        exponents = np.arange(dd - k)
        D = (XXv[..., None] ** exponents * coefs[:, k:] * fact_coefs).sum(axis=2)
        deriv_mat[:, k] = (D * weights).sum(axis=1)

    return deriv_mat
