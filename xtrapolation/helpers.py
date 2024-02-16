import numpy as np
import math
import jax
from adaXT.decision_tree import DecisionTree
from adaXT.criteria import (Squared_error,
                            Linear_regression)


##
# Functions to compute automatic derivatives
##


# First order directional derivative
def Dv(f, v):
    def fv(vx, x): return f((x + vx * v).reshape(1, -1)).reshape(())
    tmp = jax.grad(fv)
    def Dfv(x): return(tmp(0.0, x.reshape(-1, 1)))
    return jax.vmap(Dfv, 0)


# Second orders directional derivative
def DDv(f, v):
    def Dfv(vx, x): return Dv(f, v)((x + vx * v).reshape(1, -1)).reshape(())
    tmp = jax.grad(Dfv)
    def DDfv(x): return(tmp(0.0, x.reshape(-1, 1)))
    return jax.vmap(DDfv, 0)


##
# adaXT based weight function
##

def rf_weights_adaXT(X, Y,
                     Xeval=None,
                     rf_pars={},
                     num_trees=1000,
                     criteria="Squared_error",
                     verbose=0):

    if criteria == "Squared_error":
        tree = DecisionTree(tree_type="Regression",
                            criteria=Squared_error,
                            **rf_pars)
    elif criteria == "Linear_regression":
        tree = DecisionTree(tree_type="Regression",
                            criteria=Linear_regression,
                            **rf_pars)
    if verbose > 0:
        print("Fitting forest and extracting weights...")

    n, _ = X.shape
    nn = 0
    if Xeval is not None:
        nn, _ = Xeval.shape
        X = np.r_[X, Xeval]
    weight_mat = np.zeros((n+nn, n+nn))
    s = 0.5
    bn = int(n * s)

    for k in range(num_trees):
        # Draw boostrap sample
        boot_sample = np.random.choice(np.arange(n),
                                       bn, replace=False)
        split1 = boot_sample[:int(bn/2)]
        split2 = np.concatenate([boot_sample[int(bn/2):],
                                 np.arange(nn)+n])
        # Fit tree
        tree.fit(X[split1, :], Y[split1].flatten())
        # Extract weights
        weight_mat[np.ix_(split2, split2)] += tree.predict_leaf_matrix(
            X[split2, :])
    # Normalize weights (rows correspond to weights - non-symetric)
    weight_mat /= weight_mat.sum(axis=1)[:, None]
    return weight_mat


##
# Function to fit local polynomials
##

def penalized_locpol(fval, v, X, weights, degree,
                     pen=0, penalize_intercept=False):
    v = v.reshape(-1, 1)
    n = X.shape[0]
    dd = degree + 1
    if penalize_intercept:
        pen_list = list(range(0, dd))
    else:
        pen_list = list(range(1, dd))
    # Construct design matrices
    DDmat = np.zeros((n*dd, n*dd))
    DYmat = np.zeros((n*dd, 1))
    for i in range(n):
        Wi = np.sqrt(weights[i, :].reshape(-1, 1))
        # Construct DDmat (block-diagonal)
        x0v = X[i, :].dot(v)
        Di = np.tile((X.dot(v) - x0v).reshape(-1, 1), dd)**np.arange(dd) * Wi
        DDmat[(i*dd):((i+1)*dd), (i*dd):((i+1)*dd)] = (Di.T).dot(Di)
        # Construct DYmat
        DYmat[(i*dd):((i+1)*dd), :] = (Di.T).dot(
            (fval.reshape(-1, 1)) * Wi)
    Z = np.zeros((dd, dd))
    for kk in pen_list:
        Z[kk, kk] = math.factorial(kk)
    PP = np.kron(np.diag(np.sum(weights, axis=1)) - weights, Z)
    penmat = pen * (PP.T).dot(PP)
    B = np.linalg.solve(DDmat + penmat, DYmat)
    coefs = B.reshape(n, -1)
    # Extract derivatives from coefficients
    deriv_mat = coefs * np.array([math.factorial(k)
                                  for k in range(degree+1)])
    return(deriv_mat)


##
# Function to predict with local polynomials
##

def locpol_predict(coefs, weights, Xtrain, Xtest, v, degree):
    ntest = Xtest.shape[0]
    dd = degree + 1
    fhat = np.zeros(ntest)
    for i in range(ntest):
        Xi = Xtest[i, :].dot(v)
        Di = np.tile((Xi - Xtrain.dot(v)).reshape(-1, 1),
                     dd)**np.arange(dd)
        fhat[i] = np.sum(np.sum(Di*coefs, axis=1) * weights[i, :])
    return(fhat)
