import numpy as np
import copy
import math
from xtrapolation.helpers import rf_weights_adaXT
from xtrapolation.helpers import (penalized_locpol, locpol_predict, Dv)
from sklearn.model_selection import KFold


class Xtrapolation:
    """Xtrapolation class

    """
    # Initialize Xtrapolation

    def __init__(
            self,
            orders=np.array([1]),
            deriv_params={},
            extra_params={},
            verbose=1
    ):
        # Set defaults for deriv_params
        self.deriv_params_ = copy.deepcopy(deriv_params)
        if 'rf_pars' not in deriv_params:
            self.deriv_params_['rf_pars'] = {}
        if 'num_rotate' not in deriv_params:
            self.deriv_params_['num_rotate'] = 0
        if 'num_trees' not in deriv_params:
            self.deriv_params_['num_trees'] = 200
        if 'smoothed_predictions' not in deriv_params:
            self.deriv_params_['smoothed_predictions'] = False
        if 'pen' not in deriv_params:
            self.deriv_params_['pen'] = 0
        if 'penalize_intercept' not in deriv_params:
            self.deriv_params_['penalize_intercept'] = False
        # Set defaults for exra_params
        self.extra_params_ = copy.deepcopy(extra_params)
        if 'nn' not in extra_params:
            self.extra_params_['nn'] = 1
        if 'dist' not in extra_params:
            self.extra_params_['dist'] = 'rf'
        if 'num_trees' not in extra_params:
            self.extra_params_['num_trees'] = 200
        if 'alpha' not in extra_params:
            self.extra_params_['alpha'] = 0
        if 'beta' not in extra_params:
            self.extra_params_['beta'] = 0
        if 'aggregation' not in extra_params:
            self.extra_params_['aggregation'] = 'optimal-average'
        if 'rf_pars' not in extra_params:
            self.extra_params_['rf_pars'] = {}

        # Initialize remaining parameters
        self.orders_ = copy.deepcopy(orders)
        self.verbose_ = copy.deepcopy(verbose)

        # Initialize output variables
        self.weights_ = None
        self.predict_weights_ = None
        self.derivatives_ = None

        # Initialize internal variables
        self._max_order = np.max(orders)

    # Compute random forest weights
    def fit_rfweights(self, X, fval, no_xtra_features=None):
        n, d = X.shape
        fval = fval.flatten()
        if no_xtra_features is not None:
            xtra_features = [kk for kk in range(d)
                             if kk not in no_xtra_features]
            d_xtra = len(xtra_features)
        else:
            d_xtra = d
            xtra_features = list(range(d))
        weights = [None]*d_xtra

        # Compute weights for derivative estimation (only on xtra_features)
        for jj, var in enumerate(xtra_features):
            var_order = list(range(d))
            var_order = np.array([var] + var_order[:var]
                                 + var_order[var+1:])
            if self.verbose_ > 1:
                print(f"Fitting RF wrt to {var}. coordinate")
                print(var_order)
            weights[jj] = rf_weights_adaXT(
                X[:, var_order], fval,
                rf_pars=self.deriv_params_['rf_pars'],
                num_trees=self.deriv_params_['num_trees'],
                criteria="Linear_regression",
                verbose=self.verbose_ - 2)
        # Collect output
        self.weights_ = weights

    def fit_rfweights_predict(self, X, fval, x0):
        n, d = X.shape
        n0, _ = x0.shape
        fval = fval.flatten()

        if self.verbose_ > 1:
            print("Fitting weights for extrapolation points...")
        predict_weights = rf_weights_adaXT(
            X, fval, x0,
            rf_pars=self.extra_params_['rf_pars'],
            num_trees=self.extra_params_['num_trees'],
            criteria="Squared_error",
            verbose=self.verbose_ - 2)
        self.predict_weights_ = predict_weights[n:, :n]

    # Function to tune rf_pars and pen parameters
    def parameter_tuning(self, X, fval, Y,
                         no_xtra_features=None,
                         rf_pars_list=None, pen_list=None,
                         loss="mse", q=None, tol=3):
        n = X.shape[0]
        d = X.shape[1]
        eps = np.finfo(float).eps
        if rf_pars_list is None:
            rf_pars_list = [self.deriv_params_['rf_pars']]
        if pen_list is None:
            pen_list = [self.deriv_params_['pen']]
        N1 = len(rf_pars_list)
        N2 = len(pen_list)

        if no_xtra_features is not None:
            xtra_features = [kk for kk in range(d)
                             if kk not in no_xtra_features]
            d_xtra = len(xtra_features)
        else:
            xtra_features = list(range(d))
            d_xtra = d

        # Tuning of parameters derivative estimation (rf_pars and pen)
        if N1 > 1 or N2 > 1:
            kf = KFold(n_splits=5, shuffle=True)
            err_mat = np.zeros((n, N1, N2))
            for i, rf_pars in enumerate(rf_pars_list):
                self.deriv_params_['rf_pars'] = rf_pars
                self.fit_rfweights(X, fval, no_xtra_features)
                for j, pen in enumerate(pen_list):
                    self.deriv_params_['pen'] = pen
                    fhat = np.zeros(n)
                    degree = self._max_order+1
                    # Compute predictions using CV and locpol predictions
                    for (train, test) in kf.split(X):
                        for ll, var in enumerate(xtra_features):
                            vv = np.zeros((d, 1))
                            vv[var] = 1
                            weights = self.weights_[ll][:, train]
                            weights /= weights.sum(axis=1)[:, None]
                            deriv_mat = penalized_locpol(
                                fval[train], vv, X[train, :],
                                weights[train, :],
                                degree=degree,
                                pen=self.deriv_params_['pen'],
                                penalize_intercept=self.deriv_params_[
                                    'penalize_intercept'])
                            coefs = deriv_mat / np.array(
                                [math.factorial(k)
                                 for k in range(degree+1)])
                            fhat[test] += locpol_predict(
                                coefs, weights[test, :],
                                X[train, :], X[test, :],
                                vv, degree)/d_xtra
                    # Select loss
                    if loss == "quantile":
                        err_mat[:, i, j] = (q * np.abs(fhat - Y) * (Y > fhat) +
                                            (1-q) * np.abs(
                                                fhat - Y) * (Y <= fhat))
                    elif loss == "mse":
                        err_mat[:, i, j] = (fhat - Y) ** 2
            # Select optimal rf_pars
            mean_mat = err_mat.mean(axis=0)
            min_ind = np.where(mean_mat == mean_mat.min())
            ind_x = min_ind[0][0]
            ind_y = min_ind[1][0]
            var_mat = np.var(
                err_mat - err_mat[:, ind_x, ind_y][:, None, None],
                axis=0)
            se_mat = np.sqrt(var_mat)/np.sqrt(n)
            bx, by = np.where(
                mean_mat <= mean_mat[ind_x, ind_y] + tol * se_mat + eps)
            ind1 = np.min(bx)
            ind2 = np.min(by[bx == ind1])
            opt_rf_pars = rf_pars_list[ind1]
            opt_pen = pen_list[ind2]
            # print results
            if self.verbose_ > 0:
                print(mean_mat)
                print(se_mat)
        else:
            opt_rf_pars = rf_pars_list[0]
            opt_pen = pen_list[0]
        # Set optimal parameters
        self.deriv_params_['rf_pars'] = opt_rf_pars
        self.deriv_params_['pen'] = opt_pen

        return opt_rf_pars, opt_pen

    # Estimate derivatives
    def fit_derivatives(self, X, fval,
                        no_xtra_features=None, refit=False):
        n, d = X.shape
        fval = fval.flatten()

        # Handle no_xtra_features
        if no_xtra_features is not None:
            xtra_features = [kk for kk in range(d)
                             if kk not in no_xtra_features]
            d_xtra = len(xtra_features)
        else:
            d_xtra = d
            xtra_features = list(range(d))
            no_xtra_features = []

        # Can only compute higher orders if d == 1
        if (self._max_order > 1) & (d > 1):
            raise ValueError(
                "Current implimentation only allows \
                higher-order derivatives for X.shape[1] == 1.")

        # Check if weights are computed
        if self.weights_ is None or refit:
            if self.verbose_ > 0:
                print("Fitting weights for local polynomial...")
            self.fit_rfweights(X, fval, no_xtra_features)

        # Estimate derivatives with local polynomial
        derivatives = np.zeros((self._max_order + 1, n, d_xtra))
        Xtilde = X[:, xtra_features]
        for ll in range(self.deriv_params_['num_rotate'] + 1):
            # Rotate and refit weights
            if ll > 0:
                mu = derivatives[1].mean(axis=0)
                Vt = np.linalg.svd(derivatives[1]-mu[None, :])[2]
                Xtilde = X[:, xtra_features].dot(Vt.T)
                self.fit_rfweights(
                    np.c_[Xtilde, X[:, no_xtra_features]],
                    fval, no_xtra_features=list(range(d_xtra, d)))
            # Fit local polynomial
            for jj in range(d_xtra):
                vv = np.zeros((d_xtra, 1))
                vv[jj] = 1
                tmp = penalized_locpol(fval, vv,
                                       Xtilde,
                                       self.weights_[jj],
                                       degree=self._max_order+1,
                                       pen=self.deriv_params_['pen'],
                                       penalize_intercept=self.deriv_params_[
                                           'penalize_intercept'])
                for kk in range(self._max_order + 1):
                    if (kk == 0 and not
                        self.deriv_params_['smoothed_predictions']):
                        derivatives[kk, :, jj] = fval
                    else:
                        derivatives[kk, :, jj] = tmp[:, kk]
            # Rotate derivatives back
            if ll > 0:
                for kk in range(1, self._max_order + 1):
                    derivatives[kk] = derivatives[kk].dot(Vt)

        # Collect output
        self.derivatives_ = derivatives

    # Compute oracle derivatives
    def fit_derivatives_oracle(self, X, ff,
                               no_xtra_features=None):
        n, d = X.shape

        # Handle no_xtra_features
        if no_xtra_features is not None:
            xtra_features = [kk for kk in range(d)
                             if kk not in no_xtra_features]
            d_xtra = len(xtra_features)
        else:
            d_xtra = d
            xtra_features = list(range(d))
            no_xtra_features = []

        # Oracle only works up to order 2
        if self._max_order > 2:
            raise ValueError("Oracle bounds can only be \
            computed up to order 2.")

        # Compute derivatives
        self.derivatives_ = [None] * 2
        grad_mat = np.zeros((d_xtra, n))
        hessian_mat = np.zeros((d_xtra, d_xtra, n))
        if self._max_order > 0:
            for jj, var in enumerate(xtra_features):
                print(jj)
                vv = np.zeros((d, 1))
                vv[var] = 1
                grad_mat[jj, :] = Dv(ff, vv)(X)
                if self._max_order > 1:
                    for kk, var2 in enumerate(xtra_features[jj:]):
                        zz = np.zeros((d, 1))
                        zz[var2] = 1
                        hessian_mat[jj, kk, :] = Dv(Dv(ff, vv), zz)(X)
                        hessian_mat[kk, jj, :] = hessian_mat[jj, kk, :]
            self.derivatives_[0] = grad_mat
            if self._max_order > 1:
                self.derivatives_[1] = hessian_mat

    # Compute extrapolation bounds
    def prediction_bounds(self, X, fval, x0,
                          no_xtra_features=None, refit=False):
        if len(x0.shape) == 1:
            x0 = x0.reshape(-1, 1)
        n, d = X.shape
        n0 = x0.shape[0]
        fval = fval.flatten()

        # Handle no_xtra_features
        if no_xtra_features is not None:
            xtra_features = [kk for kk in range(d)
                             if kk not in no_xtra_features]
        else:
            xtra_features = list(range(d))
            no_xtra_features = []

        # Check if derivatives and weights are already computed
        if self.weights_ is None or refit:
            if self.verbose_ > 0:
                print("Fitting weights for local polynomial...")
            self.fit_rfweights(X, fval, no_xtra_features)
        if self.derivatives_ is None or refit:
            if self.verbose_ > 0:
                print("Fitting derivatives...")
            self.fit_derivatives(X, fval,
                                 no_xtra_features=no_xtra_features,
                                 refit=refit)

        # Determine weighting for extrapolation points (using rotation)
        mu = self.derivatives_[1].mean(axis=0)
        U, D, Vt = np.linalg.svd(self.derivatives_[1]-mu[None, :])
        TT = (Vt.T) * D[None, :]
        Xtilde = X[:, xtra_features].dot(TT)
        Xtilde = np.c_[Xtilde, X[:, no_xtra_features]]
        x0tilde = x0[:, xtra_features].dot(TT)
        x0tilde = np.c_[x0tilde, x0[:, no_xtra_features]]
        if self.extra_params_['dist'] in ['rf', 'rfnn']:
            self.fit_rfweights_predict(Xtilde, fval, x0tilde)
            weight_x0 = self.predict_weights_
            if self.extra_params_['dist'] == 'rfnn':
                nn = self.extra_params_['nn']
                for ii in range(n0):
                    nn_max = np.min([np.sum(weight_x0[ii, :] > 0), nn])
                    xinds = np.argsort(weight_x0[ii, :])[-nn_max:]
                    weight_x0[ii, :] = 0
                    weight_x0[ii, xinds] = 1/nn
        elif self.extra_params_['dist'] == 'euclidean':
            # find closest points between rotated points
            weight_x0 = np.zeros((n0, n))
            nn = self.extra_params_['nn']
            for ii in range(n0):
                xinds = np.argsort(
                    np.sum((x0tilde[None, ii, :] - Xtilde)**2,
                           axis=1))[:nn]
                weight_x0[ii, xinds] = 1/nn

        # Iterate over all extrapolation points and average/intersect
        bounds = np.zeros((n0, len(self.orders_), 3))
        for ll, xpt in enumerate(x0):
            xinds = np.where(weight_x0[ll, :] != 0)[0]
            if self.verbose_ > 2:
                print(f'Number of anchor points to check: {len(xinds)}')
            f_lower = np.zeros((len(xinds), len(self.orders_)))
            f_upper = np.zeros((len(xinds), len(self.orders_)))
            f_median = np.zeros((len(xinds), len(self.orders_)))
            for ii, xind in enumerate(xinds):
                xx = X[xind, :].reshape(1, -1)
                vv = (xpt - xx)[:, xtra_features]
                vv_norm = np.sqrt(np.sum(vv**2))
                # Compute directional derivatives
                deriv_mat = np.zeros((n, self._max_order+1))
                deriv_mat[:, 0] = self.derivatives_[0, :, :].mean(axis=1)
                if vv_norm > np.finfo(float).eps:
                    vv_direction = np.array(vv/vv_norm).reshape(-1, 1)
                    for kk in range(1, self._max_order + 1):
                        deriv_mat[:, kk] = self.derivatives_[kk, :, :].dot(
                            vv_direction**kk).flatten()
                # Select bounds
                deriv_min = np.quantile(deriv_mat,
                                        self.extra_params_['alpha'],
                                        axis=0)
                deriv_max = np.quantile(deriv_mat,
                                        1-self.extra_params_['alpha'],
                                        axis=0)
                deriv_median = np.quantile(deriv_mat, 0.5, axis=0)

                # Debug output
                if self.verbose_ > 3:
                    print(xx)
                    print(fval[xind].reshape(1, -1))
                    print(deriv_mat[xind, :])

                # Estimate extrapolation bounds
                mterm = 0
                kk = 0
                for oo in range(0, self._max_order+1):
                    if oo in self.orders_:
                        lo_bdd = (deriv_min[oo]*(vv_norm**oo) /
                                  math.factorial(oo))
                        up_bdd = (deriv_max[oo]*(vv_norm**oo) /
                                  math.factorial(oo))
                        median_deriv = (deriv_median[oo]*(vv_norm**oo) /
                                        math.factorial(oo))
                        f_lower[ii, kk] = (mterm + lo_bdd)
                        f_upper[ii, kk] = (mterm + up_bdd)
                        f_median[ii, kk] = (mterm + median_deriv)
                        kk += 1
                    mterm += deriv_mat[xind, oo]*(
                        vv_norm**oo)/math.factorial(oo)

            # Combine bounds over xinds
            ww = (weight_x0[ll, xinds]/np.sum(weight_x0[ll, :]))[:, None]
            f_median = np.sum(f_median*ww, axis=0)
            if self.extra_params_['aggregation'] == "average":
                f_lower = np.sum(f_lower*ww, axis=0)
                f_upper = np.sum(f_upper*ww, axis=0)
            elif self.extra_params_['aggregation'] == "intersection":
                f_lower = np.max(f_lower, axis=0)
                f_upper = np.min(f_upper, axis=0)
            elif self.extra_params_['aggregation'] == "tightest":
                ind = np.argmin(f_upper - f_lower, axis=0)
                f_lower = f_lower[ind, list(range(len(ind)))]
                f_upper = f_upper[ind, list(range(len(ind)))]
            elif self.extra_params_['aggregation'] == "quantile-intersection":
                qt = self.extra_params_['beta']
                f_lower = np.quantile(f_lower, 1-qt, axis=0)
                f_upper = np.quantile(f_upper, qt, axis=0)
            elif self.extra_params_['aggregation'] == "best":
                f_upper.sort(axis=0)
                f_lower[::-1].sort(axis=0)
                ind = np.argmax((f_upper - f_lower) >= 0, axis=0)
                f_lower = f_lower[ind, list(range(len(ind)))]
                f_upper = f_upper[ind, list(range(len(ind)))]
            elif self.extra_params_['aggregation'] == "optimal-average":
                f_lower = np.max(f_lower, axis=0)
                f_upper = np.min(f_upper, axis=0)
                ind = f_upper < f_lower
                average = (f_upper + f_lower)/2
                f_lower[ind] = average[ind]
                f_upper[ind] = average[ind]

            # collect results
            bounds[ll, :, 0] = f_lower
            bounds[ll, :, 1] = f_upper
            bounds[ll, :, 2] = f_median

        # return results
        return(bounds)

    # Compute oracle extrapolation bounds
    def prediction_bounds_oracle(self, X, ff, x0,
                                 no_xtra_features=None, refit=False):
        if len(x0.shape) == 1:
            x0 = x0.reshape(-1, 1)
        n, d = X.shape
        n0 = x0.shape[0]

        # Handle no_xtra_features
        if no_xtra_features is not None:
            xtra_features = [kk for kk in range(d)
                             if kk not in no_xtra_features]
        else:
            xtra_features = list(range(d))
            no_xtra_features = []

        # Oracle only works up to order 2
        if self._max_order > 2:
            raise ValueError("Oracle bounds can only be \
            computed up to order 2.")

        # Compute derivatives
        if self.derivatives_ is None or refit:
            if self.verbose_ > 0:
                print("Fitting derivatives...")
            self.fit_derivatives_oracle(X, ff)
        grad_mat = self.derivatives_[0]
        hessian_mat = self.derivatives_[1]

        # Pre-evaluate function
        fval = ff(X).flatten()

        # Determine weighting for extrapolation points (using rotation)
        mu = grad_mat.mean(axis=1)
        U, D, Vt = np.linalg.svd((grad_mat.T)-mu[None, :])
        TT = (Vt.T) * D[None, :]
        Xtilde = X[:, xtra_features].dot(TT)
        Xtilde = np.c_[Xtilde, X[:, no_xtra_features]]
        x0tilde = x0[:, xtra_features].dot(TT)
        x0tilde = np.c_[x0tilde, x0[:, no_xtra_features]]
        if self.extra_params_['dist'] in ['rf', 'rfnn']:
            self.fit_rfweights_predict(Xtilde, fval, x0tilde)
            weight_x0 = self.predict_weights_
            if self.extra_params_['dist'] == 'rfnn':
                nn = self.extra_params_['nn']
                for ii in range(n0):
                    xinds = np.argsort(weight_x0[ii, :])[-nn:]
                    weight_x0[ii, :] = 0
                    weight_x0[ii, xinds] = 1/nn
        elif self.extra_params_['dist'] == 'euclidean':
            # find closest points between rotated points
            weight_x0 = np.zeros((n0, n))
            nn = self.extra_params_['nn']
            for ii in range(n0):
                xinds = np.argsort(
                    np.sum((x0tilde[None, ii, :] - Xtilde)**2,
                           axis=1))[:nn]
                weight_x0[ii, xinds] = 1/nn

        # Iterate over all extrapolation points and average/intersect
        bounds = np.zeros((n0, len(self.orders_), 2))
        for ll, xpt in enumerate(x0):
            xinds = np.where(weight_x0[ll, :] != 0)[0]
            f_lower = np.zeros((len(xinds), len(self.orders_)))
            f_upper = np.zeros((len(xinds), len(self.orders_)))
            for ii, xind in enumerate(xinds):
                xx = X[xind, :].reshape(1, -1)
                vv = (xpt - xx)[:, xtra_features]
                vv_norm = np.sqrt(np.sum(vv**2))
                # Compute directional derivatives
                if vv_norm < np.finfo(float).eps:
                    deriv_mat = np.zeros((n, self._max_order+1))
                    deriv_mat[:, 0] = fval
                else:
                    vv_direction = np.array(vv/vv_norm).reshape(1, -1)
                    deriv_mat = np.zeros((n, self._max_order+1))
                    for order in range(self._max_order+1):
                        if order == 0:
                            deriv_mat[:, order] = fval
                        elif order == 1:
                            deriv_mat[:, order] = vv_direction.dot(
                                grad_mat).flatten()
                        elif order == 2:
                            deriv_mat[:, order] = (vv_direction).dot(
                                vv_direction.dot(hessian_mat)[0]).flatten()
                # Select bounds
                deriv_min = np.quantile(deriv_mat,
                                        self.extra_params_['alpha'], axis=0)
                deriv_max = np.quantile(deriv_mat,
                                        1-self.extra_params_['alpha'], axis=0)

                # Debug output
                if self.verbose_ > 3:
                    print(xx)
                    print(ff(xx.reshape(1, -1)))
                    print(deriv_mat[xind, :])
                    print(deriv_min)
                    print(deriv_max)
                    print(vv_direction)

                # Compute extrapolation bounds
                mterm = 0
                kk = 0
                for oo in range(0, self._max_order+1):
                    if oo in self.orders_:
                        lo_bdd = (deriv_min[oo]*(vv_norm**oo) /
                                  math.factorial(oo))
                        up_bdd = (deriv_max[oo]*(vv_norm**oo) /
                                  math.factorial(oo))
                        f_lower[ii, kk] = (mterm + lo_bdd)
                        f_upper[ii, kk] = (mterm + up_bdd)
                        kk += 1
                    mterm += deriv_mat[xind, oo]*(
                        vv_norm**oo)/math.factorial(oo)

            # Combine bounds over xinds
            if self.extra_params_['aggregation'] == "average":
                ww = (weight_x0[ll, xinds]/np.sum(weight_x0[ll, :]))[:, None]
                f_lower = np.sum(f_lower*ww, axis=0)
                f_upper = np.sum(f_upper*ww, axis=0)
            elif self.extra_params_['aggregation'] == "intersection":
                f_lower = np.max(f_lower, axis=0)
                f_upper = np.min(f_upper, axis=0)
            elif self.extra_params_['aggregation'] == "tightest":
                ind = np.argmin(f_upper - f_lower, axis=0)
                f_lower = f_lower[ind, list(range(len(ind)))]
                f_upper = f_upper[ind, list(range(len(ind)))]
            elif self.extra_params_['aggregation'] == "quantile-intersection":
                qt = self.extra_params_['beta']
                f_lower = np.quantile(f_lower, 1-qt, axis=0)
                f_upper = np.quantile(f_upper, qt, axis=0)
            elif self.extra_params_['aggregation'] == "best":
                f_upper.sort(axis=0)
                f_lower[::-1].sort(axis=0)
                ind = np.argmax((f_upper - f_lower) >= 0, axis=0)
                f_lower = f_lower[ind, list(range(len(ind)))]
                f_upper = f_upper[ind, list(range(len(ind)))]
            elif self.extra_params_['aggregation'] == "optimal-average":
                f_lower = np.max(f_lower, axis=0)
                f_upper = np.min(f_upper, axis=0)
                ind = f_upper < f_lower
                average = (f_upper + f_lower)/2
                f_lower[ind] = average[ind]
                f_upper[ind] = average[ind]

            # collect results
            bounds[ll, :, 0] = f_lower
            bounds[ll, :, 1] = f_upper

        # return results
        return(bounds)
