import numpy as np
import torch
from experiments.helpers.conformal.cqr import helper
from experiments.helpers.conformal.nonconformist.nc import RegressorNc
from experiments.helpers.conformal.nonconformist.nc import QuantileRegErrFunc
from sklearn.model_selection import KFold
from quantile_forest import RandomForestQuantileRegressor


###
# Quantile Regression functions
###

# CQR-QuantileForest wrapper
def cqr_QuantileForest(Xtrain, ytrain, Xtest, quantiles, params, idx=None):
    # Compute level
    alpha = 1 - (quantiles[1] - quantiles[0])

    n_train, d = Xtrain.shape

    # Data splitting
    if idx is None:
        idx = np.random.permutation(n_train)
        n_half = int(np.floor(n_train/2))
        idx_train, idx_cal = idx[:n_half], idx[n_half:2*n_half]
    else:
        idx_train, idx_cal = idx

    # Add default conformal prediction parameters
    params_full = {}
    params_full["rf_pars"] = params
    params_full["random_state"] = 1
    params_full["CV"] = False
    params_full["coverage_factor"] = 0.85
    params_full["test_ratio"] = 0.05
    params_full["range_vals"] = 30
    params_full["num_vals"] = 10

    # define QRF model
    quantile_estimator = helper.QuantileForestRegressorAdapter(
        model=None,
        fit_params=None,
        quantiles=[quantiles[0]*100,
                   quantiles[1]*100],
        params=params_full)

    # define the CQR object
    nc = RegressorNc(
        quantile_estimator, QuantileRegErrFunc())

    # run CQR procedure
    y_lower, y_upper = helper.run_icp(
        nc, Xtrain, ytrain, Xtest, idx_train, idx_cal, alpha)
    qmat = np.c_[y_lower, y_upper]

    return qmat


# QuantileNN wrapper
def QuantileNN(Xtrain, ytrain, Xtest, quantiles, params):
    n_train, d = Xtrain.shape

    # define QRF model
    quantile_estimator = helper.AllQNet_RegressorAdapter(
        model=None,
        fit_params=None,
        in_shape=d,
        hidden_size=params['hidden_size'],
        quantiles=quantiles,
        learn_func=torch.optim.Adam,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        dropout=0.1,
        lr=0.0005,
        wd=params['weight_decay'],
        test_ratio=0.05,
        random_state=1,
        use_rearrangement=False)

    # Fit quantile regression
    quantile_estimator.fit(Xtrain, ytrain)

    # Predict on test data
    qmat = quantile_estimator.predict(Xtest)

    return qmat


# CQR-QuantileNN wrapper
def cqr_QuantileNN(Xtrain, ytrain, Xtest, quantiles, params, idx=None):
    n_train, d = Xtrain.shape

    # Compute level
    alpha = 1 - (quantiles[1] - quantiles[0])

    # Data splitting
    if idx is None:
        idx = np.random.permutation(n_train)
        n_half = int(np.floor(n_train/2))
        idx_train, idx_cal = idx[:n_half], idx[n_half:2*n_half]
    else:
        idx_train, idx_cal = idx

    # define QRF model
    quantile_estimator = helper.AllQNet_RegressorAdapter(
        model=None,
        fit_params=None,
        in_shape=d,
        hidden_size=params['hidden_size'],
        quantiles=quantiles,
        learn_func=torch.optim.Adam,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        dropout=0.1,
        lr=0.0005,
        wd=params['weight_decay'],
        test_ratio=0.05,
        random_state=1,
        use_rearrangement=False)

    # define the CQR object
    nc = RegressorNc(
        quantile_estimator, QuantileRegErrFunc())

    # run CQR procedure
    y_lower, y_upper = helper.run_icp(
        nc,  Xtrain, ytrain,
        Xtest,
        idx_train, idx_cal, alpha)
    qmat = np.c_[y_lower, y_upper]

    return qmat


###
# Cross-validation for quantile regressions
###


def quantile_cross_validation(X, y, quantiles, method, n_folds=3, tol=1):
    n, d = X.shape
    kf = KFold(n_splits=n_folds, shuffle=True)
    if method == 'qrf':
        par_grid = [1, 5, 10, 20, 40, 80, 160]
        cv_scores = np.zeros((len(quantiles), n, len(par_grid)))
        for ii, md in enumerate(par_grid):
            qrf = RandomForestQuantileRegressor(n_estimators=200,
                                                min_samples_leaf=md)
            qmat = np.zeros((n, len(quantiles)))
            for (train, test) in kf.split(X):
                qrf.fit(X[train, :], y[train])
                qmat[test, :] = qrf.predict(X[test, :],
                                            quantiles=quantiles)
            for kk, q in enumerate(quantiles):
                cv_scores[kk, :, ii] = (
                    q * np.abs(qmat[:, kk] - y) * (y > qmat[:, kk]) +
                    (1-q) * np.abs(qmat[:, kk] - y) * (y <= qmat[:, kk]))
        scores = cv_scores.mean(axis=0)
        avg_scores = scores.mean(axis=0)
        min_ind = np.where(avg_scores == avg_scores.min())[0][0]
        var_mat = np.var(scores - scores[:, min_ind][:, None], axis=0)
        se_scores = np.sqrt(var_mat/n)
        print(avg_scores)
        print(se_scores)
        print(avg_scores <= np.min(avg_scores) + tol * se_scores)
        ind = np.max(
            np.where(avg_scores <= np.min(avg_scores) + tol * se_scores))
        opt_params = {'min_samples_leaf': par_grid[ind]}
    elif method == 'conf-qrf':
        par_grid = [1, 5, 10, 20, 40, 80, 160]
        cv_scores = np.zeros((len(quantiles), n, len(par_grid)))
        for ii, md in enumerate(par_grid):
            params = {'n_estimators': 200,
                      'min_samples_leaf': md}
            qmat = np.zeros((n, len(quantiles)))
            for (train, test) in kf.split(X):
                qmat[test, :] = cqr_QuantileForest(
                    X[train, :], y[train], X[test, :],
                    quantiles, params)
            for kk, q in enumerate(quantiles):
                cv_scores[kk, :, ii] = (
                    q * np.abs(qmat[:, kk] - y) * (y > qmat[:, kk]) +
                    (1-q) * np.abs(qmat[:, kk] - y) *
                    (y <= qmat[:, kk]))
        scores = cv_scores.mean(axis=0)
        avg_scores = scores.mean(axis=0)
        min_ind = np.where(avg_scores == avg_scores.min())[0][0]
        var_mat = np.var(scores - scores[:, min_ind][:, None], axis=0)
        se_scores = np.sqrt(var_mat/n)
        print(avg_scores)
        print(se_scores)
        print(avg_scores <= np.min(avg_scores) + tol * se_scores)
        ind = np.max(
            np.where(avg_scores <= np.min(avg_scores) + tol * se_scores))
        opt_params = {'min_samples_leaf': par_grid[ind]}
    elif method == 'qnn':
        wds = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        hiddens = [64, 128, 256]
        cv_scores = np.zeros((len(quantiles), n,
                              len(wds), len(hiddens)))
        params = {'batch_size': 64,
                  'epochs': 10}
        for ii, wd in enumerate(wds):
            for ll, hd in enumerate(hiddens):
                params['weight_decay'] = wd
                params['hidden_size'] = hd
                qmat = np.zeros((n, len(quantiles)))
                for (train, test) in kf.split(X):
                    qmat[test, :] = QuantileNN(X[train, :], y[train],
                                               X[test, :], quantiles,
                                               params)
                for kk, q in enumerate(quantiles):
                    cv_scores[kk, :, ii, ll] = (
                        q * np.abs(qmat[:, kk] - y) * (y > qmat[:, kk]) +
                        (1-q) * np.abs(qmat[:, kk] - y) *
                        (y <= qmat[:, kk]))
        scores = cv_scores.mean(axis=0)
        avg_scores = scores.mean(axis=0)
        min_ind = np.where(avg_scores == avg_scores.min())
        ind_x = min_ind[0][0]
        ind_y = min_ind[1][0]
        var_mat = np.var(
            scores - scores[:, ind_x, ind_y][:, None, None], axis=0)
        se_scores = np.sqrt(var_mat/n)
        print(avg_scores)
        print(se_scores)
        print(avg_scores <= avg_scores[ind_x, ind_y] + tol * se_scores)
        ind0, ind1 = np.where(avg_scores <= avg_scores[ind_x, ind_y] +
                              tol * se_scores)
        ind_x = np.min(ind0[ind1 == ind_y])
        opt_params = {'batch_size': 64,
                      'epochs': 1000,
                      'weight_decay': wds[ind_x],
                      'hidden_size': hiddens[ind_y]}
    elif method == 'conf-qnn':
        wds = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        hiddens = [64, 128, 256]
        cv_scores = np.zeros((len(quantiles), n,
                              len(wds), len(hiddens)))
        params = {'batch_size': 64,
                  'epochs': 200}
        for ii, wd in enumerate(wds):
            for ll, hd in enumerate(hiddens):
                params['weight_decay'] = wd
                params['hidden_size'] = hd
                qmat = np.zeros((n, len(quantiles)))
                for (train, test) in kf.split(X):
                    qmat[test, :] = cqr_QuantileNN(X[train, :], y[train],
                                                   X[test, :], quantiles,
                                                   params)
                for kk, q in enumerate(quantiles):
                    cv_scores[kk, :, ii, ll] = (
                        q * np.abs(qmat[:, kk] - y) * (y > qmat[:, kk]) +
                        (1-q) * np.abs(qmat[:, kk] - y) *
                        (y <= qmat[:, kk]))
        scores = cv_scores.mean(axis=0)
        avg_scores = scores.mean(axis=0)
        min_ind = np.where(avg_scores == avg_scores.min())
        ind_x = min_ind[0][0]
        ind_y = min_ind[1][0]
        var_mat = np.var(
            scores - scores[:, ind_x, ind_y][:, None, None], axis=0)
        se_scores = np.sqrt(var_mat/n)
        print(avg_scores)
        print(se_scores)
        print(avg_scores <= avg_scores[ind_x, ind_y] + tol * se_scores)
        ind0, ind1 = np.where(avg_scores <= avg_scores[ind_x, ind_y] +
                              tol * se_scores)
        ind_x = np.min(ind0[ind1 == ind_y])
        opt_params = {'batch_size': 64,
                      'epochs': 1000,
                      'weight_decay': wds[ind_x],
                      'hidden_size': hiddens[ind_y]}

    return opt_params
