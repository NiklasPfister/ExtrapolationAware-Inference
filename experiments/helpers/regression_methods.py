import numpy as np
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# SVR
def svr_regression(X, Y, rf_screen=False):
    param_grid = [
        {'C': np.logspace(-3, 3, 7),
         'gamma': np.logspace(-3, 3, 7)/X.shape[1],
         'kernel': ['rbf']},
    ]
    svr = SVR()
    svr_cv = GridSearchCV(svr, param_grid,
                          scoring='neg_mean_squared_error')
    if rf_screen:
        reg = Pipeline(
            [('rf_screening', SelectFromModel(RandomForestRegressor())),
             ('scale', StandardScaler()),
             ('reg_cv', svr_cv)])
    else:
        reg = Pipeline(
            [('scale', StandardScaler()),
             ('reg_cv', svr_cv)])
    reg.fit(X, Y)
    fval = reg.predict(X).reshape(-1)
    mse = -reg['reg_cv'].best_score_
    return reg, fval, mse


# RandomForest
def rf_regression(X, Y, param_grid=None, rf_screen=False):
    if param_grid is None:
        param_grid = [
            {'n_estimators': [500],
             'max_depth': [1, 2, 4, 8, 16, sys.maxsize]}
        ]
    rf = RandomForestRegressor()
    rf_cv = GridSearchCV(rf, param_grid,
                         scoring='neg_mean_squared_error')
    if rf_screen:
        reg = Pipeline(
            [('rf_screening', SelectFromModel(RandomForestRegressor())),
             ('reg_cv', rf_cv)])
    else:
        reg = Pipeline([('reg_cv', rf_cv)])
    reg.fit(X, Y)
    fval = reg.predict(X).reshape(-1)
    mse = -reg['reg_cv'].best_score_
    return reg, fval, mse


# Multi-layer perceptron
def mlp_regression(X, Y, rf_screen=False):
    param_grid = [
        {'solver': ['adam'],
         'hidden_layer_sizes': [(100,), (20, 20, 20,)],
         'activation': ['relu'],
         'early_stopping': [True],
         'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1]},
    ]
    mlp = MLPRegressor(max_iter=1000)
    mlp_cv = GridSearchCV(mlp, param_grid,
                          scoring='neg_mean_squared_error')
    if rf_screen:
        reg = Pipeline(
            [('rf_screening', SelectFromModel(RandomForestRegressor())),
             ('reg_cv', mlp_cv)])
    else:
        reg = Pipeline([('reg_cv', mlp_cv)])
    reg.fit(X, Y)
    fval = reg.predict(X).reshape(-1)
    mse = -reg['reg_cv'].best_score_
    return reg, fval, mse


# Ordinary least squares
def ols_regression(X, Y):
    ols = LinearRegression()
    ols.fit(X, Y)
    fval = ols.predict(X).reshape(-1)
    mse = mean_squared_error(Y, fval)
    return ols, fval, mse
