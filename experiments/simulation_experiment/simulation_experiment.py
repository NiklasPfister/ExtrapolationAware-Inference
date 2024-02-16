import numpy as np
import pandas as pd
import argparse
import os
from scipy.spatial.distance import cdist
from xtrapolation.xtrapolation import Xtrapolation
import experiments.helpers.regression_methods as reg
import experiments.helpers.examples as ex


##
# Parse experiment inputs
##

print("Working directory: " + os.getcwd())

parser = argparse.ArgumentParser(
    description="Xtrapolation simulation experiment",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--seed", type=int,
                    help="dimension of predictors")
parser.add_argument("-d", "--dimension", type=int,
                    help="dimension of predictors")
parser.add_argument("-n", "--samplesize", type=int,
                    help="sample size of data")
parser.add_argument("-r", "--runname",
                    help="name for of experiment")
args = parser.parse_args()
config = vars(args)
print(config)

seed = config['seed']
run_name = config['runname']
n = config['samplesize']
d = config['dimension']


# set seed
np.random.seed(seed)

# set output path
output_path = f"experiments/results/oos_prediction_{run_name}/"
os.makedirs(output_path, exist_ok=True)

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
Xtmp = np.random.uniform(-2, 2, 1000*d).reshape(-1, d)
slope_vec = slope_vec/np.sqrt(
    np.var(ex.piecewise_linear(Xtmp, slope_vec, grid)))


def ff(x): return(ex.piecewise_linear(x, slope_vec, grid))


##
# Generate data
##

# noise is scaled according to dimension
X, _ = ex.sample_X(n, bds)
Y = np.array(ff(X)).flatten() + np.random.normal(0, 1/10, n)


##
# Fit regression models
##

print("Fitting regressions...")

reg_rf, fval_rf, mspe_rf = reg.rf_regression(X, Y, rf_screen=True)
reg_svr, fval_svr, mspe_svr = reg.svr_regression(X, Y, rf_screen=True)
reg_mlp, fval_mlp, mspe_mlp = reg.mlp_regression(X, Y, rf_screen=True)
reg_ols, fval_ols, mspe_ols = reg.ols_regression(X, Y)


cv_mspe = {'rf_mspe': mspe_rf,
           'svr_mspe': mspe_svr,
           'mlp_mspe': mspe_mlp,
           'ols_mspe': mspe_ols}
print(cv_mspe)

# Save MSPEs
np.save(output_path + f'mspe_dim{d}_size{n}_seed{seed}.npy',
        cv_mspe, allow_pickle=True)


##
# Tuning of Xtrapolation parameters
##

tol = 3
rf_pars_list = [
    {'impurity_tol': 100},
    {'impurity_tol': 10},
    {'impurity_tol': 1},
    {'impurity_tol': 0.1},
    {'impurity_tol': 0.01},
]
pen_list = [10, 1, 0.1, 0.01, 0.001, 0]

parameters = {'orders': [1],
              'deriv_params': {
                  'num_rotate': 0,
                  'num_trees': 200,
                  'smoothed_predictions': False,
                  'penalize_intercept': False,
              },
              'extra_params': {
                  'nn': int(0.5*n),
                  'dist': 'euclidean',
                  'alpha': 0,
                  'aggregation': 'optimal-average',
              },
              'verbose': 2}

xtra = Xtrapolation(**parameters)

# RF parameters
rf_pars_rf, pen_rf = xtra.parameter_tuning(
    X, fval_rf, Y, None,
    rf_pars_list, pen_list,
    loss='mse', tol=tol)
print(rf_pars_rf)
print(pen_rf)

# SVR parameters
rf_pars_svr, pen_svr = xtra.parameter_tuning(
    X, fval_svr, Y, None,
    rf_pars_list, pen_list,
    loss='mse', tol=tol)
print(rf_pars_svr)
print(pen_svr)

# MLP parameters
rf_pars_mlp, pen_mlp = xtra.parameter_tuning(
    X, fval_mlp, Y, None,
    rf_pars_list, pen_list,
    loss='mse', tol=tol)
print(rf_pars_mlp)
print(pen_mlp)

# OLS parameters
rf_pars_ols, pen_ols = xtra.parameter_tuning(
    X, fval_ols, Y, None,
    rf_pars_list, pen_list,
    loss='mse', tol=tol)
print(rf_pars_ols)
print(pen_ols)


# bounds_rf = xtra_rf.prediction_bounds(X, fval_rf, regions[2])
# lo1 = np.max(bounds_rf[:, :, 0], axis=1)
# up1 = np.min(bounds_rf[:, :, 1], axis=1)
# yhat1 = (lo1 + up1)/2

# bounds_svr = xtra_svr.prediction_bounds(X, fval_svr, regions[2])
# lo2 = np.max(bounds_svr[:, :, 0], axis=1)
# up2 = np.min(bounds_svr[:, :, 1], axis=1)
# yhat2 = (lo2 + up2)/2

# bounds = xtra_oracle.prediction_bounds_oracle(X, ff, regions[2])
# lo = np.max(bounds[:, :, 0], axis=1)
# up = np.min(bounds[:, :, 1], axis=1)
# yhat_oracle = (lo + up)/2

# plt.scatter(X[:, 0], Y)
# plt.scatter(X[:, 0], fval_rf)
# plt.scatter(X[:, 0], fval_svr)
# plt.scatter(regions[2][:, 0], yhat1)
# plt.scatter(regions[2][:, 0], yhat2)
# plt.scatter(regions[2][:, 0], lo1)
# plt.scatter(regions[2][:, 0], up1)
# plt.scatter(regions[2][:, 0], lo2)
# plt.scatter(regions[2][:, 0], up2)
# plt.scatter(regions[2][:, 0], lo)
# plt.scatter(regions[2][:, 0], up)


##
# Apply Xtrapolation
##


# Baseline Xtrapolation
parameters_oracle = {'orders': [1],
                     'extra_params': {
                         'nn': int(0.5*n),
                         'dist': 'euclidean',
                         'alpha': 0,
                         'aggregation': 'optimal-average',
                     },
                     'verbose': 2}


# Xtrapolation for RandomForest
print("Fitting Xtrapolation derivatives for RF...")
parameters['deriv_params']['rf_pars'] = rf_pars_rf
parameters['deriv_params']['pen'] = pen_rf
xtra_rf = Xtrapolation(**parameters)
xtra_rf.fit_derivatives(X, fval_rf)

# Xtrapolation for SVR
print("Fitting Xtrapolation derivatives for SVR...")
parameters['deriv_params']['rf_pars'] = rf_pars_svr
parameters['deriv_params']['pen'] = pen_svr
xtra_svr = Xtrapolation(**parameters)
xtra_svr.fit_derivatives(X, fval_svr)

# Xtrapolation for MLP
parameters['deriv_params']['rf_pars'] = rf_pars_mlp
parameters['deriv_params']['pen'] = pen_mlp
print("Fitting Xtrapolation derivatives for MLP...")
xtra_mlp = Xtrapolation(**parameters)
xtra_mlp.fit_derivatives(X, fval_mlp)

# Xtrapolation for OLS
parameters['deriv_params']['rf_pars'] = rf_pars_ols
parameters['deriv_params']['pen'] = pen_ols
print("Fitting Xtrapolation derivatives for OLS...")
xtra_ols = Xtrapolation(**parameters)
xtra_ols.fit_derivatives(X, fval_ols)

# Xtrapolation for oracle
print("Fitting Xtrapolation derivatives for oracle...")
xtra_oracle = Xtrapolation(**parameters_oracle)
xtra_oracle.fit_derivatives_oracle(X, ff)


##
# Evaluation regions
##

ntest = 200
X_Din, X_Dout = ex.sample_X(ntest, bds)
X_full = np.random.uniform(-2, 2, ntest*d).reshape(-1, d)
regions = [X_Din, X_Dout, X_full]

##
# Evaluate different methods
##

# Regression results
prediction_results = pd.DataFrame({})
for k in range(len(regions)):
    prediction_results[f'truth_{k}'] = ff(regions[k])
    prediction_results[f'RF_{k}'] = reg_rf.predict(regions[k])
    prediction_results[f'SVR_{k}'] = reg_svr.predict(regions[k])
    prediction_results[f'MLP_{k}'] = reg_mlp.predict(regions[k])
    prediction_results[f'OLS_{k}'] = reg_ols.predict(regions[k])

# Xtrapolation results
for k in range(len(regions)):
    print(f'Xtrapolation prediction on region: {k}')
    # RF
    bounds_rf = xtra_rf.prediction_bounds(X, fval_rf, regions[k])
    prediction_results[f'XtraLo-RF_{k}'] = np.max(bounds_rf[:, :, 0], axis=1)
    prediction_results[f'XtraUp-RF_{k}'] = np.min(bounds_rf[:, :, 1], axis=1)
    # SVR
    bounds_svr = xtra_svr.prediction_bounds(X, fval_svr, regions[k])
    prediction_results[f'XtraLo-SVR_{k}'] = np.max(bounds_svr[:, :, 0], axis=1)
    prediction_results[f'XtraUp-SVR_{k}'] = np.min(bounds_svr[:, :, 1], axis=1)
    # MLP
    bounds_mlp = xtra_mlp.prediction_bounds(X, fval_mlp, regions[k])
    prediction_results[f'XtraLo-MLP_{k}'] = np.max(bounds_mlp[:, :, 0], axis=1)
    prediction_results[f'XtraUp-MLP_{k}'] = np.min(bounds_mlp[:, :, 1], axis=1)
    # OLS
    bounds_ols = xtra_ols.prediction_bounds(X, fval_ols, regions[k])
    prediction_results[f'XtraLo-OLS_{k}'] = np.max(bounds_ols[:, :, 0], axis=1)
    prediction_results[f'XtraUp-OLS_{k}'] = np.min(bounds_ols[:, :, 1], axis=1)
    # Oracle
    bounds_oracle = xtra_oracle.prediction_bounds_oracle(X, ff, regions[k])
    prediction_results[f'XtraLo-oracle_{k}'] = np.max(
        bounds_oracle[:, :, 0], axis=1)
    prediction_results[f'XtraUp-oracle_{k}'] = np.min(
        bounds_oracle[:, :, 1], axis=1)

# Nearest neighbor
for k in range(len(regions)):
    prediction_results[f'euclidean_nn_{k}'] = np.min(
        cdist(X, regions[k]), axis=0)

# Save first coordinate of regions
for k in range(len(regions)):
    prediction_results[f'X1_region_{k}'] = regions[k][:, 0]


##
# Save all results
##

prediction_results.to_csv(
    output_path + f'prediction_results_dim{d}_size{n}_seed{seed}.csv')
