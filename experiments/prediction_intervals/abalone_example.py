from ucimlrepo import fetch_ucirepo
import numpy as np
import argparse
import os
import pickle
from quantile_forest import RandomForestQuantileRegressor
from sklearn.preprocessing import StandardScaler
from experiments.helpers.cp_methods import (cqr_QuantileForest,
                                            cqr_QuantileNN,
                                            QuantileNN,
                                            quantile_cross_validation)
from xtrapolation.xtrapolation import Xtrapolation


##
# Parse experiment inputs
##

print("Working directory: " + os.getcwd())

parser = argparse.ArgumentParser(
    description="Xtrapolation simulation experiment",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--split", type=int,
                    help="number of split to run")
parser.add_argument("-m", "--method",
                    help="quantile regression method")
parser.add_argument("-r", "--runname",
                    help="name for experiment")
args = parser.parse_args()
config = vars(args)
print(config)

runname = config['runname']
split = config['split']
method = config['method']


# set output path
output_path = "experiments/results/abalone_analysis/"
os.makedirs(output_path, exist_ok=True)


##
# Load data and process
##

# fetch dataset
abalone = fetch_ucirepo(id=1)

# data (as pandas dataframes)
Xraw = abalone.data.features
yraw = abalone.data.targets

# subsample and convert
nmax = Xraw.shape[0]
X = Xraw.iloc[:nmax, :]
X.iloc[:, 0] = X['Sex'].factorize()[0]
X = X.astype(float).to_numpy()
y = yraw.iloc[:nmax].to_numpy().flatten()


##
# Setup parameters
##

# Set quantiles
quantiles = [0.1, 0.9]

# Set xtrapolation parameters
no_xtra_features = [0]
parameters = {'orders': [1],
              'deriv_params': {
                  'num_rotate': 0,
                  'rf_pars': {},
                  'num_trees': 200,
                  'smoothed_predictions': False,
                  'pen': 0,
                  'penalize_intercept': True
              },
              'extra_params': {
                  'nn': 100,
                  'dist': 'rfnn',
                  'rf_pars': {},
                  'num_trees': 200,
                  'aggregation': 'optimal-average',
                  'alpha': 0,
              },
              'verbose': 2}

# CV parameter grids
rf_pars_list = [
    {'min_samples_leaf': 10},
    {'min_samples_leaf': 5},
    {}
]

pen_list = [0.1, 0.01, 0.001]


# Tuning parameters wrapper function
def tuning_wrapper_fun(X, qmat, y):
    rf_pars = [None]*len(quantiles)
    pens = [None]*len(quantiles)
    xtra = Xtrapolation(**parameters)
    for i, qq in enumerate(quantiles):
        rf_pars[i], pens[i] = xtra.parameter_tuning(
            X, (qmat[:, 0] + qmat[:, 1])/2, y,
            no_xtra_features,
            rf_pars_list, pen_list,
            loss="quantile", q=qq, tol=1)
        print(f'quantile {qq}:')
        print(rf_pars[i])
        print(pens[i])
    return rf_pars, pens


###
# Multiple train-test splits
##

def train_test_split(train_ind, method):
    Xtrain = X[train_ind, :]
    ytrain = y[train_ind]

    # Scaling (mean and variance)
    std_scale = StandardScaler()
    std_scale.fit(Xtrain)
    Xtrain = std_scale.transform(Xtrain)
    Xscaled = std_scale.transform(X, copy=True)

    if method == "qrf":
        # Run quantile regression (with forests)
        opt_params = quantile_cross_validation(Xtrain, ytrain, quantiles,
                                               "qrf", n_folds=5)
        opt_params['n_estimators'] = 2000
        print(opt_params)
        qrf = RandomForestQuantileRegressor(**opt_params)
        qrf.fit(Xtrain, ytrain)
        qmat = qrf.predict(Xscaled, quantiles=quantiles)
    elif method == "conf-qrf":
        # Run conformalized quantile regression (with forests)
        idx = np.random.permutation(Xtrain.shape[0])
        n_half = int(np.floor(Xtrain.shape[0]/2))
        idx_train, idx_cal = (idx[:n_half], idx[n_half:2*n_half])
        opt_params = quantile_cross_validation(
            Xtrain[idx_train, :], ytrain[idx_train], quantiles,
            "conf-qrf", n_folds=5, tol=1)
        opt_params['n_estimators'] = 2000
        print(opt_params)
        qmat = cqr_QuantileForest(Xtrain, ytrain,
                                  Xscaled, quantiles,
                                  opt_params, (idx_train, idx_cal))
    elif method == "qnn":
        # Run quantile regression (with neural nets)
        opt_params = quantile_cross_validation(Xtrain, ytrain, quantiles,
                                               "qnn", n_folds=5, tol=1)
        opt_params['batch_size'] = 64
        opt_params['epochs'] = 1000
        print(opt_params)
        qmat = QuantileNN(Xtrain, ytrain,
                          Xscaled, quantiles, opt_params)
    elif method == "conf-qnn":
        # Run conformalized quantile regression (with neural nets)
        idx = np.random.permutation(Xtrain.shape[0])
        n_half = int(np.floor(Xtrain.shape[0]/2))
        idx_train, idx_cal = (idx[:n_half], idx[n_half:2*n_half])
        opt_params = quantile_cross_validation(
            Xtrain[idx_train, :], ytrain[idx_train], quantiles,
            "conf-qnn", n_folds=5, tol=1)
        opt_params['batch_size'] = 64
        opt_params['epochs'] = 1000
        print(opt_params)
        qmat = cqr_QuantileNN(Xtrain, ytrain,
                              Xscaled, quantiles,
                              opt_params, (idx_train, idx_cal))

    # Run parameter tuning
    rf_pars, pens = tuning_wrapper_fun(
        Xtrain, qmat[train_ind, :], ytrain)

    # Xtrapolation
    bounds_list = [None] * len(quantiles)
    for i, qq in enumerate(quantiles):
        # Run xtrapolation on quantile
        parameters['deriv_params']['rf_pars'] = rf_pars[i]
        parameters['deriv_params']['pen'] = pens[i]
        xtra = Xtrapolation(**parameters)
        print(f"Working on quantile {qq}")
        bounds_list[i] = xtra.prediction_bounds(
            Xtrain, qmat[train_ind, i], Xscaled,
            no_xtra_features,
            refit=True)

    # Return results
    return {'train_ind': train_ind,
            'quantiles': quantiles,
            'pars': (rf_pars, pens),
            'qmat': qmat,
            'bounds_list': bounds_list}


##
# Create splits
##

# set seed
np.random.seed(1)

num_splits = 8
split_len = int(len(y)/num_splits)
thresholds = [k*split_len for k in range(num_splits)] + [len(y)]

# Extrapolating train-test splits (depending on X)
xvec = X[:, 1]
sortX = np.argsort(xvec)
train_inds = [np.repeat(True, len(y)) for k in range(num_splits)]
for k in range(num_splits):
    train_inds[k][sortX[thresholds[k]:thresholds[k+1]]] = False

# Interpolating train-test splits (fully random)
randX = np.random.permutation(len(y))
train_inds += [(randX <= thresholds[k]) | (thresholds[k+1] < randX)
               for k in range(num_splits)]


##
# Run experiment
##

res = train_test_split(train_inds[split], method)

# Save results
results_dict = {
    'res': res,
    'parameters': parameters
}

with open(output_path + f'abalone_{runname}_{method}_{split}.pkl', 'wb') as f:
    pickle.dump(results_dict, f)
