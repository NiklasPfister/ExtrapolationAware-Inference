from ucimlrepo import fetch_ucirepo
import numpy as np
import argparse
import os
import pickle
from quantile_forest import RandomForestQuantileRegressor
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
parser.add_argument("-r", "--runname",
                    help="name for of experiment")
args = parser.parse_args()
config = vars(args)
print(config)

runname = config['runname']
split = config['split']


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
                  'penalize_intercept': False
              },
              'extra_params': {
                  'nn': 50,
                  'dist': 'rfnn',
                  'rf_pars': {},
                  'num_trees': 200,
                  'aggregation': 'optimal-average',
                  'alpha': 0,
              },
              'verbose': 2}

# CV parameter grids
rf_pars_list = [
    {'max_depth': 1},
    {'max_depth': 2},
    {'max_depth': 3},
    {'max_depth': 4},
]

pen_list = [1, 0.1, 0.01, 0]


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
            loss="quantile", q=qq, tol=3)
        print(f'quantile {qq}:')
        print(rf_pars[i])
        print(pens[i])
    return rf_pars, pens


###
# Multiple train-test splits
##

def train_test_split(train_ind):
    Xtrain = X[train_ind, :]
    ytrain = y[train_ind]

    # Run quantile regression (with forests)
    qrf = RandomForestQuantileRegressor(n_estimators=2000,
                                        max_depth=5)
    qrf.fit(Xtrain, ytrain)
    qmat = qrf.predict(X, quantiles=quantiles)

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
            Xtrain, qmat[train_ind, i], X,
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

res = train_test_split(train_inds[split])

# Save results
results_dict = {
    'res': res,
    'parameters': parameters
}

with open(output_path + f'abalone_{runname}_{split}.pkl', 'wb') as f:
    pickle.dump(results_dict, f)
