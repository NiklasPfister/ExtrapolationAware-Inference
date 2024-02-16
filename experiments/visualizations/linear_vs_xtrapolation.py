import numpy as np
from xtrapolation.xtrapolation import Xtrapolation
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib

np.random.seed(111)

# Set plotting parameters
params = {'axes.labelsize': 8,
          'font.size': 8,
          'legend.fontsize': 8,
          'xtick.labelsize': 6,
          'ytick.labelsize': 6,
          'lines.linewidth': 0.7,
          'text.usetex': True,
          'axes.unicode_minus': True}
matplotlib.rcParams.update(params)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


##
# Define some functions used later
##

# Target density for rejection sampler
def target_density(x):
    return((x < 0) * (1 + x) + (x > 0) * (1 - x))


# Proposal density for rejection sampler
def proposal_density(x):
    return(1/2)


# Nonlinear model for conditional mean
def nonlinear_model(x):
    return(1/(1+np.exp(-3*(x-0.35))))


# Linear model for conditional mean
def linear_model(x):
    return(0.5*x + 0.35)


##
# Create two models: linear (X, Y1) and nonlinear (X, Y2)
##

n = 200
d = 1
# Generate data (with rejection sampler)
X = np.zeros((n, d))
kk = 0
M = 2
while kk < n:
    xx = np.random.uniform(-1, 1, 1)
    u = np.random.uniform(0, 1, 1)
    if (target_density(xx)/proposal_density(xx) >= M * u):
        X[kk, :] = xx
        kk += 1
Y1 = linear_model(X.reshape(-1)) + np.random.normal(0, 0.1, n)
Y2 = nonlinear_model(X.reshape(-1)) + np.random.normal(0, 0.1, n)

# Selct points at which to extrapolate
xeval = np.linspace(-2, 2, 100)


##
# Fit regression models
##

# RandomForest on linear
n_estimators = 200
reg_rflinear = RandomForestRegressor(n_estimators=n_estimators,
                                     max_depth=5)
reg_rflinear.fit(X, Y1)
fval_rflinear = reg_rflinear.predict(X).reshape(-1, 1)
mrss_rflinear = np.mean((Y1 - fval_rflinear)**2)

# OLS on linear
reg_olslinear = LinearRegression()
reg_olslinear.fit(X, Y1)
fval_olslinear = reg_olslinear.predict(X).reshape(-1, 1)
mrss_olslinear = np.mean((Y1 - fval_olslinear)**2)

# RandomForest on nonlinear
n_estimators = 200
reg_rfnonlin = RandomForestRegressor(n_estimators=n_estimators,
                                     max_depth=5)
reg_rfnonlin.fit(X, Y2)
fval_rfnonlin = reg_rfnonlin.predict(X).reshape(-1, 1)
mrss_rfnonlin = np.mean((Y2 - fval_rfnonlin)**2)

# OLS on linear
reg_olsnonlin = LinearRegression()
reg_olsnonlin.fit(X, Y2)
fval_olsnonlin = reg_olsnonlin.predict(X).reshape(-1, 1)
mrss_olsnonlin = np.mean((Y2 - fval_olsnonlin)**2)

# Print OLS RMSE
print(np.sqrt(mrss_olslinear))
print(np.sqrt(mrss_olsnonlin))


##
# Apply Xtrapolation
##

tol = 1
rf_pars_list = [
    {'min_samples_leaf': 200},
    {'min_samples_leaf': 100},
    {'min_samples_leaf': 50},
    {'min_samples_leaf': 10},
]
pen_list = [0.1, 0.01, 0]

# Baseline parameters
parameters = {'orders': [1],
              'deriv_params': {
                  'rf_pars': {},
                  'num_trees': 200,
                  'smoothed_predictions': False,
                  'penalize_intercept': False,
                  'pen': 0,
              },
              'extra_params': {
                  'nn': 20,
                  'dist': 'euclidean',
                  'alpha': 0,
                  'aggregation': 'optimal-average',
              },
              'verbose': 1}

# Xtrapolation for RandomForest on linear
xtra_rflinear = Xtrapolation(**parameters)
xtra_rflinear.parameter_tuning(
    X, fval_rflinear, Y1, None,
    rf_pars_list, pen_list,
    loss='mse', tol=tol)
print(xtra_rflinear.deriv_params_)

bounds_rflinear = xtra_rflinear.prediction_bounds(X, fval_rflinear, xeval)
ybdd_rflinear = np.empty((len(xeval), 2))
ybdd_rflinear[:, 0] = np.max(bounds_rflinear[:, :, 0], axis=1)
ybdd_rflinear[:, 1] = np.min(bounds_rflinear[:, :, 1], axis=1)
yhat_rflinear = (ybdd_rflinear[:, 0] + ybdd_rflinear[:, 1])/2


# Xtrapolation for OLS on linear
xtra_olslinear = Xtrapolation(**parameters)
xtra_olslinear.parameter_tuning(
    X, fval_olslinear, Y1, None,
    rf_pars_list, pen_list,
    loss='mse', tol=tol)
print(xtra_olslinear.deriv_params_)

bounds_olslinear = xtra_olslinear.prediction_bounds(X, fval_olslinear, xeval)
ybdd_olslinear = np.empty((len(xeval), 2))
ybdd_olslinear[:, 0] = np.max(bounds_olslinear[:, :, 0], axis=1)
ybdd_olslinear[:, 1] = np.min(bounds_olslinear[:, :, 1], axis=1)
yhat_olslinear = (ybdd_olslinear[:, 0] + ybdd_olslinear[:, 1])/2


# Xtrapolation for RandomForest on nonlinear
xtra_rfnonlin = Xtrapolation(**parameters)
xtra_rfnonlin.parameter_tuning(
    X, fval_rfnonlin, Y2, None,
    rf_pars_list, pen_list,
    loss='mse', tol=tol)
print(xtra_rfnonlin.deriv_params_)

bounds_rfnonlin = xtra_rfnonlin.prediction_bounds(X, fval_rfnonlin, xeval)
ybdd_rfnonlin = np.empty((len(xeval), 2))
ybdd_rfnonlin[:, 0] = np.max(bounds_rfnonlin[:, :, 0], axis=1)
ybdd_rfnonlin[:, 1] = np.min(bounds_rfnonlin[:, :, 1], axis=1)
yhat_rfnonlin = (ybdd_rfnonlin[:, 0] + ybdd_rfnonlin[:, 1])/2


# Xtrapolation for OLS on nonlinear
xtra_olsnonlin = Xtrapolation(**parameters)
xtra_olsnonlin.parameter_tuning(
    X, fval_olsnonlin, Y2, None,
    rf_pars_list, pen_list,
    loss='mse', tol=tol)
print(xtra_olsnonlin.deriv_params_)

bounds_olsnonlin = xtra_olsnonlin.prediction_bounds(X, fval_olsnonlin, xeval)
ybdd_olsnonlin = np.empty((len(xeval), 2))
ybdd_olsnonlin[:, 0] = np.max(bounds_olsnonlin[:, :, 0], axis=1)
ybdd_olsnonlin[:, 1] = np.min(bounds_olsnonlin[:, :, 1], axis=1)
yhat_olsnonlin = (ybdd_olsnonlin[:, 0] + ybdd_olsnonlin[:, 1])/2


##
# Compute Xtrapolation bootstrap confidence intervals
##


def bootstrap_evaluation(X, Y, reg, xtra):
    boot = np.random.choice(range(n), n, replace=True)
    reg.fit(X[boot, :], Y[boot])
    fval = reg.predict(X[boot, :]).reshape(-1, 1)
    bounds = xtra.prediction_bounds(X[boot, :], fval, xeval, refit=True)
    return bounds


B = 100
ybdd1_lo = np.zeros((len(xeval), B))
ybdd1_up = np.zeros((len(xeval), B))
ybdd2_lo = np.zeros((len(xeval), B))
ybdd2_up = np.zeros((len(xeval), B))
for k in range(B):
    print(k)
    # Initialize regression and xtrapolation
    reg = RandomForestRegressor(n_estimators=500)
    xtra = Xtrapolation(**parameters)
    xtra.deriv_params_ = xtra_rflinear.deriv_params_
    # Run on linear model
    bounds = bootstrap_evaluation(X, Y1, reg, xtra)
    ybdd1_lo[:, k] = np.max(bounds[:, :, 0], axis=1)
    ybdd1_up[:, k] = np.min(bounds[:, :, 1], axis=1)
    # Initialize regression and xtrapolation
    reg = RandomForestRegressor(n_estimators=500)
    xtra = Xtrapolation(**parameters)
    xtra.deriv_params_ = xtra_rfnonlin.deriv_params_
    # Run on nonlinear model
    bounds = bootstrap_evaluation(X, Y2, reg, xtra)
    ybdd2_lo[:, k] = np.max(bounds[:, :, 0], axis=1)
    ybdd2_up[:, k] = np.min(bounds[:, :, 1], axis=1)
# Construct "basic/empirical" bootstrap CIs
bounds1_lo = (2*ybdd_rflinear[:, 0] -
              np.quantile(ybdd1_lo, 0.9, axis=1))
bounds1_up = (2*ybdd_rflinear[:, 1] -
              np.quantile(ybdd1_up, 0.1, axis=1))
bounds2_lo = (2*ybdd_rfnonlin[:, 0] -
              np.quantile(ybdd2_lo, 0.9, axis=1))
bounds2_up = (2*ybdd_rfnonlin[:, 1] -
              np.quantile(ybdd2_up, 0.1, axis=1))


###
# Plot results
###

width = 5.876
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(width, width*0.35)
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
plt.savefig('experiments/results/linear_vs_xtrapolation_part1.pdf',
            bbox_inches="tight")
# Plot OLS fits
ax[0].plot(xeval, reg_olslinear.predict(xeval.reshape(-1, 1)),
           color="darkgreen", linestyle="dashdot", label="OLS")
ax[1].plot(xeval, reg_olsnonlin.predict(xeval.reshape(-1, 1)),
           color="darkgreen", linestyle="dashdot")
fig.tight_layout()
plt.savefig('experiments/results/linear_vs_xtrapolation_part2.pdf',
            bbox_inches="tight")
# Plot RF fits
ax[0].plot(xeval, reg_rflinear.predict(xeval.reshape(-1, 1)), color="brown",
           label="RF")
ax[1].plot(xeval, reg_rfnonlin.predict(xeval.reshape(-1, 1)), color="brown")
fig.tight_layout()
plt.savefig('experiments/results/linear_vs_xtrapolation_part3.pdf',
            bbox_inches="tight")
# Plot Xtrapolation
ax[0].plot(xeval, linear_model(xeval.reshape(-1, 1)), color="black",
           linestyle='dashed', label='$\\Psi$')
ax[0].plot(xeval, bounds1_lo,
           color="mediumblue",
           label="extrapolation-aware CI")
ax[0].plot(xeval, bounds1_up,
           color="mediumblue")
ax[0].fill_between(xeval,
                   bounds1_lo,
                   bounds1_up,
                   color='mediumblue', alpha=0.2)
ax[1].plot(xeval, nonlinear_model(xeval.reshape(-1, 1)), color="black",
           linestyle='dashed')
ax[1].plot(xeval, bounds2_lo,
           color="mediumblue")
ax[1].plot(xeval, bounds2_up,
           color="mediumblue")
ax[1].fill_between(xeval, bounds2_lo,
                   bounds2_up,
                   color='mediumblue', alpha=0.2)
# Save figure
fig.tight_layout()
fig.legend(loc='lower center', ncols=4,
           bbox_to_anchor=(0.5, -0.1))
plt.savefig('experiments/results/linear_vs_xtrapolation.pdf',
            bbox_inches="tight")
plt.close()
