import numpy as np
from xtrapolation.xtrapolation import Xtrapolation
import experiments.helpers.examples as ex
import matplotlib.pyplot as plt
import matplotlib


# Set plotting parameters
params = {'backend': 'pdf',
          'axes.labelsize': 8,
          'font.size': 8,
          'legend.fontsize': 8,
          'xtick.labelsize': 6,
          'ytick.labelsize': 6,
          'lines.linewidth': 1,
          'text.usetex': True,
          'axes.unicode_minus': True}
matplotlib.rcParams.update(params)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

##
# Select experiment setting and generate data
##

n = 200
num_settings = 3
names = [None] * num_settings
ff_list = [None] * num_settings
X_list = [None] * num_settings
xeval_list = [None] * num_settings
extrapolation_regions = [None] * num_settings
support_regions = [None] * num_settings
ylim_list = [None] * num_settings

# Specify settings

# quadratic
names[0] = "$f(x)=x^2$"
ff_list[0] = ex.quadratic
X_list[0] = np.linspace(-1.5, 1.5, n).reshape(n, 1)
xeval_list[0] = np.r_[np.linspace(-3, -1.5, 25),
                      np.linspace(1.5, 3, 25)]
extrapolation_regions[0] = [np.arange(25),
                            np.arange(25, 50)]
support_regions[0] = [[-1.5, 1.5]]
ylim_list[0] = [-0.25, 9.25]

# expit
names[1] = "$f(x)=2$expit$(2(x-0.5))$"
ff_list[1] = ex.expit
X_list[1] = np.linspace(-2.5, 2.5, n).reshape(n, 1)
xeval_list[1] = np.r_[np.linspace(-5, -2.5, 25),
                      np.linspace(2.5, 5, 25)]
extrapolation_regions[1] = [np.arange(25),
                            np.arange(25, 50)]
support_regions[1] = [[-2.5, 2.5]]
ylim_list[1] = [-2.5, 4.25]

# sin
names[2] = "$f(x)=\\sin(x\\pi)$"
ff_list[2] = ex.sin
X_list[2] = (np.r_[np.linspace(-2, -0.6, 25),
                   np.linspace(0.3, 2, 25)]).reshape(50, 1)
xeval_list[2] = np.r_[np.linspace(-0.6, 0.3, 25),
                      np.linspace(2, 3, 25)]
extrapolation_regions[2] = [np.arange(25),
                            np.arange(25, 50)]
support_regions[2] = [[-2, -0.6], [0.3, 2]]
ylim_list[2] = [-2, 2]


##
# Apply Xtraploation (using oracle)
##

orders = [1, 2]
bounds = [None] * num_settings
parameters = {'orders': orders,
              'extra_params': {
                  'nn': n,
                  'dist': 'euclidean',
                  'alpha': 0,
                  'aggregation': 'optimal-average',
              },
              'verbose': 2}

xtra = Xtrapolation(**parameters)

# Iterate over each setting
for kk in range(num_settings):
    print(kk)
    xtra.fit_derivatives_oracle(X_list[kk], ff_list[kk])
    bounds[kk] = xtra.prediction_bounds_oracle(
        X_list[kk], ff_list[kk], xeval_list[kk])


##
# Create plots
##

width = 5.876

fig, ax = plt.subplots(1, num_settings, constrained_layout=True)
fig.set_size_inches(width, width/3)
# fig.tight_layout()
colors = ['blue', 'green', 'red']

# Xtrapolation bounds
for kk in range(num_settings):
    # Plot true function
    for ll, supp in enumerate(support_regions[kk]):
        ax[kk].plot(np.arange(supp[0], supp[1], 0.01),
                    ff_list[kk](np.arange(supp[0], supp[1],
                                          0.01).reshape(-1, 1)),
                    color="black",
                    label="$f$")
    # Plot bounds
    for oo, order in enumerate(orders):
        for ll, ind in enumerate(extrapolation_regions[kk]):
            if ll == 0:
                ax[kk].fill_between(xeval_list[kk][ind],
                                    bounds[kk][ind, oo, 0],
                                    bounds[kk][ind, oo, 1],
                                    color=colors[oo],
                                    alpha=0.2,
                                    label=(r'bounds $q=$' +
                                           f'{order}'))
            else:
                ax[kk].fill_between(xeval_list[kk][ind],
                                    bounds[kk][ind, oo, 0],
                                    bounds[kk][ind, oo, 1], color=colors[oo],
                                    alpha=0.2)
            ax[kk].plot(xeval_list[kk][ind],
                        bounds[kk][ind, oo, 0], color=colors[oo])
            ax[kk].plot(xeval_list[kk][ind],
                        bounds[kk][ind, oo, 1], color=colors[oo])
    # Plot support region
    for ll, supp in enumerate(support_regions[kk]):
        ax[kk].fill_between(supp, [-10, -10], [10, 10],
                            color="black", alpha=0.1,
                            label="$\\mathcal{D}$")
    # Customize plots
    ax[kk].set_title(f"{names[kk]}")
    ax[kk].set_ylim(ylim_list[kk])
    ax[kk].set_xlabel("$x$")
    ax[kk].set_ylabel("$f(x)$")
# Customize figure
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower left",
           bbox_to_anchor=(0.2, -0.15),
           ncol=len(orders)+2)


# Save plot
plt.savefig('experiments/results/xtrapolation_bounds_visualization.pdf',
            bbox_inches="tight")
