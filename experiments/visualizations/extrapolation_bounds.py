import numpy as np
from xtrapolation.xtrapolation import Xtrapolation
from xtrapolation.helpers import DerivativeConfig, ExtrapolationConfig
import experiments.helpers.examples as ex
import matplotlib.pyplot as plt
import matplotlib


# Set plotting parameters
params = {
    "backend": "pdf",
    "axes.labelsize": 8,
    "font.size": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "lines.linewidth": 1,
    "text.usetex": True,
    "axes.unicode_minus": True,
}
matplotlib.rcParams.update(params)
plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})

##
# Select experiment setting and generate data
##

n = 200
num_settings = 3

# quadratic
names0 = "$f(x)=x^2$"
ff0 = ex.quadratic
X0 = np.linspace(-1.5, 1.5, n).reshape(n, 1)
xeval0 = np.r_[np.linspace(-3, -1.5, 25), np.linspace(1.5, 3, 25)].reshape(-1, 1)
extrapolation_regions0 = [np.arange(25), np.arange(25, 50)]
support_region0 = [[-1.5, 1.5]]
ylim0 = [-0.25, 9.25]

# expit
names1 = "$f(x)=2$expit$(2(x-0.5))$"
ff1 = ex.expit
X1 = np.linspace(-2.5, 2.5, n).reshape(n, 1)
xeval1 = np.r_[np.linspace(-5, -2.5, 25), np.linspace(2.5, 5, 25)].reshape(-1, 1)
extrapolation_regions1 = [np.arange(25), np.arange(25, 50)]
support_region1 = [[-2.5, 2.5]]
ylim1 = [-2.5, 4.25]

# sin
names2 = "$f(x)=\\sin(x\\pi)$"
ff2 = ex.sin
X2 = (np.r_[np.linspace(-2, -0.6, int(n/2)), np.linspace(0.3, 2, int(n/2))]).reshape(-1, 1)
xeval2 = np.r_[np.linspace(-0.6, 0.3, int(n/2)), np.linspace(2, 3, int(n/2))].reshape(-1, 1)
extrapolation_regions2 = [np.arange(int(n/2)), np.arange(int(n/2), n)]
support_region2 = [[-2, -0.6], [0.3, 2]]
ylim2 = [-2, 2]

# Collect all settings
names = [names0, names1, names2]
ff_list = [ff0, ff1, ff2]
X_list = [X0, X1, X2]
xeval_list = [xeval0, xeval1, xeval2]
extrapolation_regions = [
    extrapolation_regions0,
    extrapolation_regions1,
    extrapolation_regions2,
]
support_regions = [support_region0, support_region1, support_region2]
ylim_list = [ylim0, ylim1, ylim2]


def main():
    # Apply Xtraploation (using oracle)
    orders = [1, 2]
    deriv_config = DerivativeConfig()
    extra_config = ExtrapolationConfig(
        nn=n,
        dist="euclidean",
        alpha=0.0,
        aggregation="intersection-corrected",
    )

    # Iterate over each setting
    bounds = []
    for kk in range(num_settings):
        print(f"Setting: {kk}")
        xtra = Xtrapolation(
            orders=orders,
            deriv_config=deriv_config,
            extra_config=extra_config,
            verbose=2,
        )
        bounds.append(
            xtra.prediction_bounds_oracle(
                X_list[kk], ff_list[kk], xeval_list[kk], refit=True
            )
        )


    ##
    # Create plots
    ##

    width = 5.876

    fig, ax = plt.subplots(1, num_settings, constrained_layout=True)
    fig.set_size_inches(width, width / 3)
    # fig.tight_layout()
    colors = ["blue", "green", "red"]

    # Xtrapolation bounds
    for kk in range(num_settings):
        # Plot true function
        for ll, supp in enumerate(support_regions[kk]):
            ax[kk].plot(
                np.arange(supp[0], supp[1], 0.01),
                ff_list[kk](np.arange(supp[0], supp[1], 0.01).reshape(-1, 1)),
                color="black",
                label="$f$",
            )
        # Plot bounds
        for oo, order in enumerate(orders):
            for ll, ind in enumerate(extrapolation_regions[kk]):
                if ll == 0:
                    ax[kk].fill_between(
                        xeval_list[kk][ind, 0],
                        bounds[kk][ind, oo, 0],
                        bounds[kk][ind, oo, 1],
                        color=colors[oo],
                        alpha=0.2,
                        label=(r"bounds $q=$" + f"{order}"),
                    )
                else:
                    ax[kk].fill_between(
                        xeval_list[kk][ind, 0],
                        bounds[kk][ind, oo, 0],
                        bounds[kk][ind, oo, 1],
                        color=colors[oo],
                        alpha=0.2,
                    )
                ax[kk].plot(
                    xeval_list[kk][ind, 0], bounds[kk][ind, oo, 0], color=colors[oo]
                )
                ax[kk].plot(
                    xeval_list[kk][ind, 0], bounds[kk][ind, oo, 1], color=colors[oo]
                )
        # Plot support region
        for ll, supp in enumerate(support_regions[kk]):
            ax[kk].fill_between(
                supp,
                [-10, -10],
                [10, 10],
                color="black",
                alpha=0.1,
                label="$\\mathcal{D}$",
            )
        # Customize plots
        ax[kk].set_title(f"{names[kk]}")
        ax[kk].set_ylim(ylim_list[kk])
        ax[kk].set_xlabel("$x$")
        ax[kk].set_ylabel("$f(x)$")
    # Customize figure
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.2, -0.15),
        ncol=len(orders) + 2,
    )

    # Save plot
    plt.savefig(
        "experiments/results/visualizations/xtrapolation_bounds_visualization.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
