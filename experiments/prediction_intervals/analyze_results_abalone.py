from ucimlrepo import fetch_ucirepo
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib

# set runname
runname = 'RUNNAME'

# set method list
method_list = ['qrf', 'conf-qrf', 'qnn', 'conf-qnn']


# Set plotting parameters
width = 5.876
params = {'axes.labelsize': 8,
          'font.size': 8,
          'legend.fontsize': 8,
          'xtick.labelsize': 6,
          'ytick.labelsize': 6,
          'lines.linewidth': 0.7,
          'text.usetex': True,
          'axes.unicode_minus': True,
          'text.latex.preamble': r'\usepackage{amsfonts}'}
matplotlib.rcParams.update(params)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


##
# Load results and data
##

# set output path
output_path = "experiments/results/abalone_analysis/"

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

# Load results
files = [f for f in os.listdir(output_path)
         if os.path.isfile(os.path.join(output_path, f))
         and runname + '_' + method_list[0] + '_' in f]

res_list = {}
for mm in method_list:
    res_list[mm] = [None] * len(files)
    for i in range(len(files)):
        with open(output_path + f'abalone_{runname}_{mm}_{i}.pkl', 'rb') as f:
            res_dict = pickle.load(f)
            res_list[mm][i] = res_dict['res']


###
# Function to adjust for coverage by averaged randomized prediction intervals
###

def averaged_randomized_PI(qmat, y, train, target_coverage):
    # Compute callibartion prob on train
    alpha_withboundary = np.mean(
        (qmat[train, 0] <= y[train]) & (y[train] <= qmat[train, 1]))
    alpha_withoutboundary = np.mean(
        (qmat[train, 0] < y[train]) & (y[train] < qmat[train, 1]))
    if target_coverage <= alpha_withoutboundary:
        prob_si = 1
    elif target_coverage >= alpha_withboundary:
        prob_si = 0
    else:
        prob_si = ((target_coverage - alpha_withboundary) /
                   (alpha_withoutboundary - alpha_withboundary))

    # compute vector that weights boundary to match coverage
    inside = (qmat[:, 0] < y) & (y < qmat[:, 1])
    boundary = (qmat[:, 0] == y) | (qmat[:, 1] == y)
    adjusted_inside = inside * 1.0
    adjusted_inside[boundary] = 1-prob_si

    return adjusted_inside


i = 7
mm = "qnn"
train = res_list[mm][i]['train_ind']
qmat = res_list[mm][i]['qmat']
bounds_list = res_list[mm][i]['bounds_list']
BBlo = np.max(bounds_list[0][:, :, 0], axis=1)
BBup = np.min(bounds_list[1][:, :, 1], axis=1)
test = ~train
ind = 1
plt.scatter(X[train, ind], y[train])
plt.scatter(X[test, ind], y[test])
plt.scatter(X[:, ind], qmat[:, 0])
plt.scatter(X[:, ind], qmat[:, 1])
plt.scatter(X[:, ind], BBlo)
plt.scatter(X[:, ind], BBup)

for mm in method_list:
    print(mm)
    for i in range(16):
        print(i)
        print(res_list[mm][i]['pars'])


##
# Create Interpolation vs Extrapolation plot
##

# Process results
quantiles = [0.1, 0.9]
num_splits = len(res_list[method_list[0]])
num_intervals = 1
coverage_train = {}
coverage_test = {}
for mm in method_list:
    coverage_train[mm] = np.zeros((num_splits, 2 * num_intervals))
    coverage_test[mm] = np.zeros((num_splits, 2 * num_intervals))
    for i in range(len(res_list[mm])):
        train_ind = res_list[mm][i]['train_ind']
        test_ind = ~res_list[mm][i]['train_ind']
        ytrain = y[train_ind]
        ytest = y[test_ind]
        qmat = res_list[mm][i]['qmat']
        bounds_list = res_list[mm][i]['bounds_list']
        for k in range(num_intervals):
            k1 = 2*k
            k2 = 2*k + 1
            level = quantiles[k2] - quantiles[k1]
            BBlo = np.max(bounds_list[k1][:, :, 0], axis=1)
            BBup = np.min(bounds_list[k2][:, :, 1], axis=1)
            # regular quantile forest
            qrf = averaged_randomized_PI(qmat, y, train_ind, level)
            coverage_train[mm][i, 2*k] = np.mean(qrf[train_ind])
            coverage_test[mm][i, 2*k] = np.mean(qrf[test_ind])
            # xtrapolation quanitles
            xtra = averaged_randomized_PI(np.c_[BBlo, BBup], y,
                                          train_ind, level)
            coverage_train[mm][i, 2*k+1] = np.mean(xtra[train_ind])
            coverage_test[mm][i, 2*k+1] = np.mean(xtra[test_ind])


# Create plot
markers = ["+", "P", "x", "X"]
colortype = {'blue': [{'color': "tab:blue"},
                      {'edgecolor': "tab:blue", 'c': "None"},
                      {'color': "tab:blue"},
                      {'edgecolor': "tab:blue", 'c': "None"}],
             'green': [{'color': "tab:green"},
                       {'edgecolor': "tab:green", 'c': "None"},
                       {'color': "tab:green"},
                       {'edgecolor': "tab:green", 'c': "None"}]}
mshift = [-0.015, -0.005, 0.005, 0.015]
kk = int(num_splits/2)
fig, ax = plt.subplots(1, 2, sharey=True, constrained_layout=True)
fig.set_size_inches(0.9*width, width*0.3)
xgrid_train = np.linspace(0, 0.40, kk)
xgrid_test = np.linspace(0.6, 1, kk)
xgrid_train = np.random.uniform(-0.02, 0.02, kk)
xgrid_test = np.linspace(0.3, 0.9, kk)
for k in range(2):
    ax[k].axhline(y=level, color='r', linestyle='dashed')
for ll, mm in enumerate(method_list):
    # extrapolation splits
    # regular quantile regression
    ax[0].scatter(xgrid_train+0.125, coverage_train[mm][:kk, 0],
                  marker=markers[ll], alpha=0.5,
                  **colortype['blue'][ll])
    ax[0].scatter(xgrid_test+mshift[ll]-0.0025, coverage_test[mm][:kk, 0],
                  marker=markers[ll], **colortype['blue'][ll])
    # xtra quantile regression
    ax[0].scatter(xgrid_train+0.175, coverage_train[mm][:kk, 1],
                  marker=markers[ll], alpha=0.5,
                  **colortype['green'][ll])
    ax[0].scatter(xgrid_test+mshift[ll]+0.0025, coverage_test[mm][:kk, 1],
                  marker=markers[ll], **colortype['green'][ll])
    # random splits
    # regular quantile regression
    ax[1].scatter(xgrid_train+0.125, coverage_train[mm][kk:, 0],
                  marker=markers[ll], alpha=0.5, **colortype['blue'][ll])
    ax[1].scatter(xgrid_test+mshift[ll]-0.0025, coverage_test[mm][kk:, 0],
                  marker=markers[ll], **colortype['blue'][ll])
    # xtra quantile regression
    ax[1].scatter(xgrid_train+0.175, coverage_train[mm][kk:, 1],
                  marker=markers[ll], alpha=0.5, **colortype['green'][ll])
    ax[1].scatter(xgrid_test+mshift[ll]+0.0025, coverage_test[mm][kk:, 1],
                  marker=markers[ll], **colortype['green'][ll])

# adjust axes
xlabels = ["train"] + [k for k in range(kk)]
xticks = np.r_[0.15, xgrid_test]
for k in range(2):
    ax[k].set_xticks(xticks)
    ax[k].set_xticklabels(xlabels)
    ax[k].set_xlabel('split index')
ax[0].set_title('\\texttt{abalone}: extrapolating splits')
ax[1].set_title('\\texttt{abalone}: random splits')
ax[0].set_ylabel('coverage')
# create legend manually
p1 = matplotlib.lines.Line2D(
    [0], [0], label='qrf',
    marker=markers[0], linestyle='', color="tab:blue")
p2 = matplotlib.lines.Line2D(
    [0], [0], label='cpqrf',
    marker=markers[1], linestyle='', mfc="white", mec="tab:blue")
p3 = matplotlib.lines.Line2D(
    [0], [0], label='qnn',
    marker=markers[2], linestyle='', color="tab:blue")
p4 = matplotlib.lines.Line2D(
    [0], [0], label='cpqnn',
    marker=markers[3], linestyle='', mfc="white", mec="tab:blue")
p11 = matplotlib.lines.Line2D(
    [0], [0], label='xtra-qrf',
    marker=markers[0], linestyle='', color="tab:green")
p22 = matplotlib.lines.Line2D(
    [0], [0], label='xtra-cpqrf',
    marker=markers[1], linestyle='', mfc="white", mec="tab:green")
p33 = matplotlib.lines.Line2D(
    [0], [0], label='xtra-qnn',
    marker=markers[2], linestyle='', color="tab:green")
p44 = matplotlib.lines.Line2D(
    [0], [0], label='xtra-cpqnn',
    marker=markers[3], linestyle='', mfc="white", mec="tab:green")

fig.legend([p1, p2, p3, p4, p11, p22, p33, p44],
           ['\\texttt{qrf}', '\\texttt{cpqrf}', '\\texttt{qnn}',
            '\\texttt{cpqnn}', '\\texttt{xtra-qrf}', '\\texttt{xtra-cpqrf}',
            '\\texttt{xtra-qnn}', '\\texttt{xtra-cpqnn}'],
           ncols=1, bbox_to_anchor=(1.19, 0.87))
fig.set_tight_layout(True)
plt.savefig(
    'experiments/results/abalone_inter_vs_extra.pdf',
    bbox_inches="tight")
plt.close()


##
# Create coverage vs extrapolation score plot
##


n = X.shape[0]
num_splits = len(res_list[method_list[0]])
qmat_xtra = {}
qmat_regr = {}
qmat_xtra_train = {}
qmat_regr_train = {}
score_xtra = {}
for mm in method_list:
    qmat_xtra[mm] = np.zeros((n, 2))
    qmat_regr[mm] = np.zeros((n, 2))
    qmat_xtra_train[mm] = np.zeros((n, 2))
    qmat_regr_train[mm] = np.zeros((n, 2))
    score_xtra[mm] = np.zeros(n)
    c_vec = np.zeros(n)
    for split in range(int(num_splits/2)):
        train_ind = res_list[mm][split]['train_ind']
        c_vec[~train_ind] = split
        qmat = res_list[mm][split]['qmat']
        bounds_list = res_list[mm][split]['bounds_list']
        # quantiles
        b_up1 = np.min(bounds_list[0][:, :, 1], axis=1)
        b_up2 = np.min(bounds_list[1][:, :, 1], axis=1)
        b_lo1 = np.max(bounds_list[0][:, :, 0], axis=1)
        b_lo2 = np.max(bounds_list[1][:, :, 0], axis=1)
        # test
        qmat_xtra[mm][~train_ind, 0] = b_lo1[~train_ind]
        qmat_xtra[mm][~train_ind, 1] = b_up2[~train_ind]
        qmat_regr[mm][~train_ind, 0] = qmat[~train_ind, 0]
        qmat_regr[mm][~train_ind, 1] = qmat[~train_ind, 1]
        # train
        qmat_xtra_train[mm][train_ind, 0] = b_lo1[train_ind]
        qmat_xtra_train[mm][train_ind, 1] = b_up2[train_ind]
        qmat_regr_train[mm][train_ind, 0] = qmat[train_ind, 0]
        qmat_regr_train[mm][train_ind, 1] = qmat[train_ind, 1]
        score_xtra[mm][~train_ind] = ((b_up1 - b_lo1) +
                                      (b_up2 - b_lo2))[~train_ind]
        print(np.median(score_xtra[mm][~train_ind]))


# Rolling window coverage
window_len = 100
coverage_xtra = {}
coverage_regr = {}
for mm in method_list:
    coverage_xtra[mm] = np.ones(n)
    coverage_regr[mm] = np.ones(n)
    perm = np.arange(n)
    np.random.shuffle(perm)
    score_sort = np.argsort(score_xtra[mm][perm])
    score_sort = np.argsort(score_xtra[mm][perm])
    for k in range(n):
        lo = np.max([int(k-window_len/2), 0])
        up = np.min([int(k+window_len/2), n])
        # test
        xtra_test = qmat_xtra[mm][perm, :][score_sort[lo:up], :]
        regr_test = qmat_regr[mm][perm, :][score_sort[lo:up], :]
        # train
        xtra_train = qmat_xtra_train[mm][perm, :][score_sort[lo:up], :]
        regr_train = qmat_regr_train[mm][perm, :][score_sort[lo:up], :]
        true_y = y[perm][score_sort[lo:up]]
        # coverage
        win_len = regr_train.shape[0]
        train_ind = np.concatenate(
            [np.repeat(np.array([True]), win_len),
             np.repeat(np.array([False]), win_len)])
        tmp = averaged_randomized_PI(np.r_[regr_train, regr_test],
                                     np.r_[true_y, true_y],
                                     train_ind, level)
        coverage_regr[mm][k] = np.mean(tmp[~train_ind])
        tmp = averaged_randomized_PI(np.r_[xtra_train, xtra_test],
                                     np.r_[true_y, true_y],
                                     train_ind, level)
        coverage_xtra[mm][k] = np.mean(tmp[~train_ind])

method_names = ['qrf', 'cpqrf', 'qnn', 'cpqnn']
for ll, mm in enumerate(method_list):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(0.5*width, width*0.35)
    ax.plot(coverage_xtra[mm],
            color="tab:green",
            label=f'\\texttt{{xtra-{method_names[ll]}}}')
    ax.plot(coverage_regr[mm],
            color="tab:blue",
            label=f'\\texttt{{{method_names[ll]}}}')
    ax.axhline(0.8, linestyle='dashed', c='red')
    ax.set_title(
        f'\\texttt{{{method_names[ll]}}} on \\texttt{{abalone}}')
    ax.set_xlabel('extrapolation score')
    ax.set_ylabel('smoothed coverage')
    plt.legend(ncol=1)
    fig.set_tight_layout(True)
    plt.savefig(
        f'experiments/results/abalone_extrapolation_score_{mm}.pdf',
        bbox_inches="tight")
    plt.close()


fig, ax = plt.subplots(4, 1, sharex=True)
fig.set_size_inches(0.5*width, width)
method_names = ['qrf', 'cpqrf', 'qnn', 'cpqnn']
for ll, mm in enumerate(method_list):
    ax[ll].plot(coverage_xtra[mm],
                color="tab:green",
                label=f'\\texttt{{xtra-{method_names[ll]}}}')
    ax[ll].plot(coverage_regr[mm],
                color="tab:blue",
                label=f'\\texttt{{{method_names[ll]}}}')
    ax[ll].axhline(0.8, linestyle='dashed', c='red')
    if ll == 0:
        ax[ll].set_title('\\texttt{abalone}')
    if ll == 3:
        ax[ll].set_xlabel('extrapolation score')
    # ax[ll].set_ylabel('smoothed coverage')
    ax[ll].legend(ncol=1)
fig.set_tight_layout(True)
plt.savefig(
    'experiments/results/abalone_extrapolation_score.pdf',
    bbox_inches="tight")
plt.close()
