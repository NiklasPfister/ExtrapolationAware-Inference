import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import re
import experiments.helpers.examples as ex
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Set plotting parameters
width = 5.876
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
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
flierprops = dict(marker='o', markersize=1, linestyle='none')


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['medians'], color='k')


# Set output path
run_name = 'RUNNAME'
results_path = f"experiments/results/oos_prediction_{run_name}/"

allfiles = [f for f in os.listdir(results_path) if
            os.path.isfile(os.path.join(results_path, f))]

# Extract results
all_results = pd.DataFrame()
for fname in allfiles:
    if 'prediction_results' in fname:
        split_name = str.split(fname, '_')
        curr_res = pd.read_csv(os.path.join(results_path, fname))
        curr_res.drop(columns=['Unnamed: 0'], inplace=True)
        d = int(re.findall(r'\d+', split_name[2])[0])
        n = int(re.findall(r'\d+', split_name[3])[0])
        seed = int(re.findall(r'\d+', split_name[4])[0])
        curr_res['d'] = d
        curr_res['n'] = n
        curr_res['seed'] = seed
        # Check whether extrapolation is identifiable or not
        oracle_up = curr_res['XtraUp-oracle_1']
        oracle_lo = curr_res['XtraLo-oracle_1']
        curr_res['extrapolation'] = (np.mean(
            np.abs(oracle_up - oracle_lo)) > 0.1)
        # Load regression MSPEs
        cv_mspe = np.load(results_path + f'mspe_dim{d}_size{n}_seed{seed}.npy',
                          allow_pickle=True).item()
        curr_res['RF_mspe'] = cv_mspe['rf_mspe']
        curr_res['SVR_mspe'] = cv_mspe['svr_mspe']
        curr_res['MLP_mspe'] = cv_mspe['mlp_mspe']
        curr_res['OLS_mspe'] = cv_mspe['ols_mspe']
        all_results = pd.concat([all_results, curr_res])
methods = ['RF', 'SVR', 'MLP', 'OLS']
regions = [0, 1, 2]
all_results.columns


# Create consistency dataframe
consistency_results = pd.DataFrame()
for reg in regions:
    for mm in methods:
        tmp_results = pd.DataFrame()
        clo = 'XtraLo-' + mm + '_' + str(reg)
        cup = 'XtraUp-' + mm + '_' + str(reg)
        olo = 'XtraLo-oracle' + '_' + str(reg)
        oup = 'XtraUp-oracle' + '_' + str(reg)
        se_lo = (all_results[clo] - all_results[olo])**2
        se_up = (all_results[cup] - all_results[oup])**2
        tmp_results['MSE'] = np.concatenate((se_up, se_lo))
        tmp_results['bound'] = np.repeat(["upper", "lower"], len(se_lo))
        tmp_results['region'] = reg
        tmp_results['method'] = mm
        tmp_results['n'] = np.tile(all_results['n'], 2)
        tmp_results['d'] = np.tile(all_results['d'], 2)
        tmp_results['seed'] = np.tile(all_results['seed'], 2)
        tmp_results['extrapolation'] = np.tile(all_results['extrapolation'], 2)
        consistency_results = pd.concat([consistency_results, tmp_results])
# Average over points in region
consistency_results = consistency_results.groupby(
    ['bound', 'region', 'method', 'n', 'd', 'seed']).mean().reset_index()


# Create MSE dataframe
mse_results = pd.DataFrame()
for reg in regions:
    ytrue = all_results['truth_' + str(reg)]
    # Add oracle
    tmp_results = pd.DataFrame()
    oracle_lo = all_results['XtraLo-oracle' + '_' + str(reg)]
    oracle_up = all_results['XtraUp-oracle' + '_' + str(reg)]
    yhat_mm_xtra = (oracle_up + oracle_lo)/2
    tmp_results['MSE'] = (yhat_mm_xtra-ytrue)**2
    tmp_results['xtra-score'] = (np.abs(oracle_up - oracle_lo) /
                                 np.sqrt(all_results[mm + '_mspe']))
    tmp_results['worstMSE'] = np.max(np.c_[(yhat_mm_xtra-oracle_lo)**2,
                                           (yhat_mm_xtra-oracle_up)**2],
                                     axis=1)
    tmp_results['region'] = reg
    tmp_results['method'] = 'Xtra-oracle'
    tmp_results['n'] = all_results['n']
    tmp_results['d'] = all_results['d']
    tmp_results['seed'] = all_results['seed']
    tmp_results['extrapolation'] = all_results['extrapolation']
    mse_results = pd.concat([mse_results, tmp_results])
    # Add regression methods
    for mm in methods:
        tmp_results = pd.DataFrame()
        yhat_mm = all_results[mm + '_' + str(reg)]
        tmp1 = all_results['XtraLo-' + mm + '_' + str(reg)]
        tmp2 = all_results['XtraUp-' + mm + '_' + str(reg)]
        yhat_mm_xtra = (tmp1 + tmp2)/2
        tmp_results['MSE'] = np.concatenate(
            ((yhat_mm-ytrue)**2, (yhat_mm_xtra-ytrue)**2))
        tmp_results['worstMSE'] = np.r_[
            np.max(np.c_[(yhat_mm-oracle_lo)**2,
                         (yhat_mm-oracle_up)**2],
                   axis=1),
            np.max(np.c_[(yhat_mm_xtra-oracle_lo)**2,
                         (yhat_mm_xtra-oracle_up)**2],
                   axis=1)]
        tmp_results['xtra-score'] = np.concatenate(
            (all_results['euclidean_nn_' + str(reg)],
             np.abs(tmp2 - tmp1)/np.sqrt(all_results[mm + '_mspe'])))
        tmp_results['region'] = reg
        tmp_results['method'] = np.repeat([mm, 'Xtra-' + mm], len(ytrue))
        tmp_results['n'] = np.tile(all_results['n'], 2)
        tmp_results['d'] = np.tile(all_results['d'], 2)
        tmp_results['seed'] = np.tile(all_results['seed'], 2)
        tmp_results['extrapolation'] = np.tile(all_results['extrapolation'], 2)
        mse_results = pd.concat([mse_results, tmp_results])
# Average over points in region
mse_full = mse_results
mse_results = mse_results.groupby(
    ['region', 'method', 'n', 'd', 'seed', 'extrapolation']
).mean().reset_index()


###
# Create plot to visualize possible data generating functions
##

n = 200
d = 2
seeds = [21, 45, 39]

fig, ax = plt.subplots(1, 3, constrained_layout=True)
ax = ax.flatten()
fig.set_size_inches(width, width*0.35)
for kk, seed in enumerate(seeds):
    # Generate data (as in simulation)
    np.random.seed(seed)
    bds = np.random.choice([0, 1, 2, 3], d, replace=True)
    grid = np.linspace(-5, 5, 11)
    slopes = np.random.uniform(-10, 10, 3)
    slope_vec = np.random.choice(slopes, 10, replace=True)
    ind = np.delete(list(range(3, 7)), bds[0])
    slope_vec[ind] = slopes
    Xtmp = np.random.uniform(-2, 2, 1000*d).reshape(-1, d)
    slope_vec = slope_vec/np.sqrt(
        np.var(ex.piecewise_linear(Xtmp, slope_vec, grid)))
    X, _ = ex.sample_X(n, bds)
    Y = np.array(ex.piecewise_linear(
        X, slope_vec, grid)).flatten() + np.random.normal(0, 1/10, n)
    # Load results
    res_examples = all_results.query(
        f'n == {n} and d == {d} and seed == {seed}')
    # Create plot
    ax[kk].set_xlabel('$X^1$')
    ax[kk].set_ylabel('$Y$')
    ax[kk].scatter(X[:, 0], Y, color=default_colors[0], alpha=0.3, s=4)
    sort_ind = np.argsort(res_examples['X1_region_2'])
    xgrid = res_examples['X1_region_2'][sort_ind]
    ax[kk].plot(xgrid, res_examples['truth_2'][sort_ind], color='k')
    ax[kk].plot(xgrid, res_examples['XtraLo-SVR_2'][sort_ind],
                color='green', linestyle='dashed')
    ax[kk].plot(xgrid, res_examples['XtraUp-SVR_2'][sort_ind],
                color='green', linestyle='dashed')
    ax[kk].plot(xgrid, res_examples['XtraLo-oracle_2'][sort_ind],
                color='red')
    ax[kk].plot(xgrid, res_examples['XtraUp-oracle_2'][sort_ind],
                color='red')
line0 = Line2D([0], [0],
               label='$\\Psi_0$',
               color='k')
line1 = Line2D([0], [0],
               label='based on \\texttt{rf}',
               color='green', linestyle='dashed')
line2 = Line2D([0], [0],
               label='oracle bounds',
               color='red')
fig.legend(handles=[line0, line1, line2], ncols=3,
           bbox_to_anchor=(0.8, 0.05))
fig.set_tight_layout(True)
plt.savefig(f'experiments/results/example_simulations_{run_name}.pdf',
            bbox_inches="tight")
plt.close()


###
# Consistency (xtrapolation bounds on full data support)
###

method_names = ['based on \\texttt{rf}', 'based on \\texttt{svr}',
                'based on \\texttt{mlp}', 'based on \\texttt{ols}']
ds = np.unique(consistency_results['d'])
ns = np.unique(consistency_results['n'])
fig, ax = plt.subplots(1, len(ds)*2, sharey=True)
fig.set_size_inches(width, width*0.35)
grid = [(0, ds[0]), (0, ds[1]), (1, ds[0]), (1, ds[1])]
for kk, (reg, dd) in enumerate(grid):
    ax[kk].set_xlabel("$n$")
    ax[kk].set_yscale('log')
    ax[kk].set_xscale('log')
    if kk == 0:
        ax[kk].set_ylabel("RMSE")
    if reg == 0:
        title = (f'$d = {dd}$' + ' and ' +
                 '$\\widehat{\\mathcal{D}}_{\\mathrm{in}}$')
    else:
        title = (f'$d = {dd}$' + ' and ' +
                 '$\\widehat{\\mathcal{D}}_{\\mathrm{out}}$')
    ax[kk].set_title(title)
    for ii, mm in enumerate(methods):
        cond1 = f'method=="{mm}" and d=={dd} and region=={reg}'
        cond2 = ' and bound=="lower"'
        cond3 = ' and bound=="upper"'
        cond_lower = cond1 + cond2
        cond_upper = cond1 + cond3
        tmp_lower = consistency_results.query(cond_lower)
        tmp_upper = consistency_results.query(cond_upper)
        df_tmp = pd.DataFrame()
        df_tmp['n'] = tmp_lower['n']
        df_tmp['RMSE'] = np.sqrt(tmp_lower['MSE']) + np.sqrt(tmp_lower['MSE'])
        dfRMSE = df_tmp.groupby('n')[['n', 'RMSE']].mean()
        dfSD = df_tmp.groupby('n')[['RMSE']].std()
        dfcount = df_tmp.groupby('n')[['RMSE']].count()
        ax[kk].plot(dfRMSE['n'], dfRMSE['RMSE'], 'o-',
                    label=method_names[ii], markersize=1)
        ax[kk].fill_between(
            dfRMSE['n'],
            dfRMSE['RMSE'] - dfSD['RMSE']/np.sqrt(dfcount['RMSE']),
            dfRMSE['RMSE'] + dfSD['RMSE']/np.sqrt(dfcount['RMSE']),
            alpha=0.2)
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncols=4,
           bbox_to_anchor=(0.5, -0.1))
fig.tight_layout()
plt.savefig(f'experiments/results/consistency_{run_name}.pdf',
            bbox_inches="tight")
plt.close()


###
# Out-of-support prediction performance - Plain regression vs xtrapolation
###

method_names = ['\\texttt{rf}', '\\texttt{svr}', '\\texttt{mlp}']
methods = ['RF', 'SVR', 'MLP']
titels = ['$\\widehat{\\mathcal{D}}_{\\mathrm{in}}$',
          '$\\widehat{\\mathcal{D}}_{\\mathrm{out}}$ - identified',
          '$\\widehat{\\mathcal{D}}_{\\mathrm{out}}$ - unidentified']
fig, ax = plt.subplots(1, 3, constrained_layout=True)
ax = ax.flatten()
fig.set_size_inches(1.2*width, width*0.35)
# Set axis
ax[0].set_ylabel("RMSE")
for kk in [0, 1, 2]:
    ax[kk].set_title(titels[kk])
    if kk == 0:
        cond = 'n==1600 and d==2 and region==0'
    elif kk == 1:
        cond = 'n==1600 and d==2 and region==1 and extrapolation==False'
    elif kk == 2:
        cond = 'n==1600 and d==2 and region==1 and extrapolation==True'
    data_short = mse_results.query(cond)
    data_oracle = data_short.query('method=="Xtra-oracle"')
    MSE_oracle = data_oracle['MSE'].median()
    MSE_reg = np.empty((len(data_oracle), len(methods)))
    MSE_xtra = np.empty((len(data_oracle), len(methods)))
    for ll, mm in enumerate(methods):
        cond1 = f'method=="{mm}"'
        cond2 = f'method=="Xtra-{mm}"'
        MSE_reg[:, ll] = data_short.query(cond1)['MSE']
        MSE_xtra[:, ll] = data_short.query(cond2)['MSE']
    ax[kk].axhline(y=np.sqrt(MSE_oracle), color='k', linestyle='dashed')
    bpl = ax[kk].boxplot(np.sqrt(MSE_reg),
                         positions=np.array(range(MSE_reg.shape[1]))*2.0-0.4,
                         widths=0.6, patch_artist=True, flierprops=flierprops)
    bpr = ax[kk].boxplot(np.sqrt(MSE_xtra),
                         positions=np.array(range(MSE_xtra.shape[1]))*2.0+0.4,
                         widths=0.6, patch_artist=True, flierprops=flierprops)
    set_box_color(bpl, default_colors[0])
    set_box_color(bpr, default_colors[1])
    ax[kk].set_xticks(range(0, len(methods) * 2, 2), method_names)
patch1 = mpatches.Patch(color=default_colors[0],
                        label='$\\widehat{f}_{\\mathrm{reg}}$')
patch2 = mpatches.Patch(
    color=default_colors[1],
    label='$\\widehat{f}_{\\mathrm{xtra}}$')
line = Line2D([0], [0],
              label='median of oracle based $\\widehat{f}_{\\mathrm{xtra}}$',
              color='k', linestyle='dashed')
fig.legend(handles=[patch1, patch2, line], ncols=3,
           bbox_to_anchor=(0.8, 0.05))
fig.set_tight_layout(True)
plt.savefig(
    f'experiments/results/out_of_support_prediction_{run_name}.pdf',
    bbox_inches="tight")
plt.close()


###
# Worst-case loss - Plain regression vs xtrapolation
###


method_names = ['\\texttt{rf}', '\\texttt{svr}', '\\texttt{mlp}']
methods = ['RF', 'SVR', 'MLP']
titels = ['$\\widehat{\\mathcal{D}}_{\\mathrm{in}}$',
          '$\\widehat{\\mathcal{D}}_{\\mathrm{out}}$ - identified',
          '$\\widehat{\\mathcal{D}}_{\\mathrm{out}}$ - unidentified']
fig, ax = plt.subplots(1, 3, constrained_layout=True)
ax = ax.flatten()
fig.set_size_inches(1.2*width, width*0.35)
# Set axis
ax[0].set_ylabel("worst-case RMSE")
for kk in [0, 1, 2]:
    ax[kk].set_title(titels[kk])
    if kk == 0:
        cond = 'n==1600 and d==2 and region==0'
    elif kk == 1:
        cond = 'n==1600 and d==2 and region==1 and extrapolation==False'
    elif kk == 2:
        cond = 'n==1600 and d==2 and region==1 and extrapolation==True'
    data_short = mse_results.query(cond)
    data_oracle = data_short.query('method=="Xtra-oracle"')
    MSE_oracle = data_oracle['MSE'].median()
    MSE_reg = np.empty((len(data_oracle), len(methods)))
    MSE_xtra = np.empty((len(data_oracle), len(methods)))
    for ll, mm in enumerate(methods):
        cond1 = f'method=="{mm}"'
        cond2 = f'method=="Xtra-{mm}"'
        MSE_reg[:, ll] = data_short.query(cond1)['worstMSE']
        MSE_xtra[:, ll] = data_short.query(cond2)['worstMSE']
    ax[kk].axhline(y=np.sqrt(MSE_oracle), color='k', linestyle='dashed')
    bpl = ax[kk].boxplot(np.sqrt(MSE_reg),
                         positions=np.array(range(MSE_reg.shape[1]))*2.0-0.4,
                         widths=0.6, patch_artist=True, flierprops=flierprops)
    bpr = ax[kk].boxplot(np.sqrt(MSE_xtra),
                         positions=np.array(range(MSE_xtra.shape[1]))*2.0+0.4,
                         widths=0.6, patch_artist=True, flierprops=flierprops)
    set_box_color(bpl, default_colors[0])
    set_box_color(bpr, default_colors[1])
    ax[kk].set_xticks(range(0, len(methods) * 2, 2), method_names)
patch1 = mpatches.Patch(color=default_colors[0],
                        label='$\\widehat{f}_{\\mathrm{reg}}$')
patch2 = mpatches.Patch(color=default_colors[1],
                        label='$\\widehat{f}_{\\mathrm{xtra}}$')
line = Line2D([0], [0],
              label='median of oracle based $\\widehat{f}_{\\mathrm{xtra}}$',
              color='k', linestyle='dashed')
fig.legend(handles=[patch1, patch2, line], ncols=3,
           bbox_to_anchor=(0.8, 0.05))
fig.set_tight_layout(True)
plt.savefig(
    f'experiments/results/out_of_support_prediction_worstcase_{run_name}.pdf',
    bbox_inches="tight")
plt.close()


###
# Analyze xtrapolation score vs Euclidean distance
###


# Create plot
B = 1000
dds = [2, 8]
methods = ["RF", "SVR", "MLP"]
method_names = ['\\texttt{rf}', '\\texttt{svr}', '\\texttt{mlp}']
line_types = ["-", "dotted", "dashed"]
fig = plt.figure()
ax = [None] * 3
ax[0] = fig.add_subplot(1, 3, 1)
ax[1] = fig.add_subplot(1, 3, 2, sharey=ax[0])
ax[2] = fig.add_subplot(1, 3, 3)
fig.set_size_inches(width, width*0.35)
ax[0].set_ylabel("cumulative RMSE")
ax[0].set_xlabel("fraction of observations")
ax[1].set_xlabel("fraction of observations")
ax[2].set_ylabel("RMSE")
ax[2].set_title('$d=8$')
for ii, mm in enumerate(methods):
    for k, dd in enumerate(dds):
        ax[k].set_title(f'$d={dd}$')
        xtra_score_df = mse_full.query(
            f'method=="Xtra-{mm}" and n==1600 and d=={dd} and region!=2')
        mm_score_df = mse_full.query(
            f'method=="{mm}" and n==1600 and d=={dd} and region!=2')
        df_res = pd.DataFrame(
            {'s-score': xtra_score_df['xtra-score'].to_numpy(),
             'e-score': mm_score_df['xtra-score'].to_numpy(),
             'xtra-mse': xtra_score_df['MSE'].to_numpy(),
             'mse': mm_score_df['MSE'].to_numpy(),
             'seed': mm_score_df['seed'].to_numpy()})
        # Computations for xtra
        score_grid1 = np.logspace(
            -10,
            np.log10(np.max(df_res['s-score'])), B)
        n_grid1 = np.zeros(B)
        mse_grid1 = np.zeros(B)
        for i, score in enumerate(score_grid1):
            ind = df_res['s-score'] <= score
            n_grid1[i] = np.mean(ind)
            mse_grid1[i] = np.sqrt(np.mean(df_res['xtra-mse'][ind]))
        mse_grid1[np.isnan(mse_grid1)] = mse_grid1[
            np.argmin(np.isnan(mse_grid1))]
        mse_grid1 = np.concatenate(([mse_grid1[0]], mse_grid1))
        n_grid1 = np.concatenate(([0], n_grid1))
        # Computations for regression
        score_grid2 = np.logspace(
            np.log10(np.quantile(df_res['e-score'], 0.05)),
            np.log10(np.max(df_res['e-score'])), B)
        n_grid2 = np.zeros(B)
        mse_grid2 = np.zeros(B)
        for i, score in enumerate(score_grid2):
            ind = df_res['e-score'] <= score
            n_grid2[i] = np.mean(ind)
            mse_grid2[i] = np.sqrt(np.mean(df_res['xtra-mse'][ind]))
        mse_grid2[np.isnan(mse_grid2)] = mse_grid2[
            np.argmin(np.isnan(mse_grid2))]
        mse_grid2 = np.concatenate(([mse_grid2[0]], mse_grid2))
        n_grid2 = np.concatenate(([0], n_grid2))
        # Add to plot
        ax[k].plot(n_grid1, mse_grid1,
                   c=default_colors[0], linestyle=line_types[ii])
        ax[k].plot(n_grid2, mse_grid2,
                   c=default_colors[1], linestyle=line_types[ii])
        # Boxplot
        if dd == 8:
            df_res = df_res.groupby('seed').mean()
            bpl = ax[2].boxplot(
                np.sqrt(df_res['xtra-mse'][df_res['s-score'] <= 1]),
                positions=[ii-0.15],
                widths=0.2, patch_artist=True, flierprops=flierprops)
            bpr = ax[2].boxplot(
                np.sqrt(df_res['xtra-mse'][df_res['s-score'] > 1]),
                positions=[ii+0.15],
                widths=0.2, patch_artist=True, flierprops=flierprops)
            set_box_color(bpl, default_colors[2])
            set_box_color(bpr, default_colors[3])
            ax[2].set_xticks(range(0, len(methods), 1), method_names)

# Construct legend
patch1 = mpatches.Patch(color=default_colors[0],
                        label='with $\\widehat{S}$')
patch2 = mpatches.Patch(color=default_colors[1],
                        label='with $\\widehat{E}$')
line1 = Line2D([0], [0], label='\\texttt{rf}',
               color='k', linestyle='-')
line2 = Line2D([0], [0], label='\\texttt{mlp}',
               color='k', linestyle='dotted')
line3 = Line2D([0], [0], label='\\texttt{svr}',
               color='k', linestyle='dashed')
patch3 = mpatches.Patch(color=default_colors[2],
                        label='$\\widehat{S}(x)\\leq 1$')
patch4 = mpatches.Patch(color=default_colors[3],
                        label='$\\widehat{S}(x)> 1$')
fig.legend(handles=[line1, line2, line3, patch1, patch2, patch3, patch4],
           ncols=7, bbox_to_anchor=(1.0, 0.03))
fig.set_tight_layout(True)
plt.savefig(
    f'experiments/results/xtrapolation_score_{run_name}.pdf',
    bbox_inches="tight")
plt.close()
