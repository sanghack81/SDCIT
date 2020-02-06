import pickle
import time
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from scipy.stats import gamma, pearson3
from tqdm import trange

from experiments.exp_setup import *
from experiments.run_KCIPT import test_chaotic
from experiments.exp_utils import read_chaotic
from sdcit.kcipt import c_KCIPT
from sdcit.sdcit_mod import SDCIT
from sdcit.utils import p_value_of, K2D


def experiment(obj_filename):
    if not os.path.exists(obj_filename):
        trial = 0
        gamma_param = 0.0
        N = 400
        independent = 1
        initial_B = 100
        kx, ky, kz, Dz = read_chaotic(independent, gamma_param, trial, N)

        # Compare SDCIT and KCIPT100
        print('SDCIT ... ')
        sdcit_mmd, sdcit_pval, sdcit_null = SDCIT(kx, ky, kz, with_null=True, seed=trial, to_shuffle=False)
        print('KCIPT {} ... '.format(initial_B))
        _, mmds100, _, outer_null100 = c_KCIPT(kx, ky, kz, K2D(kz), initial_B, 10000, 10000, n_jobs=PARALLEL_JOBS, seed=trial)

        # Infer desired B
        desired_B = int(initial_B * (outer_null100.std() / sdcit_null.std()) ** 2)
        print('Desired B: {}'.format(desired_B))

        # Prepare outer null distribution
        print('KCIPT {} ... '.format(desired_B))
        _, mmds_B, _, outer_null_B = c_KCIPT(kx, ky, kz, K2D(kz), desired_B, 10000, 10000, n_jobs=PARALLEL_JOBS, seed=trial)

        print('TS distributions for KCIPT {} ... '.format(desired_B))
        time.sleep(1)
        distr_boot = np.zeros((1000,))
        for ii in trange(len(distr_boot)):
            _, mmds_B, _, _ = c_KCIPT(kx, ky, kz, K2D(kz), desired_B, 0, 0, n_jobs=PARALLEL_JOBS, seed=ii)
            distr_boot[ii] = mmds_B.mean()

        with open(obj_filename, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([sdcit_mmd, sdcit_null, mmds100, outer_null100, desired_B, mmds_B, outer_null_B, distr_boot], f)

        print(independent, gamma_param, N)
        outs = [test_chaotic(independent, gamma_param, tt, N, B=desired_B, n_jobs=PARALLEL_JOBS) for tt in trange(300)]
        with open(SDCIT_RESULT_DIR + '/kcipt_chaotic_{}.csv'.format(desired_B), 'a') as f:
            for out in outs:
                print(*out, sep=',', file=f, flush=True)


def main():
    assert exists(SDCIT_RESULT_DIR + '/kcipt_chaotic_5000.csv'), 'run_SDCIT first'
    assert exists(SDCIT_RESULT_DIR + '/kcipt_chaotic_20000.csv'), 'run_SDCIT first'

    from experiments.draw_figures import color_palettes, method_color_codes

    obj_filename = SDCIT_RESULT_DIR + '/right_power.pickle'
    experiment(obj_filename)

    time.sleep(3)

    with open(obj_filename, 'rb') as f:  # Python 3: open(..., 'rb')
        sdcit_mmd, sdcit_null, mmds100, outer_null100, desired_B, mmds_B, outer_null_B, distr_boot = pickle.load(f)

    print(desired_B)
    print('SKEW SDCIT NULL: {}'.format(scipy.stats.skew(sdcit_null)))
    print('SKEW KCIPT NULL: {}'.format(scipy.stats.skew(outer_null_B)))

    names_kcipt_chaotic = ['independent', 'gamma', 'trial', 'N', 'statistic', 'pvalue', 'B']
    names_sdcit_chaotic = ['independent', 'gamma', 'trial', 'N', 'statistic', 'pvalue']

    df_kcipt_desired_B = pd.read_csv(SDCIT_RESULT_DIR + '/kcipt_chaotic_{}.csv'.format(desired_B), names=names_kcipt_chaotic, )
    df_kcipt_5000 = pd.read_csv(SDCIT_RESULT_DIR + '/kcipt_chaotic_5000.csv', names=names_kcipt_chaotic, )
    df_kcipt_20000 = pd.read_csv(SDCIT_RESULT_DIR + '/kcipt_chaotic_20000.csv', names=names_kcipt_chaotic, )
    df_sdcit = pd.read_csv(SDCIT_RESULT_DIR + '/sdcit_chaotic.csv', names=names_sdcit_chaotic, )
    df_sdcit = df_sdcit[df_sdcit['N'] == 400]
    df_sdcit = df_sdcit[df_sdcit['independent'] == 1]
    df_sdcit = df_sdcit[df_sdcit['gamma'] == 0.0]
    assert len(df_sdcit) == 300
    xs_sdcit = np.linspace(1.3 * sdcit_null.min(), 1.3 * sdcit_null.max(), 1000)
    ys_sdcit_pearson3 = pearson3.pdf(xs_sdcit, *pearson3.fit(sdcit_null))

    xs_kcipt = np.linspace(1.3 * outer_null_B.min(), 1.3 * outer_null_B.max(), 1000)
    ys_kcipt_pearson3 = pearson3.pdf(xs_kcipt, *pearson3.fit(outer_null_B))

    # 20000's null is inferred from known one...
    factor_20000 = np.sqrt(20000 / desired_B)
    ys_kcipt_20000_gamma = gamma.pdf(xs_kcipt, *gamma.fit(outer_null_B / factor_20000))

    sns.set(style='white', font_scale=1.2)
    paper_rc = {'lines.linewidth': 0.8, 'lines.markersize': 2, 'patch.linewidth': 1}
    sns.set_context("paper", rc=paper_rc)
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{cmbright}')

    if True:
        fig = plt.figure(figsize=[5, 3.5])
        ##################################
        fig.add_subplot(2, 2, 1, adjustable='box')

        plt.plot(xs_sdcit, ys_sdcit_pearson3, label='SDCIT null', lw=1.5, color=color_palettes[method_color_codes['SDCIT']])
        plt.plot([sdcit_mmd, sdcit_mmd], [0, 1000], label='SDCIT TS', color=color_palettes[method_color_codes['SDCIT']])
        plt.plot(xs_kcipt, ys_kcipt_pearson3, label='KCIPT null', lw=1.5, color=color_palettes[method_color_codes['KCIPT']])
        sns.distplot(distr_boot, hist=True, kde=False, hist_kws={'histtype': 'stepfilled'}, norm_hist=True, label='KCIPT TS', color=color_palettes[method_color_codes['KCIPT']])
        plt.gca().set_xlim([-0.0003, 0.0005])
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.gca().set_ylabel('density')
        plt.setp(plt.gca(), 'yticklabels', [])
        plt.legend(loc=1)
        ##################################
        fig.add_subplot(2, 2, 2, adjustable='box')

        pvals_B = [p_value_of(t, outer_null_B) for t in distr_boot]
        pval_sdcit = p_value_of(sdcit_mmd, sdcit_null)

        sns.distplot(pvals_B, bins=20, hist=True, kde=False, hist_kws={'histtype': 'stepfilled'}, norm_hist=True, color=color_palettes[method_color_codes['KCIPT']], label='KCIPT p-values')
        plt.plot([pval_sdcit, pval_sdcit], [0, 1], label='SDCIT p-value', color=color_palettes[method_color_codes['SDCIT']])
        plt.gca().set_ylim([0, 2.2])
        plt.gcf().subplots_adjust(wspace=0.3)
        plt.legend(loc=2)
        sns.despine()

        ##################################
        fig.add_subplot(2, 2, 3, adjustable='box')
        sns.distplot(df_sdcit['statistic'], hist=True, bins=20, kde=False, color=color_palettes[method_color_codes['SDCIT']], label='SDCIT TS')
        sns.distplot(df_kcipt_desired_B['statistic'], hist=True, bins=20, kde=False, color=color_palettes[method_color_codes['KCIPT']], label='KCIPT TS')
        plt.legend()
        plt.gca().set_xlim([-0.0003, 0.0005])
        plt.gca().set_xlabel('MMD')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.gca().set_ylabel('density')
        plt.setp(plt.gca(), 'yticklabels', [])

        ##################################
        fig.add_subplot(2, 2, 4, adjustable='box')

        sns.distplot(df_sdcit['pvalue'], hist=True, bins=20, kde=False, color=color_palettes[method_color_codes['SDCIT']], norm_hist=True, label='SDCIT p-values')
        sns.distplot(df_kcipt_desired_B['pvalue'], hist=True, bins=20, kde=False, color=color_palettes[method_color_codes['KCIPT']], norm_hist=True, label='KCIPT p-values')
        plt.gca().set_xlabel('p-value')
        plt.gcf().subplots_adjust(wspace=0.3, hspace=0.3)
        plt.gca().set_ylim([0, 2.2])
        plt.legend(loc=0)
        sns.despine()
        plt.savefig(SDCIT_FIGURE_DIR + '/kcipt_{}_ps.pdf'.format(desired_B), transparent=True, bbox_inches='tight', pad_inches=0.02)
        plt.close()

    ###############################################
    ###############################################
    ###############################################
    ###############################################
    ###############################################
    ###############################################
    if True:
        sns.set(style='white', font_scale=1.2)
        paper_rc = {'lines.linewidth': 0.8, 'lines.markersize': 2, 'patch.linewidth': 1}
        sns.set_context("paper", rc=paper_rc)
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{cmbright}')
        fig = plt.figure(figsize=[5, 1.6])
        ##################################
        fig.add_subplot(1, 2, 1, adjustable='box')
        sns.distplot(df_kcipt_5000['statistic'], hist=True, bins=20, kde=False, color=color_palettes[method_color_codes['KCIPT']], label='TS')
        plt.legend()
        plt.gca().set_xlabel('MMD')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.gca().set_ylabel('density')
        plt.gca().set_xlim([-0.0002, 0.0003])
        plt.setp(plt.gca(), 'yticklabels', [])
        ##
        fig.add_subplot(1, 2, 2, adjustable='box')
        sns.distplot(df_kcipt_5000['pvalue'], hist=True, bins=20, kde=False, color=color_palettes[method_color_codes['KCIPT']], norm_hist=True, label='p-value')
        plt.gca().set_xlabel('p-value')
        plt.gcf().subplots_adjust(wspace=0.3, hspace=0.3)
        plt.legend(loc=0)
        sns.despine()
        plt.savefig(SDCIT_FIGURE_DIR + '/kcipt_5000_ps.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
        plt.close()

    if True:
        sns.set(style='white', font_scale=1.2)
        paper_rc = {'lines.linewidth': 0.8, 'lines.markersize': 2, 'patch.linewidth': 1}
        sns.set_context("paper", rc=paper_rc)
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{cmbright}')
        fig = plt.figure(figsize=[5, 1.6])

        # left subplot
        fig.add_subplot(1, 2, 1, adjustable='box')
        plt.plot(xs_sdcit, ys_sdcit_pearson3, label='SDCIT null', lw=1.5, color=color_palettes[method_color_codes['SDCIT']])
        plt.plot(xs_kcipt, ys_kcipt_20000_gamma, label='KCIPT null', lw=1.5, color=color_palettes[method_color_codes['KCIPT']])
        sns.distplot(df_kcipt_20000['statistic'], hist=True, bins=20, kde=False, norm_hist=True, color=color_palettes[method_color_codes['KCIPT']], label='KCIPT TS')
        plt.legend(loc=1)
        plt.gca().set_xlabel('MMD')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.gca().set_ylabel('density')
        plt.gca().set_xlim([-0.0002, 0.0003])
        plt.setp(plt.gca(), 'yticklabels', [])

        # right subplot
        fig.add_subplot(1, 2, 2, adjustable='box')
        sns.distplot(df_kcipt_20000['pvalue'], hist=True, bins=20, kde=False, color=color_palettes[method_color_codes['KCIPT']], norm_hist=True, label='KCIPT p')
        sns.distplot([p_value_of(ss, sdcit_null) for ss in df_kcipt_20000['statistic']], hist=True, bins=20, kde=False, color='k', norm_hist=True, label='KCIPT p on SDCIT null')
        plt.gca().set_xlabel('p-value')
        plt.gcf().subplots_adjust(wspace=0.3, hspace=0.3)
        plt.legend(loc=0)

        sns.despine()
        plt.savefig(SDCIT_FIGURE_DIR + '/kcipt_20000_ps.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
        plt.close()


if __name__ == "__main__":
    main()
