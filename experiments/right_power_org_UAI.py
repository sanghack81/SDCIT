import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from scipy.stats import gamma, pearson3
from tqdm import trange

from experiments.testing_all_KCIPT import test_chaotic
from experiments.testing_utils import read_chaotic
from kcipt.algo import c_KCIPT
from kcipt.sdcit2 import bias_reduced_SDCIT
from kcipt.utils import p_value_of, K2D


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
        sdcit_mmd, sdcit_pval, sdcit_null = bias_reduced_SDCIT(kx, ky, kz, with_null=True, seed=trial)
        print('KCIPT {} ... '.format(initial_B))
        _, mmds100, _, outer_null100 = c_KCIPT(kx, ky, kz, K2D(kz), initial_B, 10000, 10000, n_jobs=32, seed=trial)

        # Infer desired B
        desired_B = int(initial_B * (outer_null100.std() / sdcit_null.std()) ** 2)
        print('Desired B: {}'.format(desired_B))

        # Prepare outer null distribution
        print('KCIPT {} ... '.format(desired_B))
        _, mmds_B, _, outer_null_B = c_KCIPT(kx, ky, kz, K2D(kz), desired_B, 10000, 10000, n_jobs=32, seed=trial)

        print('TS distributions for KCIPT {} ... '.format(desired_B))
        time.sleep(1)
        distr_boot = np.zeros((1000,))
        for ii in trange(len(distr_boot)):
            _, mmds_B, _, _ = c_KCIPT(kx, ky, kz, K2D(kz), desired_B, 0, 0, n_jobs=32, seed=ii)
            distr_boot[ii] = mmds_B.mean()

        with open(obj_filename, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([sdcit_mmd, sdcit_null, mmds100, outer_null100, desired_B, mmds_B, outer_null_B, distr_boot], f)

        print(independent, gamma_param, N)
        outs = [test_chaotic(independent, gamma_param, tt, N, B=desired_B, n_jobs=32) for tt in trange(300)]
        with open('../results/kcipt_chaotic_{}.csv'.format(desired_B), 'a') as f:
            for out in outs:
                print(*out, sep=',', file=f, flush=True)


def main():
    from experiments.drawing import cp, cps

    # cp = sns.color_palette('Set1', 5)
    # cps = {'CHSIC': 3, 'KCIT': 2, 'SDCIT': 0, 'KCIPT': 1}

    obj_filename = '../results/right_power.pickle'
    experiment(obj_filename)

    time.sleep(3)

    with open(obj_filename, 'rb') as f:  # Python 3: open(..., 'rb')
        sdcit_mmd, sdcit_null, mmds100, outer_null100, desired_B, mmds_B, outer_null_B, distr_boot = pickle.load(f)

    print(desired_B)
    print('SKEW SDCIT NULL: {}'.format(scipy.stats.skew(sdcit_null)))
    print('SKEW KCIPT NULL: {}'.format(scipy.stats.skew(outer_null_B)))

    names_kcipt_chaotic = ['independent', 'gamma', 'trial', 'N', 'statistic', 'pvalue', 'B']
    names_sdcit_chaotic = ['independent', 'gamma', 'trial', 'N', 'statistic', 'pvalue']

    df = pd.read_csv('../results/kcipt_chaotic_{}.csv'.format(desired_B), names=names_kcipt_chaotic, )
    df5000 = pd.read_csv('../results/kcipt_chaotic_5000.csv', names=names_kcipt_chaotic, )
    df20000 = pd.read_csv('../results/kcipt_chaotic_20000.csv', names=names_kcipt_chaotic, )
    df_sdcit = pd.read_csv('../results/sdcit_chaotic.csv', names=names_sdcit_chaotic, )
    df_sdcit = df_sdcit[df_sdcit['N'] == 400]
    df_sdcit = df_sdcit[df_sdcit['independent'] == 1]
    df_sdcit = df_sdcit[df_sdcit['gamma'] == 0.0]
    assert len(df_sdcit) == 300
    xs_sdcit = np.linspace(1.3 * sdcit_null.min(), 1.3 * sdcit_null.max(), 1000)
    ys_sdcit = pearson3.pdf(xs_sdcit, *pearson3.fit(sdcit_null))

    xs_B = np.linspace(1.3 * outer_null_B.min(), 1.3 * outer_null_B.max(), 1000)
    ys_B = pearson3.pdf(xs_B, *pearson3.fit(outer_null_B))

    # 20000's null is inferred from known one...
    factor_20000 = np.sqrt(20000 / desired_B)
    ys_20000 = gamma.pdf(xs_B, *gamma.fit(outer_null_B / factor_20000))

    sns.set(style='white', font_scale=1.2)
    paper_rc = {'lines.linewidth': 0.8, 'lines.markersize': 2, 'patch.linewidth': 1}
    sns.set_context("paper", rc=paper_rc)
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{cmbright}')

    if True:
        fig = plt.figure(figsize=[5, 3.5])
        ##################################
        fig.add_subplot(2, 2, 1, adjustable='box')

        plt.plot(xs_sdcit, ys_sdcit, label='SDCIT null', lw=1.5, color=cp[cps['SDCIT']])
        plt.plot([sdcit_mmd, sdcit_mmd], [0, 1000], label='SDCIT TS', color=cp[cps['SDCIT']])
        plt.plot(xs_B, ys_B, label='KCIPT null', lw=1.5, color=cp[cps['KCIPT']])
        sns.distplot(distr_boot, hist=True, kde=False, hist_kws={'histtype': 'stepfilled'}, norm_hist=True, label='KCIPT TS', color=cp[cps['KCIPT']])
        plt.gca().set_xlim([-0.0003, 0.0005])
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.gca().set_ylabel('density')
        plt.setp(plt.gca(), 'yticklabels', [])
        plt.legend(loc=1)
        ##################################
        fig.add_subplot(2, 2, 2, adjustable='box')

        pvals_B = [p_value_of(t, outer_null_B) for t in distr_boot]
        pval_sdcit = p_value_of(sdcit_mmd, sdcit_null)

        sns.distplot(pvals_B, bins=20, hist=True, kde=False, hist_kws={'histtype': 'stepfilled'}, norm_hist=True, color=cp[cps['KCIPT']], label='KCIPT p-values')
        plt.plot([pval_sdcit, pval_sdcit], [0, 1], label='SDCIT p-value', color=cp[cps['SDCIT']])
        plt.gca().set_ylim([0, 2.2])
        plt.gcf().subplots_adjust(wspace=0.3)
        plt.legend(loc=2)
        sns.despine()

        ##################################
        fig.add_subplot(2, 2, 3, adjustable='box')
        sns.distplot(df_sdcit['statistic'], hist=True, bins=20, kde=False, color=cp[cps['SDCIT']], label='SDCIT TS')
        sns.distplot(df['statistic'], hist=True, bins=20, kde=False, color=cp[cps['KCIPT']], label='KCIPT TS')
        plt.legend()
        plt.gca().set_xlim([-0.0003, 0.0005])
        plt.gca().set_xlabel('MMD')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.gca().set_ylabel('density')
        plt.setp(plt.gca(), 'yticklabels', [])

        ##################################
        fig.add_subplot(2, 2, 4, adjustable='box')

        sns.distplot(df_sdcit['pvalue'], hist=True, bins=20, kde=False, color=cp[cps['SDCIT']], norm_hist=True, label='SDCIT p-values')
        sns.distplot(df['pvalue'], hist=True, bins=20, kde=False, color=cp[cps['KCIPT']], norm_hist=True, label='KCIPT p-values')
        plt.gca().set_xlabel('p-value')
        plt.gcf().subplots_adjust(wspace=0.3, hspace=0.3)
        plt.gca().set_ylim([0, 2.2])
        plt.legend(loc=0)
        sns.despine()
        plt.savefig('../results/kcipt_{}_ps.pdf'.format(desired_B), transparent=True, bbox_inches='tight', pad_inches=0.02)
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
        sns.distplot(df5000['statistic'], hist=True, bins=20, kde=False, color=cp[cps['KCIPT']], label='TS')
        plt.legend()
        plt.gca().set_xlabel('MMD')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.gca().set_ylabel('density')
        plt.gca().set_xlim([-0.0002, 0.0003])
        plt.setp(plt.gca(), 'yticklabels', [])
        ##
        fig.add_subplot(1, 2, 2, adjustable='box')
        sns.distplot(df5000['pvalue'], hist=True, bins=20, kde=False, color=cp[cps['KCIPT']], norm_hist=True, label='p-value')
        plt.gca().set_xlabel('p-value')
        # plt.gca().set_ylabel('density')
        plt.gcf().subplots_adjust(wspace=0.3, hspace=0.3)
        plt.legend(loc=0)
        sns.despine()
        plt.savefig('../results/kcipt_5000_ps.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
        plt.close()

    if True:
        sns.set(style='white', font_scale=1.2)
        paper_rc = {'lines.linewidth': 0.8, 'lines.markersize': 2, 'patch.linewidth': 1}
        sns.set_context("paper", rc=paper_rc)
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{cmbright}')
        fig = plt.figure(figsize=[5, 1.6])
        ##################################
        fig.add_subplot(1, 2, 1, adjustable='box')
        plt.plot(xs_sdcit, ys_sdcit, label='SDCIT null', lw=1.5, color=cp[cps['SDCIT']])
        plt.plot(xs_B, ys_20000, label='KCIPT null', lw=1.5, color=cp[cps['KCIPT']])
        sns.distplot(df20000['statistic'], hist=True, bins=20, kde=False, norm_hist=True, color=cp[cps['KCIPT']], label='KCIPT TS')
        plt.legend(loc=1)
        plt.gca().set_xlabel('MMD')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.gca().set_ylabel('density')
        plt.gca().set_xlim([-0.0002, 0.0003])
        plt.setp(plt.gca(), 'yticklabels', [])
        ##
        fig.add_subplot(1, 2, 2, adjustable='box')
        sns.distplot(df20000['pvalue'], hist=True, bins=20, kde=False, color=cp[cps['KCIPT']], norm_hist=True, label='KCIPT p')
        sns.distplot([p_value_of(ss, sdcit_null) for ss in df20000['statistic']], hist=True, bins=20, kde=False, color='k', norm_hist=True, label='KCIPT p on SDCIT null')

        plt.gca().set_xlabel('p-value')
        plt.gcf().subplots_adjust(wspace=0.3, hspace=0.3)
        plt.legend(loc=0)
        sns.despine()
        plt.savefig('../results/kcipt_20000_ps.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
        plt.close()

    if True:
        #
        sns.set()
        sns.distplot(sdcit_null, norm_hist=True, kde=False)
        plt.plot(xs_sdcit, ys_sdcit, lw=1.5, color=cp[cps['SDCIT']])
        plt.savefig('../results/inspect_sdcit_null.pdf')
        plt.close()


if __name__ == "__main__":
    main()
