import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from scipy.stats import gamma

from kcipt.utils import p_value_of

if __name__ == '__main__':
    cp = sns.color_palette('Set1', 5)
    cps = {'CHSIC': 3, 'KCIT': 2, 'SDCIT': 0, 'adj_kcipt': 4, 'KCIPT': 1}

    obj_filename = 'right_power.pickle'
    # if not os.path.exists(obj_filename):
    #     trial = 1
    #     gamma = 0.0
    #     N = 400
    #     independent = 0
    #
    #     b = 1000
    #     initial_B = 100
    #
    #     mat_load = scipy.io.loadmat(os.path.expanduser('~/kcipt_data/{}_{}_{}_{}_chaotic.mat'.format(gamma, trial, independent, N)), squeeze_me=True, struct_as_record=False)
    #     data = mat_load['data']
    #     if independent:
    #         X = data.Xt1
    #         Y = data.Yt
    #         Z = data.Xt[:, 0:2]
    #     else:
    #         X = data.Yt1
    #         Y = data.Xt
    #         Z = data.Yt[:, 0: 2]
    #
    #     kx, ky, kz = toK(X, Y, Z)
    #
    #     np.random.seed(trial)
    #     lee_mmd, lee_pval, lee_null = lee_KCIPT(kx, ky, kz, size_of_null_sample=b, with_null=True)
    #
    #     np.random.seed(trial)
    #     pval100, mmds100, inner_null100, outer_null100 = c_KCIPT(kx, ky, kz, K2D(kz), initial_B, b, b, n_jobs=32)
    #     desired_B = int(initial_B * (outer_null100.std() / lee_null.std()) ** 2)
    #     print(desired_B)
    #
    #     np.random.seed(trial)
    #     pval_B, mmds_B, inner_null_B, outer_null_B = c_KCIPT(kx, ky, kz, K2D(kz), desired_B, max(100, 10000 / desired_B), b, n_jobs=32)
    #
    #     how_many = b
    #     distr_boot = np.zeros((how_many))
    #     for ii in trange(how_many):
    #         _, mmds_B, _, _ = c_KCIPT(kx, ky, kz, K2D(kz), desired_B, 0, 0, n_jobs=32)
    #         distr_boot[ii] = mmds_B.mean()
    #
    #     with open(obj_filename, 'wb') as f:  # Python 3: open(..., 'wb')
    #         pickle.dump([lee_mmd, lee_null, mmds100, inner_null100, outer_null100, desired_B, mmds_B, inner_null_B, outer_null_B, distr_boot], f)
    #
    # import time
    # time.sleep(3)
    with open(obj_filename, 'rb') as f:  # Python 3: open(..., 'rb')
        lee_mmd, lee_null, mmds100, inner_null100, outer_null100, desired_B, mmds_B, inner_null_B, outer_null_B, distr_boot = pickle.load(f)
    print(desired_B)
    print(scipy.stats.skew(lee_null))
    print(scipy.stats.skew(outer_null_B))
    names_pykcipt_chaotic = ['independent', 'gamma', 'trial', 'N', 'statistic', 'pvalue', 'B']
    names_lee_chaotic = ['independent', 'gamma', 'trial', 'N', 'statistic', 'pvalue']
    df = pd.read_csv('kcipt_chaotic_1470.csv', names=names_pykcipt_chaotic, )
    df5000 = pd.read_csv('kcipt_chaotic_5000.csv', names=names_pykcipt_chaotic, )
    df20000 = pd.read_csv('kcipt_chaotic_20000.csv', names=names_pykcipt_chaotic, )
    df_sdcit = pd.read_csv('sdcit_chaotic.csv', names=names_lee_chaotic, )
    df_sdcit = df_sdcit[df_sdcit['N'] == 400]
    df_sdcit = df_sdcit[df_sdcit['independent'] == 0]
    df_sdcit = df_sdcit[df_sdcit['gamma'] == 0.0]
    assert len(df_sdcit) == 300
    xs_lee = np.linspace(2 * lee_null.min(), 2 * lee_null.max(), 1000)
    ys_lee = gamma.pdf(xs_lee, *gamma.fit(lee_null))

    xs_B = np.linspace(2 * outer_null_B.min(), 2 * outer_null_B.max(), 1000)
    ys_B = gamma.pdf(xs_B, *gamma.fit(outer_null_B))

    factor_20000 = np.sqrt(20000/1470)
    ys_20000 = gamma.pdf(xs_B, *gamma.fit(outer_null_B/factor_20000))

    if False:
        sns.set(style='white', font_scale=1.2)
        paper_rc = {'lines.linewidth': 0.8, 'lines.markersize': 2, 'patch.linewidth': 1}
        sns.set_context("paper", rc=paper_rc)
        # fig = plt.figure(figsize=[4, 3])
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{cmbright}')

        fig = plt.figure(figsize=[5, 4])
        ##################################
        ax = fig.add_subplot(2, 2, 1, adjustable='box')

        xs_T = np.linspace(2 * distr_boot.min(), 2 * distr_boot.max(), 1000)
        ys_T = gamma.pdf(xs_T, *gamma.fit(distr_boot))

        # fig.suptitle('comparison of empirical distributions')
        plt.plot(xs_lee, ys_lee, label='SDCIT null', lw=1.5, color=cp[cps['SDCIT']])
        plt.plot([lee_mmd, lee_mmd], [0, 1000], label='SDCIT TS', color=cp[cps['SDCIT']])
        plt.plot(xs_B, ys_B, label='KCIPT null', lw=1.5, color=cp[cps['KCIPT']])
        # plt.plot(xs_T, ys_T, color='gray', label='KCIPT TS', ls=':', lw=2)
        sns.distplot(distr_boot, hist=True, kde=False, hist_kws={'histtype': 'stepfilled'}, norm_hist=True, label='KCIPT TS', color=cp[cps['KCIPT']])
        plt.gca().set_xlim([-0.0003, 0.0005])
        # plt.gca().set_xlabel('MMD')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.gca().set_ylabel('density')
        plt.setp(plt.gca(), 'yticklabels', [])
        plt.legend(loc=1)
        ##################################
        ax = fig.add_subplot(2, 2, 2, adjustable='box')

        pvals_B = [p_value_of(t, outer_null_B) for t in distr_boot]
        pval_lee = p_value_of(lee_mmd, lee_null)

        sns.distplot(pvals_B, bins=20, hist=True, kde=False, hist_kws={'histtype': 'stepfilled'}, norm_hist=True, color=cp[cps['KCIPT']], label='KCIPT p-values')
        plt.plot([pval_lee, pval_lee], [0, 1], label='SDCIT p-value', color=cp[cps['SDCIT']])
        # plt.gca().set_xlabel('p-value')
        # plt.gca().set_ylabel('density')
        plt.gcf().subplots_adjust(wspace=0.3)
        plt.legend(loc=2)
        sns.despine()

        ##################################
        ax = fig.add_subplot(2, 2, 3, adjustable='box')
        sns.distplot(df_sdcit['statistic'], hist=True, bins=20, kde=False, color=cp[cps['SDCIT']], label='SDCIT TS')
        sns.distplot(df['statistic'], hist=True, bins=20, kde=False, color=cp[cps['KCIPT']], label='KCIPT TS')
        plt.legend()
        plt.gca().set_xlim([-0.0003, 0.0005])
        plt.gca().set_xlabel('MMD')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.gca().set_ylabel('density')
        plt.setp(plt.gca(), 'yticklabels', [])

        ##################################
        ax = fig.add_subplot(2, 2, 4, adjustable='box')

        sns.distplot(df_sdcit['pvalue'], hist=True, bins=20, kde=False, color=cp[cps['SDCIT']], norm_hist=True, label='SDCIT p-values')
        sns.distplot(df['pvalue'], hist=True, bins=20, kde=False, color=cp[cps['KCIPT']], norm_hist=True, label='KCIPT p-values')
        plt.gca().set_xlabel('p-value')
        # plt.gca().set_ylabel('density')
        plt.gcf().subplots_adjust(wspace=0.3, hspace=0.3)
        plt.legend(loc=0)
        sns.despine()
        plt.savefig('kcipt_1470_ps.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
        plt.close()

    ###############################################
    ###############################################
    ###############################################
    ###############################################
    ###############################################
    ###############################################
    if False:
        sns.set(style='white', font_scale=1.2)
        paper_rc = {'lines.linewidth': 0.8, 'lines.markersize': 2, 'patch.linewidth': 1}
        sns.set_context("paper", rc=paper_rc)
        # fig = plt.figure(figsize=[4, 3])
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{cmbright}')
        fig = plt.figure(figsize=[5, 2])
        ##################################
        ax = fig.add_subplot(1, 2, 1, adjustable='box')
        sns.distplot(df5000['statistic'], hist=True, bins=20, kde=False, color=cp[cps['KCIPT']], label='5000')
        plt.legend()
        plt.gca().set_xlabel('MMD')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.gca().set_ylabel('density')
        plt.gca().set_xlim([-0.0002, 0.0003])
        plt.setp(plt.gca(), 'yticklabels', [])
        ##
        ax = fig.add_subplot(1, 2, 2, adjustable='box')
        sns.distplot(df5000['pvalue'], hist=True, bins=20, kde=False, color=cp[cps['KCIPT']], norm_hist=True, label='5000')
        plt.gca().set_xlabel('p-value')
        # plt.gca().set_ylabel('density')
        plt.gcf().subplots_adjust(wspace=0.3, hspace=0.3)
        plt.legend(loc=0)
        sns.despine()
        plt.savefig('kcipt_5000_ps.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
        plt.close()


    if True:
        sns.set(style='white', font_scale=1.2)
        paper_rc = {'lines.linewidth': 0.8, 'lines.markersize': 2, 'patch.linewidth': 1}
        sns.set_context("paper", rc=paper_rc)
        # fig = plt.figure(figsize=[4, 3])
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{cmbright}')
        fig = plt.figure(figsize=[5, 2])
        ##################################
        ax = fig.add_subplot(1, 2, 1, adjustable='box')
        plt.plot(xs_lee, ys_lee, label='SDCIT null', lw=1.5, color=cp[cps['SDCIT']])
        plt.plot(xs_B, ys_20000, label='KCIPT null', lw=1.5, color=cp[cps['KCIPT']])
        # sns.distplot(df5000['statistic'], hist=True, bins=20, kde=False, color=cp[cps['KCIPT']], label='5000')
        sns.distplot(df20000['statistic'], hist=True, bins=20, kde=False, norm_hist=True, color=cp[cps['KCIPT']], label='KCIPT TS')
        plt.legend(loc=1)
        plt.gca().set_xlabel('MMD')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.gca().set_ylabel('density')
        plt.gca().set_xlim([-0.0003, 0.0005])
        plt.setp(plt.gca(), 'yticklabels', [])
        ##
        ax = fig.add_subplot(1, 2, 2, adjustable='box')
        sns.distplot(df20000['pvalue'], hist=True, bins=20, kde=False, color=cp[cps['KCIPT']], norm_hist=True, label='KCIPT p')
        sns.distplot([p_value_of(ss, lee_null) for ss in df20000['statistic']], hist=True, bins=20, kde=False, color='k', norm_hist=True, label='KCIPT p on SDCIT null')

        plt.gca().set_xlabel('p-value')
        # plt.gca().set_ylabel('density')
        plt.gcf().subplots_adjust(wspace=0.3, hspace=0.3)
        plt.legend(loc=0)
        sns.despine()
        plt.savefig('kcipt_20000_ps.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
        plt.close()
