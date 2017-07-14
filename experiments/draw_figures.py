import collections

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import seaborn as sns
from os.path import exists

from experiments.exp_setup import SDCIT_RESULT_DIR, SDCIT_FIGURE_DIR
from sdcit.utils import AUPC

names_chsic_chaotic = ['independent', 'gamma', 'noise', 'trial', 'N', 'runtime', 'statistic', 'pvalue']
names_chsic_postnonlinear = ['independent', 'noise', 'trial', 'N', 'runtime', 'statistic', 'pvalue']
names_kcit_chaotic = ['independent', 'gamma', 'noise', 'trial', 'N', 'runtime', 'statistic', 'boot_p_value', 'appr_p_value']
names_kcit_postnonlinear = ['independent', 'noise', 'trial', 'N', 'runtime', 'statistic', 'boot_p_value', 'appr_p_value']
names_sdcit_chaotic = ['independent', 'gamma', 'trial', 'N', 'statistic', 'pvalue']
names_sdcit_postnonlinear = ['independent', 'noise', 'trial', 'N', 'statistic', 'pvalue']
names_kcipt_chaotic = ['independent', 'gamma', 'trial', 'N', 'statistic', 'pvalue', 'B']
names_kcipt_postnonlinear = ['independent', 'noise', 'trial', 'N', 'statistic', 'pvalue', 'B']

names = {('CHSIC', 'chaotic'): names_chsic_chaotic,
         ('CHSIC', 'postnonlinear'): names_chsic_postnonlinear,
         ('KCIT', 'chaotic'): names_kcit_chaotic,
         ('KCIT', 'postnonlinear'): names_kcit_postnonlinear,
         ('KCIT2', 'chaotic'): names_kcit_chaotic,
         ('KCIT2', 'postnonlinear'): names_kcit_postnonlinear,
         ('SDCIT', 'chaotic'): names_sdcit_chaotic,
         ('SDCIT', 'postnonlinear'): names_sdcit_postnonlinear,
         ('KCIPT', 'chaotic'): names_kcipt_chaotic,
         ('KCIPT', 'postnonlinear'): names_kcipt_postnonlinear,
         }

pvalue_column = collections.defaultdict(lambda: 'pvalue')
pvalue_column['KCIT'] = 'boot_p_value'
pvalue_column['KCIT2'] = 'boot_p_value'

color_palettes = sns.color_palette('Paired', 10)
method_color_codes = {'KCIT': 3, 'SDCIT': 5, 'KCIPT': 1, 'CHSIC': 9, 'KCIT2': 2}
markers = collections.defaultdict(lambda: 'o')
markers.update({'KCIT': 'o', 'SDCIT': 's', 'KCIPT': '*', 'CHSIC': '^', 'KCIT2': 'o'})
all_algos = ['KCIT', 'SDCIT', 'KCIPT', 'CHSIC', 'KCIT2']


def algo_name(org_name):
    map = {'KCIT2': 'KCIT', 'KCIT': 'KCIT (org.)'}
    if org_name in map:
        return map[org_name]
    else:
        return org_name


def draw_aupc_chaotic():
    data = 'chaotic'

    aupc_data = []
    for algo in all_algos:
        df = pd.read_csv(SDCIT_RESULT_DIR + '/' + algo.lower() + '_' + data + '.csv', names=names[(algo, data)])

        for group_key, group_df in df.groupby(by=['gamma', 'independent', 'N']):
            group_key = (int(group_key[0] * 10) / 10, *group_key[1:])
            if group_key[1] == 0:
                aupc_data.append([algo, *group_key, AUPC(group_df[pvalue_column[algo]])])

    print(draw_aupc_chaotic.__name__)
    [print(xx) for xx in aupc_data]

    aupc_data = np.array(aupc_data)
    aupc_df = pd.DataFrame({'algorithm': aupc_data[:, 0],
                            'gamma': aupc_data[:, 1],
                            'independent': aupc_data[:, 2],
                            'N': aupc_data[:, 3],
                            'AUPC': aupc_data[:, 4]})
    aupc_df['gamma'] = aupc_df['gamma'].astype(float)
    aupc_df['independent'] = aupc_df['independent'].astype(int)
    aupc_df['N'] = aupc_df['N'].map(int)
    aupc_df['AUPC'] = aupc_df['AUPC'].astype(float)

    aupc_df = aupc_df[aupc_df['independent'] == 0]
    aupc_df["algo-N"] = aupc_df["algorithm"].map(str) + aupc_df["N"].map(lambda xxx: ' (' + str(xxx) + ')')
    sns_setting()
    for k, gdf in aupc_df.groupby(['algorithm', 'N']):
        print('chaotic', k, gdf['AUPC'])
        if k[1] == 400:
            plt.plot(gdf['gamma'], gdf['AUPC'], markers[(k[0])], c=color_palettes[method_color_codes[k[0]]] if k[1] == 400 else color_palettes[-0 + method_color_codes[k[0]]],
                     ls='-' if k[1] == 400 else ':', label=algo_name(str(k[0])))
        else:
            plt.plot(gdf['gamma'], gdf['AUPC'], markers[(k[0])], c=color_palettes[method_color_codes[k[0]]] if k[1] == 400 else color_palettes[-0 + method_color_codes[k[0]]],
                     ls='-' if k[1] == 400 else ':', label='_nolegend_')

    plt.axes().set_xlabel(r'$\gamma$')
    plt.axes().set_ylabel('Area Under Power Curve')
    plt.axes().set_ylim([0.45, 1.05])

    handles, labels = plt.axes().get_legend_handles_labels()
    # plt.axes().legend(handles[::-1], labels[::-1])

    sns.despine()
    plt.savefig(SDCIT_FIGURE_DIR + '/{}_aupc.pdf'.format(data), transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def draw_calib_chaotic():
    data = 'chaotic'
    calib_data = []
    for algo in all_algos:
        df = pd.read_csv(SDCIT_RESULT_DIR + '/' + algo.lower() + '_' + data + '.csv', names=names[(algo, data)])
        for k, gdf in df.groupby(by=['independent', 'gamma', 'N']):
            if float(k[0]) == 1:
                D, _ = scipy.stats.kstest(gdf[pvalue_column[algo]], 'uniform')
                calib_data.append([algo, float(k[1]), int(k[2]), D])

    print(draw_calib_chaotic.__name__)
    [print(xx) for xx in calib_data]

    df = pd.DataFrame(calib_data, columns=['algo', 'gamma', 'N', 'D'])
    df['gamma'] = df['gamma'].astype(float)
    df['N'] = df['N'].map(int)
    df['D'] = df['D'].astype(float)
    sns_setting()
    for k, gdf in df.groupby(['algo', 'N']):
        if k[1] == 400:
            plt.plot(gdf['gamma'], gdf['D'], markers[(k[0])], c=color_palettes[method_color_codes[k[0]]] if k[1] == 400 else color_palettes[-0 + method_color_codes[k[0]]],
                     ls='-' if k[1] == 400 else ':', label=algo_name(str(k[0])))
        else:
            plt.plot(gdf['gamma'], gdf['D'], markers[(k[0])], c=color_palettes[method_color_codes[k[0]]] if k[1] == 400 else color_palettes[-0 + method_color_codes[k[0]]],
                     ls='-' if k[1] == 400 else ':', label='_nolegend_')
    handles, labels = plt.axes().get_legend_handles_labels()
    plt.axes().legend(handles[::-1], labels[::-1], ncol=2)
    plt.axes().set_xlabel(r'$\gamma$')
    plt.axes().set_ylabel('KS test statistic')
    plt.axes().set_ylim([0.0, 0.5])
    plt.axes().invert_yaxis()
    plt.axes().set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    handles, labels = plt.axes().get_legend_handles_labels()
    # plt.axes().legend(handles[::-1], labels[::-1])

    sns.despine()
    plt.savefig(SDCIT_FIGURE_DIR + '/chaotic_calib.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def draw_type_I_error_chaotic():
    data = 'chaotic'
    calib_data = []
    for algo in all_algos:
        df = pd.read_csv(SDCIT_RESULT_DIR + '/' + algo.lower() + '_' + data + '.csv', names=names[(algo, data)])
        for k, gdf in df.groupby(by=['independent', 'gamma', 'N']):
            if float(k[0]) == 1:
                calib_data.append([algo, float(k[1]), int(k[2]), np.mean(gdf[pvalue_column[algo]] <= 0.05)])

    print(draw_type_I_error_chaotic.__name__)
    [print(xx) for xx in calib_data]

    df = pd.DataFrame(calib_data, columns=['algo', 'gamma', 'N', 'D'])
    df['gamma'] = df['gamma'].astype(float)
    df['N'] = df['N'].map(int)
    df['D'] = df['D'].astype(float)
    sns_setting()
    for k, gdf in df.groupby(['algo', 'N']):
        if k[1] == 400:
            plt.plot(gdf['gamma'], gdf['D'], markers[(k[0])], c=color_palettes[method_color_codes[k[0]]] if k[1] == 400 else color_palettes[-0 + method_color_codes[k[0]]],
                     ls='-' if k[1] == 400 else ':', label=algo_name(str(k[0])))
        else:
            plt.plot(gdf['gamma'], gdf['D'], markers[(k[0])], c=color_palettes[method_color_codes[k[0]]] if k[1] == 400 else color_palettes[-0 + method_color_codes[k[0]]],
                     ls='-' if k[1] == 400 else ':', label='_nolegend_')
    plt.axes().set_xlabel(r'$\gamma$')
    plt.axes().set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    plt.axes().set_ylabel('Type I error')
    plt.axes().set_ylim([0.0, 0.2])
    sns.despine()
    plt.savefig(SDCIT_FIGURE_DIR + '/chaotic_type_I.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def draw_aupc_postnonlinear():
    data = 'postnonlinear'
    aupc_data = []
    for algo in all_algos:
        df = pd.read_csv(SDCIT_RESULT_DIR + '/' + algo.lower() + '_' + data + '.csv', names=names[(algo, data)])

        for group_key, group_df in df.groupby(by=['noise', 'independent', 'N']):
            group_key = (int(group_key[0] * 10) / 10, int(group_key[1]), int(group_key[2]))
            aupc_data.append([algo, *group_key, AUPC(group_df[pvalue_column[algo]])])

    print(draw_aupc_postnonlinear.__name__)
    [print(xx) for xx in aupc_data]

    aupc_data = np.array(aupc_data)
    aupc_df = pd.DataFrame({'algorithm': [str(v) for v in aupc_data[:, 0]],
                            'noise': [int(float(v)) for v in aupc_data[:, 1]],
                            'independent': [int(v) for v in aupc_data[:, 2]],
                            'N': [int(v) for v in aupc_data[:, 3]],
                            'AUPC': [float(v) for v in aupc_data[:, 4]]})
    aupc_df['dimension'] = (aupc_df['noise'] + 1).astype(int)

    aupc_df = aupc_df[aupc_df['independent'] == 0]
    aupc_df["algo-N"] = aupc_df["algorithm"].map(str) + aupc_df["N"].map(lambda xxx: ' (' + str(xxx) + ')')
    sns_setting()
    for k, gdf in aupc_df.groupby(['algorithm', 'N']):
        gdf = gdf[gdf['dimension'] <= 5]
        if k[1] == 400:
            plt.plot(gdf['dimension'], gdf['AUPC'], markers[(k[0])], c=color_palettes[method_color_codes[k[0]]] if k[1] == 400 else color_palettes[-0 + method_color_codes[k[0]]],
                     ls='-' if k[1] == 400 else ':', label=algo_name(str(k[0])))
        else:
            plt.plot(gdf['dimension'], gdf['AUPC'], markers[(k[0])], c=color_palettes[method_color_codes[k[0]]] if k[1] == 400 else color_palettes[-0 + method_color_codes[k[0]]],
                     ls='-' if k[1] == 400 else ':', label='_nolegend_')
    plt.axes().set_xlabel('dimension')
    plt.axes().set_ylabel('Area Under Power Curve')
    plt.axes().set_ylim([0.45, 1.05])
    sns.despine()
    plt.savefig(SDCIT_FIGURE_DIR + '/postnonlinear_aupc.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def draw_aupc_postnonlinear_highdim():
    data = 'postnonlinear'
    aupc_data = []
    for algo in all_algos:
        df = pd.read_csv(SDCIT_RESULT_DIR + '/' + algo.lower() + '_' + data + '.csv', names=names[(algo, data)])

        for group_key, group_df in df.groupby(by=['noise', 'independent', 'N']):
            group_key = (int(group_key[0] * 10) / 10, int(group_key[1]), int(group_key[2]))
            aupc_data.append([algo, *group_key, AUPC(group_df[pvalue_column[algo]])])

    print(draw_aupc_postnonlinear_highdim.__name__)
    [print(xx) for xx in aupc_data]

    aupc_data = np.array(aupc_data)
    aupc_df = pd.DataFrame({'algorithm': [str(v) for v in aupc_data[:, 0]],
                            'noise': [int(float(v)) for v in aupc_data[:, 1]],
                            'independent': [int(v) for v in aupc_data[:, 2]],
                            'N': [int(v) for v in aupc_data[:, 3]],
                            'AUPC': [float(v) for v in aupc_data[:, 4]]})
    aupc_df['dimension'] = (aupc_df['noise'] + 1).astype(int)

    aupc_df = aupc_df[aupc_df['independent'] == 0]
    aupc_df["algo-N"] = aupc_df["algorithm"].map(str) + aupc_df["N"].map(lambda xxx: ' (' + str(xxx) + ')')
    sns_setting()
    for k, gdf in aupc_df.groupby(['algorithm', 'N']):
        if k[1] == 400:
            plt.plot([int(v) for v in gdf['dimension']], gdf['AUPC'], markers[(k[0])], c=color_palettes[method_color_codes[k[0]]] if k[1] == 400 else color_palettes[-0 + method_color_codes[k[0]]],
                     ls='-' if k[1] == 400 else ':',
                     label=algo_name(str(k[0])))

    plt.axes().set_xlabel('dimension')
    plt.axes().set_ylabel('Area Under Power Curve')
    plt.axes().set_ylim([0.95, 1.01])
    plt.axes().set_xscale('log')
    plt.xticks([1, 5, 10, 20, 50], [1, 5, 10, 20, 50])
    sns.despine()
    plt.savefig(SDCIT_FIGURE_DIR + '/postnonlinear_aupc_highdim.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def draw_calib_postnonlinear():
    data = 'postnonlinear'
    calib_data = []
    for algo in all_algos:
        df = pd.read_csv(SDCIT_RESULT_DIR + '/' + algo.lower() + '_' + data + '.csv', names=names[(algo, data)])
        for k, gdf in df.groupby(by=['independent', 'noise', 'N']):
            if float(k[0]) == 1:
                D, _ = scipy.stats.kstest(gdf[pvalue_column[algo]], 'uniform')
                calib_data.append([algo, float(k[1]), int(k[2]), D])

    print(draw_calib_postnonlinear.__name__)
    [print(xx) for xx in calib_data]

    df = pd.DataFrame(calib_data, columns=['algo', 'noise', 'N', 'D'])
    df['noise'] = df['noise'].map(int)
    df['dimension'] = (df['noise'] + 1).astype(int)
    df['N'] = df['N'].map(int)
    df['D'] = df['D'].astype(float)
    sns_setting()
    for k, gdf in df.groupby(['algo', 'N']):
        gdf = gdf[gdf['dimension'] <= 5]
        if k[1] == 400:
            plt.plot([int(v) for v in gdf['dimension']], gdf['D'], markers[(k[0])], c=color_palettes[method_color_codes[k[0]]] if k[1] == 400 else color_palettes[-0 + method_color_codes[k[0]]],
                     ls='-' if k[1] == 400 else ':',
                     label=algo_name(str(k[0])))
        else:
            plt.plot([int(v) for v in gdf['dimension']], gdf['D'], markers[(k[0])], c=color_palettes[method_color_codes[k[0]]] if k[1] == 400 else color_palettes[-0 + method_color_codes[k[0]]],
                     ls='-' if k[1] == 400 else ':',
                     label='_nolegend_')
    plt.axes().set_xlabel('dimension')
    plt.axes().set_ylabel('KS test statistic')
    plt.axes().set_ylim([0.0, 0.5])
    plt.axes().invert_yaxis()
    plt.axes().set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    sns.despine()
    plt.savefig(SDCIT_FIGURE_DIR + '/postnonlinear_calib.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def sns_setting():
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 2}
    sns.set_context("paper", rc=paper_rc)
    sns.set(style='white', font_scale=1.4)
    plt.figure(figsize=[4, 3])
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{cmbright}')


def draw_calib_postnonlinear_highdim():
    data = 'postnonlinear'
    calib_data = []
    for algo in all_algos:
        df = pd.read_csv(SDCIT_RESULT_DIR + '/' + algo.lower() + '_' + data + '.csv', names=names[(algo, data)])
        for k, gdf in df.groupby(by=['independent', 'noise', 'N']):
            if float(k[0]) == 1 and k[2] == 400:
                dd, _ = scipy.stats.kstest(gdf[pvalue_column[algo]], 'uniform')
                calib_data.append([algo, float(k[1]), int(k[2]), dd])

    print(draw_calib_postnonlinear_highdim.__name__)
    [print(xx) for xx in calib_data]

    df = pd.DataFrame(calib_data, columns=['algo', 'noise', 'N', 'D'])
    df['noise'] = df['noise'].map(int)
    df['dimension'] = (df['noise'] + 1).astype(int)
    df['N'] = df['N'].map(int)
    df['D'] = df['D'].astype(float)
    sns_setting()
    for k, gdf in df.groupby(['algo', 'N']):
        print('postnonlinear', k, gdf['D'])
        if k[1] == 400:
            plt.plot(gdf['dimension'], gdf['D'], markers[(k[0])], c=color_palettes[method_color_codes[k[0]]] if k[1] == 400 else color_palettes[-0 + method_color_codes[k[0]]],
                     ls='-' if k[1] == 400 else ':', label=algo_name(str(k[0])))
        else:
            plt.plot(gdf['dimension'], gdf['D'], markers[(k[0])], c=color_palettes[method_color_codes[k[0]]] if k[1] == 400 else color_palettes[-0 + method_color_codes[k[0]]],
                     ls='-' if k[1] == 400 else ':', label='_nolegend_')
    plt.axes().set_xlabel('dimension')
    plt.axes().set_ylabel('KS test statistic')
    plt.axes().set_xscale('log')
    plt.axes().set_ylim([0.0, 0.5])
    plt.axes().invert_yaxis()
    plt.xticks([1, 5, 10, 20, 50], [1, 5, 10, 20, 50])
    plt.axes().set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    sns.despine()
    plt.savefig(SDCIT_FIGURE_DIR + '/postnonlinear_calib_highdim.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def draw_type_I_postnonlinear_highdim():
    data = 'postnonlinear'
    calib_data = []
    for algo in all_algos:
        df = pd.read_csv(SDCIT_RESULT_DIR + '/' + algo.lower() + '_' + data + '.csv', names=names[(algo, data)])
        for k, gdf in df.groupby(by=['independent', 'noise', 'N']):
            if float(k[0]) == 1 and k[2] == 400:
                dd = np.mean(gdf[pvalue_column[algo]] <= 0.05)
                calib_data.append([algo, float(k[1]), int(k[2]), dd])

    print(draw_type_I_postnonlinear_highdim.__name__)
    [print(xx) for xx in calib_data]

    df = pd.DataFrame(calib_data, columns=['algo', 'noise', 'N', 'D'])
    df['noise'] = df['noise'].map(int)
    df['dimension'] = (df['noise'] + 1).astype(int)
    df['N'] = df['N'].map(int)
    df['D'] = df['D'].astype(float)
    sns_setting()
    for k, gdf in df.groupby(['algo', 'N']):
        if k[1] == 400:
            plt.plot(gdf['dimension'], gdf['D'], markers[(k[0])], c=color_palettes[method_color_codes[k[0]]] if k[1] == 400 else color_palettes[-0 + method_color_codes[k[0]]],
                     ls='-' if k[1] == 400 else ':', label=algo_name(str(k[0])))
        else:
            plt.plot(gdf['dimension'], gdf['D'], markers[(k[0])], c=color_palettes[method_color_codes[k[0]]] if k[1] == 400 else color_palettes[-0 + method_color_codes[k[0]]],
                     ls='-' if k[1] == 400 else ':', label='_nolegend_')
    plt.axes().set_xlabel('dimension')
    plt.axes().set_xscale('log')
    plt.xticks([1, 5, 10, 20, 50], [1, 5, 10, 20, 50])
    plt.axes().set_ylim([0.0, 0.2])
    handles, labels = plt.axes().get_legend_handles_labels()
    plt.axes().legend(handles[::-1], labels[::-1])
    sns.despine()
    plt.savefig(SDCIT_FIGURE_DIR + '/postnonlinear_type_I_highdim.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close()


if __name__ == '__main__':
    for data in ['chaotic', 'postnonlinear']:
        for algo in all_algos:
            assert exists(SDCIT_RESULT_DIR + '/' + algo.lower() + '_' + data + '.csv'), 'run tests first -- missing {}'.format(algo.lower() + '_' + data + '.csv')
    if True:
        # chaotic series
        draw_aupc_chaotic()
        draw_calib_chaotic()

        # postnonlinear-noise
        draw_aupc_postnonlinear()
        draw_calib_postnonlinear()
        draw_aupc_postnonlinear_highdim()
        draw_calib_postnonlinear_highdim()

        # type I for both
        draw_type_I_error_chaotic()
        draw_type_I_postnonlinear_highdim()
