import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

names_chsic_chaotic = ['independent', 'gamma', 'noise', 'trial', 'N', 'runtime', 'statistic', 'pvalue']
names_chsic_postnonlinear = ['independent', 'noise', 'trial', 'N', 'runtime', 'statistic', 'pvalue']
names_kcit_chaotic = ['independent', 'gamma', 'noise', 'trial', 'N', 'runtime', 'statistic', 'boot_p_value', 'appr_p_value']
names_kcit_postnonlinear = ['independent', 'noise', 'trial', 'N', 'runtime', 'statistic', 'boot_p_value', 'appr_p_value']
names_lee_chaotic = ['independent', 'gamma', 'trial', 'N', 'statistic', 'pvalue']
names_lee_postnonlinear = ['independent', 'noise', 'trial', 'N', 'statistic', 'pvalue']
# names_adj_kcipt_chaotic = ['independent', 'gamma', 'trial', 'N', 'statistic', 'pvalue', 'B']
# names_adj_kcipt_postnonlinear = ['independent', 'noise', 'trial', 'N', 'statistic', 'pvalue', 'B']
names_pykcipt_chaotic = ['independent', 'gamma', 'trial', 'N', 'statistic', 'pvalue', 'B']
names_pykcipt_postnonlinear = ['independent', 'noise', 'trial', 'N', 'statistic', 'pvalue', 'B']

names = {('CHSIC', 'chaotic'): names_chsic_chaotic,
         ('CHSIC', 'postnonlinear'): names_chsic_postnonlinear,
         ('KCIT', 'chaotic'): names_kcit_chaotic,
         ('KCIT', 'postnonlinear'): names_kcit_postnonlinear,
         ('SDCIT', 'chaotic'): names_lee_chaotic,
         ('JACK', 'chaotic'): names_lee_chaotic,
         ('SDCIT', 'postnonlinear'): names_lee_postnonlinear,
         ('JACK', 'postnonlinear'): names_lee_postnonlinear,
         # ('adj_kcipt', 'chaotic'): names_adj_kcipt_chaotic,
         # ('adj_kcipt', 'postnonlinear'): names_adj_kcipt_postnonlinear,
         ('KCIPT', 'chaotic'): names_pykcipt_chaotic,
         ('KCIPT', 'postnonlinear'): names_pykcipt_postnonlinear,
         }

pvalue_column = {'CHSIC': 'pvalue', 'KCIT': 'boot_p_value', 'SDCIT': 'pvalue', 'adj_kcipt': 'pvalue', 'KCIPT': 'pvalue', 'JACK': 'pvalue'}

cp = sns.color_palette('Set1', 5)
cps = {'CHSIC': 3, 'KCIT': 2, 'SDCIT': 0, 'JACK': 4, 'KCIPT': 1}
all_algos = ['SDCIT', 'KCIT', 'JACK']


def aupc(pvals):
    ttt = [(uniq_v, np.mean(pvals <= uniq_v)) for uniq_v in np.unique(pvals)]
    area = 0
    prev_x, prev_y = 0, 0
    for x, y in ttt:
        area += (x - prev_x) * prev_y
        prev_x, prev_y = x, y
    area += (1 - prev_x) * prev_y
    return area


def draw_aupc_chaotic():
    data = 'chaotic'

    aupc_data = []
    for algo in all_algos:
        df = pd.read_csv(algo.lower() + '_' + data + '.csv', names=names[(algo, data)])

        for group_key, group_df in pd.groupby(df, by=['gamma', 'independent', 'N']):
            group_key = (int(group_key[0] * 10) / 10, *group_key[1:])
            aupc_data.append([algo, *group_key, aupc(group_df[pvalue_column[algo]])])
            if group_key[0] == 0.0:
                print(algo, *group_key, np.mean(group_df[pvalue_column[algo]] <= 0.1))

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
    for k, gdf in pd.groupby(aupc_df, ['algorithm', 'N']):
        if k[1] == 400:
            plt.plot(gdf['gamma'], gdf['AUPC'], c=cp[cps[k[0]]], ls='-' if k[1] == 400 else ':', label=str(k[0]))
        else:
            plt.plot(gdf['gamma'], gdf['AUPC'], c=cp[cps[k[0]]], ls='-' if k[1] == 400 else ':', label='_nolegend_')
    handles, labels = plt.axes().get_legend_handles_labels()
    plt.axes().legend(handles[::-1], labels[::-1])
    # plt.legend()
    plt.axes().set_xlabel(r'$\gamma$')
    plt.axes().set_ylabel('Area Under Power Curve')
    plt.axes().set_ylim([0.45, 1.05])
    # plt.title('Chaotic series')
    sns.despine()
    plt.savefig('JACK_{}_aupc.pdf'.format(data), transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def draw_calib_chaotic():
    data = 'chaotic'
    calib_data = []
    for algo in all_algos:
        df = pd.read_csv(algo.lower() + '_' + data + '.csv', names=names[(algo, data)])
        for k, gdf in pd.groupby(df, by=['independent', 'gamma', 'N']):
            if float(k[0]) == 1:
                D, _ = scipy.stats.kstest(gdf[pvalue_column[algo]], 'uniform')
                calib_data.append([algo, float(k[1]), int(k[2]), D])

    df = pd.DataFrame(calib_data, columns=['algo', 'gamma', 'N', 'D'])
    df['gamma'] = df['gamma'].astype(float)
    df['N'] = df['N'].map(int)
    df['D'] = df['D'].astype(float)
    sns_setting()
    for k, gdf in pd.groupby(df, ['algo', 'N']):
        if k[1] == 400:
            plt.plot(gdf['gamma'], gdf['D'], c=cp[cps[k[0]]], ls='-' if k[1] == 400 else ':', label=str(k[0]))
        else:
            plt.plot(gdf['gamma'], gdf['D'], c=cp[cps[k[0]]], ls='-' if k[1] == 400 else ':', label='_nolegend_')
    # plt.legend()
    plt.axes().set_xlabel(r'$\gamma$')
    plt.axes().set_ylabel('KS test statistic')
    plt.axes().invert_yaxis()
    plt.axes().set_yticks([0.1, 0.2, 0.3])
    # plt.title('Chaotic series -- independent')
    sns.despine()
    plt.savefig('JACK_chaotic_calib.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def draw_Type_I_error_chaotic():
    data = 'chaotic'
    calib_data = []
    for algo in all_algos:
        df = pd.read_csv(algo.lower() + '_' + data + '.csv', names=names[(algo, data)])
        for k, gdf in pd.groupby(df, by=['independent', 'gamma', 'N']):
            if float(k[0]) == 1:
                calib_data.append([algo, float(k[1]), int(k[2]), np.mean(gdf[pvalue_column[algo]] <= 0.05)])
    df = pd.DataFrame(calib_data, columns=['algo', 'gamma', 'N', 'D'])
    df['gamma'] = df['gamma'].astype(float)
    df['N'] = df['N'].map(int)
    df['D'] = df['D'].astype(float)
    sns_setting()
    for k, gdf in pd.groupby(df, ['algo', 'N']):
        if k[1] == 400:
            plt.plot(gdf['gamma'], gdf['D'], c=cp[cps[k[0]]], ls='-' if k[1] == 400 else ':', label=str(k[0]))
        else:
            plt.plot(gdf['gamma'], gdf['D'], c=cp[cps[k[0]]], ls='-' if k[1] == 400 else ':', label='_nolegend_')
    # plt.legend()
    plt.axes().set_xlabel(r'$\gamma$')
    plt.axes().set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    plt.axes().set_ylabel('Type I error')
    sns.despine()
    plt.savefig('JACK_chaotic_type_I.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def draw_aupc_postnonlinear():
    data = 'postnonlinear'
    aupc_data = []
    for algo in all_algos:
        df = pd.read_csv(algo.lower() + '_' + data + '.csv', names=names[(algo, data)])

        for group_key, group_df in pd.groupby(df, by=['noise', 'independent', 'N']):
            group_key = (int(group_key[0] * 10) / 10, int(group_key[1]), int(group_key[2]))
            aupc_data.append([algo, *group_key, aupc(group_df[pvalue_column[algo]])])

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
    for k, gdf in pd.groupby(aupc_df, ['algorithm', 'N']):
        gdf = gdf[gdf['dimension'] <= 5]
        if k[1] == 400:
            plt.plot(gdf['dimension'], gdf['AUPC'], c=cp[cps[k[0]]], ls='-' if k[1] == 400 else ':', label=str(k[0]))
        else:
            plt.plot(gdf['dimension'], gdf['AUPC'], c=cp[cps[k[0]]], ls='-' if k[1] == 400 else ':', label='_nolegend_')
    # plt.legend()
    plt.axes().set_xlabel('dimension')
    plt.axes().set_ylabel('Area Under Power Curve')
    plt.axes().set_ylim([0.45, 1.05])
    # plt.title('Postnonlinear')
    sns.despine()
    plt.savefig('JACK_postnonlinear_aupc.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def draw_aupc_postnonlinear_highdim():
    data = 'postnonlinear'
    aupc_data = []
    for algo in all_algos:
        df = pd.read_csv(algo.lower() + '_' + data + '.csv', names=names[(algo, data)])

        for group_key, group_df in pd.groupby(df, by=['noise', 'independent', 'N']):
            group_key = (int(group_key[0] * 10) / 10, int(group_key[1]), int(group_key[2]))
            aupc_data.append([algo, *group_key, aupc(group_df[pvalue_column[algo]])])

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
    for k, gdf in pd.groupby(aupc_df, ['algorithm', 'N']):
        if k[1] == 400:
            plt.plot([int(v) for v in gdf['dimension']], gdf['AUPC'], c=cp[cps[k[0]]], ls='-' if k[1] == 400 else ':', label=str(k[0]))

    # plt.legend()
    plt.axes().set_xlabel('dimension')
    plt.axes().set_ylabel('Area Under Power Curve')
    plt.axes().set_ylim([0.95, 1.01])
    plt.axes().set_xscale('log')
    plt.xticks([1, 5, 10, 20, 50], [1, 5, 10, 20, 50])
    # plt.title('Postnonlinear')
    sns.despine()
    plt.savefig('JACK_postnonlinear_aupc_highdim.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def draw_calib_postnonlinear():
    data = 'postnonlinear'
    calib_data = []
    for algo in all_algos:
        df = pd.read_csv(algo.lower() + '_' + data + '.csv', names=names[(algo, data)])
        for k, gdf in pd.groupby(df, by=['independent', 'noise', 'N']):
            if float(k[0]) == 1:
                D, _ = scipy.stats.kstest(gdf[pvalue_column[algo]], 'uniform')
                calib_data.append([algo, float(k[1]), int(k[2]), D])

    df = pd.DataFrame(calib_data, columns=['algo', 'noise', 'N', 'D'])
    df['noise'] = df['noise'].map(int)
    df['dimension'] = (df['noise'] + 1).astype(int)
    df['N'] = df['N'].map(int)
    df['D'] = df['D'].astype(float)
    sns_setting()
    for k, gdf in pd.groupby(df, ['algo', 'N']):
        gdf = gdf[gdf['dimension'] <= 5]
        if k[1] == 400:
            plt.plot([int(v) for v in gdf['dimension']], gdf['D'], c=cp[cps[k[0]]], ls='-' if k[1] == 400 else ':', label=str(k[0]))
        else:
            plt.plot([int(v) for v in gdf['dimension']], gdf['D'], c=cp[cps[k[0]]], ls='-' if k[1] == 400 else ':', label='_nolegend_')
    # plt.legend()
    plt.axes().set_xlabel('dimension')
    plt.axes().set_ylabel('KS test statistic')
    plt.axes().invert_yaxis()
    plt.axes().set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    # plt.title('Postnonlinear')
    sns.despine()
    plt.savefig('JACK_postnonlinear_calib.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
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
        df = pd.read_csv(algo.lower() + '_' + data + '.csv', names=names[(algo, data)])
        for k, gdf in pd.groupby(df, by=['independent', 'noise', 'N']):
            if float(k[0]) == 1 and k[2] == 400:
                dd, _ = scipy.stats.kstest(gdf[pvalue_column[algo]], 'uniform')
                calib_data.append([algo, float(k[1]), int(k[2]), dd])

    df = pd.DataFrame(calib_data, columns=['algo', 'noise', 'N', 'D'])
    df['noise'] = df['noise'].map(int)
    df['dimension'] = (df['noise'] + 1).astype(int)
    df['N'] = df['N'].map(int)
    df['D'] = df['D'].astype(float)
    sns_setting()
    for k, gdf in pd.groupby(df, ['algo', 'N']):
        if k[1] == 400:
            plt.plot(gdf['dimension'], gdf['D'], c=cp[cps[k[0]]], ls='-' if k[1] == 400 else ':', label=str(k[0]))
        else:
            plt.plot(gdf['dimension'], gdf['D'], c=cp[cps[k[0]]], ls='-' if k[1] == 400 else ':', label='_nolegend_')
    # plt.legend()
    plt.axes().set_xlabel('dimension')
    plt.axes().set_ylabel('KS test statistic')
    plt.axes().set_xscale('log')
    plt.axes().invert_yaxis()
    plt.xticks([1, 5, 10, 20, 50], [1, 5, 10, 20, 50])
    plt.axes().set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    # plt.title('Postnonlinear')
    sns.despine()
    plt.savefig('JACK_postnonlinear_calib_highdim.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def draw_type_I_postnonlinear_highdim():
    data = 'postnonlinear'
    calib_data = []
    for algo in all_algos:
        df = pd.read_csv(algo.lower() + '_' + data + '.csv', names=names[(algo, data)])
        for k, gdf in pd.groupby(df, by=['independent', 'noise', 'N']):
            if float(k[0]) == 1 and k[2] == 400:
                dd = np.mean(gdf[pvalue_column[algo]] <= 0.05)
                calib_data.append([algo, float(k[1]), int(k[2]), dd])
    df = pd.DataFrame(calib_data, columns=['algo', 'noise', 'N', 'D'])
    df['noise'] = df['noise'].map(int)
    df['dimension'] = (df['noise'] + 1).astype(int)
    df['N'] = df['N'].map(int)
    df['D'] = df['D'].astype(float)
    sns_setting()
    for k, gdf in pd.groupby(df, ['algo', 'N']):
        if k[1] == 400:
            plt.plot(gdf['dimension'], gdf['D'], c=cp[cps[k[0]]], ls='-' if k[1] == 400 else ':', label=str(k[0]))
        else:
            plt.plot(gdf['dimension'], gdf['D'], c=cp[cps[k[0]]], ls='-' if k[1] == 400 else ':', label='_nolegend_')
    # plt.legend()
    plt.axes().set_xlabel('dimension')
    # plt.axes().set_ylabel('Type I error')
    plt.axes().set_xscale('log')
    plt.xticks([1, 5, 10, 20, 50], [1, 5, 10, 20, 50])
    plt.axes().set_ylim([0.0, 0.2])
    handles, labels = plt.axes().get_legend_handles_labels()
    plt.axes().legend(handles[::-1], labels[::-1])
    # plt.title('Postnonlinear')
    sns.despine()
    plt.savefig('JACK_postnonlinear_type_I_highdim.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def report_some():
    data = 'chaotic'

    aupc_data = []
    for algo in all_algos:
        df = pd.read_csv(algo.lower() + '_' + data + '.csv', names=names[(algo, data)])

        for group_key, group_df in pd.groupby(df, by=['gamma', 'independent', 'N']):
            group_key = (int(group_key[0] * 10) / 10, *group_key[1:])
            aupc_data.append([algo, *group_key, aupc(group_df[pvalue_column[algo]])])
    for x in aupc_data:
        print(x)


def report_other():
    data = 'postnonlinear'
    aupc_data = []
    for algo in all_algos:
        df = pd.read_csv(algo.lower() + '_' + data + '.csv', names=names[(algo, data)])

        for group_key, group_df in pd.groupby(df, by=['noise', 'independent', 'N']):
            group_key = (int(group_key[0] * 10) / 10, int(group_key[1]), int(group_key[2]))
            aupc_data.append([algo, *group_key, aupc(group_df[pvalue_column[algo]])])
    for x in aupc_data:
        print(x)


if __name__ == '__main__':
    draw_Type_I_error_chaotic()
    # report_some()
    # report_other()
    # draw_Type_I_error_chaotic()
    # draw_type_I_postnonlinear_highdim()
    draw_aupc_chaotic()
    draw_calib_chaotic()
    # # #
    draw_aupc_postnonlinear()
    draw_calib_postnonlinear()
    # # #
    draw_aupc_postnonlinear_highdim()
    draw_calib_postnonlinear_highdim()
