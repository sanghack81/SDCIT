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

names = {('chsic', 'chaotic'): names_chsic_chaotic,
         ('chsic', 'postnonlinear'): names_chsic_postnonlinear,
         ('kcit', 'chaotic'): names_kcit_chaotic,
         ('kcit', 'postnonlinear'): names_kcit_postnonlinear,
         ('lee', 'chaotic'): names_lee_chaotic,
         ('lee', 'postnonlinear'): names_lee_postnonlinear}

pvalue_column = {'chsic': 'pvalue', 'kcit': 'boot_p_value', 'lee': 'pvalue'}


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
    cp = sns.color_palette('Set1', 3)
    cps = {'chsic': 0, 'kcit': 1, 'lee': 2}
    aupc_data = []
    for algo in ['chsic', 'kcit', 'lee']:
        df = pd.read_csv(algo + '_' + data + '.csv', names=names[(algo, data)])

        for group_key, group_df in pd.groupby(df, by=['gamma', 'independent', 'N']):
            group_key = (int(group_key[0] * 10) / 10, *group_key[1:])
            aupc_data.append([algo, *group_key, aupc(group_df[pvalue_column[algo]])])

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
    sns.set()
    for k, gdf in pd.groupby(aupc_df, ['algorithm', 'N']):
        plt.plot(gdf['gamma'], gdf['AUPC'], c=cp[cps[k[0]]], ls='-' if k[1] == 200 else ':', label=str(k[0]) + ' (' + str(k[1]) + ')')
    plt.legend()
    plt.axes().set_xlabel('gamma')
    plt.axes().set_ylabel('Area Under Power Curve')
    plt.axes().set_ylim([0, 1])
    plt.title('Chaotic series')
    plt.savefig('{}_aupc.pdf'.format(data))
    plt.close()


def draw_calib_chaotic():
    data = 'chaotic'
    cp = sns.color_palette('Set1', 3)
    cps = {'chsic': 0, 'kcit': 1, 'lee': 2}
    calib_data = []
    for algo in ['chsic', 'kcit', 'lee']:
        df = pd.read_csv(algo + '_' + data + '.csv', names=names[(algo, data)])
        for k, gdf in pd.groupby(df, by=['independent', 'gamma', 'N']):
            if float(k[0]) == 1:
                _, p = scipy.stats.kstest(gdf[pvalue_column[algo]], 'uniform')
                calib_data.append([algo, float(k[1]), int(k[2]), p])

    df = pd.DataFrame(calib_data, columns=['algo', 'gamma', 'N', 'p'])
    df['gamma'] = df['gamma'].astype(float)
    df['N'] = df['N'].map(int)
    df['p'] = df['p'].astype(float)
    sns.set()
    for k, gdf in pd.groupby(df, ['algo', 'N']):
        plt.plot(gdf['gamma'], gdf['p'], c=cp[cps[k[0]]], ls='-' if k[1] == 200 else ':', label=str(k[0]) + ' (' + str(k[1]) + ')')
    plt.legend()
    plt.axes().set_xlabel('gamma')
    plt.axes().set_ylabel('Kolmogorov p-value')
    plt.axes().set_yscale('log')
    plt.title('Chaotic series -- independent')
    plt.savefig('chaotic_calib.pdf')
    plt.close()


def draw_aupc_postnonlinear():
    data = 'postnonlinear'
    cp = sns.color_palette('Set1', 3)
    cps = {'chsic': 0, 'kcit': 1, 'lee': 2}
    aupc_data = []
    for algo in ['chsic', 'kcit', 'lee']:
        df = pd.read_csv(algo + '_' + data + '.csv', names=names[(algo, data)])

        for group_key, group_df in pd.groupby(df, by=['noise', 'independent', 'N']):
            group_key = (int(group_key[0] * 10) / 10, int(group_key[1]), int(group_key[2]))
            aupc_data.append([algo, *group_key, aupc(group_df[pvalue_column[algo]])])

    aupc_data = np.array(aupc_data)
    aupc_df = pd.DataFrame({'algorithm': [str(v) for v in aupc_data[:, 0]],
                            'noise': [int(float(v)) for v in aupc_data[:, 1]],
                            'independent': [int(v) for v in aupc_data[:, 2]],
                            'N': [int(v) for v in aupc_data[:, 3]],
                            'AUPC': [float(v) for v in aupc_data[:, 4]]})
    aupc_df['dimension'] = aupc_df['noise'] + 1

    aupc_df = aupc_df[aupc_df['independent'] == 0]
    aupc_df["algo-N"] = aupc_df["algorithm"].map(str) + aupc_df["N"].map(lambda xxx: ' (' + str(xxx) + ')')
    sns.set()
    for k, gdf in pd.groupby(aupc_df, ['algorithm', 'N']):
        gdf = gdf[gdf['dimension'] <= 5]
        plt.plot(gdf['dimension'], gdf['AUPC'], c=cp[cps[k[0]]], ls='-' if k[1] == 200 else ':', label=str(k[0]) + ' (' + str(k[1]) + ')')
    plt.legend()
    plt.axes().set_xlabel('dimension')
    plt.axes().set_ylabel('Area Under Power Curve')
    plt.axes().set_ylim([0, 1])
    plt.title('Postnonlinear')
    plt.savefig('postnonlinear_aupc.pdf')
    plt.close()


def draw_aupc_postnonlinear_highdim():
    data = 'postnonlinear'
    cp = sns.color_palette('Set1', 3)
    cps = {'chsic': 0, 'kcit': 1, 'lee': 2}
    aupc_data = []
    for algo in ['kcit', 'lee']:
        df = pd.read_csv(algo + '_' + data + '.csv', names=names[(algo, data)])

        for group_key, group_df in pd.groupby(df, by=['noise', 'independent', 'N']):
            group_key = (int(group_key[0] * 10) / 10, int(group_key[1]), int(group_key[2]))
            aupc_data.append([algo, *group_key, aupc(group_df[pvalue_column[algo]])])

    aupc_data = np.array(aupc_data)
    aupc_df = pd.DataFrame({'algorithm': [str(v) for v in aupc_data[:, 0]],
                            'noise': [int(float(v)) for v in aupc_data[:, 1]],
                            'independent': [int(v) for v in aupc_data[:, 2]],
                            'N': [int(v) for v in aupc_data[:, 3]],
                            'AUPC': [float(v) for v in aupc_data[:, 4]]})
    aupc_df['dimension'] = aupc_df['noise'] + 1

    aupc_df = aupc_df[aupc_df['independent'] == 0]
    aupc_df["algo-N"] = aupc_df["algorithm"].map(str) + aupc_df["N"].map(lambda xxx: ' (' + str(xxx) + ')')
    sns.set()
    for k, gdf in pd.groupby(aupc_df, ['algorithm', 'N']):
        if k[1] == 400:
            gdf = gdf[gdf['dimension'] > 5]
            plt.plot(gdf['dimension'], gdf['AUPC'], c=cp[cps[k[0]]], ls='-' if k[1] == 200 else ':', label=str(k[0]) + ' (' + str(k[1]) + ')')
            print(k, gdf['AUPC'])

    plt.legend()
    plt.axes().set_xlabel('dimension')
    plt.axes().set_ylabel('Area Under Power Curve')
    plt.axes().set_ylim([0, 1])
    plt.title('Postnonlinear')
    plt.savefig('postnonlinear_aupc_highdim.pdf')
    plt.close()



def draw_calib_postnonlinear():
    data = 'postnonlinear'
    cp = sns.color_palette('Set1', 3)
    cps = {'chsic': 0, 'kcit': 1, 'lee': 2}
    calib_data = []
    for algo in ['chsic', 'kcit', 'lee']:
        df = pd.read_csv(algo + '_' + data + '.csv', names=names[(algo, data)])
        for k, gdf in pd.groupby(df, by=['independent', 'noise', 'N']):
            if float(k[0]) == 1:
                _, p = scipy.stats.kstest(gdf[pvalue_column[algo]], 'uniform')
                calib_data.append([algo, float(k[1]), int(k[2]), p])

    df = pd.DataFrame(calib_data, columns=['algo', 'noise', 'N', 'p'])
    df['noise'] = df['noise'].map(int)
    df['dimension'] = df['noise'] + 1
    df['N'] = df['N'].map(int)
    df['p'] = df['p'].astype(float)
    sns.set()
    for k, gdf in pd.groupby(df, ['algo', 'N']):
        gdf = gdf[gdf['dimension'] <= 5]
        plt.plot(gdf['dimension'], gdf['p'], c=cp[cps[k[0]]], ls='-' if k[1] == 200 else ':', label=str(k[0]) + ' (' + str(k[1]) + ')')
    plt.legend()
    plt.axes().set_xlabel('dimension')
    plt.axes().set_ylabel('Kolmogorov p-value')
    plt.axes().set_yscale('log')
    plt.title('Postnonlinear')
    plt.savefig('postnonlinear_calib.pdf')
    plt.close()


def draw_calib_postnonlinear_highdim():
    data = 'postnonlinear'
    cp = sns.color_palette('Set1', 3)
    cps = {'chsic': 0, 'kcit': 1, 'lee': 2}
    calib_data = []
    for algo in ['kcit', 'lee']:
        df = pd.read_csv(algo + '_' + data + '.csv', names=names[(algo, data)])
        for k, gdf in pd.groupby(df, by=['independent', 'noise', 'N']):
            if float(k[0]) == 1 and k[2] == 400:
                _, p = scipy.stats.kstest(gdf[pvalue_column[algo]], 'uniform')
                calib_data.append([algo, float(k[1]), int(k[2]), p])

    df = pd.DataFrame(calib_data, columns=['algo', 'noise', 'N', 'p'])
    df['noise'] = df['noise'].map(int)
    df['dimension'] = df['noise'] + 1
    df['N'] = df['N'].map(int)
    df['p'] = df['p'].astype(float)
    sns.set()
    for k, gdf in pd.groupby(df, ['algo', 'N']):
        gdf = gdf[gdf['dimension'] > 5]
        plt.plot(gdf['dimension'], gdf['p'], c=cp[cps[k[0]]], ls='-' if k[1] == 200 else ':', label=str(k[0]) + ' (' + str(k[1]) + ')')
        print(k, gdf['p'])
    plt.legend()
    plt.axes().set_xlabel('dimension')
    plt.axes().set_ylabel('Kolmogorov p-value')
    plt.axes().set_yscale('log')
    plt.axes().set_xscale('log')
    plt.title('Postnonlinear')
    plt.savefig('postnonlinear_calib_highdim.pdf')
    plt.close()


if __name__ == '__main__':
    draw_aupc_postnonlinear_highdim()
    draw_calib_postnonlinear_highdim()
