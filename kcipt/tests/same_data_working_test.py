import numpy as np
from tqdm import trange

from kcipt.algo import c_adj_KCIPT, adj_KCIPT, one_shot_KCIPT, c_KCIPT
from kcipt.utils import data_gen_one, K2D

if __name__ == '__main__':
    cnt = 0
    B = 100
    if False:
        kx, ky, kz, _, _, _ = data_gen_one(50, 0, 0.0)
        DZ = K2D(kz)
        with open('testout33.csv', 'a') as f:
            for i in trange(1000):
                np.random.seed(i)
                _, ms, _, _ = c_KCIPT(kx, ky, kz, DZ, B, 0, 0, n_jobs=5)
                _, m3, _, _ = c_adj_KCIPT(kx, ky, kz, DZ, B, 0, 0, False, n_jobs=5)
                _, m1, _, _ = c_adj_KCIPT(kx, ky, kz, DZ, B, 0, 0, n_jobs=5)
                _, m2, _, _ = adj_KCIPT(kx, ky, kz, B, 0, 0)
                mmd3 = one_shot_KCIPT(kx, ky, kz, with_null=False)  # not twiced!!!!
                print(np.mean(ms), m3, m1, m2, mmd3, sep=',', file=f)
    else:
        import seaborn as sns
        import pandas as pd

        palette = sns.color_palette('Set1', 5)
        sns.set(palette=palette)
        data = pd.read_csv('testout33.csv', names=['KCIPT', 'ADJ-ONCE', 'DOUBLE-ADJ', 'PY-DOUBLE-ADJ', 'ONESHOT'])
        sns.distplot(data['KCIPT'], hist=False, label='KCIPT')
        sns.distplot(data['ADJ-ONCE'], hist=False, label='ADJ-ONCE')
        sns.distplot(data['DOUBLE-ADJ'], hist=False, label='DOUBLE-ADJ')
        # sns.distplot(data['PY-DOUBLE-ADJ'], hist=False, label='PY-DOUBLE-ADJ')
        # sns.distplot(data['ONESHOT'], hist=False, label='ONESHOT')

        sns.plt.plot([np.mean(data['KCIPT']), np.mean(data['KCIPT'])], [0, 50], c=palette[0])
        sns.plt.plot([np.mean(data['ADJ-ONCE']), np.mean(data['ADJ-ONCE'])], [0, 50], c=palette[1])
        sns.plt.plot([np.mean(data['DOUBLE-ADJ']), np.mean(data['DOUBLE-ADJ'])], [0, 50], c=palette[2])
        # sns.plt.plot([np.mean(data['PY-DOUBLE-ADJ']), np.mean(data['PY-DOUBLE-ADJ'])], [0, 50], c=palette[3], alpha=0.5)
        sns.plt.plot([np.mean(data['ONESHOT']), np.mean(data['ONESHOT'])], [0, 50], c=palette[4],label='ONESHOT')

        sns.plt.legend()
        sns.plt.axes().set_xlim([-0.015, 0.015])
        sns.plt.axes().set_ylim([-5, 750])
        sns.plt.axes().set_xlabel('MMD')
        sns.plt.axes().set_ylabel('density')
        sns.plt.title('data generated from null hypothesis, different MMD estimates plotted')
        sns.plt.savefig('testout33.pdf')
        sns.plt.close()

