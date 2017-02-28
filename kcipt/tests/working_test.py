import numpy as np
from tqdm import trange

from kcipt.algo import c_adj_KCIPT, adj_KCIPT, one_shot_KCIPT, c_KCIPT
from kcipt.utils import data_gen_one, K2D

if __name__ == '__main__':
    cnt = 0
    B = 100
    if False:
        with open('testout22.csv', 'a') as f:
            for i in trange(2000):
                kx, ky, kz, _, _, _ = data_gen_one(50, i, 0.0)
                np.random.seed(i)
                DZ = K2D(kz)
                _, ms, _, _ = c_KCIPT(kx, ky, kz, DZ, B, 0, 0, n_jobs=32)
                _, m3, _, _ = c_adj_KCIPT(kx, ky, kz, DZ, B, 0, 0, False, n_jobs=32)
                _, m1, _, _ = c_adj_KCIPT(kx, ky, kz, DZ, B, 0, 0, n_jobs=32)
                _, m2, _, _ = adj_KCIPT(kx, ky, kz, B, 0, 0)
                mmd3 = one_shot_KCIPT(kx, ky, kz, with_null=False)  # not twiced!!!!
                print(np.mean(ms), m3, m1, m2, mmd3, sep=',', file=f)
    else:
        kx, ky, kz, _, _, _ = data_gen_one(50, 0, 0.0)
        _, _, _, onull = c_adj_KCIPT(kx, ky, kz, K2D(kz), 100, 10000, 10000)
        import seaborn as sns
        import pandas as pd

        data = pd.read_csv('testout22.csv', names=['KCIPT', 'ADJ-ONCE', 'DOUBLE-ADJ', 'PY-DOUBLE-ADJ', 'ONESHOT'])

        for i in range(4):
            palette = sns.color_palette('Set1', 5)
            sns.set(palette=palette)
            if i >= 0:
                sns.distplot(data['KCIPT'], hist=False, label='KCIPT')
            if i >= 1:
                sns.distplot(data['ADJ-ONCE'], hist=False, label='ADJ-ONCE')
            if i >= 2:
                sns.distplot(data['DOUBLE-ADJ'], hist=False, label='DOUBLE-ADJ')
            if i >= 3:
                sns.distplot(data['ONESHOT'], hist=False, label='ONESHOT')
            sns.distplot(onull, hist=False, label='null', color=palette[4])

            if i >= 0:
                sns.plt.plot([np.median(data['KCIPT']), np.median(data['KCIPT'])], [0, 50], c=palette[0], alpha=0.5)
            if i >= 1:
                sns.plt.plot([np.median(data['ADJ-ONCE']), np.median(data['ADJ-ONCE'])], [0, 50], c=palette[1], alpha=0.5)
            if i >= 2:
                sns.plt.plot([np.median(data['DOUBLE-ADJ']), np.median(data['DOUBLE-ADJ'])], [0, 50], c=palette[2], alpha=0.5)
            if i >= 3:
                sns.plt.plot([np.median(data['ONESHOT']), np.median(data['ONESHOT'])], [0, 50], c=palette[3], alpha=0.5)

            sns.plt.legend()
            sns.plt.axes().set_xlim([-0.015, 0.015])
            sns.plt.axes().set_ylim([-5, 750])
            sns.plt.axes().set_xlabel('MMD')
            sns.plt.axes().set_ylabel('density')
            # sns.plt.title('data generated from null hypothesis, different MMD estimates plotted')
            sns.plt.title('')
            sns.plt.savefig('testout22_{}.pdf'.format(i))
            sns.plt.close()
