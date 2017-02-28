import multiprocessing
import os
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PyPDF2 import PdfFileMerger
from tqdm import tqdm

from kcipt.algo import c_KCIPT
from kcipt.utils import data_gen_one, K2D

if __name__ == '__main__':
    file_names_withinner = list()
    file_names_noinner = list()

    kx, ky, kz, _, _, _ = data_gen_one(50, 0, 0.0)
    DZ = K2D(kz)
    b, M = 1000, 20000
    for B in [20, 100, 500, 2500]:
        for trial in range(1, 25 + 1):
            print(B, trial)
            _, mmds, inners, outer_null = c_KCIPT(kx, ky, kz, DZ, B, b, M, n_jobs=multiprocessing.cpu_count())

            palette = sns.color_palette('Set1', 4)
            sns.set(palette=palette)

            if len(mmds) > 1:
                sns.distplot(mmds, color='k', rug=True, kde=False, hist=False, rug_kws={'alpha': np.log(10) / np.log(B)})
            plt.plot([np.mean(mmds), np.mean(mmds)], [0, 1 * np.sqrt(B)], c=palette[0], label='test statistic')
            for i0, inner in enumerate(inners):
                if i0 == 0:
                    sns.distplot(inner, hist=False, label='inner null*', color=palette[1], kde_kws={'lw': 0.5, 'alpha': 0.3})
                else:
                    sns.distplot(inner, hist=False, color=palette[1], kde_kws={'lw': 0.5, 'alpha': 0.3})
                if i0 > 20:
                    break
            sns.distplot(outer_null, hist=False, label='bootstrap null', color=palette[2])

            plt.legend()
            plt.axes().set_xlim([-0.075, 0.17])
            plt.axes().set_xlabel('MMD')
            plt.axes().set_ylabel('density')
            plt.title('null hypothesis, B = {}, trial = {}'.format(B, trial))
            tempf = NamedTemporaryFile(suffix='.pdf', delete=False)
            plt.savefig(tempf.name)
            plt.close()
            file_names_withinner.append(tempf.name)

            if len(mmds) > 1:
                sns.distplot(mmds, color='k', rug=True, kde=False, hist=False, rug_kws={'alpha': np.log(10) / np.log(B)})
            plt.plot([np.mean(mmds), np.mean(mmds)], [0, 1 * np.sqrt(B)], c=palette[0], label='test statistic')
            sns.distplot(outer_null, hist=False, label='bootstrap null', color=palette[2])

            plt.legend()
            plt.axes().set_xlim([-0.15 / np.sqrt(B), 0.15 / np.sqrt(B)])
            plt.axes().set_xlabel('MMD')
            plt.axes().set_ylabel('density')
            plt.title('null hypothesis, B = {}, trial = {}'.format(B, trial))
            tempf = NamedTemporaryFile(suffix='.pdf', delete=False)
            plt.savefig(tempf.name)
            plt.close()
            file_names_noinner.append(tempf.name)

    output_withinner = PdfFileMerger()
    for f in tqdm(file_names_withinner, desc='writing with inner'):
        output_withinner.append(f)
    output_withinner.write(open("with_inner_multi_trial.pdf", "wb"))
    output_withinner.close()

    output_noinner = PdfFileMerger()
    for f in tqdm(file_names_noinner, desc='writing no inner'):
        output_noinner.append(f)
    output_noinner.write(open("no_inner_multi_trial.pdf", "wb"))
    output_noinner.close()

    print('cleaning up')
    dummy = [os.remove(f) for f in file_names_noinner]
    dummy = [os.remove(f) for f in file_names_withinner]
