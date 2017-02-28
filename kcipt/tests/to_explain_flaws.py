import os
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PyPDF2 import PdfFileMerger
from tqdm import tqdm
from tqdm import trange

from kcipt.algo import c_KCIPT
from kcipt.utils import data_gen_one, K2D


def append_pdf(input, output):
    [output.addPage(input.getPage(page_num)) for page_num in range(input.numPages)]


if __name__ == '__main__':
    file_names_withinner = list()
    file_names_noinner = list()

    cnt = 0

    fullB = 1000
    fullb = 5000
    kx, ky, kz, _, _, _ = data_gen_one(50, 0, 0.0)
    DZ = K2D(kz)
    _, ms, all_inner, outer = c_KCIPT(kx, ky, kz, DZ, fullB, fullb, 0, n_jobs=32)
    all_inner = all_inner.reshape((fullB, fullb))
    outer_null = None
    palette = sns.color_palette('Set1', 4)
    sns.set(palette=palette)
    for B in trange(1, fullB + 1):
        mmds = ms[0:B]
        inners = all_inner[0:B, :]
        if outer_null is None:
            outer_null = np.array(inners[0, :], copy=True)
        else:
            outer_null = outer_null * ((B - 1) / B) + np.random.choice(all_inner[B - 1, :], fullb) / B

        if B <= 100:
            if len(mmds) > 1:
                sns.distplot(mmds, color='k', rug=True, kde=False, hist=False, rug_kws={'alpha': 0.4})
            plt.plot([np.mean(mmds), np.mean(mmds)], [0, 1 * np.sqrt(B)], c=palette[0], label='test statistic')
            for i0, inner in enumerate(inners):
                if i0 == 0:
                    sns.distplot(inner, hist=False, label='inner null', color=palette[1], kde_kws={'lw': 0.5, 'alpha': 0.3})
                else:
                    sns.distplot(inner, hist=False, color=palette[1], kde_kws={'lw': 0.5, 'alpha': 0.3})
            sns.distplot(outer_null, hist=False, label='bootstrap null', color=palette[2])

            plt.legend()
            plt.axes().set_xlim([-0.075, 0.17])
            plt.axes().set_xlabel('MMD')
            plt.axes().set_ylabel('density')
            plt.title('data generated from null hypothesis, B = {}'.format(B))
            tempf = NamedTemporaryFile(suffix='.pdf', delete=False)
            plt.savefig(tempf.name)
            plt.close()
            file_names_withinner.append(tempf.name)

        if len(mmds) > 1:
            sns.distplot(mmds, color='k', rug=True, kde=False, hist=False, rug_kws={'alpha': 0.4})
        plt.plot([np.mean(mmds), np.mean(mmds)], [0, 1 * np.sqrt(B)], c=palette[0], label='test statistic')
        sns.distplot(outer_null, hist=False, label='bootstrap null', color=palette[2])

        plt.legend()
        plt.axes().set_xlim([-0.15 / np.sqrt(B), 0.15 / np.sqrt(B)])
        plt.axes().set_xlabel('MMD')
        plt.axes().set_ylabel('density')
        plt.title('data generated from null hypothesis, B = {}'.format(B))
        tempf = NamedTemporaryFile(suffix='.pdf', delete=False)
        plt.savefig(tempf.name)
        plt.close()
        file_names_noinner.append(tempf.name)

    output_withinner = PdfFileMerger()
    for f in tqdm(file_names_withinner, desc='writing with inner'):
        output_withinner.append(f)
    output_withinner.write(open("with_inner.pdf", "wb"))
    output_withinner.close()

    output_noinner = PdfFileMerger()
    for f in tqdm(file_names_noinner, desc='writing no inner'):
        output_noinner.append(f)
    output_noinner.write(open("no_inner.pdf", "wb"))
    output_noinner.close()

    print('cleaning up')
    dummy = [os.remove(f) for f in file_names_noinner]
    dummy = [os.remove(f) for f in file_names_withinner]
