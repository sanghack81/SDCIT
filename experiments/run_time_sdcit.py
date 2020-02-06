import time
from os.path import exists

import pandas as pd
import scipy.io

from experiments.exp_setup import *
from sdcit.sdcit_mod import c_SDCIT
from sdcit.utils import rbf_kernel_median, K2D

if __name__ == '__main__':
    # experiments
    fname = SDCIT_RESULT_DIR + '/c_sdcit_time.csv'
    independent = 1
    if not exists(fname):
        with open(fname, 'a') as f:
            for N in [200, 400]:
                for b in [500, 1000]:
                    for trial in range(300):
                        mat_load = scipy.io.loadmat(os.path.expanduser(SDCIT_DATA_DIR + '/{}_{}_{}_{}_chaotic.mat'.format('0.0', trial, independent, N)), squeeze_me=True, struct_as_record=False)
                        data = mat_load['data']
                        X = data.Yt1
                        Y = data.Xt
                        Z = data.Yt[:, 0: 2]

                        start = time.time()

                        kkk = rbf_kernel_median(X, Y, Z)
                        Dz = K2D(kkk[-1])
                        c_SDCIT(*kkk, Dz=Dz, size_of_null_sample=b, seed=trial, to_shuffle=False)

                        endtime = time.time()
                        print(endtime - start, trial, N, b, file=f, sep=',', flush=True)

    # analyze
    df = pd.read_csv(fname, names=['time', 'trial', 'N', 'b'])
    for key, gdf in df.groupby(by=['N', 'b']):
        print('{}: {:.2f} +- {:.2f}'.format(key, gdf['time'].mean(), gdf['time'].std()))

    df = pd.read_csv(fname, names=['time', 'trial', 'N', 'b'])
    for key, gdf in df.groupby(by=['N', 'b']):
        print('{}: {:.5f} +- {:.5f}'.format(key, gdf['time'].mean(), gdf['time'].std()))

    if exists(SDCIT_RESULT_DIR + '/kcit_chaotic_timing.csv'):
        print()
        print('KCIT')
        df_kcit = pd.read_csv(SDCIT_RESULT_DIR + '/kcit_chaotic_timing.csv', names=['independent', 'gamma', 'noise', 'trial', 'N', 'runtime', 'statistic', 'boot_p_value', 'appr_p_value'])
        df_kcit = df_kcit[df_kcit['independent'] == independent]
        df_kcit = df_kcit[df_kcit['gamma'] == 0.0]
        for key, gdf in df_kcit.groupby(by=['N']):
            assert len(gdf) == 300
            print('{}: {:.2f} +- {:.2f}'.format(key, gdf['runtime'].mean(), gdf['runtime'].std()))

        for key, gdf in df_kcit.groupby(by=['N']):
            assert len(gdf) == 300
            print('{}: {:.5f} +- {:.5f}'.format(key, gdf['runtime'].mean(), gdf['runtime'].std()))
