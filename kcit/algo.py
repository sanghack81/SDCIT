import os


def kcit(x, y, z, seed=None, mateng=None, installed_at='~/Dropbox/research/2014 rcm/workspace/python/KCIPT2017/kcit'):
    import matlab.engine

    if mateng is None:
        mateng = matlab.engine.start_matlab()
        dir_at = os.path.expanduser(installed_at)
        mateng.addpath(mateng.genpath(dir_at))

        if seed is not None:
            mateng.RandStream.setGlobalStream(mateng.RandStream('mcg16807', 'Seed', seed))
        statistic, v2, boot_p_value, v3, appr_p_value = mateng.CInd_test_new_withGP(x, y, z, 0.01, 0, nargout=5)
        mateng.quit()

        return statistic, v2, boot_p_value, v3, appr_p_value
    else:
        if seed is not None:
            mateng.RandStream.setGlobalStream(mateng.RandStream('mcg16807', 'Seed', seed))

        statistic, v2, boot_p_value, v3, appr_p_value = mateng.CInd_test_new_withGP(x, y, z, 0.01, 0, nargout=5)
        return statistic, v2, boot_p_value, v3, appr_p_value


def kcit_lee(Kx, Ky, Kz, seed=None, mateng=None, installed_at='~/Dropbox/research/2014 rcm/workspace/python/KCIPT2017/kcit'):
    import matlab.engine

    if mateng is None:
        mateng = matlab.engine.start_matlab()
        dir_at = os.path.expanduser(installed_at)
        mateng.addpath(mateng.genpath(dir_at))

        if seed is not None:
            mateng.RandStream.setGlobalStream(mateng.RandStream('mcg16807', 'Seed', seed))
        _, _, _, _, appr_p_value = mateng.CInd_test_new_withGP_Lee(Kx, Ky, Kz, 0.01, nargout=5)
        mateng.quit()

        return appr_p_value
    else:
        if seed is not None:
            mateng.RandStream.setGlobalStream(mateng.RandStream('mcg16807', 'Seed', seed))

        _, _, _, _, appr_p_value = mateng.CInd_test_new_withGP_Lee(Kx, Ky, Kz, 0.01, nargout=5)
        return appr_p_value
