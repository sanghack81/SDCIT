from setuptools import setup, Extension
from sys import platform

import numpy
from Cython.Build import cythonize

blossom_v_dir = 'blossom5/'

new_extension = Extension("sdcit.cython_impl.cy_sdcit",
                          sources=[
                              'sdcit/cython_impl/cy_sdcit.pyx',
                              'sdcit/cython_impl/KCIPT.cpp',
                              'sdcit/cython_impl/SDCIT.cpp',
                              'sdcit/cython_impl/HSIC.cpp',
                              'sdcit/cython_impl/permutation.cpp',
                              blossom_v_dir + 'GEOM/GPMinit.cpp',
                              blossom_v_dir + 'GEOM/GPMinterface.cpp',
                              blossom_v_dir + 'GEOM/GPMkdtree.cpp',
                              blossom_v_dir + 'GEOM/GPMmain.cpp',
                              blossom_v_dir + 'MinCost/MinCost.cpp',
                              blossom_v_dir + 'misc.cpp',
                              blossom_v_dir + 'PMduals.cpp',
                              blossom_v_dir + 'PMexpand.cpp',
                              blossom_v_dir + 'PMinit.cpp',
                              blossom_v_dir + 'PMinterface.cpp',
                              blossom_v_dir + 'PMmain.cpp',
                              blossom_v_dir + 'PMrepair.cpp',
                              blossom_v_dir + 'PMshrink.cpp',
                          ],
                          language="c++",
                          include_dirs=[numpy.get_include(), 'sdcit/cython_impl', blossom_v_dir,
                                        blossom_v_dir + 'MinCost', blossom_v_dir + 'GEOM'],
                          # extra_compile_args=["-std=c++11", "-stdlib=libc++", "-mmacosx-version-min=10.7"] if platform == "darwin" else ["-std=c++11"],
                          # extra_link_args=["-std=c++11", "-stdlib=libc++", "-mmacosx-version-min=10.7"] if platform == "darwin" else ["-std=c++11"]
                          extra_compile_args=["-std=c++17", "-mmacosx-version-min=10.9"] if platform == "darwin" else ["-std=c++11"],
                          extra_link_args=["-std=c++17", "-mmacosx-version-min=10.9"] if platform == "darwin" else ["-std=c++11"]
                          )

setup(
    name='SDCIT',
    packages=['sdcit', 'sdcit.cython_impl'],
    version='2.0.0',
    description='Self-Discrepancy Conditional Independence Test',
    author='Sanghack Lee',
    author_email='sanghack.lee@gmail.com',
    url='https://github.com/sanghack81/SDCIT',
    keywords=['independence test', 'conditional independence', 'machine learning', 'statistical test'],
    classifiers=[],
    # force=True: always regenerate the .cpp from the .pyx, so a stale Cython
    # output (e.g. generated against a different NumPy version) is never reused.
    ext_modules=cythonize([new_extension], force=True, language_level='3'),
    python_requires='>=3.9',
    install_requires=['numpy', 'scipy>=1.5', 'scikit-learn'],
    extras_require={
        # GP-based tests (KCIT, FCIT, GP residualization) require the gpflow 2.x API.
        'gp': ['gpflow>=2.0', 'tensorflow>=2.0'],
    },
)
# python setup.py build_ext --inplace
# pip install -e .
