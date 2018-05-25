from distutils.core import setup
from distutils.extension import Extension
from sys import platform

import numpy
from Cython.Distutils import build_ext

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
                          extra_compile_args=["-std=c++11", "-stdlib=libc++", "-mmacosx-version-min=10.7"] if platform == "darwin" else ["-std=c++11"],
                          extra_link_args=["-std=c++11", "-stdlib=libc++", "-mmacosx-version-min=10.7"] if platform == "darwin" else ["-std=c++11"]
                          )

setup(
    name='SDCIT',
    packages=['sdcit', 'sdcit.cython_impl'],
    version='1.1.1',
    description='Self-Discrepancy Conditional Independence Test',
    author='Sanghack Lee',
    author_email='sanghack.lee@gmail.com',
    url='https://github.com/sanghack81/SDCIT',
    keywords=['independence test', 'conditional independence', 'machine learning', 'statistical test'],
    classifiers=[],
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        new_extension
    ], requires=['numpy'],
)
# python setup.py build_ext --inplace
# pip install -e .
