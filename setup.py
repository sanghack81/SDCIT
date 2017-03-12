from distutils.core import setup
from distutils.extension import Extension
from sys import platform

import numpy
from Cython.Distutils import build_ext

blossom_v_extension = Extension("kcipt.blossom_v.cy_blossom_v",
                                sources=['kcipt/blossom_v/' + f for f in [
                                    'cy_blossom_v.pyx',
                                    'c_cy_blossom_v.cpp',
                                    'PMduals.cpp',
                                    'PMexpand.cpp',
                                    'PMinit.cpp',
                                    'PMinterface.cpp',
                                    'PMmain.cpp',
                                    'PMrepair.cpp',
                                    'PMshrink.cpp',
                                    'example_lee.cpp',
                                    'misc.cpp',
                                    'MinCost/MinCost.cpp',
                                    'GEOM/GPMinit.cpp',
                                    'GEOM/GPMinterface.cpp',
                                    'GEOM/GPMkdtree.cpp',
                                    'GEOM/GPMmain.cpp'
                                ]],
                                language="c++",
                                include_dirs=[numpy.get_include(), 'kcipt/blossom_v/MinCost',
                                              'kcipt/blossom_v/GEOM'],
                                extra_compile_args=["-std=c++11"],
                                extra_link_args=["-std=c++11"]
                                )

new_extension = Extension("kcipt.cython_impl.cy_kcipt",
                          sources=[
                              'kcipt/cython_impl/cy_kcipt.pyx',
                              'kcipt/cython_impl/KCIPT.cpp',
                              'kcipt/cython_impl/permutation.cpp',
                              'kcipt/blossom_v/GEOM/GPMinit.cpp',
                              'kcipt/blossom_v/GEOM/GPMinterface.cpp',
                              'kcipt/blossom_v/GEOM/GPMkdtree.cpp',
                              'kcipt/blossom_v/GEOM/GPMmain.cpp',
                              'kcipt/blossom_v/MinCost/MinCost.cpp',
                              'kcipt/blossom_v/misc.cpp',
                              'kcipt/blossom_v/PMduals.cpp',
                              'kcipt/blossom_v/PMexpand.cpp',
                              'kcipt/blossom_v/PMinit.cpp',
                              'kcipt/blossom_v/PMinterface.cpp',
                              'kcipt/blossom_v/PMmain.cpp',
                              'kcipt/blossom_v/PMrepair.cpp',
                              'kcipt/blossom_v/PMshrink.cpp',
                          ],
                          language="c++",
                          include_dirs=[numpy.get_include(), 'kcipt/cython_impl', 'kcipt/blossom_v',
                                        'kcipt/blossom_v/MinCost', 'kcipt/blossom_v/GEOM'],
                          extra_compile_args=["-std=c++11", "-stdlib=libc++", "-mmacosx-version-min=10.7"] if platform == "darwin" else ["-std=c++11"],
                          extra_link_args=["-std=c++11", "-stdlib=libc++", "-mmacosx-version-min=10.7"] if platform == "darwin" else ["-std=c++11"]
                          )

setup(
    name='SDCIT',
    packages=['kcipt'],
    version='0.1',
    description='Self-Discrepancy Conditional Independence Test',
    author='Sanghack Lee',
    author_email='sanghack.lee@gmail.com',
    url='https://github.com/sanghack81/SDCIT',
    keywords=['independence test', 'conditional independence', 'machine learning', 'statistical test'],
    classifiers=[],

    cmdclass={'build_ext': build_ext},
    ext_modules=[
        blossom_v_extension,
        new_extension
    ],

)
# source activate tensorflow
# python3 setup.py build_ext --inplace
# pip install -e .

# ~/anaconda/bin/python3 setup.py build_ext --inplace
# ~/anaconda/bin/pip install -e .
