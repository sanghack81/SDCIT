#!/bin/bash
BLOSSOM_V_URL=http://pub.ist.ac.at/~vnk/software/blossom5-v2.05.src.tar.gz
BLOSSOM_V_ZIPFILE=blossom5-v2.05.src.tar.gz
wget ${BLOSSOM_V_URL} -O ${BLOSSOM_V_ZIPFILE}
tar xzvf ${BLOSSOM_V_ZIPFILE}
mv blossom5-v2.05.src blossom5

python3 setup.py build_ext --inplace
pip install -e .

KCITURL=http://people.tuebingen.mpg.de/kzhang/KCI-test.zip
KCITZIPFILE=KCI-test.zip

wget ${KCITURL} -O ${KCITZIPFILE}
unzip ${KCITZIPFILE} algorithms/* gpml-matlab/*
mv gpml-matlab kcit
mv algorithms kcit
rmdir algorithms
cp kcit/CInd_test_new_withGP_Lee.m kcit/algorithms/
