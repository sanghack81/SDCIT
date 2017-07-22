#!/bin/bash
BLOSSOM_V_URL=http://pub.ist.ac.at/~vnk/software/blossom5-v2.05.src.tar.gz
BLOSSOM_V_ZIPFILE=blossom5-v2.05.src.tar.gz
wget ${BLOSSOM_V_URL} -O ${BLOSSOM_V_ZIPFILE}
tar xzvf ${BLOSSOM_V_ZIPFILE}
mv blossom5-v2.05.src blossom5
