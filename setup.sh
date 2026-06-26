#!/bin/bash
# Download and unpack Blossom-V (external C++ dependency) into ./blossom5/.
# Uses curl so it works on both macOS (no wget by default) and Linux.
set -e

BLOSSOM_V_URL=https://pub.ista.ac.at/~vnk/software/blossom5-v2.05.src.tar.gz
BLOSSOM_V_ZIPFILE=blossom5-v2.05.src.tar.gz

if [ -d blossom5 ]; then
    echo "blossom5/ already exists; skipping download."
    exit 0
fi

# -f: fail on HTTP errors, -S: show errors, -L: follow redirects
curl -fSL "${BLOSSOM_V_URL}" -o "${BLOSSOM_V_ZIPFILE}"
tar xzf "${BLOSSOM_V_ZIPFILE}"
mv blossom5-v2.05.src blossom5
rm -f "${BLOSSOM_V_ZIPFILE}"
echo "Blossom-V ready in ./blossom5/"
