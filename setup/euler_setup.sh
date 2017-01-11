#!/bin/sh -e
module load eth_proxy gcc/4.9.2 python/3.3.3 openblas/0.2.13_par
export BLAS=/cluster/apps/openblas/0.2.13_par/x86_64/gcc_4.9.2/lib/libopenblas.so
export LAPACK=/cluster/apps/openblas/0.2.13_par/x86_64/gcc_4.9.2/lib/libopenblas.so
export ATLAS=/cluster/apps/openblas/0.2.13_par/x86_64/gcc_4.9.2/lib/libopenblas.so

export C_INCLUDE_PATH="$C_INCLUDE_PATH:/cluster/apps/python/3.3.3/x86_64/include/python3.3m/"
export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/cluster/apps/python/3.3.3/x86_64/include/python3.3m/"

python3 -m venv "$HOME"/.venv
. "$HOME"/.venv/bin/activate    

wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py

pip3 install cython
pip3 install numpy
pip3 install scipy
pip3 install word2vec
pip3 install etaprogress


wget https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp34-cp34m-linux_x86_64.whl
mv tensorflow-0.8.0-cp34-cp34m-linux_x86_64.whl tensorflow-0.8.0-cp33-cp33m-linux_x86_64.whl
pip3 install --upgrade tensorflow-0.8.0-cp33-cp33m-linux_x86_64.whl

module load binutils

wget http://ftp.gnu.org/gnu/glibc/glibc-2.23.tar.gz

tar xfz glibc-2.23.tar.gz

mkdir "$HOME"/ext

cd glibc-2.23
# TODO: use the cluster to compile?
# XXX: remove the -j flag if it fails
mkdir build && cd build && ../configure --prefix="$HOME"/ext && make -j && make -j install && cd "$HOME"

module load zlib

# test if tensorflow is available
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/ext/lib" "$HOME"/ext/lib/ld-2.23.so "$HOME"/.venv/bin/python3 -c 'import tensorflow'
