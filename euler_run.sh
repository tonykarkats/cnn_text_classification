#!/bin/sh -e
module load eth_proxy gcc/4.9.2 python/3.3.3 openblas/0.2.13_par
export BLAS=/cluster/apps/openblas/0.2.13_par/x86_64/gcc_4.9.2/lib/libopenblas.so
export LAPACK=/cluster/apps/openblas/0.2.13_par/x86_64/gcc_4.9.2/lib/libopenblas.so
export ATLAS=/cluster/apps/openblas/0.2.13_par/x86_64/gcc_4.9.2/lib/libopenblas.so

export C_INCLUDE_PATH="$C_INCLUDE_PATH:/cluster/apps/python/3.3.3/x86_64/include/python3.3m/"
export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/cluster/apps/python/3.3.3/x86_64/include/python3.3m/"

. "$HOME"/.venv/bin/activate    

module load zlib

export PARAM_SET="para_default"
if [ ! -f "cnn_text/""$PARAM_SET"".py" ]; then
  echo "No such set of parameters, exiting."
  exit 1
fi
bsub -W 24:00 -n 48 < euler_run_main.sh
