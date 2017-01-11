#!/bin/sh -e
ROOT="${0%/*}"

. "$HOME"/.bash_profile

if [ "$ROOT" != "." ]; then
  echo "Execute in the directory: ./pyvenv_setup.sh"
  exit 1
fi

WD="$(pwd)"

clean(){
  rm -rf "$WD"/tmp
}

trap clean 1 2 3 15

PIP3_PACKAGES="Cython numpy scipy sklearn word2vec etaprogress"

install_python(){
  mkdir tmp
  cd tmp
  wget https://www.python.org/ftp/python/3.5.1/Python-3.5.1.tar.xz
  tar xvfJ Python-3.5.1.tar.xz
  cd Python-3.5.1
  ./configure --prefix="$WD"/.local
  make -j 4 && make -j 4 install

  if ! which python3 || ! which pyvenv; then
cat >> "$HOME"/.bash_profile <<-EOF
PATH=\$PATH:$WD/.local/bin
export PATH
EOF
  fi

  cd ..
  cd ..
  rm -rf tmp
}

if which python3; then
  echo "python3 already available"
else
  echo "Installing python3 in $WD/python3"
  install_python
fi

. "$HOME"/.bash_profile

if ! python3 -m venv venv; then
  echo "venv not available, trying to install"
  curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
cat >> "$HOME"/.bash_profile <<-EOF
export PATH="/home/ubuntu/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
EOF
python3 -m venv venv
fi


. venv/bin/activate
printf "Virtual environment activated, run the command \"deactivate\" to disable it.\n"
PIP="pip install"
$PIP --upgrade pip
for pck in $PIP3_PACKAGES; do
  $PIP "$pck" || $PIP --no-cache-dir "$pck"
done

# tensorflow
# Uses a renaming workaround for https://github.com/tensorflow/tensorflow/issues/2188
# Cuda does not work for now

# Ubuntu/Linux 64-bit, CPU only:
wget https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp34-cp34m-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled. Requires CUDA toolkit 7.5 and CuDNN v4.
# wget https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0-cp34-cp34m-linux_x86_64.whl

ver=$(python3 -c 'import sys; print(sys.version_info[1])')
if [ "$ver" -eq 5 ]; then
  mv tensorflow-0.8.0-cp34-cp34m-linux_x86_64.whl tensorflow-0.8.0-cp35-cp35m-linux_x86_64.whl
  pip3 install tensorflow-0.8.0-cp35-cp35m-linux_x86_64.whl
  rm tensorflow-0.8.0-cp35-cp35m-linux_x86_64.whl
else
  # Assuming python v3.4
  pip3 install tensorflow-0.8.0-cp34-cp34m-linux_x86_64.whl
fi

deactivate
