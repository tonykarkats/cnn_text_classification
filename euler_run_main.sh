#!/bin/sh -e

main(){
  echo "Starting now on euler"
  date
  cd word2vec && ./create_embeddings.sh && cd ..
  cd cnn_text && LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/ext/lib" "$HOME"/ext/lib/ld-2.23.so "$HOME"/.venv/bin/python3 train_cnn.py "$PARAM_SET" && cd ..
  echo "Just finished on euler"
  date
}

mkdir -p euler_output

file="euler_output/out_$(date -Iseconds)"

main 2>&1 | tee "$file"
