#!/bin/sh -e

compute_w2v_embeddings(){
  cd word2vec
  ./create_embeddings.sh
  cd ..
}

run_model_basic_embeddings(){
  compute_w2v_embeddings
  python3 model.py
}

run_model_cnn(){
  PARAM_SET="$1"
  if [ ! -f "cnn_text/""$PARAM_SET"".py" ]; then
    echo "No such set of parameters, exiting."
    exit 1
  fi
  compute_w2v_embeddings
  cd cnn_text
  python3 train_cnn.py "$PARAM_SET"
  cd ..
}



mkdir -p output
run_model_cnn para_no_overfit_guard_2 | tee output/output_pyvenv_run_"$1"
