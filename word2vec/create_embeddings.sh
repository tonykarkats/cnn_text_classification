#!/bin/sh -e
ROOT="${0%/*}"

if [ "$ROOT" != "." ]; then
  echo "Execute in the directory: ./create_embeddings.sh"
  exit 1
fi

if [ -f "embeddings_done" ]; then
    echo "w2v embeddings already computed, exiting"
    exit 0
fi

mkdir -p tmp

if [ ! -s tmp/train_full.txt ]; then
  cat ../data/train_pos_full.txt ../data/train_neg_full.txt > tmp/train_full.txt
fi

FILES_TO_GROUP="tmp/train_full.txt ../data/test_data.txt ../data/train_neg_full.txt ../data/train_neg.txt ../data/train_pos_full.txt ../data/train_pos.txt"

for f in $FILES_TO_GROUP; do
  python3 -c "import wv_cus; wv_cus.group_words('$f')"
done

DIMENSIONS="25 50 100 200"
echo "Training with respect to the dimensions $DIMENSIONS."

for dim in $DIMENSIONS; do
    echo "Dimension $dim..."
    python3 -c "import wv_cus; wv_cus.train_model($dim)"
done

echo "Done."

touch "embeddings_done"
