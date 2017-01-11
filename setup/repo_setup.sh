#!/bin/sh -e
ROOT="${0%/*}"
EMB_SET="glove.twitter.27B.zip"
WD="$(pwd)"

if [ "$ROOT" != "./setup" ]; then
  echo "Execute from the git root repository"
  exit 1
fi

# On the ETH computers, clone in /local (open to anybody)
echo "Make sure you have enough space (clone in /local if at ETH)"
chmod -R go-rwx "$WD"


dl_glove_data(){
  NAME_CHECK="stanford_twitter_data_dled"
  cd data
  if [ ! -f "$NAME_CHECK" ]; then
    wget nlp.stanford.edu/data/$EMB_SET
    unzip $EMB_SET && rm $EMB_SET && touch $NAME_CHECK
  fi
}

dl_twitter_dataset(){
  wget http://cil.inf.ethz.ch/material/exercise/twitter-datasets.zip
  unzip twitter-datasets.zip
  mv twitter-datasets data
}

if [ -e data ]; then
  echo "data already exists, skipping extraction of data"
  exit 0
fi

dl_twitter_dataset &

wait
