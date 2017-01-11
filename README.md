# Short version

Provided that everything is already installed on your computer, just run
    ./run.py
(or use shell script that it calls).

The following provides instructions on how to install the required libraries and how to run the program. It reflects how we have worked to try to find better parameters; we have thus found it useful to leave it here.

# Setup

Run:
    ./setup/repo_setup.sh

It should extract the data such that the files are in a folder *data*.

You can also manually download the tweet datasets from the CIL webpage here:
    http://cil.inf.ethz.ch/material/exercise/twitter-datasets.zip
After that the configuration depends on the machine.
In both cases, a parameter file has to be provided. Examples of such files are provided in *cnn_text/*.

## Setup on the ETH (and possibly other) computers

Run
    ./pyvenv_setup.sh

This project uses a python virtual environment (pyvenv) to be able to run on the ETH machines without root access.
This configuration has also worked on the other machines tested so far (apart from Euler).
If Python 3 is not detected, its sources will be downloaded, compiled and installed locally, which can take a bit of time.

### Setup on Euler

Run
    ./setup/euler_setup.sh

There might be some other stuff to do...

This script is based on the one provided by the TAs.

# Running
## Running on the ETH (and possibly other) computers

Run
    ./pyvenv_run.sh name_of_parameter_file
where the the parameter file can be found in the folder *cnn_text*

The submission is written in the *data* folder.

## Running on Euler
Change the variable *PARAM_SET* in the file *euler_run.sh* to specify the set of parameters to run the CNN on.
This file has to be in the folder *cnn_text*.
Then run:
    ./euler_run.sh
Of course depending on how big the parameters are, you might want to change the default runtime limit to allow the job to actually finish.
The output is redirected to a file in the folder *euler_output*, use *tail -f output_filename* to read it in real time.

The submission is written in the *data* folder.


# Baseline
To run the baseline, the virtual environment created by the *pyvenv_setup.sh* should do. After having activated it with
    . venv/bin/activate
create the word2vec embeddings (used by default):
    cd word2vec && ./create_embeddings
The glove embeddings can also be computed with the script *baseline/vocab/create_embeddings.sh*.
Then run the model in the *baseline* folder:
    ./baseline.py

# The char CNN
To run the char CNN, one has to change the 'unit' to 'char' in the *train_cnn.py* file.
