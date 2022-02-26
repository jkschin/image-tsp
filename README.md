# image-tsp

This work is currently under review.

## Setup
Ensure `PYTHONPATH` env variable is set first.

`export PYTHONPATH="/insert/path/of/image-tsp"`

## Data Generation
All data generation code can be found in `src/main/generate_data.py`. The command line script to run would be `python src/main/generate_data.py generate`. Generating the data first entails generating the pickle files. the `draw` and `drawall` commands can then be used to draw the TSP images based on the pickle files. It is necessary to draw the pickle files as the training process will be simpler. Reading an image in from disk is easier than generating it every time is needed (probably faster too).

On MIT Supercloud, the file `generate.sh` has to be edited. and then `sbatch generate.sh` should be run. The function signature inside `generate.sh` should also be helpful in understanding how to run this.

After generating the pickle files, run
`python src/main/generate_data.py draw TSP400.pickle 400 "(1024, 1024, 3)"`
to draw the TSP images.

## Training
To train a model:
1. Edit the argument in `run.sh` that represents the number of cities you would like to train on.
2. `./run.sh` - Configured for MIT Supercloud.

To look deeper in the code, see `src/nn/tspconv.py`.

## Testing
To run the test samples on the model:
1. `LLsub -i -g volta:1` to acquire a GPU node. Only for MIT Supercloud.
2. `python src/nn/tspconv_eval.py`

Don't forget to configure the experiment folders inside `tspconv_eval.py`. If it doesn't run, check your `PYTHONPATH`.

## Collaborators
1. Matthias Winkenbach (mwinkenb@mit.edu)


