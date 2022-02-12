# image-tsp

## Setup
Ensure `PYTHONPATH` env variable is set first.

`export PYTHONPATH="/insert/path/of/image-tsp"`

## Data Generation
All data generation code can be found in `src/main/generate_data.py`. The command line script to run would be `python src/main/generate_data.py generate`. Generating the data first entails generating the pickle files. the `draw` and `drawall` commands can then be used to draw the TSP images based on the pickle files. It is necessary to draw the pickle files as the training process will be simpler. Reading an image in from disk is easier than generating it every time is needed (probably faster too).


