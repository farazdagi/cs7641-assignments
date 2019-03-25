# Unsupervised Learning and Dimensionality Reduction

## Requirements
You will need to use python 3.x with this code, and to pip install the packages in `requirements.txt`. The main addition here is the tables module which _does_ require HDF5. If you are using OS X with Homebrew you can simply `brew install hdf5` before installing the requirements. 
If this does not work for you, try the `requirements-no-tables.txt` file. Windows users have noted the need to install the tables module but on some systems this is not required. 

## Install miniconda (optional)
curl -o Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh ./Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda create --name ml python=3.7 -y
source ~/miniconda3/bin/activate ml

## Install project code

git clone https://github.com/farazdagi/cs7641-assignments
cd cs7641-assignments/assignment3
pip install -r requirements.txt
pip install --upgrade matplotlib seaborn

## Overall Flow
1. Update `run_experiment.py` to use your data sets for dataset1 and dataset2. Also set `best_nn_params` for your data sets (lines 94 and 101).
2. Run the various experiments (perhaps via `python run_experiment.py --all`)
3. Plot the results so far via `python run_experiment.py --plot`
4. Update the dim values in `run_clustering.sh` based on the optimal values found in 2 (perhaps by looking at the scree graphs)
5. Run `run_clustering.sh`
6. One final run to plot the rest `python run_experiment.py --plot`

There are clearly some redundant steps here but thankfully the plotting is pretty fast compared to the data generation.

## Output
Output CSVs and images are written to `./output` and `./output/images` respectively. Sub-folders will be created for each DR algorithm (ICA, PCA, etc) as well as the benchmark.

If these folders do not exist the experiments module will attempt to create them.

## Clustering Experiments

The experiments will output modified versions of the data sets after applying the DR methods. The script `run_clustering.sh` can be used to perform clustering on these modified datasets, using a specific number of components for the DR method.

**BE SURE TO UPDATE THE VALUES IN THIS SCRIPT FOR YOUR DATASETS**. 

There are different optimal values for each algorithm and each dataset, and using the wrong value will make you a sad panda.

## Graphing

The run_experiment script can be use to generate plots via:

```
python run_experiment.py --plot
```

Since the files output from the experiments follow a common naming scheme this will determine the problem, algorithm,
and parameters as needed and write the output to sub-folders in `./output/images`.

