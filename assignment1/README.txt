# Assignment 1

## Install miniconda (optional)
curl -o Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh ./Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda create --name ml python=3.7 -y
source ~/miniconda3/bin/activate ml

## Install project code

git clone https://github.com/farazdagi/cs7641-assignments
cd cs7641-assignments/assignment1
pip install -r requirements.txt
pip install --upgrade matplotlib seaborn

## Run Experiment 1
python run_experiment.py --threads -1 --all --dataset spam

## Run Experiment 2
python run_experiment.py --threads -1 --all --dataset htru
