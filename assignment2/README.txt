# Assignment 2

## Install miniconda (optional)
curl -o Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh ./Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda create --name ml python=3.7 -y
source ~/miniconda3/bin/activate ml

## Install Jython (if not already installed)
wget http://search.maven.org/remotecontent?filepath=org/python/jython-installer/2.7.0/jython-installer-2.7.0.jar -O jython-installer.jar
java -jar jython-installer.jar --console
ln -s /usr/local/lib/jython/bin/jython /usr/local/bin/jython

## Install project code
git clone https://github.com/farazdagi/cs7641-assignments
cd cs7641-assignments/assignment2
pip install -r requirements.txt
pip install --upgrade matplotlib seaborn


## Run Experiment: Part 1
jython NN-Backprop.py
jython NN-GA.py
jython NN-RHC.py
jython NN-SA.py


## Run Experiment: Part 2
jython continuoutpeaks.py
jython flipflop.py
jython tsp.py

## Generate reports
python plotting.py
