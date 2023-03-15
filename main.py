import csv

import numpy as np
import sklearn
from sklearn.datasets import make_classification
import warnings
import pandas as pd
import argparse
import sys

sys.path.insert(0, "Dynamic")
from Dynamic.dynamic import DES

warnings.filterwarnings("ignore")

rng = np.random.RandomState(1)

# Generate a classification dataset
X, y = make_classification(n_samples=2000,
                           n_classes=3,
                           n_informative=6,
                           random_state=rng)

#General Dictionaries for easier config files setup
method = {
    "des": "1"
}



parser = argparse.ArgumentParser()
parser.add_argument('--config')

#function to read the config file and return its configs
def readConfigFile(filename):
    configArgs=list()
    if not (filename.endswith('.txt')):
        filename += '.txt'
    f = open("Configs/" + filename, "r")
    for line in f:
        configArgs.append(line.lower().strip())
    return configArgs



if __name__ == '__main__':
   # read config file if there is an argument mentioning one
   arg = parser.parse_args()
   configArgs =list()
   if arg.config is not None:
        configArgs = readConfigFile(arg.config)
   currentArg = 0

   loopin=0
   while loopin==0 and currentArg==0:
       # read first argument
        if configArgs.__len__()==0:
            inp = input("Run test mode for DES by typing 1\nExit by typing 0")
        else:
            inp = method[configArgs[currentArg]]
            currentArg += 1
        if inp=="0":
            loopin=1
        elif inp=="1":
            DES(configArgs,currentArg)
            currentArg+=1

        if  currentArg!=0:
            loopin=1