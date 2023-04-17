import csv

import numpy as np
import sklearn
from sklearn.datasets import make_classification
import warnings
import pandas as pd
import argparse
import sys

sys.path.insert(0, "Dynamic")
sys.path.insert(0, "Imb")
sys.path.insert(0, "Multiclass")
sys.path.insert(0, "CD")

from Dynamic.dynamic import DES,dynamic

from Imb.imb import imb

from Multiclass.multiclass import OneVo

from CD.cd import CD


warnings.filterwarnings("ignore")

rng = np.random.RandomState(1)

# Generate a classification dataset
X, y = make_classification(n_samples=2000,
                           n_classes=3,
                           n_informative=6,
                           random_state=rng)



#General Dictionaries for easier config files setup
method = {
    "des": "1",
    "mc":"2"
}

from river import datasets
from scipy.io import arff

#data = datasets.MaliciousURL()
#data.download()
#dataset=arff.loadarff(data.path)
#dataset = pd.DataFrame(dataset[0])
#dataset=data
#print(dataset.head().to_string())

#y= dataset.pop("class")
#X=dataset.iloc[:,:-1]

fileWithData = "Dynamic/Test/iris.csv"

data = pd.read_csv(fileWithData)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                            random_state=rng)

X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train,
                                                            test_size=0.50,
                                                            random_state=rng)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                            random_state=rng)



kekwait = imb("ROS",dynamic("OLA", X_train, y_train,X_dsel,y_dsel),set(y))

from river import evaluate
from river import metrics

metric = metrics.F1()

dataset= np.append(X_train,y_train[:,None],axis=1)

evaluate.progressive_val_score(dataset,kekwait,metric)

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

   #loopin=0
   loopin=1
   while loopin==0 and currentArg==0:
       # read first argument
        if configArgs.__len__()==0:
            inp = input("Run test mode for DES by typing 1\nRun test mode for Multiclass by typing 2\nRun test mode for Concept Drift by typing 3\nExit by typing 0")
        else:
            inp = method[configArgs[currentArg]]
            currentArg += 1
        if inp=="0":
            loopin=1
        elif inp=="1":
            DES(configArgs,currentArg)
            currentArg+=1
        elif inp=="2":
            print("ha")
        elif inp=="3":
            print("ss")
        if not(currentArg==0):
            loopin=1