import csv

import numpy as np
import sklearn
from sklearn.datasets import make_classification
import warnings
import pandas as pd
import argparse
import sys

sys.path.insert(0, "Dynamic")
from Dynamic.dynamic import dynamic

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

desMethds = ["RANK", "OLA", "LCA", "DESKNN","KNORRAE","KNORRAU","KNOP","METADES","Oracle"]

parser = argparse.ArgumentParser()
parser.add_argument('--string')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   # read config file if there is an argument mentioning one
   arg = parser.parse_args()
   configArgs = list()
   currentArg = 0
   if arg.string is not None:
      f = open("Configs/"+arg.string,"r")

      for line in f:
          configArgs.append(line.lower().strip())




   #read the config file and save the arguments



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
            if currentArg==0:
                fileWithData = input("Type the name of the file to use from the Dynamic/Test folder, (.csv can be ommitted)")
            else :
                fileWithData = configArgs[currentArg]

            if not(fileWithData.endswith('.csv')) :
                fileWithData += '.csv'
            fileWithData = "Dynamic/Test/" + fileWithData
            try:
                data = pd.read_csv(fileWithData)
                X = data.iloc[:,:-1]
                y = data.iloc[:,-1]
                inp_des = "1"
                currentArg +=1
            except:
                print("Failed to read file "+fileWithData)
                inp_des = "0"
            if inp_des == "1":
                if currentArg == 0:
                    desMethod = input("Input one of the following methods (with same capitalizations) or the word all:\n\nBaselines: RANK, OLA, LCA, DESKNN\nSoA: KNORRAE,KNORRAU,KNOP,METADES,Oracle")
                else:
                    desMethod = configArgs[currentArg]

                if desMethod.lower() =="all":
                    for desMethod in desMethds:
                        dynamic(desMethod, X, y)
                else:
                    dynamic(desMethod,X,y)
        if  currentArg!=0:
            loopin=1