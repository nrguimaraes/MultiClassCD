import csv

import numpy as np
import sklearn
from sklearn.datasets import make_classification
import warnings
import pandas as pd
import argparse
import sys

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "Dynamic")
sys.path.insert(0, "Imb")
sys.path.insert(0, "Multiclass")
sys.path.insert(0, "CD")

from Dynamic.dynamic import DES, dynamic

from Imb.imb import imb

from Multiclass.multiclass import OneVo

from CD.cd import CD

rng = 1

import pandas as pd
from io import StringIO

from river.compat import convert_sklearn_to_river

# after testing replace StringIO(temp) to filename


batch1 = "globalDatasets/gas/batch1.dat"

print("Reading Dataset...")
with open(batch1) as f:
    data = f.readlines()
dataS = []
for lines in data:
    dataS.append(lines.replace(";", " "))

df = pd.read_csv(StringIO(dataS[0]),
                 sep="\s+",  # separator whitespace
                 header=None)


for i in range(1,len(dataS)):
    tmp = pd.read_csv(StringIO(dataS[i]),
                 sep="\s+",  # separator whitespace
                 header=None)
    df=df.append(tmp)

for c in df.columns.values:
    if c > 1:
        df[c] = df[c].apply(lambda x: float(str(x).split(':')[1]))

data = df
y = data.iloc[:, 0]
X = data.drop(0,axis=1)

from river import drift

adwin = drift.ADWIN()

for val in data :
    adwin.update(val)
    if adwin.drift_detected:
        print("drift")
print("Balancing Dataset...")

import imblearn

oversample = imblearn.over_sampling.SMOTE()
X, y = oversample.fit_resample(X, y)


print("Training test models...")

from river import linear_model

test = linear_model.LogisticRegression()

kekwait = OneVo(test)

models = [kekwait]

from river import forest

test = forest.ARFRegressor(seed=rng)

kekwait = OneVo(test)

models.append(kekwait)

from river import naive_bayes
testwait = naive_bayes.GaussianNB()

kekwait = OneVo(test)

models.append(kekwait)

tts = y.factorize()

from river import model_selection

from river import metrics

metric = metrics.BalancedAccuracy()

test = model_selection.GreedyRegressor(models, metric)

#test = OneVo(testwait)

for i in range(1, len(X.iloc[:,0])):
    test.learn_one(X.iloc[i, :].to_dict(), tts[0][i])

print( test.best_model)

batch1 = "globalDatasets/gas/batch2.dat"

print("Reading Dataset 2...")
with open(batch1) as f:
    data = f.readlines()
dataS = []
for lines in data:
    dataS.append(lines.replace(";", " "))

df = pd.read_csv(StringIO(dataS[0]),
                 sep="\s+",  # separator whitespace
                 header=None)


for i in range(1,len(dataS)):
    tmp = pd.read_csv(StringIO(dataS[i]),
                 sep="\s+",  # separator whitespace
                 header=None)
    df=df.append(tmp)

for c in df.columns.values:
    if c > 1:
        df[c] = df[c].apply(lambda x: float(str(x).split(':')[1]))

data = df
y = data.iloc[:, 0]
X = data.drop(0,axis=1)

predicted_y = []

for n in range(0,len(y)):
    predicted_y.append(test.predict_one(X.iloc[n,:]))

from sklearn.metrics import balanced_accuracy_score

testOutputName = "testfileMultipleModels.txt"

#testOutputName = "singleClassifier.txt"

f = open(testOutputName,"w")

f.write(str(balanced_accuracy_score(y,predicted_y)))

f.close()

f = open(testOutputName,"a")

f.write("\ntest")

f.close()

driftDetected = False

for val in data :
    adwin.update(val)
    if adwin.drift_detected:
        driftDetected = True

if driftDetected:
    print("treinar com batch i-1 mas neste caso nada é feito pois já estava treinado")




