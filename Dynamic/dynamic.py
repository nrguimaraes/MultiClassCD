import deslib

# baselines
from deslib.dcs.rank import Rank
from deslib.dcs.ola import OLA
from deslib.dcs.lca import LCA
from deslib.des.des_knn import DESKNN

# SoA

from deslib.des.knora_e import KNORAE
from deslib.des.knora_u import KNORAU
from deslib.des.knop import KNOP
from deslib.des.meta_des import METADES

# Oracle cause why not

from deslib.static.oracle import Oracle

# sklearn and numpy imports

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split


# temporary import of trees for DES ensembles

from sklearn.tree import DecisionTreeClassifier

desMethdNames = ["RANK", "OLA", "LCA", "DESKNN","KNORRAE","KNORRAU","KNOP","METADES","ORACLE"]

rng = np.random.RandomState(1)
baseClassifiers = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                    n_estimators=100,
                                    random_state=rng)


def dynamic(model_name, X_train, y_train,X_dsel,y_dsel):

    baseClassifiers.fit(X_train,y_train)

    if model_name == "RANK":
        method = Rank(pool_classifiers=baseClassifiers, random_state=rng)

    elif model_name == "OLA":
        method = OLA(pool_classifiers=baseClassifiers, random_state=rng)

    elif model_name == "LCA":
        method = LCA(pool_classifiers=baseClassifiers, random_state=rng)

    elif model_name == "DESKNN":
        method = DESKNN(pool_classifiers=baseClassifiers, random_state=rng)

    elif model_name == "KNORRAE":
        method = KNORAE(pool_classifiers=baseClassifiers, random_state=rng)

    elif model_name == "KNORRAU":
        method = KNORAU(pool_classifiers=baseClassifiers, random_state=rng)

    elif model_name == "KNOP":
        method = KNOP(pool_classifiers=baseClassifiers, random_state=rng)

    elif model_name == "METADES":
        method = METADES(pool_classifiers=baseClassifiers, random_state=rng)

    elif model_name == "ORACLE":
        method = Oracle(pool_classifiers=baseClassifiers)
    else:
        print("No valid method selected")
        return

    return method
#.fit(X_dsel, y_dsel)

def methodMetrics(method,X_test,y_test,metric,methodName):

    print(methodName)
    if(metric=="acc" or metric=="accuracy"):
        print("Classification accuracy {} = {}".format(methodName, method.score(X_test, y_test)))

def DES(configArgs,currentArg):
    if currentArg == 0:
        fileWithData = input("Type the name of the file to use from the Dynamic/Test folder, (.csv can be ommitted)")
    else:
        fileWithData = configArgs[currentArg]

    if not (fileWithData.endswith('.csv')):
        fileWithData += '.csv'
    fileWithData = "Dynamic/Test/" + fileWithData
    try:
        data = pd.read_csv(fileWithData)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                            random_state=rng)

        X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train,
                                                            test_size=0.50,
                                                            random_state=rng)
        inp_des = "1"
        currentArg += 1
    except:
        print("Failed to read file " + fileWithData)
        return
    if inp_des == "1":
        if currentArg == 0:
            desMethod = input(
                "Input one of the following methods (with same capitalizations) or the word all:\n\nBaselines: RANK, OLA, LCA, DESKNN\nSoA: KNORRAE,KNORRAU,KNOP,METADES,Oracle")
        else:
            desMethod = configArgs[currentArg]
            currentArg+=1

        desMthds = list()
        if desMethod.lower() == "all":
            for desMethod in desMethdNames:
                desMthds.append(dynamic(desMethod, X_train, y_train,X_dsel,y_dsel))
            desMethod = desMethdNames
        else:
            desMthds.append(dynamic(desMethod.upper(), X_train, y_train,X_dsel,y_dsel))



        if currentArg == 0:
            metric = input(
                "Input one of the following evaluation metrics:\n Accuracy (Acc) ")
        else:
            metric = configArgs[currentArg]
            currentArg+=1
        counter = 0
        for methods in desMthds:
            metodo = desMethod
            if desMthds.__len__()>1:
                metodo = desMethod[counter]
            if methods is not None:
                methodMetrics(methods,X_test,y_test,metric,metodo)
            counter+=1

