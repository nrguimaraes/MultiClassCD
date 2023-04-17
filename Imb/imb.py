import river
from river import imblearn
from river import linear_model
import pandas as pd

#imblearn.RandomOverSampler(linear_model.LogisticRegression(), desired_dist=dic, seed = 1)

def imb(methodName,model,classNames):
    dic = {}
    size = 1 / len(classNames)
    for classe in classNames:
        dic[classe] = size
    methodName=methodName.upper()
    if methodName=="ROS":
        return imblearn.RandomOverSampler(model, desired_dist=dic, seed=1)
    elif methodName=="RUS":
        return imblearn.RandomUnderSampler(model, desired_dist=dic, seed=1)
