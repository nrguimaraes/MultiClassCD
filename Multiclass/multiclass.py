import river
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import multiclass
from river import preprocessing

dataset = datasets.ImageSegments()

scaler = preprocessing.StandardScaler()
#test = linear_model.LogisticRegression()



def OneVo(test):
    ovo = multiclass.OneVsOneClassifier(test)
    model = scaler | ovo

    return model
