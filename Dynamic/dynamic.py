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
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split


# temporary import of trees for DES ensembles

from sklearn.tree import DecisionTreeClassifier


rng = np.random.RandomState(1)
baseClassifiers = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                    n_estimators=100,
                                    random_state=rng)


def dynamic(model_name, X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=rng)

    X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train,
                                                        test_size=0.50,
                                                        random_state=rng)

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

    elif model_name == "Oracle":
        method = Oracle(pool_classifiers=baseClassifiers)
    else:
        print("No valid method selected")
        return

    method.fit(X_dsel, y_dsel)

    print("Classification accuracy {} = {}".format(model_name, method.score(X_test, y_test)))


