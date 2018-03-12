#!/usr/bin/env python3

"""
sklearn_pack.py: Do Model training with large sparse matrix with sklearn
"""

__author__ = "Yanshi Luo"
__license__ = "GPL"
__email__ = "yluo82@wisc.edu"

import pandas as pd


def random_forest(finalX_train, finalY_train, finalX_test, n_parallel=1, write_csv=False, write_filename='rf_pref.csv'):
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=10, n_jobs=n_parallel)
    clf = clf.fit(finalX_train, finalY_train)
    finalY_pred = clf.predict(finalX_test)
    finalY_pred_DF = pd.DataFrame(finalY_pred)

    if write_csv:
        finalY_pred_DF.to_csv(write_filename, index=False)

    return finalY_pred_DF


def decision_tree(finalX_train, finalY_train, finalX_test, dot=False):
    from sklearn import tree

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(finalX_train, finalY_train)
    if dot:
        tree.export_graphviz(clf, out_file='sklearn_tree.dot')

    finalY_pred = clf.predict(finalX_test)

    return pd.DataFrame(finalY_pred)


def neural_network(finalX_train, finalY_train, finalX_test):
    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(solver='adam', alpha=1e-5, random_state=1)
    clf = clf.fit(finalX_train, finalY_train)
    finalY_pred = clf.predict(finalX_test)
    return pd.DataFrame(finalY_pred)


