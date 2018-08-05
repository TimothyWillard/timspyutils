import pytest
from timspyutils.ensemble import *
from sklearn.linear_model import RidgeClassifier, LogisticRegression

import numpy as np

def test_raise_fit():
    pvc = PrefittedVotingClassifier(estimators = [])
    with pytest.raises(NotImplementedError):
        pvc.fit()

def test_numeric_classification():
    y = np.random.randint(0, 10, 100)
    X = np.random.randn(100, 4)
    ridge_classifier = RidgeClassifier()
    ridge_classifier.fit(X, y)
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X, y)
    pvc = PrefittedVotingClassifier(estimators = [
        ridge_classifier,
        logistic_regression
    ])
    pred = pvc.predict(X)
    assert pred.dtype == int

def test_string_classification():
    y = np.random.randint(0, 10, 100)
    X = np.random.randn(100, 4)
    ridge_classifier = RidgeClassifier()
    ridge_classifier.fit(X, y)
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X, y)
    pvc = PrefittedVotingClassifier(estimators = [
        ridge_classifier,
        logistic_regression
    ], labels = np.arange(0, 10).astype(str))
    pred = pvc.predict(X)
    assert pred.dtype.type is np.str_

def test_raise_without_labels():
    y = np.random.randint(0, 10, 100).astype(str)
    X = np.random.randn(100, 4)
    ridge_classifier = RidgeClassifier()
    ridge_classifier.fit(X, y)
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X, y)
    pvc = PrefittedVotingClassifier(estimators = [
        ridge_classifier,
        logistic_regression
    ])
    with pytest.raises(ValueError):
        pvc.predict(X)