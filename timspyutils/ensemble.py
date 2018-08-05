import numpy as np 
from sklearn.preprocessing import LabelEncoder
import warnings

class PrefittedVotingClassifier(object):
    _str_labels = False

    def __init__(self, estimators, labels = None):
        self.estimators = estimators
        if type(labels) != type(None):
            self._str_labels = True
            self._le = LabelEncoder().fit(labels)

    def predict(self, X):
        Y = np.zeros([X.shape[0], len(self.estimators)], dtype = int)
        for i, clf in enumerate(self.estimators):
            if self._str_labels:
                Y[:, i] = self._le.transform(clf.predict(X))
            else:
                yy = clf.predict(X)
                print(yy.dtype)
                if yy.dtype.type is np.str_:
                    raise ValueError("An estimator returned string labels, if estimators are "
                                    +"trained with string labels then labels needed to be passed "
                                    +"to PrefittedVotingClassifier via labels with init.")
                Y[:, i] = clf.predict(X)
        y = np.zeros(X.shape[0], dtype = int)
        for i in range(X.shape[0]):
            y[i] = np.argmax(np.bincount(Y[i,:]))
        if self._str_labels:
            return self._le.inverse_transform(y).astype(str)
        else:
            return y
    
    def fit(self, **args):
        raise NotImplementedError("This classifier assumes the estimators given are all ready fitted. If fitting "
                                 +"needs to be done use sklearn.ensemble.VotingClassifier instead.")