import numpy as np
from sklearn import metrics as skmetrics


def mean_squared_error(y_pred, y_true):
    return skmetrics.mean_squared_error(y_true, y_pred)

def accuracy_err(y_pred, y_true):
    return 1 - skmetrics.accuracy_score(np.argmax(y_true, axis=1),
                                        np.argmax(y_pred, axis=1),
                                        normalize=True)
