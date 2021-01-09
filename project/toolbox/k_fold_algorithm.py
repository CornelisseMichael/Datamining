import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold    
from sklearn import metrics
from sklearn import tree    
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

def k_fold(X_data, y, folds, start, stop, step, method):
    """
    Returns accuracy mean for train and test data
    Parameters
    ----------
    X_data : 
        train data
    y_np : numpy array
        test data
    fold : int
        number of folds
    start : int
        start index
    stop : int
        stop index
    step : int
        step index
    method : str
        algorithm to run k_fold on
        """
    kf = StratifiedKFold(n_splits=folds, shuffle = True)
    accuracy_mean_train = np.array([])
    accuracy_mean_test = np.array([])
    for i in range(start, stop, step):
        accuracy_train = np.array([])
        accuracy_test = np.array([])
        for train, test in kf.split(X_data, y):
            if(method == 'tree'):
                clf = tree.DecisionTreeClassifier(criterion='gini', max_depth = i)
            elif(method == 'rf'):
                clf = RandomForestClassifier(n_estimators = 100, criterion='gini', max_depth = i)
            elif(method == 'rf_estimators'):
                clf = RandomForestClassifier(n_estimators = i, criterion='gini', max_depth = 4)
            elif(method == 'xgb'):
                clf = xgb.XGBClassifier(n_estimators = 100, max_depth = i, eval_metric='error', use_label_encoder=False) 
            elif(method == 'xgb_estimators'):
                clf = xgb.XGBClassifier(n_estimators = i, max_depth = 5, eval_metric='error', use_label_encoder=False)
            
            clf = clf.fit(X_data[train], y[train])
            accuracy_train = np.append(accuracy_train, metrics.accuracy_score(y[train], clf.predict(X_data[train])))
            accuracy_test = np.append(accuracy_test, metrics.accuracy_score(y[test], clf.predict(X_data[test]))) 
                
        accuracy_mean_train = np.append(accuracy_mean_train, np.mean(accuracy_train))
        accuracy_mean_test = np.append(accuracy_mean_test, np.mean(accuracy_test))
    return accuracy_mean_train, accuracy_mean_test

def plot_classification_error(start, stop, step, accuracy_mean_train, accuracy_mean_test, function):
    """
    Returns accuracy mean for train and test data
    Parameters
    ----------
    start : int
        start index
    stop : int
        stop index
    step : int
        step index
    accuracy_mean_train : numpy array
        accuracy_mean values of train data
    accuracy_mean_test : numpy array
        accuracy_mean values of test data
    function : str
        function description
        """
    plt.plot(range(start,stop, step), (1-accuracy_mean_train), label='accuracy train set')
    plt.plot(range(start,stop, step), (1-accuracy_mean_test), label='accuracy test set')
    plt.legend()
    plt.xlabel('Maximum {}'.format(function))
    plt.ylabel('Classification error')
    plt.title('Classification error of a decision tree as function of {}'.format(function))
    plt.show()
