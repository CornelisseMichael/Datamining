
from sklearn.model_selection import StratifiedKFold    
from sklearn import metrics
from sklearn import tree    
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
def k_fold(X_data, y, folds, start, stop, step, method):
    kf = StratifiedKFold(n_splits=folds, shuffle = True)
    accuracy_mean_train = np.array([])
    accuracy_mean_test = np.array([])
    for i in range(start, stop, step):
        accuracy_train = np.array([])
        accuracy_test = np.array([])
        for train, test in kf.split(X_data, y):
            
            if(method == 'tree'):
                split_clf = tree.DecisionTreeClassifier(criterion='gini', max_depth = i)
                split_clf = split_clf.fit(X_data[train], y[train])
                accuracy_train = np.append(
                    accuracy_train, metrics.accuracy_score(y[train], split_clf.predict(X_data[train])))
                accuracy_test = np.append(
                    accuracy_test, metrics.accuracy_score(y[test], split_clf.predict(X_data[test]))) 
            
            elif(method == 'rf'):
                rf = RandomForestClassifier(n_estimators = 100, criterion='gini', max_depth = i)
                rf = rf.fit(X_data[train], y[train])
                accuracy_train = np.append(
                    accuracy_train, metrics.accuracy_score(y[train], rf.predict(X_data[train])))
                accuracy_test = np.append(
                    accuracy_test, metrics.accuracy_score(y[test], rf.predict(X_data[test]))) 
                
            elif(method == 'rf_estimators'):
                    rf = RandomForestClassifier(n_estimators = i, criterion='gini', max_depth = 4)
                    rf = rf.fit(X_data[train], y[train])
                    accuracy_train = np.append(
                        accuracy_train, metrics.accuracy_score(y[train], rf.predict(X_data[train])))
                    accuracy_test = np.append(
                        accuracy_test, metrics.accuracy_score(y[test], rf.predict(X_data[test]))) 
                
        accuracy_mean_train = np.append(accuracy_mean_train, np.mean(accuracy_train))
        accuracy_mean_test = np.append(accuracy_mean_test, np.mean(accuracy_test))
    return accuracy_mean_train, accuracy_mean_test



def plot_classification_error(start, stop, step, accuracy_mean_train, accuracy_mean_test, function):
    plt.plot(range(start,stop, step), (1-accuracy_mean_train), label='accuracy train set')
    plt.plot(range(start,stop, step), (1-accuracy_mean_test), label='accuracy test set')
    plt.legend()
    plt.xlabel('Maximum {}'.format(function))
    plt.ylabel('Classification error')
    plt.title('Classification error of a decision tree as function of {}'.format(function))
    plt.show()
