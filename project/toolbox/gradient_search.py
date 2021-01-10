from sklearn import metrics
from sklearn import tree
import xgboost as xgb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def grid_search_classifier(classifier, parameters, x_train, x_test, y_train, y_test):
    if classifier == 'tree':
        clf1 = tree.DecisionTreeClassifier()
        cols_to_keep = ['param_criterion', 'param_max_depth', 'param_min_samples_leaf' , 'param_random_state', 'mean_train_score', 'std_train_score']
    elif classifier == 'rf':
        clf1 = RandomForestClassifier()
        cols_to_keep = ['param_criterion', 'param_n_estimators', 'param_max_depth', 'param_min_samples_leaf' , 'param_random_state', 'mean_train_score', 'std_train_score']
    elif classifier == 'xgb':
        clf1 = xgb.XGBClassifier()
        cols_to_keep = ['param_eval_metric', 'param_n_estimators',  'param_max_depth', 'param_random_state', 'mean_train_score', 'std_train_score']
    
    clf = GridSearchCV(clf1, parameters, n_jobs = -1, cv = 10, return_train_score = True)
    clf.fit(x_train, y_train)
    
    results = pd.DataFrame.from_dict(clf.cv_results_)
    results = results[cols_to_keep]
    best_estimator = clf.best_estimator_
    
    prediction = clf.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, prediction)
    
    print('The best parameters for this model are: \n{}'.format(best_estimator))
    print('This gives an accuracy of {} and an error of {}'.format(accuracy, 1-accuracy))
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(results)
    return prediction, best_estimator
