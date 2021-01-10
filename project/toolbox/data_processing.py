import numpy as np
import pandas as pd
def load_data():
    X_train = pd.read_csv('data/archive/data_set_ALL_AML_train.csv')
    X_test = pd.read_csv('data/archive/data_set_ALL_AML_independent.csv')
    y = pd.read_csv('data/archive/actual.csv', index_col = 'patient')
    return X_train, X_test, y

def clean_data(train, test, y):
    """
    returns a cleaned data set
    Parameters
    ----------
    train : DataFrame
        train data
    test : DataFrame
        test data
    y : DataFrame
        y data
        """
    # Drop the call collumns from both data sets
    call_cols_train = [col for col in train.columns if 'call' in col]
    train = train.drop(call_cols_train, axis = 1)
    call_cols_test = [col for col in test.columns if 'call' in col]
    test = test.drop(call_cols_test, axis = 1)
    # Drop "Gene Description" and "Gene Accession Number"
    cols_to_drop = ['Gene Description', 'Gene Accession Number']
    train = train.drop(cols_to_drop, axis = 1)
    test = test.drop(cols_to_drop, axis = 1)
    # Transpose both train and test data_sets
    train = train.T
    test = test.T
    # Replace cancer labels with numeric values
    y = y.replace({'ALL':0,'AML':1})
    return train, test, y

def sort_X_train_and_test_data(train, test):
    train.index = pd.to_numeric(train.index) 
    test.index = pd.to_numeric(test.index)
    #y.index = pd.to_numeric(y.index)
    train.sort_index(inplace=True) 
    test.sort_index(inplace=True)
    return train, test


def get_y_train_and_test_data(y):
    """
    returns y_train and y_test data sets
    Parameters
    ----------
    train : DataFrame
        train data
    test : DataFrame
        test data
    y : DataFrame
        y data
        """
    y_train = y[y.index <= 38].reset_index(drop=True) 
    y_test= y[y.index > 38].reset_index(drop=True)
    return y_train, y_test