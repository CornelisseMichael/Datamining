import numpy as np
def clean_data(train, test, y):
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


def get_X_train_and_test_data(train, test, y):
    X_train = train.reset_index(drop=True)
    X_test = test.reset_index(drop=True)
    return X_train, X_test

def get_y_train_and_test_data(y):
    y_train = y[y.index <= 38].reset_index(drop=True) 
    y_test= y[y.index > 38].reset_index(drop=True)
    #y_train = y['cancer'][:38]
    #y_test = y['cancer'][38:]
    return y_train, y_test


def merge_train_and_test_data(train, test):
    train = train.replace(np.inf, np.nan)
    train = train.fillna(value = train.values.mean())
    test = test.replace(np.inf, np.nan)
    test = test.fillna(value = train.values.mean())
    complete_data = train.append(test)
    return complete_data