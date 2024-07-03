# Module to read and load data

# Import modules
import numpy as np
import pandas as pd

# Define functions 
def db_name_classes(db):
    # if-else sentence 
    if db == 'colon':
        classes = ['normal', 'colon']
    elif db == 'leukemia':
        classes = ['ALL', 'AML']
    elif db == 'crohn':
        classes = ['normal', 'crohn']
    return classes

def clean_data(global_path, name_db, classes, fs_method):
    # Define paths to clean data
    path_X_train = global_path + name_db + '/X_train_' + name_db + '.xlsx'
    path_y_train = global_path + name_db + '/y_train_' + name_db + '.xlsx'
    path_X_test = global_path + name_db + '/X_test_' + name_db + '.xlsx'
    path_y_test = global_path + name_db + '/y_test_' + name_db + '.xlsx'
    path_t_cv_idx = global_path + name_db + '/' + name_db + '_t_idx_cv'  + '.xlsx'
    path_v_cv_idx = global_path + name_db + '/' + name_db + '_v_idx_cv'  + '.xlsx'
    # Load data
    X_train = pd.read_excel(path_X_train, index_col=0).T
    y_train = pd.read_excel(path_y_train, index_col=0)
    y_train_dummy = y_train.replace({classes[0]:0, classes[1]:1})
    X_test = pd.read_excel(path_X_test, index_col=0).T
    y_test = pd.read_excel(path_y_test, index_col=0) 
    y_test_dummy = y_test.replace({classes[0]:0, classes[1]:1})
    t_cv_idx = pd.read_excel(path_t_cv_idx, index_col=0)
    v_cv_idx = pd.read_excel(path_v_cv_idx, index_col=0)
    # Return data
    return X_train, y_train_dummy, t_cv_idx, v_cv_idx, X_test, y_test_dummy

def n_vars_to_select():
    # Define number of variables to be selected
    x = [10, 20, 30, 40, 50, 75, 100, 150, 200]
    return x

def n_seeds():
    # Define seeds to be used
    x = [0, 1, 2]
    return x