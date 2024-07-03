# Code to run ML and obtain metrics for the selected variables in each model

# Import modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
import mlflow

# Define functions
def data_scaler(X_train, X_test):
    # Normalize train  and test with MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns.tolist())
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns.tolist())
    # Return partitions and normalized data
    return X_train_scaled, X_test_scaled

def np_conversion(X_train, y_train, X_test, y_test):
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values
    return X_train, y_train, X_test, y_test

def select_relevant_features(data_path, db, fs_method, seed, sel_vars, X):
    # Define data path
    path = data_path + db + '/feature_importance/' + fs_method + '/'
    name_file = db + '_' + fs_method + '_fi_seed_' + str(seed) + '_n_vars_' + str(sel_vars) + '.xlsx'
    path_file = path + name_file
    # Read file
    selected_vars = pd.read_excel(path_file) 
    # Select variables
    selected_vars = selected_vars[selected_vars.columns[0]].tolist()
    # Find vars idxs in the X_train dataframe
    col_idxs = [X.columns.get_loc(col) for col in selected_vars]
    # Remove duplicate vars selection
    col_idxs = list(set(col_idxs))
    return col_idxs
    
def ML_classifiers(seed_number):
    classifiers = {
            'NB': GaussianNB(),
            'SVC': SVC(kernel='rbf', C=1, random_state=seed_number),
            'RF': RandomForestClassifier(criterion='entropy', random_state=seed_number),
            'KNN': KNeighborsClassifier(),
            'LR': LogisticRegression(solver='liblinear', random_state=seed_number),
            'XGB': XGBClassifier(random_state=seed_number)
            }
    return classifiers

def report_classification_results(y_true, y_pred, classes):
    """ Print confusion matrix and show metrics """
    # Extract confusion matrix
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    conf_mat = confusion_matrix(y_true, y_pred)
    print(conf_mat)
    # Compute metrics
    headers = []
    g_TP, g_TN, g_FN, g_FP = [],[],[],[]
    ss, ee, pp, nn, ff, aa = [],[],[],[],[],[]
    S, E, PPV, NPV, F, ACC = ['Sensitivity'], ['Specificity'], ['PPV (precision)'], ['NPV'], ['F1-score'], ['Accuracy'] #Positive Predictive Value/Negative Predictive Value
    i = 1
    headers.append(classes[i])
    # Extract indicators per class
    TP = conf_mat[i,i]
    #TN = sum(sum(conf_mat))-sum(conf_mat[i,:])-sum(conf_mat[:,i])
    FN = sum(conf_mat[i,:])-conf_mat[i,i]
    FP = sum(conf_mat[:,i])-conf_mat[i,i]
    TN = sum(sum(conf_mat))-TP-FP-FN
    # Extract metrics per class
    sen = TP/(TP+FN)
    spe = TN/(TN+FP)
    ppv = TP/(TP+FP)
    npv = TN/(TN+FN)
    f1_s = 2*(ppv*sen/(ppv+sen))
    acc = (TP+TN)/(TP+TN+FP+FN)
    # Create table for printing
    S.append(str(round(sen,4)))
    E.append(str(round(spe,4)))
    PPV.append(str(round(ppv,4)))
    NPV.append(str(round(npv,4)))
    F.append(str(round(f1_s,4)))
    ACC.append(str(round(acc,4)))
    auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    metrics_array = np.array([acc, sen, spe, ppv, npv, f1_s, auc])
    return metrics_array

def train_pred_loop(classifiers, X_train, y_train, X_test, y_test, classes, seed, sel_vars, args):
    # Loop for train and predictions
    for name, classifier in classifiers.items():
        # Create pipeline with the classifier
        pipeline = Pipeline([
                    ('clf', classifier)
                ])
        # Train the classifier
        pipeline.fit(X_train, y_train)
        # Apply to the test set
        y_pred = pipeline.predict(X_test)
        # Compute metrics
        results = report_classification_results(y_test, y_pred, classes)
        # mlflow run name
        if sel_vars != X_train.shape[1]:
            mlflow_run_name = args.db + '_' + args.fs_method + '_seed_' + str(seed) + '_sel_vars_' + str(sel_vars) + '_model_' + name
            with mlflow.start_run(run_name=mlflow_run_name) as mlflow_run:
                # Log parameters
                mlflow.log_param('dataset', args.db)
                mlflow.log_param('fs_method', args.fs_method)
                mlflow.log_param('seed', seed)
                mlflow.log_param('n_sel_vars', sel_vars)
                mlflow.log_param('model', name)
                # Log metrics
                mlflow.log_metric('ACC', results[0])
                mlflow.log_metric('SEN', results[1])
                mlflow.log_metric('SPE', results[2])
                mlflow.log_metric('PPV', results[3])
                mlflow.log_metric('NPV', results[4])
                mlflow.log_metric('F1', results[5])
                mlflow.log_metric('AUC', results[6])
        else:
            mlflow_run_name = args.db + '_' + args.fs_method + '_seed_' + str(seed) + '_sel_vars_' + str(sel_vars) + '_model_' + name
            with mlflow.start_run(run_name=mlflow_run_name) as mlflow_run:
                # Log parameters
                mlflow.log_param('dataset', args.db)
                mlflow.log_param('fs_method', args.fs_method)
                mlflow.log_param('seed', seed)
                mlflow.log_param('n_sel_vars', sel_vars)
                mlflow.log_param('model', name)
                # Log metrics
                mlflow.log_metric('ACC', results[0])
                mlflow.log_metric('SEN', results[1])
                mlflow.log_metric('SPE', results[2])
                mlflow.log_metric('PPV', results[3])
                mlflow.log_metric('NPV', results[4])
                mlflow.log_metric('F1', results[5])
                mlflow.log_metric('AUC', results[6])
            

