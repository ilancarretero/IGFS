# Integrated Gradients Feature Selection 

# Import modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import optim

import igfs_attr_utils


# Class for the IG-FS selector
class IGFS:
    def __init__(self, X, y, t_cv_idx, v_cv_idx, args):
        # Definition of basic parameters
        self.X = X
        self.y = y
        self.t_cv_idx = t_cv_idx 
        self.v_cv_idx = v_cv_idx
        self.min_max = MinMaxScaler()
        self.args = args
        
    def fit(self, s_number):
        # Define general variables
        results_cv = np.array([])
        total_y_true = np.array([])
        total_y_pred = np.array([])
        df_attr_full_class_1 = pd.DataFrame()
        df_attr_full_class_0 = pd.DataFrame()
        df_attr_acc_class_1 = pd.DataFrame()
        df_attr_acc_class_0 = pd.DataFrame()
        # Start cv loop
        for i in range(self.t_cv_idx.shape[1]):
            # Define seed for reproducibility
            igfs_attr_utils.set_seed(s_number)
            # Create train and validation splits
            na_value = 900
            t_idx = self.t_cv_idx.iloc[:, i]
            t_idx = t_idx.drop(t_idx[t_idx > na_value].index)
            v_idx = self.v_cv_idx.iloc[:, i]
            v_idx = v_idx.drop(v_idx[v_idx > na_value].index)
            X_train = self.X.iloc[t_idx.tolist()]
            y_train = self.y.iloc[t_idx.tolist()]
            X_val = self.X.iloc[v_idx.tolist()]
            y_val = self.y.iloc[v_idx.tolist()]
            # Convert to torch tensors
            X_train_t, y_train_t, X_val_t, y_val_t = igfs_attr_utils.torch_tensors(X_train,
                                                                               y_train,
                                                                               X_val,
                                                                               y_val)
            # Load parameters and hyperparameters.
            criterion_rec, criterion_clf, learning_rate, epochs, neurons, classes = igfs_attr_utils.optim_param_hparam_nn(self.args)
            # Define NN model and optimizer
            model = igfs_attr_utils.autoencoder_E2E(neurons)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            # Adjust pytorch settings
            X_train_t, y_train_t, X_val_t, y_val_t, criterion_rec, criterion_clf, model = igfs_attr_utils.transfer_device(X_train_t,
                                                                              y_train_t,
                                                                              X_val_t,
                                                                              y_val_t,
                                                                              criterion_rec,
                                                                              criterion_clf,
                                                                              model)
            # Train and evaluate model
            labels, out = igfs_attr_utils.train_eval_loop(epochs,
                                                      X_train_t,
                                                      y_train_t,
                                                      X_val_t,
                                                      y_val_t,
                                                      model,
                                                      optimizer,
                                                      criterion_rec,
                                                      criterion_clf,
                                                      i)
            
            # Obtain metrics
            y_pred, y_true, results = igfs_attr_utils.clf_results(labels, out, classes)
            if i == 0:
                results_cv = np.concatenate((results_cv, results), axis=0)
                results_cv = results_cv.reshape((1, results.shape[0]))
            else:
                results_cv = np.concatenate((results_cv, [results]), axis=0)
            total_y_true = np.concatenate((total_y_true, y_true), axis=None)
            total_y_pred = np.concatenate((total_y_pred, y_pred), axis=None)
            
            # Define submodel and obtain predictions
            submodel = igfs_attr_utils.SubModel(model)
            submodel.eval()
            out_probs = submodel(X_val_t)
            round_pred = torch.round(out_probs.to('cpu')).detach().numpy()
            
            # Obtain attribution for validation set
            val_samples = v_idx.shape[0]
            sorted_attr_df, correct_idx_preds = igfs_attr_utils.IG_attributions(submodel, X_val, y_val, round_pred, val_samples = val_samples)
            subject_number = y_val.index.tolist()
            subject_number.insert(0, 'attr_mean')
            subject_number.insert(0, 'variable')
            col_names = [str(element) for element in subject_number]
            sorted_attr_df.columns = col_names
            
            # Data wrangling and separation by class and correct predictions
            cv_correct_vars_class_1 = igfs_attr_utils.selected_correct_vars_attr(self.y, sorted_attr_df, correct_idx_preds, class_1=True, cv=i, full=False)
            cv_correct_vars_class_0 = igfs_attr_utils.selected_correct_vars_attr(self.y, sorted_attr_df, correct_idx_preds, class_1=False, cv=i, full=False)
            cv_full_vars_class_1 = igfs_attr_utils.selected_correct_vars_attr(self.y, sorted_attr_df, correct_idx_preds, class_1=True, cv=i, full=True)
            cv_full_vars_class_0 = igfs_attr_utils.selected_correct_vars_attr(self.y, sorted_attr_df, correct_idx_preds, class_1=False, cv=i, full=True)
                
            # Concatenate the created datasets and merge it
            if i == 0:
                df_attr_acc_class_1 = pd.concat([df_attr_acc_class_1, cv_correct_vars_class_1],  axis=1)
                df_attr_acc_class_0 = pd.concat([df_attr_acc_class_0, cv_correct_vars_class_0], axis=1)
                df_attr_full_class_1 = pd.concat([df_attr_full_class_1, cv_full_vars_class_1],  axis=1)
                df_attr_full_class_0 = pd.concat([df_attr_full_class_0, cv_full_vars_class_0], axis=1)
            else:
                df_attr_acc_class_1 = df_attr_acc_class_1.merge(cv_correct_vars_class_1, on='variable', how='outer')
                df_attr_acc_class_0 = df_attr_acc_class_0.merge(cv_correct_vars_class_0, on='variable', how='outer')
                df_attr_full_class_1 = df_attr_full_class_1.merge(cv_full_vars_class_1, on='variable', how='outer')
                df_attr_full_class_0 = df_attr_full_class_0.merge(cv_full_vars_class_0, on='variable', how='outer')
                
        return df_attr_acc_class_0, df_attr_acc_class_1, df_attr_full_class_0, df_attr_full_class_1
                
# Save data
def save_data(df_acc, seed, args, full=True, cl=1):
    # Path
    path = args.root_data_path + args.db + '/feature_importance_9_11_2023/igfs_attr/' 
    if full:   
        name = args.db + '_igfs_full_seed_' + str(seed) + '_class_' + str(cl)  +'.xlsx' 
    else:
        name = args.db + '_igfs_acc_seed_' + str(seed) + '_class_' + str(cl)  +'.xlsx' 
    path_name = path + name
    # Save data
    df_acc.to_excel(path_name, index=False)
    
