# Integrated Gradients Feature Selection 

# Import modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import optuna
from torch import optim

import igfs_opt_utils


# Class for the IG-FS selector
class IGFS:
    def __init__(self, X, y, t_cv_idx, v_cv_idx, args, classes):
        # Definition of basic parameters
        self.X = X
        self.y = y
        self.t_cv_idx = t_cv_idx 
        self.v_cv_idx = v_cv_idx
        self.args = args
        self.classes = classes
        
    def fit(self, s_number, params):
        # Loss validation array 
        val_AUC_cv = np.zeros(self.t_cv_idx.shape[1])
        # Start cv loop
        for i in range(self.t_cv_idx.shape[1]):
            # Define seed for reproducibility
            igfs_opt_utils.set_seed(s_number)
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
            X_train_t, y_train_t, X_val_t, y_val_t = igfs_opt_utils.torch_tensors(X_train,
                                                                               y_train,
                                                                               X_val,
                                                                               y_val)
            # Load parameters and hyperparameters.
            criterion_rec, criterion_clf, layers, afunc = igfs_opt_utils.param_hparam_nn(params, self.args, self.X, self.y)
            epochs = 2500
            # Define NN model, optimizer and scheduler
            model = igfs_opt_utils.autoencoder_E2E(layers, afunc, params)
            if params['optimizer'] == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=float(params['learning_rate']))
            elif params['optimizer'] == 'Nadam':
                optimizer = optim.NAdam(model.parameters(), lr=float(params['learning_rate']))
            elif params['optimizer'] == 'adamW':
                optimizer = optim.AdamW(model.parameters(), lr=float(params['learning_rate']))
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50, 
                                                             threshold=0.0001)
            # Adjust pytorch settings
            X_train_t, y_train_t, X_val_t, y_val_t, criterion_rec, criterion_clf, model = igfs_opt_utils.transfer_device(X_train_t,
                                                                              y_train_t,
                                                                              X_val_t,
                                                                              y_val_t,
                                                                              criterion_rec,
                                                                              criterion_clf,
                                                                              model)
            # Train and evaluate model
            best_v_AUC = igfs_opt_utils.train_eval_loop(epochs,
                                                      X_train_t,
                                                      y_train_t,
                                                      X_val_t,
                                                      y_val_t,
                                                      model,
                                                      optimizer,
                                                      scheduler,
                                                      criterion_rec,
                                                      criterion_clf,
                                                      params,
                                                      i,
                                                      self.classes)
            
            # Save validation loss in a np array
            val_AUC_cv[i] = best_v_AUC
            
        mean_val_AUC_cv = np.mean(val_AUC_cv)
        return mean_val_AUC_cv
            
# Optuna functions
def objective(trial, X, y, t_cv_idx, v_cv_idx, args, s_number, classes):
    # Define params
    params = {
        'learning_rate': trial.suggest_categorical('learning_rate', ['0.001', '0.0005', '0.0001', '0.00005', '0.00001']),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'Nadam', 'adamW']),
        'ae_arch': trial.suggest_categorical('ae_arch', ['small', 'big']),
        'latent_space': trial.suggest_int('latent_space', 8, 32, step=8),
        'lambda_1': trial.suggest_categorical('lambda_1', ['0.1', '0.01', '0.001', '0.0001', '0.00001', '0']),
        'lambda_2': trial.suggest_categorical('lambda_2', ['0.1', '0.01', '0.001', '0.0001', '0.00001', '0']),
        'activation_function': trial.suggest_categorical('activation_function', ['relu', 'silu', 'prelu']),
        'batch_normalization': trial.suggest_categorical('batch_normalization', ['true', 'false']),
        'classifier_arch': trial.suggest_categorical('classifier_arch', ['no_hidden', 'hidden']),
        'batch_size': trial.suggest_int('batch_size', 0, 32, step=8),
        'weight_recon': trial.suggest_int('weight_recon', 1, 100, step=1),
        'weight_cl': trial.suggest_int('weight_cl', 1, 100, step=1),
        'cl_weights': trial.suggest_categorical('cl_weights', ['true', 'false'])}   
    # Define NN and train and validate it
    igfs = IGFS(X, y, t_cv_idx, v_cv_idx, args, classes)
    mean_val_loss_cv = igfs.fit(s_number, params)
    return mean_val_loss_cv


    
    
    
