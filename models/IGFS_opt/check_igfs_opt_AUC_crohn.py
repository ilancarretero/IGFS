# Integrated Gradients Feature Selection 

# Import modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import optim
import mlflow
import os

import check_igfs_opt_utils_AUC_crohn

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
        
    def fit(self, s_number):
        # AUC validation array 
        val_AUC_cv = np.zeros(self.t_cv_idx.shape[1])
        # Start cv loop
        for i in range(self.t_cv_idx.shape[1]):
            # Define seed for reproducibility
            check_igfs_opt_utils_AUC_crohn.set_seed(s_number)
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
            X_train_t, y_train_t, X_val_t, y_val_t = check_igfs_opt_utils_AUC_crohn.torch_tensors(X_train,
                                                                               y_train,
                                                                               X_val,
                                                                               y_val)
            # Load parameters and hyperparameters.
            params = check_igfs_opt_utils_AUC_crohn.get_params(self.args)
            criterion_rec, criterion_clf, layers, afunc = check_igfs_opt_utils_AUC_crohn.param_hparam_nn(params, self.args, self.X, self.y)
            epochs = 2500
            # Define NN model and optimizer
            model = check_igfs_opt_utils_AUC_crohn.autoencoder_E2E(layers, afunc, params)
            if params['optimizer'] == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=float(params['learning_rate']))
            elif params['optimizer'] == 'Nadam':
                optimizer = optim.NAdam(model.parameters(), lr=float(params['learning_rate']))
            elif params['optimizer'] == 'adamW':
                optimizer = optim.AdamW(model.parameters(), lr=float(params['learning_rate']))
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50, 
                                                             threshold=0.0001)
            # Adjust pytorch settings
            X_train_t, y_train_t, X_val_t, y_val_t, criterion_rec, criterion_clf, model = check_igfs_opt_utils_AUC_crohn.transfer_device(X_train_t,
                                                                              y_train_t,
                                                                              X_val_t,
                                                                              y_val_t,
                                                                              criterion_rec,
                                                                              criterion_clf,
                                                                              model)
            # Train and evaluate model
            mlflow_run_name = self.args.db + '_NN_optimization_seed_' + str(s_number) + '_fold_' + str(i)
            mlflow.start_run(run_name=mlflow_run_name)
            best_v_AUC, best_v_loss, best_t_loss, best_results_t, best_results_v, y_true_total_t, best_y_pred_total_t, y_true_v, best_y_pred_v, best_epochs = check_igfs_opt_utils_AUC_crohn.train_eval_loop(epochs,
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
            # Results 
            if not os.path.exists(self.args.root_data_path + 'artifacts'):
                os.makedirs(self.args.root_data_path + 'artifacts')
            name_txt = self.args.root_data_path + 'artifacts/predictions.txt'
            with open(name_txt, 'w') as f:
                name_vecs = ['y_true_train', 'y_pred_train', 'y_true_val', 'y_pred_val']
                vecs = [y_true_total_t, best_y_pred_total_t, y_true_v.squeeze().tolist(), best_y_pred_v.squeeze().tolist()]
                for k in zip(name_vecs, vecs):
                    vec_str = k[0] + str(k[1])
                    f.write(vec_str + '\n')
            mlflow.log_artifact(self.args.root_data_path + 'artifacts/predictions.txt')
            mlflow.log_metric('best train loss', best_t_loss)
            mlflow.log_metric('best val loss', best_v_loss)
            mlflow.log_metric('best epochs', best_epochs)
            mlflow.log_metric('best train ACC', best_results_t[0])
            mlflow.log_metric('best train SEN', best_results_t[1])
            mlflow.log_metric('best train SPE', best_results_t[2])
            mlflow.log_metric('best train PPV', best_results_t[3])
            mlflow.log_metric('best train NPV', best_results_t[4])
            mlflow.log_metric('best train F1', best_results_t[5])
            mlflow.log_metric('best train AUC', best_results_t[6])
            mlflow.log_metric('best val ACC', best_results_v[0])
            mlflow.log_metric('best val SEN', best_results_v[1])
            mlflow.log_metric('best val SPE', best_results_v[2])
            mlflow.log_metric('best val PPV', best_results_v[3])
            mlflow.log_metric('best val NPV', best_results_v[4])
            mlflow.log_metric('best val F1', best_results_v[5])
            mlflow.log_metric('best val AUC', best_results_v[6])
            
            mlflow.end_run()
            
            # Save validation loss in a np array
            val_AUC_cv[i] = best_v_AUC
            
        mean_val_loss_cv = np.mean(val_AUC_cv)
        return mean_val_loss_cv