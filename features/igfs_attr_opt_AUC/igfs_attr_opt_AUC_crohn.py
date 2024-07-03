# Integrated Gradients Feature Selection 

# Import modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import optim
import os

import igfs_attr_opt_utils_AUC_crohn


# Class for the IG-FS selector
class IGFS:
    def __init__(self, X, y, t_cv_idx, v_cv_idx, args, classes):
        # Definition of basic parameters
        self.X = X
        self.y = y
        self.t_cv_idx = t_cv_idx 
        self.v_cv_idx = v_cv_idx
        self.min_max = MinMaxScaler()
        self.args = args
        self.classes = classes
        
    def fit(self, s_number):
        # AUC validation array 
        val_AUC_cv = np.zeros(self.t_cv_idx.shape[1])
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
            igfs_attr_opt_utils_AUC_crohn.set_seed(s_number)
            # Define model path and name
            save_model_path = self.args.root_models_path + self.args.fs_method + '/' + self.args.db + '/' + 'aeE2E_'  + self.args.db + '_seed_' + str(s_number) + '_fold_' + str(i) + '.pth'
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
            X_train_t, y_train_t, X_val_t, y_val_t = igfs_attr_opt_utils_AUC_crohn.torch_tensors(X_train,
                                                                               y_train,
                                                                               X_val,
                                                                               y_val)
            # Load parameters and hyperparameters.
            params = igfs_attr_opt_utils_AUC_crohn.get_params(self.args)
            criterion_rec, criterion_clf, layers, afunc = igfs_attr_opt_utils_AUC_crohn.param_hparam_nn(params, self.args, self.X, self.y)
            epochs = 2500
            # Define NN model and optimizer
            model = igfs_attr_opt_utils_AUC_crohn.autoencoder_E2E(layers, afunc, params)
            if params['optimizer'] == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=float(params['learning_rate']))
            elif params['optimizer'] == 'Nadam':
                optimizer = optim.NAdam(model.parameters(), lr=float(params['learning_rate']))
            elif params['optimizer'] == 'adamW':
                optimizer = optim.AdamW(model.parameters(), lr=float(params['learning_rate']))
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50, 
                                                             threshold=0.0001)
            # Adjust pytorch settings
            X_train_t, y_train_t, X_val_t, y_val_t, criterion_rec, criterion_clf, model = igfs_attr_opt_utils_AUC_crohn.transfer_device(X_train_t,
                                                                              y_train_t,
                                                                              X_val_t,
                                                                              y_val_t,
                                                                              criterion_rec,
                                                                              criterion_clf,
                                                                              model)
            # Train and evaluate model
            if not os.path.exists(save_model_path):
                print(f'--------------------- TRAIN MODEL AND SAVE IT INTO {save_model_path} -------------------------------------')
                labels, out, epoch = igfs_attr_opt_utils_AUC_crohn.train_eval_loop(epochs,
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
                                                        self.classes,
                                                        save_model_path)
                # Obtain metrics
                y_pred, y_true, results = igfs_attr_opt_utils_AUC_crohn.clf_results(labels, torch.sigmoid(out), self.classes)
                if i == 0:
                    results_cv = np.concatenate((results_cv, results), axis=0)
                    results_cv = results_cv.reshape((1, results.shape[0]))
                else:
                    results_cv = np.concatenate((results_cv, [results]), axis=0)
                total_y_true = np.concatenate((total_y_true, y_true), axis=None)
                total_y_pred = np.concatenate((total_y_pred, y_pred), axis=None)
            else:
                print(f'LOADING MODEL: {save_model_path}')
            # Load model, define submodel and obtain predictions
            loaded_model = igfs_attr_opt_utils_AUC_crohn.autoencoder_E2E(layers, afunc, params)
            loaded_model.load_state_dict(torch.load(save_model_path))
            submodel = igfs_attr_opt_utils_AUC_crohn.SubModel(loaded_model)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            submodel.to(device)
            submodel.eval()
            out = submodel(X_val_t)
            round_pred = torch.round(torch.sigmoid(out).to('cpu')).detach().numpy()
            
            # Obtain attribution for validation set
            val_samples = v_idx.shape[0]
            sorted_attr_df, correct_idx_preds = igfs_attr_opt_utils_AUC_crohn.IG_attributions(submodel, X_val, y_val, round_pred, val_samples = val_samples)
            subject_number = y_val.index.tolist()
            subject_number.insert(0, 'attr_mean')
            subject_number.insert(0, 'variable')
            col_names = [str(element) for element in subject_number]
            sorted_attr_df.columns = col_names
            
            # Data wrangling and separation by class and correct predictions
            cv_correct_vars_class_1 = igfs_attr_opt_utils_AUC_crohn.selected_correct_vars_attr(self.y, sorted_attr_df, correct_idx_preds, class_1=True, cv=i, full=False)
            cv_correct_vars_class_0 = igfs_attr_opt_utils_AUC_crohn.selected_correct_vars_attr(self.y, sorted_attr_df, correct_idx_preds, class_1=False, cv=i, full=False)
            cv_full_vars_class_1 = igfs_attr_opt_utils_AUC_crohn.selected_correct_vars_attr(self.y, sorted_attr_df, correct_idx_preds, class_1=True, cv=i, full=True)
            cv_full_vars_class_0 = igfs_attr_opt_utils_AUC_crohn.selected_correct_vars_attr(self.y, sorted_attr_df, correct_idx_preds, class_1=False, cv=i, full=True)
                
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
    path = args.root_data_path + args.db + '/feature_importance/igfs_attr/' 
    if full:   
        name = args.db + '_igfs_full_seed_' + str(seed) + '_class_' + str(cl)  +'.xlsx' 
    else:
        name = args.db + '_igfs_acc_seed_' + str(seed) + '_class_' + str(cl)  +'.xlsx' 
    path_name = path + name
    # Save data
    df_acc.to_excel(path_name, index=False)
    
