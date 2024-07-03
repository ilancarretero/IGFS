# Main code to compute ML models and obtain metrics

# Import modules
import sys
import argparse
import mlflow

# Add the paths to the script directory
sys.path.append('./src/data')

# Import scripts
import load_data
import ML_models, ML_models_utils

# Main process
def main_pipeline(args):
    # Load data
    classes = load_data.db_name_classes(args.db)
    X_train, y_train_dummy, t_cv_idx, v_cv_idx, X_test, y_test_dummy = load_data.clean_data(args.root_data_path,
                                                                                            args.db,
                                                                                            classes,
                                                                                            args.fs_method)
    # Preprocess data
    X_train_scaled, X_test_scaled = ML_models.data_scaler(X_train, X_test)
    # Numpy conversion
    X_train_scaled, y_train, X_test_scaled, y_test = ML_models.np_conversion(X_train_scaled,
                                                                             y_train_dummy,
                                                                             X_test_scaled,
                                                                             y_test_dummy)
    # Define vars selected and seeds
    n_vars = load_data.n_vars_to_select()
    seeds = load_data.n_seeds()
    # mlflow tracking uri and exp name
    mlflow.set_tracking_uri("http://158.42.170.77:5000")
    mlflow_exp_name = 'DB:' + str.upper(args.db) + '_FS:' + str.upper(args.fs_method)
    if not mlflow.get_experiment_by_name(mlflow_exp_name):
        experiment_id = mlflow.create_experiment(name=mlflow_exp_name)
        mlflow.set_experiment(experiment_id=experiment_id)
    else:
        mlflow.set_experiment(experiment_name=mlflow_exp_name)
    # Loop for ML methods
    for seed in seeds:
        classifiers = ML_models.ML_classifiers(seed)
        if args.fs_method != 'all':
            for sel_vars in n_vars:
                #TODO: implement functions with MLFlow variables 
                # mlflow 
                # Select variables
                col_idxs = ML_models.select_relevant_features(args.root_data_path,
                                                                   args.db,
                                                                   args.fs_method,
                                                                   seed,
                                                                   sel_vars,
                                                                   X_train)
                X_train_scaled_sel_vars = X_train_scaled[:, col_idxs]
                X_test_scaled_sel_vars = X_test_scaled[:, col_idxs]
                # Train models
                ML_models.train_pred_loop(classifiers,
                                          X_train_scaled_sel_vars,
                                          y_train,
                                          X_test_scaled_sel_vars,
                                          y_test,
                                          classes,
                                          seed,
                                          sel_vars,
                                          args)
        else:
            # Complete dataset
            sel_vars = X_train_scaled.shape[1]
            # Train models
            ML_models.train_pred_loop(classifiers,
                                      X_train_scaled,
                                      y_train,
                                      X_test_scaled,
                                      y_test,
                                      classes,
                                      seed,
                                      sel_vars,
                                      args)
            pass
            
        
def main():
    parser = argparse.ArgumentParser()
    # Constants
    GLOBAL_DATA_PATH = './data/processed/'
    # Arguments
    parser.add_argument('--root_data_path', default=GLOBAL_DATA_PATH, type=str)
    parser.add_argument('--db', default='colon', type=str)
    parser.add_argument('--fs_method', default='igfs_mean_acc_neg', type=str)
    # Define args and handle unknown args
    args, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        print(f'Unknown arguments: {unknown}')
    main_pipeline(args=args)
    
if __name__ == "__main__":
    main()
    
     