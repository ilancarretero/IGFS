# Main code to select important features with IGFS method

# Import modules 
import sys
import argparse
import numpy as np
import pandas as pd
import mlflow

# Add the paths to the script directory
sys.path.append('./src/data')

# Import scripts
import load_data
import check_igfs_opt, check_igfs_opt_utils

# Main process
def main_pipeline(args):
    # Load data
    classes = load_data.db_name_classes(args.db)
    X_train, y_train_dummy, t_cv_idx, v_cv_idx, _, _ = load_data.clean_data(args.root_data_path,
                                                                                            args.db,
                                                                                            classes,
                                                                                            args.fs_method)
    # Preprocess train data
    X_train_idxs = X_train.index.tolist()
    X_train = check_igfs_opt_utils.data_scaler(X_train)
    X_train.index = X_train_idxs
    assert X_train.index.tolist() == y_train_dummy.index.tolist(), 'Different indices in X and y'
    # Define seeds 
    seeds = load_data.n_seeds()
    mlflow.set_tracking_uri("http://158.42.170.77:5000")
    mlflow_exp_name = 'DB:' + str.upper(args.db) + '_OPTIMAL_NN'
    if not mlflow.get_experiment_by_name(mlflow_exp_name):
        experiment_id = mlflow.create_experiment(name=mlflow_exp_name)
        mlflow.set_experiment(experiment_id=experiment_id)
    else:
        mlflow.set_experiment(experiment_name=mlflow_exp_name)
    # Loop for igfs method
    for seed in seeds:
        # Run NN and extract IG per validation sample full and correct
        ig_fs = check_igfs_opt.IGFS(X_train, y_train_dummy, t_cv_idx, v_cv_idx, args, classes)
        mean_val_loss_cv = ig_fs.fit(seed)
           
def main():
    parser = argparse.ArgumentParser()
    # Constants
    GLOBAL_DATA_PATH = './data/processed/'
    # Arguments
    parser.add_argument('--root_data_path', default=GLOBAL_DATA_PATH, type=str)
    parser.add_argument('--db', default='crohn', type=str)
    parser.add_argument('--fs_method', default='igfs', type=str)
    # Define args and handle unknown args
    args, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        print(f'Unknown arguments: {unknown}')
    main_pipeline(args=args)
    
if __name__ == "__main__":
    main()


