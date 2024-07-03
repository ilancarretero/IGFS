# Main code to select important features with IGFS method

# Import modules 
import sys
import argparse
import numpy as np
import pandas as pd
import optuna

# Add the paths to the script directory
sys.path.append('./src/data')

# Import scripts
import load_data
import igfs_opt_crohn, igfs_opt_utils_crohn

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
    X_train = igfs_opt_utils_crohn.data_scaler(X_train)
    X_train.index = X_train_idxs
    assert X_train.index.tolist() == y_train_dummy.index.tolist(), 'Different indices in X and y'
    # Define seeds
    seeds = load_data.n_seeds()
    seed = seeds[0]
    # Create optuna study
    study = optuna.create_study(study_name=args.db + '_' + args.fs_method + '_model_AUC',
                                storage='sqlite:////Workspace/models/igfs/opt/' + args.db + '_' + args.fs_method + '_model_optimization_AUC.db',
                                direction='maximize')
    # Optimize study
    study.optimize(lambda trial: igfs_opt_crohn.objective(trial, X_train, y_train_dummy,
                                                    t_cv_idx, v_cv_idx,
                                                    args, seed, classes), n_trials=args.n_trials)
    
            
def main():
    parser = argparse.ArgumentParser()
    # Constants
    GLOBAL_DATA_PATH = './data/processed/'
    # Arguments
    parser.add_argument('--root_data_path', default=GLOBAL_DATA_PATH, type=str)
    parser.add_argument('--db', default='crohn', type=str)
    parser.add_argument('--fs_method', default='igfs', type=str)
    parser.add_argument('--n_trials', default=1000, type=int)
    # Define args and handle unknown args
    args, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        print(f'Unknown arguments: {unknown}')
    main_pipeline(args=args)
    
if __name__ == "__main__":
    main()


