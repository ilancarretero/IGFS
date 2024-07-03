# Main code to select important features with IGFS method

# Import modules 
import sys
import argparse
import numpy as np
import pandas as pd

# Add the paths to the script directory
sys.path.append('./src/data')

# Import scripts
import load_data
import igfs_attr_opt_val_loss_auc, igfs_attr_opt_utils_val_loss_auc

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
    X_train = igfs_attr_opt_utils_val_loss_auc.data_scaler(X_train)
    X_train.index = X_train_idxs
    assert X_train.index.tolist() == y_train_dummy.index.tolist(), 'Different indices in X and y'
    # Define seeds
    seeds = load_data.n_seeds()
    # Loop for igfs method
    for seed in seeds:
    # Run NN and extract IG per validation sample full and correct
        ig_fs = igfs_attr_opt_val_loss_auc.IGFS(X_train, y_train_dummy, t_cv_idx, v_cv_idx, args, classes)
        df_attr_acc_class_0, df_attr_acc_class_1, df_attr_full_class_0, df_attr_full_class_1 = ig_fs.fit(seed)
        # Save data
        igfs_attr_opt_val_loss_auc.save_data(df_attr_acc_class_0, seed, args, full=False, cl=0)
        igfs_attr_opt_val_loss_auc.save_data(df_attr_acc_class_1, seed, args, full=False, cl=1)
        igfs_attr_opt_val_loss_auc.save_data(df_attr_full_class_0, seed, args, full=True, cl=0)
        igfs_attr_opt_val_loss_auc.save_data(df_attr_full_class_1, seed, args, full=True, cl=1)
            
def main():
    parser = argparse.ArgumentParser()
    # Constants
    GLOBAL_DATA_PATH = './data/processed/'
    GLOBAL_MODELS_PATH = './models/'
    # Arguments
    parser.add_argument('--root_data_path', default=GLOBAL_DATA_PATH, type=str)
    parser.add_argument('--root_models_path', default=GLOBAL_MODELS_PATH, type=str)
    parser.add_argument('--db', default='colon', type=str)
    parser.add_argument('--fs_method', default='igfs', type=str)
    # Define args and handle unknown args
    args, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        print(f'Unknown arguments: {unknown}')
    main_pipeline(args=args)
    
if __name__ == "__main__":
    main()


