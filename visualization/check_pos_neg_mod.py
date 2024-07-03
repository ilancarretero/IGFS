# Code to check the positive-negative module variable selection 

# Import modules
import os
import argparse
import numpy as np
import pandas as pd

# Basic functions

def load_data(args):
    # Define paths
    global_path = args.root_data_path + args.db + '/' + 'feature_importance/'
    path_pos = global_path + 'igfs_' + args.igfs_method + '_acc_pos/'
    name_pos = args.db + '_igfs_' + args.igfs_method + '_acc_pos_fi_seed_' + str(args.seed) + '_n_vars_' + str(args.vars) + '.xlsx'
    path_neg = global_path + 'igfs_' + args.igfs_method + '_acc_neg/'
    name_neg = args.db + '_igfs_' + args.igfs_method + '_acc_neg_fi_seed_' + str(args.seed) + '_n_vars_' + str(args.vars) + '.xlsx'
    path_pos_neg = global_path + 'igfs_' + args.igfs_method + '_acc_' + args.vars_method + '/'
    name_pos_neg = args.db + '_igfs_' + args.igfs_method + '_acc_' + args.vars_method  +'_fi_seed_' + str(args.seed) + '_n_vars_' + str(args.vars) + '.xlsx'
    # Load data
    df_pos = pd.read_excel(path_pos + name_pos)
    df_neg = pd.read_excel(path_neg + name_neg)
    df_pos_neg = pd.read_excel(path_pos_neg + name_pos_neg)
    # Return dfs
    return df_neg, df_pos, df_pos_neg

def common_vars(df_neg, df_pos, df_pos_neg):
    # Filter by variable and convert into list
    vars_neg = df_neg['VARIABLES'].tolist()
    vars_pos = df_pos['VARIABLES'].tolist()
    vars_pos_neg = df_pos_neg['VARIABLES'].tolist()
    # Obtain common variables
    common_pos_neg = list(set(vars_pos).intersection(set(vars_neg)))
    pos_neg_vars = list(filter(lambda x: x not in common_pos_neg, vars_pos_neg))
    common_pos = list(set(vars_pos).intersection(set(pos_neg_vars)))
    common_neg = list(set(vars_neg).intersection(set(pos_neg_vars)))
    # Return common variables
    return common_neg, common_pos, common_pos_neg

# Main function
def main(args):
    # Load data
    df_neg, df_pos, df_pos_neg = load_data(args)
    # Obtain common variables
    c_neg, c_pos, c_pos_neg = common_vars(df_neg, df_pos, df_pos_neg)
    # Operations
    print(f'Positive: {len(c_pos)}')
    print(f'% Positive: {len(c_pos)/(len(c_pos) + len(c_neg) + len(c_pos_neg))}')
    print(f'Negative: {len(c_neg)}')
    print(f'% Negative: {len(c_neg)/(len(c_pos) + len(c_neg) + len(c_pos_neg))}')
    print(f'Positive-Negative: {len(c_pos_neg)}')
    print(f'% Positive-Negative: {len(c_pos_neg)/(len(c_pos) + len(c_neg) + len(c_pos_neg))}')
    
# Run script
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Constants
    GLOBAL_DATA_PATH = './data/processed/'
    # Arguments
    parser.add_argument('--root_data_path', default=GLOBAL_DATA_PATH, type=str)
    parser.add_argument('--db', default='crohn', type=str)
    parser.add_argument('--vars', default=30, type=int)
    parser.add_argument('--igfs_method', default='median', type=str)
    parser.add_argument('--vars_method', default='pos_neg_mod', type=str)
    parser.add_argument('--seed', default=2, type=int)
    
    args = parser.parse_args()
    main(args)