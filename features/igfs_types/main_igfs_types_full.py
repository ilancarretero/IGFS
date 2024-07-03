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
import igfs_types

# Main process
def main_pipeline(args):
    # Define vars selected and seeds
    n_vars = load_data.n_vars_to_select()
    seeds = load_data.n_seeds()
    # Loop for igfs method
    for seed in seeds:
        # Load data
        df_cl_0, df_cl_1 = igfs_types.load_igfs_data(args, seed)
        # Mean and median sorted 
        df_cl_0_mean, df_cl_1_mean = igfs_types.sorted_mean_median(df_cl_0,
                                                                   df_cl_1,
                                                                   sorted_by='mean')
        df_cl_0_median, df_cl_1_median = igfs_types.sorted_mean_median(df_cl_0,
                                                                       df_cl_1,
                                                                       sorted_by='median')
        for sel_vars in n_vars:
            # Define mean, mode and median with pos, neg and pos-neg final excels
            # 1. MEAN CASE
            df_pos_mean = igfs_types.pos_or_neg_attr(df_cl_1_mean, sel_vars)
            df_neg_mean = igfs_types.pos_or_neg_attr(df_cl_0_mean, sel_vars)
            df_pos_neg_mean = igfs_types.pos_neg_attr(df_cl_0_mean, df_cl_1_mean, sel_vars)
            # 2. MEDIAN CASE
            df_pos_median = igfs_types.pos_or_neg_attr(df_cl_1_median, sel_vars)
            df_neg_median = igfs_types.pos_or_neg_attr(df_cl_0_median, sel_vars)
            df_pos_neg_median = igfs_types.pos_neg_attr(df_cl_0_median, df_cl_1_median, sel_vars)
            # 3. OCC_MEAN CASE
            df_cl_0_occ_mean = igfs_types.sorted_occ_mean_median(df_cl_0,
                                                                 sel_vars,
                                                                 sorted_by='mean')
            df_cl_1_occ_mean = igfs_types.sorted_occ_mean_median(df_cl_1,
                                                                 sel_vars,
                                                                 sorted_by='mean')
            df_pos_occ_mean = igfs_types.pos_or_neg_attr(df_cl_1_occ_mean, sel_vars, occ=True)
            df_neg_occ_mean = igfs_types.pos_or_neg_attr(df_cl_0_occ_mean, sel_vars, occ=True)
            df_pos_neg_occ_mean = igfs_types.pos_neg_attr_occ(df_cl_0_occ_mean, df_cl_1_occ_mean, sel_vars)
            # 3. OCC_MEDIAN CASE
            df_cl_0_occ_median = igfs_types.sorted_occ_mean_median(df_cl_0,
                                                                 sel_vars,
                                                                 sorted_by='median')
            df_cl_1_occ_median = igfs_types.sorted_occ_mean_median(df_cl_1,
                                                                 sel_vars,
                                                                 sorted_by='median')
            df_pos_occ_median = igfs_types.pos_or_neg_attr(df_cl_1_occ_median, sel_vars, occ=True)
            df_neg_occ_median = igfs_types.pos_or_neg_attr(df_cl_0_occ_median, sel_vars, occ=True)
            df_pos_neg_occ_median = igfs_types.pos_neg_attr_occ(df_cl_0_occ_median, df_cl_1_occ_median, sel_vars)
            # Save excels
            # 1. MEAN CASE
            igfs_types.save_data(df_pos_mean, seed, sel_vars, args, 'mean', 'pos')
            igfs_types.save_data(df_neg_mean, seed, sel_vars, args, 'mean', 'neg')
            igfs_types.save_data(df_pos_neg_mean, seed, sel_vars, args, 'mean', 'pos_neg')
            # 2. MEDIAN CASE
            igfs_types.save_data(df_pos_median, seed, sel_vars, args, 'median', 'pos')
            igfs_types.save_data(df_neg_median, seed, sel_vars, args, 'median', 'neg')
            igfs_types.save_data(df_pos_neg_median, seed, sel_vars, args, 'median', 'pos_neg')
            # 3. OCC_MEAN CASE
            igfs_types.save_data(df_pos_occ_mean, seed, sel_vars, args, 'mode_mean', 'pos')
            igfs_types.save_data(df_neg_occ_mean, seed, sel_vars, args, 'mode_mean', 'neg')
            igfs_types.save_data(df_pos_neg_occ_mean, seed, sel_vars, args, 'mode_mean', 'pos_neg')
            # 4. OCC_MEDIAN CASE
            igfs_types.save_data(df_pos_occ_median, seed, sel_vars, args, 'mode_median', 'pos')
            igfs_types.save_data(df_neg_occ_median, seed, sel_vars, args, 'mode_median', 'neg')
            igfs_types.save_data(df_pos_neg_occ_median, seed, sel_vars, args, 'mode_median', 'pos_neg')
                 
def main():
    parser = argparse.ArgumentParser()
    # Constants
    GLOBAL_DATA_PATH = './data/processed/'
    # Arguments
    parser.add_argument('--root_data_path', default=GLOBAL_DATA_PATH, type=str)
    parser.add_argument('--db', default='leukemia', type=str)
    parser.add_argument('--fs_method', default='igfs', type=str)
    parser.add_argument('--igfs_type', default='full', type=str)
    # Define args and handle unknown args
    args, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        print(f'Unknown arguments: {unknown}')
    main_pipeline(args=args)
    
if __name__ == "__main__":
    main()


