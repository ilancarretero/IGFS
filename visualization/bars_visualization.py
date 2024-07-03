# CODE TO REPRESENT THE PERFORMANCE CURVES FOR EACH DATABASE 

# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

import bars_visualization_utils

# Main function
def main(args):
    # Define path
    viz_path = './reports/figures/' 
    trial_val_loss = '_best_v_loss_acc_mod'
    trial_AUC = '_AUC_acc_mod'
    trial = trial_val_loss
    if args.db == 'all':
        res_path_colon_mm = args.root_data_path + 'colon' + '/' + 'results/' + 'colon' + '_results_igfs_' + 'mean_median' + trial + '.csv' 
        res_path_colon_mmm = args.root_data_path + 'colon' + '/' + 'results/' + 'colon' + '_results_igfs_' + 'mode_mean_median' + trial + '.csv' 
        res_path_leukemia_mm = args.root_data_path + 'leukemia' + '/' + 'results/' + 'leukemia' + '_results_igfs_' + 'mean_median' + trial + '.csv' 
        res_path_leukemia_mmm = args.root_data_path + 'leukemia' + '/' + 'results/' + 'leukemia' + '_results_igfs_' + 'mode_mean_median' + trial + '.csv' 
        res_path_crohn_mm = args.root_data_path + 'crohn' + '/' + 'results/' + 'crohn' + '_results_igfs_' + 'mean_median' + trial + '.csv' 
        res_path_crohn_mmm = args.root_data_path + 'crohn' + '/' + 'results/' + 'crohn' + '_results_igfs_' + 'mode_mean_median' + trial + '.csv'
        res_path_colon_soa = './data/processed/' + 'colon' + '/' + 'results/' + 'colon' + '_results_SOA_all_good.csv' 
        res_path_leukemia_soa = './data/processed/' + 'leukemia' + '/' + 'results/' + 'leukemia' + '_results_SOA_all_good.csv' 
        res_path_crohn_soa = './data/processed/' + 'crohn' + '/' + 'results/' + 'crohn' + '_results_SOA_all_good.csv' 
    else:
        res_path_mm = args.root_data_path + args.db + '/' + 'results/' + args.db + '_results_igfs_' + 'mean_median' + trial + '.csv'
        res_path_mmm = args.root_data_path + args.db + '/' + 'results/' + args.db + '_results_igfs_' + 'mode_mean_median' + trial + '.csv'
        res_path_soa = './data/processed/' + args.db + '/' + 'results/' + args.db + '_results_SOA_all_good.csv' 
        
    # Read csv
    if args.db == 'all':
        # Colon
        colon_df_mm = pd.read_csv(res_path_colon_mm)
        colon_df_mmm = pd.read_csv(res_path_colon_mmm)
        colon_df_soa = pd.read_csv(res_path_colon_soa)
        colon_df = pd.concat([colon_df_mm, colon_df_mmm, colon_df_soa])
        # Leukemia
        leukemia_df_mm = pd.read_csv(res_path_leukemia_mm)
        leukemia_df_mmm = pd.read_csv(res_path_leukemia_mmm)
        leukemia_df_soa = pd.read_csv(res_path_leukemia_soa)
        leukemia_df = pd.concat([leukemia_df_mm, leukemia_df_mmm, leukemia_df_soa])
        # Crohn
        crohn_df_mm = pd.read_csv(res_path_crohn_mm)
        crohn_df_mmm = pd.read_csv(res_path_crohn_mmm)
        crohn_df_soa = pd.read_csv(res_path_crohn_soa)
        crohn_df = pd.concat([crohn_df_mm, crohn_df_mmm, crohn_df_soa])
        # All
        df = pd.concat([colon_df, leukemia_df, crohn_df])
    else:
        # Read db
        df_mm = pd.read_csv(res_path_mm)
        df_mmm = pd.read_csv(res_path_mmm)
        df_soa = pd.read_csv(res_path_soa)
        df = pd.concat([df_mm, df_mmm, df_soa])
        
    # Data wrangling
    if args.comparison == 'selector':
        regex = ['igfs_mean', 'igfs_median', 'igfs_mode_mean', 'igfs_mode_median']
    elif args.comparison == 'type':
        regex = ['igfs_' + args.type + '_acc', 'igfs_' + args.type + '_full']
    elif args.comparison == 'samples':
        regex = ['igfs_' + args.type + '_' + args.samples + '_neg', 'igfs_' + args.type + '_' + args.samples + '_pos', 'igfs_' + args.type + '_' + args.samples + '_pos_neg']
    elif args.comparison == 'all_rand_igfs':
        regex = ['igfs_mode_mean_acc_pos_neg', 'all', 'random']
        
    df_list = []
    for i in range(len(regex)):
        result_df = bars_visualization_utils.data_filter_all_vars(df, regex[i], args.vars)
        df_list.append(result_df)
    df_list_mean_std = bars_visualization_utils.data_wrangling(df_list)
       
    # Visualization
    igfs_names = [x.replace('igfs_', '') for x in regex]
    # igfs_names = ['Mean', 'Median', 'Mode (mean)', 'Mode (median)']
    bars_visualization_utils.viz_res_igfs(df_list_mean_std, regex, args, viz_path)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Constants
    GLOBAL_DATA_PATH = './data/processed/'
    # Arguments
    parser.add_argument('--root_data_path', default=GLOBAL_DATA_PATH, type=str)
    parser.add_argument('--db', default='colon', type=str)
    parser.add_argument('--vars', default=30, type=int)
    parser.add_argument('--comparison', default='all_rand_igfs', type=str)
    parser.add_argument('--type', default='mode_mean', type=str)
    parser.add_argument('--samples', default='acc', type=str)
    
    args = parser.parse_args()
    main(args)
    
