# CODE TO REPRESENT THE PERFORMANCE CURVES FOR EACH DATABASE 

# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

import performance_curves_all_utils

# Main function
def main(args):
    # Define path
    viz_path = './reports/figures/' 
    trial_val_loss = '_best_v_loss_acc_mod'
    trial_auc = '_AUC_acc_mod'
    fs_methods_full = ['snel_batch_full', 'fsnet_recon_1_acc_1_batch_8', 'cae_u_batch_full', 'cae_s_recon_1_acc_1_batch_full', 'igfs_median_acc_pos_neg_mod']
    fs_methods_8 = ['snel_batch_full', 'fsnet_recon_1_acc_1_batch_8', 'cae_u_batch_8', 'cae_s_recon_1_acc_1_batch_8', 'igfs_median_acc_pos_neg']
    fs_methods = fs_methods_full
    if args.db == 'all':
        res_path_colon_soa = './data/processed/' + 'colon' + '/' + 'results/' + 'colon' + '_results_SOA_all_good.csv' 
        res_path_leukemia_soa = './data/processed/' + 'leukemia' + '/' + 'results/' + 'leukemia' + '_results_SOA_all_good.csv' 
        res_path_crohn_soa = './data/processed/' + 'crohn' + '/' + 'results/' + 'crohn' + '_results_SOA_all_good.csv'
        res_path_colon_igfs = './data/processed/' + 'colon' + '/' + 'results/' + 'colon' + '_results_igfs_mean_median' + trial_val_loss + '.csv'
        res_path_leukemia_igfs = './data/processed/' + 'leukemia' + '/' + 'results/' + 'leukemia' + '_results_igfs_mean_median' + trial_val_loss + '.csv' 
        res_path_crohn_igfs = './data/processed/' + 'crohn' + '/' + 'results/' + 'crohn' + '_results_igfs_mean_median' + trial_val_loss + '.csv'
    else:
        res_path_soa = './data/processed/' + args.db + '/' + 'results/' + args.db + '_results_SOA_all_good.csv' 
        res_path_igfs = './data/processed/' + args.db + '/' + 'results/' + args.db + '_results_igfs_mean_median' + trial_val_loss + '.csv' 
        
    
    # Read csv
    if args.db == 'all':
        fs_datasets_colon = performance_curves_all_utils.read_data(res_path_colon_soa, res_path_colon_igfs, fs_methods)
        fs_datasets_leukemia = performance_curves_all_utils.read_data(res_path_leukemia_soa, res_path_leukemia_igfs, fs_methods)
        fs_datasets_crohn = performance_curves_all_utils.read_data(res_path_crohn_soa, res_path_crohn_igfs, fs_methods)
        fs_datasets = []
        for i in range(len(fs_methods)):
            df_concat = pd.concat([fs_datasets_colon[i], fs_datasets_leukemia[i], fs_datasets_crohn[i]])
            fs_datasets.append(df_concat)
    else:
        fs_datasets = performance_curves_all_utils.read_data(res_path_soa, res_path_igfs, fs_methods)
        
    # Data wrangling
    df_list_mean_std = performance_curves_all_utils.data_wrangling(fs_datasets)
    
    # Visualization
    fs_methods = ['SNeL-FS', 'FsNet', 'CAEu', 'CAEs', 'IGFS']
    performance_curves_all_utils.viz_igfs(df_list_mean_std, fs_methods, args, viz_path)
    print('debug')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='crohn', type=str)
    args = parser.parse_args()
    main(args)
    
