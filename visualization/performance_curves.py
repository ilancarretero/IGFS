# CODE TO REPRESENT THE PERFORMANCE CURVES FOR EACH DATABASE 

# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

import performance_curves_utils

# Main function
def main(args):
    # Define path
    viz_path = './reports/figures/' 
    if args.db == 'all':
        res_path_colon = './data/processed/' + 'colon' + '/' + 'results/' + 'colon' + '_results_SOA_all_good.csv' 
        res_path_leukemia = './data/processed/' + 'leukemia' + '/' + 'results/' + 'leukemia' + '_results_SOA_all_good.csv' 
        res_path_crohn = './data/processed/' + 'crohn' + '/' + 'results/' + 'crohn' + '_results_SOA_all_good.csv'
    else:
        res_path = './data/processed/' + args.db + '/' + 'results/' + args.db + '_results_SOA_all_good.csv' 
        
    
    # Read csv
    if args.db == 'all':
        fs_methods, fs_datasets_colon = performance_curves_utils.read_data(res_path_colon)
        fs_methods, fs_datasets_leukemia = performance_curves_utils.read_data(res_path_leukemia)
        fs_methods, fs_datasets_crohn = performance_curves_utils.read_data(res_path_crohn)
        fs_datasets = []
        for i in range(len(fs_methods)):
            df_concat = pd.concat([fs_datasets_colon[i], fs_datasets_leukemia[i], fs_datasets_crohn[i]])
            fs_datasets.append(df_concat)
    else:
        fs_methods, fs_datasets = performance_curves_utils.read_data(res_path)
        
    # Data wrangling
    df_list_mean_std = performance_curves_utils.data_wrangling(fs_datasets)
    
    # Visualization
    performance_curves_utils.viz_igfs(df_list_mean_std, fs_methods, args, viz_path)
    print('debug')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='all', type=str)
    
    args = parser.parse_args()
    main(args)
    
