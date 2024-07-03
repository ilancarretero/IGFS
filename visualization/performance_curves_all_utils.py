# FUNCTIONS TO REPRESENT THE PERFORMANCE CURVES FOR EACH DATABASE 

# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

# Load all .xlsx
def read_data(res_path_soa, res_path_igfs, fs_methods):
    df_results_soa = pd.read_csv(res_path_soa)
    df_results_igfs = pd.read_csv(res_path_igfs)
    df_results = pd.concat([df_results_soa, df_results_igfs])
    fs_datasets = []
    for fs_method in fs_methods:
        df_fs_method = df_results[df_results['fs_method'] == fs_method]
        fs_datasets.append(df_fs_method)
    #all_index = fs_methods.index('all')
    #fs_methods.pop(all_index)
    #fs_datasets.pop(all_index)
    return fs_datasets

# Data wrangling
def data_wrangling(df_list):
    df_list_mean_std = ['none'] * len(df_list)
    for i in range(len(df_list)):
        df_grouped = df_list[i].groupby(['n_sel_vars','model'])['AUC'].mean().reset_index()
        df_grouped = df_grouped.reset_index()
        df_grouped = df_grouped.groupby('n_sel_vars')['AUC'].agg(['mean', 'std']).reset_index()
        df_list_mean_std[i] = df_grouped
    return df_list_mean_std

# Visualization
def viz_igfs(df_list_mean_std, fs_methods, args, viz_path):
    #fig = plt.figure()
    #plt.rc('lines', linewidth=4)
    #plt.rc('axes', prop_cycle=(cycler('color', ['indianred', 'gold', 'olive', 'turquoise', 'darkviolet'])))
    plt.style.use('seaborn-v0_8-whitegrid')
    high_contrast_palette = ['#EE334E', '#228B22', '#FFA500', '#6A1B9A', '#0077BB']
    plt.rcParams['axes.prop_cycle'] = cycler(color=high_contrast_palette)
    fig, ax = plt.subplots()
    # plt.subplots_adjust(left=0.04, right=0.05, top=0.9, bottom=0.1)
    equidistant_pos = df_list_mean_std[0]['n_sel_vars']#np.arange(df_list_mean_std[0].shape[0])
    for i in range(len(df_list_mean_std)):
        #plt.errorbar(equidistant_pos, df_list_mean_std[i]['mean'], yerr=df_list_mean_std[i]['std'], 
        #             elinewidth=2, capsize=4, barsabove=True, label=igfs_types[i])
        upper_bound = df_list_mean_std[i]['mean'] + df_list_mean_std[i]['std']
        lower_bound = df_list_mean_std[i]['mean'] - df_list_mean_std[i]['std']
        if i == 4:
            ax.scatter(equidistant_pos, df_list_mean_std[i]['mean'], s=45, marker="*", color=high_contrast_palette[i], alpha=0.9) # color='gray'
            ax.plot(equidistant_pos, df_list_mean_std[i]['mean'], label=fs_methods[i], color=high_contrast_palette[i], alpha=0.6, linewidth=2)
        else:
            ax.scatter(equidistant_pos, df_list_mean_std[i]['mean'], s=40, marker="*", color=high_contrast_palette[i], alpha=0.8) # color='gray'
            ax.plot(equidistant_pos, df_list_mean_std[i]['mean'], label=fs_methods[i], color=high_contrast_palette[i], alpha=0.5)
        ax.fill_between(equidistant_pos, lower_bound, upper_bound, alpha=0.1, color=high_contrast_palette[i])
    
    start_value = 0.5
    end_value = 1.05
    step = 0.05
    y_ticks = np.arange(start_value, end_value, step)
    ax.set_ylim(bottom=0.5, top=1)
    ax.set_yticks(y_ticks)
    ax.set_xticks(equidistant_pos, df_list_mean_std[0]['n_sel_vars'])
    ax.set_xlabel('Variables', fontsize=11)
    ax.set_ylabel('AUC', fontsize=11)
    #ax.set_title('SoA FS ' + args.db + ' results')
    ax.set_title('Crohn', fontsize=14)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.15,
                 box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=5)
    #plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.savefig(viz_path + 'fs_curves_' + args.db + '.png', dpi=1000)
    plt.show()
        
    