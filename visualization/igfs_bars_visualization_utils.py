# FUNCTIONS TO REPRESENT THE PERFORMANCE CURVES FOR EACH DATABASE 

# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

# Define functions
# Data filter
def data_filter(df_, regex, n_sel_vars):
    filter = (df_['fs_method'].str.startswith(regex)) & (df_['n_sel_vars'] == n_sel_vars)
    result_df = df_[filter]
    return result_df

def data_filter_all_vars(df_, regex):
    filter = (df_['fs_method'].str.startswith(regex))
    result_df = df_[filter]
    return result_df

# Data wrangling
def data_wrangling(df_list):
    df_list_mean_std = ['none'] * len(df_list)
    for i in range(len(df_list)):
        df_grouped = df_list[i].groupby(['n_sel_vars','model'])['AUC'].agg(['mean', 'std']).reset_index()
        df_list_mean_std[i] = df_grouped
    return df_list_mean_std

# Visualization
def viz_res_igfs(df_list_mean_std, legend_names, args, viz_path):
    #plt.rc('axes', prop_cycle=(cycler('color', ['mediumaquamarine', 'turquoise', 'teal', 'lightskyblue'])))
    
    ML_models_name = df_list_mean_std[0]['model'].tolist()
    width = 1 / (len(df_list_mean_std) + 1)
    mov = [width*x for x in range(len(df_list_mean_std))]
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rc('axes', prop_cycle=(cycler('color', ['lightcoral', 'powderblue',  'orange', 'palegreen'])))
    # Definimos una paleta de colores azules que son visualmente distintos y complementarios
    complementary_blue_palette = [
        '#002366',  # Azul oscuro, como el de la marina
        '#48D1CC',   # Cian medio, un azul verdoso
        '#0057B7',  # Azul cobalto, un tono medio de azul puro  
        '#89CFF0',  # Azul bebé, claro y con alto brillo 
    ]
    # plt.rcParams['axes.prop_cycle'] = cycler(color=complementary_blue_palette)
    fig, ax = plt.subplots()
    #ax = fig.add_axes([0,0,1,1])
    equidistant_pos = np.arange(df_list_mean_std[0]['model'].unique().shape[0])
    for i in range(len(df_list_mean_std)):
        #plt.errorbar(equidistant_pos, df_list_mean_std[i]['mean'], yerr=df_list_mean_std[i]['std'], 
        #             elinewidth=2, capsize=4, barsabove=True, label=igfs_types[i])
        # if i == 0:
        #     y = df_list_mean_std[i][df_list_mean_std[i]['VARIABLES'] == df_list_mean_std[i]['VARIABLES'].max()]
        #     plt.bar(equidistant_pos + mov[i], y['mean'], width=0.2, label=legend_names[i])
        #     plt.errorbar(equidistant_pos + mov[i], y['mean'], yerr=y['std'], fmt='o',
        #                  color='gray', alpha=0.4, barsabove=True)
        # else:
        y = df_list_mean_std[i] #[df_list_mean_std[i]['VARIABLES'] == args.vars]
        ax.bar(equidistant_pos + mov[i], y['mean'], width=width, label=legend_names[i], alpha=0.7)
        ax.errorbar(equidistant_pos + mov[i], y['mean'], yerr=y['std'], fmt='o',
                        color='gray', alpha = 0.4, barsabove=True)
    
    start_value = 0.5
    end_value = 1.05
    step = 0.05
    y_ticks = np.arange(start_value, end_value, step)
    ax.set_ylim(bottom=0.5, top=1.01)
    ax.set_yticks(y_ticks)    
    ax.set_xticks(equidistant_pos + mov[-1]/2, ML_models_name)
    ax.set_xlabel('ML models', fontsize=11)
    ax.set_ylabel('AUC', fontsize=11)
    #ax.set_title('ML models ' + args.db + ' results')
    ax.set_title('Average', fontsize=14)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.85])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=4)
    #plt.tight_layout()
    plt.savefig(viz_path + 'ML_models_' + args.db + '.png', dpi=1000)
    plt.show()

