# Code to make random variable selection 

# Import modules
import random
import numpy as np
import pandas as pd

# Define functions
def random_selection(X, seed, sel_vars):
    # Random selection of variables
    col_names = X.columns.tolist()
    # Set seed
    random.seed(seed)
    # Random selection
    selected_vars = random.sample(col_names, sel_vars)
    # Convert into dataframe
    selected_vars_df = pd.DataFrame({'VARIABLES': selected_vars})
    return selected_vars_df

def save_dataframe(X_df, seed, sel_vars, args):
    # Define path 
    path = args.root_data_path + args.db + '/feature_importance/' + args.fs_method + '/'
    name = args.db + '_' + args.fs_method + '_fi_seed_' + str(seed) + '_n_vars_' + str(sel_vars) + '.xlsx'
    path_file = path + name
    # Save dataframe
    X_df.to_excel(path_file, index=False)
