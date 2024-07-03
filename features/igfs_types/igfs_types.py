# Functios to apply in the main script

# Load modules
import os
import numpy as np
import pandas as pd

# Define functions
def load_igfs_data(args, seed):
   # Define paths
   global_path = args.root_data_path + args.db + '/feature_importance/' + args.fs_method + '_attr/'
   name_0 = args.db + '_' + args.fs_method + '_' + args.igfs_type + '_seed_' + str(seed) + '_class_0.xlsx'
   name_1 = args.db + '_' + args.fs_method + '_' + args.igfs_type +'_seed_' + str(seed) + '_class_1.xlsx'
   path_cl_0 = global_path + name_0
   path_cl_1 = global_path + name_1
   # Load data
   df_cl_0 = pd.read_excel(path_cl_0)
   df_cl_1 = pd.read_excel(path_cl_1)
   # Return data
   return df_cl_0, df_cl_1

def sorted_mean_median(df_cl_0, df_cl_1, sorted_by='mean'):
    # Add mean/median column
    if sorted_by == 'mean':
        df_cl_0['mean'] = df_cl_0.filter(like='obs').mean(axis=1)
        df_cl_1['mean'] = df_cl_1.filter(like='obs').mean(axis=1)
    else: 
        df_cl_0['median'] = df_cl_0.filter(like='obs').median(axis=1)
        df_cl_1['median'] = df_cl_1.filter(like='obs').median(axis=1)
    # Sort by mean/median column
    df_cl_0_sorted = df_cl_0.sort_values(by=sorted_by, ascending=True) 
    df_cl_1_sorted = df_cl_1.sort_values(by=sorted_by, ascending=False) 
    # Cols selected
    columns_to_select = ['variable', sorted_by]
    # Dataframes created
    df_cl_0_sorted = df_cl_0_sorted[columns_to_select]
    df_cl_1_sorted = df_cl_1_sorted[columns_to_select]
    # Return dataframes
    return df_cl_0_sorted, df_cl_1_sorted

def pos_or_neg_attr(df_cl_type, n_vars, occ=False):
    # Select only the first n_vars
    df_top_n = df_cl_type.head(n_vars)
    if occ:
        df_top_n.columns = ['VARIABLES', 'SCORE', 'FREQUENCY']
    else:
        df_top_n.columns = ['VARIABLES', 'SCORE']
    # return df
    return df_top_n

def pos_neg_attr(df_cl_0_type, df_cl_1_type, n_vars):
    # Select only the first n_vars
    half_n_vars = n_vars//2
    df_top_n_cl_0 = df_cl_0_type.head(n_vars)
    df_top_n_cl_1 = df_cl_1_type.head(n_vars)
    df_top_n_cl_0.columns = ['VARIABLES', 'SCORE']
    df_top_n_cl_1.columns = ['VARIABLES', 'SCORE']
    # Scores of cl 0 in absolute value
    df_top_n_cl_0['SCORE'] = df_top_n_cl_0['SCORE'].abs()
    # Intersection between both dataframes variables
    merged_df = pd.merge(df_top_n_cl_0.head(half_n_vars), df_top_n_cl_1.head(half_n_vars), on='VARIABLES', how='outer', suffixes=('_cl_0', '_cl_1'))
    merged_df['SCORE']=np.where(
        merged_df[['SCORE_cl_0', 'SCORE_cl_1']].notna().all(axis=1),
        merged_df[['SCORE_cl_0', 'SCORE_cl_1']].mean(axis=1),
        np.nan
    )
    intersect_df = merged_df[merged_df['SCORE'].notna()]
    if not intersect_df.empty:
        df_top_n = intersect_df[['VARIABLES', 'SCORE']]
        n_intersect = len(df_top_n)
        vars_to_remove = df_top_n['VARIABLES'].tolist()
        df_top_n_cl_0_filter = df_top_n_cl_0[~df_top_n_cl_0.isin(vars_to_remove).any(axis=1)]
        df_top_n_cl_1_filter = df_top_n_cl_1[~df_top_n_cl_1.isin(vars_to_remove).any(axis=1)]
        half_n_vars = (n_vars - n_intersect) // 2
        if n_vars % 2 != 0:
            df_top_n_cl_0_half = df_top_n_cl_0_filter.head(half_n_vars+1)
            df_top_n_cl_1_half = df_top_n_cl_1_filter.head(half_n_vars+1)
            last_cl_0 = df_top_n_cl_0_half.iloc[-1]['SCORE']
            last_cl_1 = df_top_n_cl_1_half.iloc[-1]['SCORE']
            if last_cl_0 > last_cl_1:
                df_top_n_cl_1_half.drop(df_top_n_cl_1_half.index[-1], inplace=True)  
            else:
                df_top_n_cl_0_half.drop(df_top_n_cl_0_half.index[-1], inplace=True) 
            df_top_n = pd.concat([df_top_n, df_top_n_cl_0_half, df_top_n_cl_1_half])
        else:
            df_top_n_cl_0_half = df_top_n_cl_0_filter.head(half_n_vars)
            df_top_n_cl_1_half = df_top_n_cl_1_filter.head(half_n_vars)
            df_top_n = pd.concat([df_top_n, df_top_n_cl_0_half, df_top_n_cl_1_half])  
    else:
        if n_vars % 2 != 0:
            df_top_n_cl_0_half = df_top_n_cl_0.head(half_n_vars+1)
            df_top_n_cl_1_half = df_top_n_cl_1.head(half_n_vars+1)
            df_top_n_cl_0_half.columns = ['VARIABLES', 'SCORE']
            df_top_n_cl_1_half.columns = ['VARIABLES', 'SCORE']
            last_cl_0 = df_top_n_cl_0_half.iloc[-1]['SCORE']
            last_cl_1 = df_top_n_cl_1_half.iloc[-1]['SCORE']
            if last_cl_0 > last_cl_1:
                df_top_n_cl_1_half.drop(df_top_n_cl_1_half.index[-1], inplace=True)  
            else:
                df_top_n_cl_0_half.drop(df_top_n_cl_0_half.index[-1], inplace=True) 
            df_top_n = pd.concat([df_top_n_cl_0_half, df_top_n_cl_1_half])
        else:
            df_top_n_cl_0_half = df_top_n_cl_0.head(half_n_vars)
            df_top_n_cl_1_half = df_top_n_cl_1.head(half_n_vars)
            df_top_n = pd.concat([df_top_n_cl_0_half, df_top_n_cl_1_half])
    df_top_n.columns = ['VARIABLES', 'SCORE']
    # return df
    return df_top_n

def pos_neg_attr_occ(df_cl_0_type, df_cl_1_type, n_vars):
    # Select only the first n_vars
    half_n_vars = n_vars//2
    df_top_n_cl_0 = df_cl_0_type.head(n_vars)
    df_top_n_cl_1 = df_cl_1_type.head(n_vars)
    # Scores of cl 0 in absolute value
    df_top_n_cl_0 = df_top_n_cl_0['SCORE'].abs()
    # Intersection between both dataframes variables
    merged_df = pd.merge(df_top_n_cl_0.head(half_n_vars), df_top_n_cl_1.head(half_n_vars), on='VARIABLES', how='outer', suffixes=('_cl_0', '_cl_1'))
    merged_df['SCORE']=np.where(
        merged_df[['SCORE_cl_0', 'SCORE_cl_1']].notna().all(axis=1),
        merged_df[['SCORE_cl_0', 'SCORE_cl_1']].mean(axis=1),
        np.nan
    )
    merged_df['FREQUENCY']=np.where(
        merged_df[['FREQUENCY_cl_0', 'FREQUENCY_cl_1']].notna().all(axis=1),
        merged_df[['FREQUENCY_cl_0', 'FREQUENCY_cl_1']].mean(axis=1),
        np.nan
    )
    intersect_df = merged_df[merged_df['SCORE'].notna()]
    if not intersect_df.empty:
        df_top_n = intersect_df[['VARIABLES', 'SCORE', 'FREQUENCY']]
        n_intersect = len(df_top_n)
        vars_to_remove = df_top_n['VARIABLES'].tolist()
        df_top_n_cl_0_filter = df_top_n_cl_0[~df_top_n_cl_0.isin(vars_to_remove).any(axis=1)]
        df_top_n_cl_1_filter = df_top_n_cl_1[~df_top_n_cl_1.isin(vars_to_remove).any(axis=1)]
        half_n_vars = (n_vars - n_intersect) // 2
        if n_vars % 2 != 0:
            df_top_n_cl_0_half = df_top_n_cl_0_filter.head(half_n_vars+1)
            df_top_n_cl_1_half = df_top_n_cl_1_filter.head(half_n_vars+1)
            last_cl_0 = df_top_n_cl_0_half.iloc[-1]['SCORE'] 
            last_cl_1 = df_top_n_cl_1_half.iloc[-1]['SCORE']  
            if last_cl_0 > last_cl_1:
                df_top_n_cl_1_half.drop(df_top_n_cl_1_half.index[-1], inplace=True)  
            else:
                df_top_n_cl_0_half.drop(df_top_n_cl_0_half.index[-1], inplace=True) 
            df_top_n = pd.concat([df_top_n, df_top_n_cl_0_half, df_top_n_cl_1_half])
        else:
            df_top_n_cl_0_half = df_top_n_cl_0_filter.head(half_n_vars)
            df_top_n_cl_1_half = df_top_n_cl_1_filter.head(half_n_vars)
            df_top_n = pd.concat([df_top_n, df_top_n_cl_0_half, df_top_n_cl_1_half])  
    else:
        if n_vars % 2 != 0:
            df_top_n_cl_0_half = df_top_n_cl_0.head(half_n_vars+1)
            df_top_n_cl_1_half = df_top_n_cl_1.head(half_n_vars+1)
            last_cl_0 = df_top_n_cl_0_half.iloc[-1]['SCORE'] 
            last_cl_1 = df_top_n_cl_1_half.iloc[-1]['SCORE']  
            if last_cl_0 > last_cl_1:
                df_top_n_cl_1_half.drop(df_top_n_cl_1_half.index[-1], inplace=True)  
            else:
                df_top_n_cl_0_half.drop(df_top_n_cl_0_half.index[-1], inplace=True) 
            df_top_n = pd.concat([df_top_n_cl_0_half, df_top_n_cl_1_half])
        else:
            df_top_n_cl_0_half = df_top_n_cl_0.head(half_n_vars)
            df_top_n_cl_1_half = df_top_n_cl_1.head(half_n_vars)
            df_top_n = pd.concat([df_top_n_cl_0_half, df_top_n_cl_1_half])
    df_top_n.columns = ['VARIABLES', 'SCORE', 'FREQUENCY']
    # return df
    return df_top_n

def sorted_occ_mean_median(df_cl, sel_vars, cl='0', sorted_by='mean'):
    # Creation of a new dataframe saving variables and values per occurrence
    obs_cols = df_cl.filter(regex='^obs').columns.tolist()
    df_sorted_occ = pd.DataFrame()
    for obs_col in obs_cols:
        if cl == '0':
            df_sorted = df_cl.sort_values(by=obs_col, ascending=True)
        else:
            df_sorted = df_cl.sort_values(by=obs_col, ascending=False)
        df_sorted_redux = df_sorted[['variable', obs_col]]
        df_sorted_redux.columns = [obs_col + '_variable', obs_col]
        df_sorted_redux = df_sorted_redux.reset_index(drop=True)
        df_sorted_occ = pd.concat([df_sorted_occ, df_sorted_redux], axis=1)
        df_sorted_occ = df_sorted_occ.reset_index(drop=True)
    # Filter by the number of selected vars
    df_sorted_occ = df_sorted_occ.head(sel_vars)
    # Divide dataset into categorical and numerical variables
    df_sorted_occ_cat = df_sorted_occ.filter(regex='_variable$')
    df_sorted_occ_num = df_sorted_occ.filter(regex='^(?!.*_variable$)')
    # Find the variables that are more frequent
    most_frequent_vars = df_sorted_occ_cat.stack().value_counts()
    attr_values = []
    for value, count in most_frequent_vars.items():
        indices = [(i, j) for i in range(df_sorted_occ_cat.shape[0]) for j in range(df_sorted_occ_cat.shape[1]) if df_sorted_occ_cat.iloc[i,j] == value]
        df_sorted_occ_num_filter = [df_sorted_occ_num.iloc[i, j] for i, j in indices]
        if sorted_by == 'mean':
            attr_value = pd.Series(df_sorted_occ_num_filter).mean()
            attr_values.append(attr_value)
        else:
            attr_value = pd.Series(df_sorted_occ_num_filter).median()
            attr_values.append(attr_value)
    # Normalize frequency values
    most_frequent_vars = most_frequent_vars/len(obs_cols)
    occ_data = {'VARIABLES': most_frequent_vars.index.tolist(),
                'SCORE': attr_values,
                'FREQUENCY': most_frequent_vars.tolist()}
    df_occ_cl = pd.DataFrame(occ_data)
    df_occ_cl_sorted = df_occ_cl.sort_values(by=['FREQUENCY', 'SCORE'], ascending=[False, False])
    return df_occ_cl_sorted

def save_data(X_df, seed, sel_vars, args, selector, type_selector):
    # Define global path
    global_path = args.root_data_path + args.db + '/feature_importance/' 
    specific_path = args.fs_method + '_' + selector + '_' + args.igfs_type + '_' + type_selector + '/'
    if not os.path.exists(global_path + specific_path):
        os.makedirs(global_path + specific_path)
    name = args.db + '_' + args.fs_method + '_' + selector + '_' + args.igfs_type + '_' + type_selector + '_fi_seed_' + str(seed) + '_n_vars_' + str(sel_vars) + '.xlsx'
    path = global_path + specific_path + name
    # Save data
    X_df.to_excel(path, index=False)
    
    
def pos_neg_attr_good(df_cl_0_type, df_cl_1_type, n_vars):
    # Select only the first n_vars
    half_n_vars = n_vars//2
    df_top_n_cl_0 = df_cl_0_type.head(n_vars)
    df_top_n_cl_1 = df_cl_1_type.head(n_vars)
    df_top_n_cl_0.columns = ['VARIABLES', 'SCORE']
    df_top_n_cl_1.columns = ['VARIABLES', 'SCORE']
    # Scores of cl 0 in absolute value
    df_top_n_cl_0['SCORE'] = df_top_n_cl_0['SCORE'].abs()
    # Intersection between both dataframes variables
    merged_df = pd.merge(df_top_n_cl_0.head(half_n_vars), df_top_n_cl_1.head(half_n_vars), on='VARIABLES', how='outer', suffixes=('_cl_0', '_cl_1'))
    merged_df['SCORE']=np.where(
        merged_df[['SCORE_cl_0', 'SCORE_cl_1']].notna().all(axis=1),
        merged_df[['SCORE_cl_0', 'SCORE_cl_1']].mean(axis=1),
        np.nan
    )
    intersect_df = merged_df[merged_df['SCORE'].notna()]
    if not intersect_df.empty:
        df_top_n = intersect_df[['VARIABLES', 'SCORE']]
        n_intersect = len(df_top_n)
        vars_to_remove = df_top_n['VARIABLES'].tolist()
        df_top_n_cl_0_filter = df_top_n_cl_0[~df_top_n_cl_0.isin(vars_to_remove).any(axis=1)]
        df_top_n_cl_1_filter = df_top_n_cl_1[~df_top_n_cl_1.isin(vars_to_remove).any(axis=1)]
        half_n_vars = (n_vars - n_intersect) // 2
        while not intersect_df.empty:
            # Intersection between both dataframes variables
            merged_df = pd.merge(df_top_n_cl_0_filter.head(half_n_vars), df_top_n_cl_1_filter.head(half_n_vars), on='VARIABLES', how='outer', suffixes=('_cl_0', '_cl_1'))
            merged_df['SCORE']=np.where(
                merged_df[['SCORE_cl_0', 'SCORE_cl_1']].notna().all(axis=1),
                merged_df[['SCORE_cl_0', 'SCORE_cl_1']].mean(axis=1),
                np.nan
            )
            intersect_df = merged_df[merged_df['SCORE'].notna()]
            df_subtop_n = intersect_df[['VARIABLES', 'SCORE']]
            n_intersect = len(df_top_n) + len(df_subtop_n)
            vars_to_remove = df_subtop_n['VARIABLES'].tolist()
            df_top_n_cl_0_filter = df_top_n_cl_0_filter[~df_top_n_cl_0_filter.isin(vars_to_remove).any(axis=1)]
            df_top_n_cl_1_filter = df_top_n_cl_1_filter[~df_top_n_cl_1_filter.isin(vars_to_remove).any(axis=1)]
            half_n_vars = (n_vars - n_intersect) // 2
            n_vars_rem = n_vars - n_intersect
            df_top_n = pd.concat([df_top_n, df_subtop_n], ignore_index=True)
        if n_vars_rem % 2 != 0:
            df_top_n_cl_0_half = df_top_n_cl_0_filter.head(half_n_vars+1)
            df_top_n_cl_1_half = df_top_n_cl_1_filter.head(half_n_vars+1)
            last_cl_0 = df_top_n_cl_0_half.iloc[-1]['SCORE']
            last_cl_1 = df_top_n_cl_1_half.iloc[-1]['SCORE']
            if last_cl_0 > last_cl_1:
                df_top_n_cl_1_half.drop(df_top_n_cl_1_half.index[-1], inplace=True)  
            else:
                df_top_n_cl_0_half.drop(df_top_n_cl_0_half.index[-1], inplace=True) 
            df_top_n = pd.concat([df_top_n, df_top_n_cl_0_half, df_top_n_cl_1_half], ignore_index=True)
        else:
            df_top_n_cl_0_half = df_top_n_cl_0_filter.head(half_n_vars)
            df_top_n_cl_1_half = df_top_n_cl_1_filter.head(half_n_vars)
            df_top_n = pd.concat([df_top_n, df_top_n_cl_0_half, df_top_n_cl_1_half], ignore_index=True)  
    else:
        if n_vars % 2 != 0:
            df_top_n_cl_0_half = df_top_n_cl_0.head(half_n_vars+1)
            df_top_n_cl_1_half = df_top_n_cl_1.head(half_n_vars+1)
            df_top_n_cl_0_half.columns = ['VARIABLES', 'SCORE']
            df_top_n_cl_1_half.columns = ['VARIABLES', 'SCORE']
            last_cl_0 = df_top_n_cl_0_half.iloc[-1]['SCORE']
            last_cl_1 = df_top_n_cl_1_half.iloc[-1]['SCORE']
            if last_cl_0 > last_cl_1:
                df_top_n_cl_1_half.drop(df_top_n_cl_1_half.index[-1], inplace=True)  
            else:
                df_top_n_cl_0_half.drop(df_top_n_cl_0_half.index[-1], inplace=True) 
            df_top_n = pd.concat([df_top_n_cl_0_half, df_top_n_cl_1_half])
        else:
            df_top_n_cl_0_half = df_top_n_cl_0.head(half_n_vars)
            df_top_n_cl_1_half = df_top_n_cl_1.head(half_n_vars)
            df_top_n = pd.concat([df_top_n_cl_0_half, df_top_n_cl_1_half])
    df_top_n.columns = ['VARIABLES', 'SCORE']
    # return df
    return df_top_n

def pos_neg_attr_occ_good(df_cl_0_type, df_cl_1_type, n_vars):
    # Select only the first n_vars
    half_n_vars = n_vars//2
    df_top_n_cl_0 = df_cl_0_type.head(n_vars)
    df_top_n_cl_1 = df_cl_1_type.head(n_vars)
    # Scores of cl 0 in absolute value
    df_top_n_cl_0['SCORE'] = df_top_n_cl_0['SCORE'].abs()
    # Intersection between both dataframes variables
    merged_df = pd.merge(df_top_n_cl_0.head(half_n_vars), df_top_n_cl_1.head(half_n_vars), on='VARIABLES', how='outer', suffixes=('_cl_0', '_cl_1'))
    merged_df['SCORE']=np.where(
        merged_df[['SCORE_cl_0', 'SCORE_cl_1']].notna().all(axis=1),
        merged_df[['SCORE_cl_0', 'SCORE_cl_1']].mean(axis=1),
        np.nan
    )
    merged_df['FREQUENCY']=np.where(
        merged_df[['FREQUENCY_cl_0', 'FREQUENCY_cl_1']].notna().all(axis=1),
        merged_df[['FREQUENCY_cl_0', 'FREQUENCY_cl_1']].mean(axis=1),
        np.nan
    )
    intersect_df = merged_df[merged_df['SCORE'].notna()]
    if not intersect_df.empty:
        df_top_n = intersect_df[['VARIABLES', 'SCORE', 'FREQUENCY']]
        n_intersect = len(df_top_n)
        vars_to_remove = df_top_n['VARIABLES'].tolist()
        df_top_n_cl_0_filter = df_top_n_cl_0[~df_top_n_cl_0.isin(vars_to_remove).any(axis=1)]
        df_top_n_cl_1_filter = df_top_n_cl_1[~df_top_n_cl_1.isin(vars_to_remove).any(axis=1)]
        half_n_vars = (n_vars - n_intersect) // 2
        while not intersect_df.empty:
            # Intersection between both dataframes variables
            merged_df = pd.merge(df_top_n_cl_0_filter.head(half_n_vars), df_top_n_cl_1_filter.head(half_n_vars), on='VARIABLES', how='outer', suffixes=('_cl_0', '_cl_1'))
            merged_df['SCORE']=np.where(
                merged_df[['SCORE_cl_0', 'SCORE_cl_1']].notna().all(axis=1),
                merged_df[['SCORE_cl_0', 'SCORE_cl_1']].mean(axis=1),
                np.nan
            )
            merged_df['FREQUENCY']=np.where(
                merged_df[['FREQUENCY_cl_0', 'FREQUENCY_cl_1']].notna().all(axis=1),
                merged_df[['FREQUENCY_cl_0', 'FREQUENCY_cl_1']].mean(axis=1),
                np.nan
            )
            intersect_df = merged_df[merged_df['SCORE'].notna()]
            df_subtop_n = intersect_df[['VARIABLES', 'SCORE', 'FREQUENCY']]
            n_intersect = len(df_top_n) + len(df_subtop_n)
            vars_to_remove = df_subtop_n['VARIABLES'].tolist()
            df_top_n_cl_0_filter = df_top_n_cl_0_filter[~df_top_n_cl_0_filter.isin(vars_to_remove).any(axis=1)]
            df_top_n_cl_1_filter = df_top_n_cl_1_filter[~df_top_n_cl_1_filter.isin(vars_to_remove).any(axis=1)]
            half_n_vars = (n_vars - n_intersect) // 2
            n_vars_rem = n_vars - n_intersect
            df_top_n = pd.concat([df_top_n, df_subtop_n], ignore_index=True)
        if n_vars_rem % 2 != 0:
            df_top_n_cl_0_half = df_top_n_cl_0_filter.head(half_n_vars+1)
            df_top_n_cl_1_half = df_top_n_cl_1_filter.head(half_n_vars+1)
            last_cl_0 = df_top_n_cl_0_half.iloc[-1]['SCORE'] + df_top_n_cl_0_half.iloc[-1]['FREQUENCY'] * 100
            last_cl_1 = df_top_n_cl_1_half.iloc[-1]['SCORE'] + df_top_n_cl_0_half.iloc[-1]['FREQUENCY'] * 100
            if last_cl_0 > last_cl_1:
                df_top_n_cl_1_half.drop(df_top_n_cl_1_half.index[-1], inplace=True)  
            else:
                df_top_n_cl_0_half.drop(df_top_n_cl_0_half.index[-1], inplace=True) 
            df_top_n = pd.concat([df_top_n, df_top_n_cl_0_half, df_top_n_cl_1_half], ignore_index=True)
        else:
            df_top_n_cl_0_half = df_top_n_cl_0_filter.head(half_n_vars)
            df_top_n_cl_1_half = df_top_n_cl_1_filter.head(half_n_vars)
            df_top_n = pd.concat([df_top_n, df_top_n_cl_0_half, df_top_n_cl_1_half], ignore_index=True)  
    else:
        if n_vars % 2 != 0:
            df_top_n_cl_0_half = df_top_n_cl_0.head(half_n_vars+1)
            df_top_n_cl_1_half = df_top_n_cl_1.head(half_n_vars+1)
            last_cl_0 = df_top_n_cl_0_half.iloc[-1]['SCORE'] + df_top_n_cl_0_half.iloc[-1]['FREQUENCY'] * 100
            last_cl_1 = df_top_n_cl_1_half.iloc[-1]['SCORE'] + df_top_n_cl_0_half.iloc[-1]['FREQUENCY'] * 100
            if last_cl_0 > last_cl_1:
                df_top_n_cl_1_half.drop(df_top_n_cl_1_half.index[-1], inplace=True)  
            else:
                df_top_n_cl_0_half.drop(df_top_n_cl_0_half.index[-1], inplace=True) 
            df_top_n = pd.concat([df_top_n_cl_0_half, df_top_n_cl_1_half])
        else:
            df_top_n_cl_0_half = df_top_n_cl_0.head(half_n_vars)
            df_top_n_cl_1_half = df_top_n_cl_1.head(half_n_vars)
            df_top_n = pd.concat([df_top_n_cl_0_half, df_top_n_cl_1_half])
    df_top_n.columns = ['VARIABLES', 'SCORE', 'FREQUENCY']
    # return df
    return df_top_n

def pos_neg_attr_module(df_cl_0_type, df_cl_1_type, n_vars, occ=False):
    # Select only the first n_vars
    df_top_n_cl_0 = df_cl_0_type.head(n_vars)
    df_top_n_cl_1 = df_cl_1_type.head(n_vars)
    if occ:
        df_top_n_cl_0.columns = ['VARIABLES', 'SCORE', 'FREQUENCY']
        df_top_n_cl_1.columns = ['VARIABLES', 'SCORE', 'FREQUENCY']
        # Scores of cl 0 in absolute value
        df_top_n_cl_0['SCORE'] = df_top_n_cl_0['SCORE'].abs()
        # Concatenate both datasets
        df_top_n = pd.concat([df_top_n_cl_1, df_top_n_cl_0], ignore_index=True)
        # Sort by frequency and score
        df_top_n = df_top_n.sort_values(by=['FREQUENCY', 'SCORE'], ascending=[False, False])
        # Delete duplicates
        df_top_n = df_top_n.drop_duplicates(subset='VARIABLES', keep='first')
        # Select variables
        df_top_n = df_top_n.head(n_vars)
        # Return df_top_n
        return df_top_n
    else:
        df_top_n_cl_0.columns = ['VARIABLES', 'SCORE']
        df_top_n_cl_1.columns = ['VARIABLES', 'SCORE']
        # Scores of cl 0 in absolute value
        df_top_n_cl_0['SCORE'] = df_top_n_cl_0['SCORE'].abs()
        # Concatenate both datasets
        df_top_n = pd.concat([df_top_n_cl_1, df_top_n_cl_0], ignore_index=True)
        # Sort by score
        df_top_n = df_top_n.sort_values(by='SCORE', ascending=False)
        # Delete duplicates
        df_top_n = df_top_n.drop_duplicates(subset='VARIABLES', keep='first')
        # Select variables
        df_top_n = df_top_n.head(n_vars)
        # Return df_top_n
        return df_top_n

        