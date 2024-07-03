# Main code to select random features with random method

# Import modules
import sys
import argparse

# Add the paths to the script directory
sys.path.append('./src/data')

# Import scripts
import load_data
import random_var_sel

# Main process
def main_pipeline(args):
    # Load data
    classes = load_data.db_name_classes(args.db)
    X_train, _, _, _, _, _ = load_data.clean_data(args.root_data_path,
                                                  args.db,
                                                  classes,
                                                  args.fs_method)
    # Define vars selected and seeds
    n_vars = load_data.n_vars_to_select()
    seeds = load_data.n_seeds()
    # Loop for random variable selection method
    for seed in seeds:
        for sel_vars in n_vars:
            # Select random variables
            selected_vars_df = random_var_sel.random_selection(X_train,
                                                               seed,
                                                               sel_vars)
            # Save dataframe
            random_var_sel.save_dataframe(selected_vars_df,
                                          seed,
                                          sel_vars,
                                          args)
    
def main():
    parser = argparse.ArgumentParser()
    # Constants
    GLOBAL_DATA_PATH = './data/processed/'
    # Arguments
    parser.add_argument('--root_data_path', default=GLOBAL_DATA_PATH, type=str)
    parser.add_argument('--db', default='colon', type=str)
    parser.add_argument('--fs_method', default='random', type=str)
    # Define args and handle unknown args
    args, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        print(f'Unknown arguments: {unknown}')
    main_pipeline(args=args)
    
if __name__ == "__main__":
    main()