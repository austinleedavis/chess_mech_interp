import os
import warnings
from tqdm.auto import tqdm
import pandas as pd

def make_4th_step_feathers():
    source_folder = 'chess_data/feathers'  # Path to the folder with the original feather files
    target_folder = 'chess_data/feathers_step4_eval'  # Path to the folder where the modified feather files will be saved

    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)

    # Iterate through each feather file in the source folder
    for filename in tqdm(sorted(os.listdir(source_folder))):
        if filename.endswith('.feather'):
            # Load the dataframe from the feather file
            df = pd.read_feather(os.path.join(source_folder, filename))
            
            # Select every 4th row
            filtered_df = df.iloc[2::80, :]
            
            # Save the filtered dataframe to a new feather file in the target folder
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                filtered_df.reset_index(drop=True).to_feather(os.path.join(target_folder, filename))

    # This code will go through each feather file in the source folder, filter out every 4th row,
    # and save these filtered dataframes into new feather files in the target folder.

def merge_probe_datasets():
    # Path to the directory containing the feather files
    directory_path = 'chess_data/feathers_step4_eval'

    # List all feather files in the directory
    feather_files = [f for f in os.listdir(directory_path) if f.endswith('.feather')]

    # Initialize an empty DataFrame to hold the concatenated dataset
    full_dataset = pd.DataFrame()
    i = 1
    # Loop through the feather files and concatenate them into the full dataset
    for i, file_name in tqdm(enumerate(feather_files),total = len(feather_files)):
        file_path = os.path.join(directory_path, file_name)
        batch_df = pd.read_feather(file_path)
        full_dataset = pd.concat([full_dataset, batch_df], ignore_index=True)
    full_dataset.to_feather(f'chess_data/eval_probe_dataset.feather')
            
            
if __name__ == '__main__':
    merge_probe_datasets()