import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# Paths to your original ECG data
folders = {
    'train': '../train_folder',
    'validation': '../validation_folder'
}

# Paths to the folders where you want to save the cropped ECG data
new_folders = {
    'train': '../cropped_train_folder',
    'validation': '../cropped_validation_folder'
}

# Function to randomly crop a 1D ECG array
def random_crop(ecg_data, crop_length):
    assert ecg_data.shape[0] >= crop_length
    start_idx = np.random.randint(0, ecg_data.shape[0] - crop_length)
    cropped_ecg = ecg_data[start_idx : start_idx + crop_length]
    return cropped_ecg

# Function to upsample a 1D ECG array
def upsample_ecg(cropped_ecg, original_length):
    # Create an array of indices for the original ECG data
    old_indices = np.linspace(0, cropped_ecg.shape[0] - 1, cropped_ecg.shape[0])

    # Create an array of indices for the upsampled ECG data
    new_indices = np.linspace(0, cropped_ecg.shape[0] - 1, original_length)

    # Use interpolation to create the upsampled ECG data
    interpolator = interp1d(old_indices, cropped_ecg, axis=0)
    upsampled_ecg = interpolator(new_indices)
    return upsampled_ecg

# Loop over each type of data (train and validation)
for data_type in folders.keys():
    folder = folders[data_type]
    new_folder = new_folders[data_type]

    # Create the new folder if it does not exist
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    # Loop over each file in the folder
    for file_name in os.listdir(folder):
        # Ensure we're working with CSV files
        if file_name.endswith('.csv'):
            # Construct the full file path
            file_path = os.path.join(folder, file_name)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Convert the DataFrame to a NumPy array
            ecg_data = df.values

            # Crop the ECG data
            cropped_ecg = random_crop(ecg_data, crop_length=2500)  

            # Upsample the cropped ECG data
            upsampled_ecg = upsample_ecg(cropped_ecg, original_length=5000)  

            # Convert the upsampled ECG data back to a DataFrame
            df_upsampled = pd.DataFrame(upsampled_ecg)

            # Construct the output file path
            new_file_path = os.path.join(new_folder, file_name)

            # Save the upsampled ECG data to a new CSV file
            df_upsampled.to_csv(new_file_path, index=False)


# Put ECG and age, sex in the same csv file for training
df = pd.read_excel('../no9055_Pt_hash_id.xlsx')
output_df = pd.read_excel('../deleted.xlsx')

# Fetch all file names in the train and validation directories
train_files = os.listdir('../cropped_train_folder')
valid_files = os.listdir('../cropped_validation_folder')

# Create two dataframes for train and validation based on file names in respective folders
train_df = df[df['出力ファイル名'].isin(train_files)]
valid_df = df[df['出力ファイル名'].isin(valid_files)]

# Remove rows in train_df and valid_df if they're in output_df
train_df = train_df[~train_df['出力ファイル名'].isin(output_df['出力ファイル名'])]
valid_df = valid_df[~valid_df['出力ファイル名'].isin(output_df['出力ファイル名'])]

# Add prefix to 出力ファイル名 column for the first column
train_df['path'] = '../cropped_train_folder/' + train_df['出力ファイル名']
valid_df['path'] = '../cropped_validation_folder/' + valid_df['出力ファイル名']

# Map gender to 0 and 1 for the third column
# Replace 'gender' with actual gender column name in your file if it's different
train_df['Female'] = train_df['gender'].map({'F': 1, 'M': 0})
valid_df['Female'] = valid_df['gender'].map({'F': 1, 'M': 0})

# Rename 'age_at_ecg' column to 'Age'
train_df = train_df.rename(columns={'age_at_ecg': 'Age'})
valid_df = valid_df.rename(columns={'age_at_ecg': 'Age'})

# Keep only necessary columns and in the right order
train_df = train_df[['path', 'Age', 'Female']]
valid_df = valid_df[['path', 'Age', 'Female']]


# Write 1/3 to CSV files
train_df[:int(len(train_df)/3)].to_csv('../croppedTrain_Data.csv', index=False)
valid_df[:int(len(valid_df)/3)].to_csv('../croppedValidation_Data.csv', index=False)


