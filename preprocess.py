import os
import pandas as pd

# Define the base and new directories
base_folder = '../wave_csvs'  # Replace with your source directory
new_base_folder = '../new_folder'  # Replace with your target directory

# Create the new directory if it doesn't exist
os.makedirs(new_base_folder, exist_ok=True)

# Counter for csv files with -00 ending
counter_00 = 0

# Iterate over all directories and files in the base directory
for dirpath, dirnames, filenames in os.walk(base_folder):
    # Check each file
    for filename in filenames:
        # If the file is a csv file and doesn't end with -00
        if filename.endswith('.csv') and not filename.endswith('_00.csv'):
            # Load the CSV data
            df = pd.read_csv(os.path.join(dirpath, filename))
            
            # Drop the last six columns
            df = df.iloc[:, :-6]
            
            # Create the same structure in the new directory
            structure = os.path.join(new_base_folder, os.path.relpath(dirpath, base_folder))
            os.makedirs(structure, exist_ok=True)
            
            # Save the modified data to the new location
            df.to_csv(os.path.join(structure, filename), index=False)
        
        # Count the csv files with -00 ending
        elif filename.endswith('_00.csv'):
            counter_00 += 1




print(f"Number of csv files skipped: {counter_00}")


# read the excel file
df = pd.read_excel(os.path.join('..', 'no9055_Pt_hash_id.xlsx'))

# find the rows where the 5th and 6th column are null
empty_rows = df[df.iloc[:, 4:6].isnull().any(axis=1)]

# get the 2nd column values of these rows
second_column_values = empty_rows.iloc[:, 1]

# write these values to a new Excel file
second_column_values.to_excel(os.path.join('..', 'output.xlsx'), index=False)

# Specify the directory path
new_folder_path = '../new_folder'
output_file_path = '../output.xlsx'

# Load the Excel file
excel_data = pd.read_excel(output_file_path, header=None)

# Initialize a count of the files deleted
files_deleted = 0

# Iterate over each subdirectory in the new_folder
for subdir, dirs, files in os.walk(new_folder_path):
    # Iterate over the DataFrame
    for index, row in excel_data.iterrows():
        # Construct the file path
        file_path = os.path.join(subdir, str(row[0]) )
        
        # Check if the file exists
        if os.path.exists(file_path):
            # Delete the file
            os.remove(file_path)
            
            # Increment the count of files deleted
            files_deleted += 1

print(f"Deleted {files_deleted} files.")

import os
import random
import shutil

# Set the path to the new_folder containing the subfolders
folder_path = "../new_folder"

# Set the destination paths for the training, validation, and testing sets
train_path = "../train_folder"
val_path = "../validation_folder"
test_path = "../test_folder"

# Set the desired proportions for splitting (7:1:2)
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

# Create destination directories if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Set the random seed for reproducibility
random.seed(42)

# Function to recursively search for CSV files in the directory
def find_csv_files(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files

# Find all CSV files in the subfolders
csv_files = find_csv_files(folder_path)

# Shuffle the list of CSV files
random.shuffle(csv_files)

# Calculate the number of files for each set
num_files = len(csv_files)
num_train = int(train_ratio * num_files)
num_val = int(val_ratio * num_files)
num_test = int(test_ratio * num_files)

# Split the files into the training, validation, and testing sets
train_files = csv_files[:num_train]
val_files = csv_files[num_train:num_train + num_val]
test_files = csv_files[num_train + num_val:]

# Move the files to their respective sets without subfolders
def move_files(file_list, dest_path):
    for file in file_list:
        dest_file = os.path.join(dest_path, os.path.basename(file))
        shutil.copy2(file, dest_file)

move_files(train_files, train_path)
move_files(val_files, val_path)
move_files(test_files, test_path)



# Put ECG and age, sex in the same csv file for training
df = pd.read_excel('../no9055_Pt_hash_id.xlsx')
output_df = pd.read_excel('../output.xlsx')

# Fetch all file names in the train and validation directories
train_files = os.listdir('../train_folder')
valid_files = os.listdir('../validation_folder')

# Create two dataframes for train and validation based on file names in respective folders
train_df = df[df['出力ファイル名'].isin(train_files)]
valid_df = df[df['出力ファイル名'].isin(valid_files)]

# Remove rows in train_df and valid_df if they're in output_df
train_df = train_df[~train_df['出力ファイル名'].isin(output_df['出力ファイル名'])]
valid_df = valid_df[~valid_df['出力ファイル名'].isin(output_df['出力ファイル名'])]

# Add prefix to 出力ファイル名 column for the first column
train_df['path'] = '../train_folder/' + train_df['出力ファイル名']
valid_df['path'] = '../validation_folder/' + valid_df['出力ファイル名']

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

# Write to CSV files
train_df.to_csv('../Train_Data.csv', index=False)
valid_df.to_csv('../Validation_Data.csv', index=False)

