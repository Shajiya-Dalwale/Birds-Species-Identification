import os
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt

# Define the base directory
base_dir = "C:\\BSD"

# Define the paths for train, test, and validation directories
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
validate_dir = os.path.join(base_dir, "validate")

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(validate_dir, exist_ok=True)

# Define the bird folders
bird_folders = ["cuckoo", "sparrow", "crow", "laughing_dove", "ambient"]

# Define the percentage split
train_split = 0.7
test_split = 0.2
validate_split = 0.1

# Initialize lists to store the number of files for each bird
num_train_files = []
num_test_files = []
num_validate_files = []

# Loop through each bird folder
for bird_folder in bird_folders:
    # Define the path to the bird dataset directory
    dataset_directory = os.path.join(base_dir, "dataset", bird_folder)
    
    if not os.path.exists(dataset_directory):
        print(f"Dataset directory for {bird_folder} does not exist.")
        continue
    
    # Get the list of unique files in the dataset directory
    file_list = list(set(os.listdir(dataset_directory)))
    
    # Shuffle the file list
    np.random.shuffle(file_list)
    
    # Create subdirectories in train, test, and validate directories for the bird
    os.makedirs(os.path.join(train_dir, bird_folder), exist_ok=True)
    os.makedirs(os.path.join(test_dir, bird_folder), exist_ok=True)
    os.makedirs(os.path.join(validate_dir, bird_folder), exist_ok=True)
    
    # Calculate the number of files for each set
    num_files = len(file_list)
    num_train = int(train_split * num_files)
    num_test = int(test_split * num_files)
    num_validate = num_files - num_train - num_test
    
    # Distribute files to train set
    for i in range(num_train):
        source_filepath = os.path.join(dataset_directory, file_list[i])
        dest_filepath = os.path.join(train_dir, bird_folder, file_list[i])
        copyfile(source_filepath, dest_filepath)
    
    # Distribute files to test set
    for i in range(num_train, num_train + num_test):
        source_filepath = os.path.join(dataset_directory, file_list[i])
        dest_filepath = os.path.join(test_dir, bird_folder, file_list[i])
        copyfile(source_filepath, dest_filepath)
    
    # Distribute files to validation set
    for i in range(num_train + num_test, num_files):
        source_filepath = os.path.join(dataset_directory, file_list[i])
        dest_filepath = os.path.join(validate_dir, bird_folder, file_list[i])
        copyfile(source_filepath, dest_filepath)
    
    # Append the number of files for each set to the respective lists
    num_train_files.append(num_train)
    num_test_files.append(num_test)
    num_validate_files.append(num_validate)

print("Dataset is divided into a Train Dataset, Validation Dataset and Test Dataset in the ratio of 70:10:20.")

# Plotting the dataset split for existing bird folders
labels = [bird_folders[i] for i in range(len(bird_folders)) if os.path.exists(os.path.join(base_dir, "dataset", bird_folders[i]))]
x = np.arange(len(labels))
width = 0.3

plt.figure(figsize=(10, 6))
plt.bar(x - width, num_train_files, width, label='Train')
plt.bar(x, num_validate_files, width, label='Validate')
plt.bar(x + width, num_test_files, width, label='Test')
plt.xlabel('Bird Species')
plt.ylabel('Number of Files')
plt.title('Division of Dataset by Bird Species')
plt.xticks(x, labels)
plt.legend()
plt.tight_layout()
plt.show()
