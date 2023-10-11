import os

# Define the root directory
root_dir = '/pscratch/sd/s/sanjeevr/MODELPATH'

# Define the old and new file names
old_filename = 'best_model.pth'
new_filename = 'best_ckpt.pth'

# Function to rename files
def rename_files(root_dir):
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == old_filename:
                file_path = os.path.join(foldername, filename)
                new_path = os.path.join(foldername, new_filename)
                os.rename(file_path, new_path)
                print(f'Renamed: {file_path} -> {new_path}')

# Call the function to start renaming
rename_files(root_dir)