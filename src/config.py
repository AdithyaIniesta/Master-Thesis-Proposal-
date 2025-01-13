import os
import torch
import sys
import numpy as np
# Isolation forest
from sklearn.ensemble import IsolationForest 
from sklearn.svm import OneClassSVM 
import sys
import subprocess

# Define the parent folder for video frames dataset
parent_folder = "/scratch/anara12s/nn_env/IsolationForest/"

# Create paths for dataset, model and src files
normal_dataset_parent_folder = '/scratch/anara12s/nn_env/fresh_dataset/'
test_dataset_parent_folder = '/scratch/anara12s/nn_env/test_data/'
isolation_forest_folder = '/scratch/anara12s/nn_env/IsolationForest/isolation_models'
model_folder = os.path.join(parent_folder, "model_files/")

src_folder = os.path.join(parent_folder, "src/") 
feature_folder = os.path.join(src_folder, "feature/")
plots_folder = os.path.join(parent_folder, "plots/")
# Set the number of frames to uniformly sample from each video
num_frames_to_sample = 64

# SVM parameters  
kernel = 'rbf'
nu = 0.8

if torch.cuda.is_available():
    device = torch.device('cuda:0')  # Use GPU if available
else:
    device = torch.device('cpu')     # Use CPU if GPU is not available

# String to remove
remove_string = 'corn_batch_5'

feature_dir = "/scratch/anara12s/nn_env/IsolationForest/src/feature/"

# Remove the specified string from the list
items = os.listdir(normal_dataset_parent_folder)
items.remove(remove_string)

# Filter out subfolders
normal_dataset_batches = sorted([os.path.join(normal_dataset_parent_folder, item) \
    for item in items if os.path.isdir(os.path.join(normal_dataset_parent_folder, item))])


np.set_printoptions(threshold=sys.maxsize)

# Get a list of all subfolders
subfolders = [os.path.join(test_dataset_parent_folder, f)\
               for f in os.listdir(test_dataset_parent_folder) \
              if os.path.isdir(os.path.join(test_dataset_parent_folder, f))]

# Filter subfolders with the name "anomaly"
absolute_anomaly_batches = [subfolder for subfolder in subfolders \
                      if 'absolute_anomaly' in subfolder.lower()]

normal_batches = [subfolder for subfolder in subfolders \
                      if 'normal' in subfolder.lower()]

min_distance = 300
normal = 1
anomaly = 0



# Set flags to choose which features to fuse
audio_flag = 1
depth_flag = 1
video_flag = 1

enable_coreset = 0
corest_idx = 110
disable_dl = False

model_name = "svm"
model = None

if model_name == "isolation_forest":
    # model_folder = isolation_forest_folder
    model = IsolationForest(contamination=0.001)
    model_filename = (
        f'isolation_forest_model_audio{audio_flag}_'
        f'depth{depth_flag}_'
        f'video{video_flag}_{corest_idx}.pkl'
    )

    roc_plot = f'isolation_model_audio{audio_flag}'\
        f'_depth{depth_flag}'\
        f'_video{video_flag}_roc'

elif model_name == "svm":
    model = OneClassSVM(kernel = kernel, nu = nu)
    model_filename = (
        f'svm_model_audio{audio_flag}_'
        f'depth{depth_flag}_'
        f'video{video_flag}_{corest_idx}.pkl'
    )

    roc_plot = f'svm_audio{audio_flag}'\
        f'_depth{depth_flag}'\
        f'_video{video_flag}_roc'
