# NumPy library
import numpy as np  

import os 

# Isolation forest
from sklearn.ensemble import IsolationForest 

# Configuration file
import config

from pre_process import (
    fuse_features,
    scale_features,
    process_batch
)

import pandas as pd

# Process normal batches
train_audio_features, train_depth_features, train_video_features, \
    train_meta_data, train_labels_list \
    = process_batch(
        config.normal_dataset_batches,  # List of normal batches
        config.normal,  # Label for normal data
        "start_end_time_new.npy"  # File extension for timestamp
    )

# Convert lists to numpy arrays
train_audio_features = np.array(train_audio_features)
train_depth_features = np.array(train_depth_features)
train_video_features = np.array(train_video_features)
train_labels = np.array(train_labels_list)

# Display shapes of audio and video features
print(f'Audio features shape {train_audio_features.shape}')
print(f'Depth features shape {train_depth_features.shape}')
print(f'Video features shape {train_video_features.shape}')

# Fit the scaler on the data and transform it
normalized_train_audio_features = scale_features(train_audio_features)
normalized_train_depth_features = scale_features(train_depth_features)
normalized_train_video_features = scale_features(train_video_features)
# Create a dictionary to hold different types of features
feature_dict = {'audio': normalized_train_audio_features,  # Audio features
                'depth': normalized_train_depth_features,  # Depth features (set to None in this case)
                'video': normalized_train_video_features}  # Video features

# Fuse different types of features using the defined function
normalized_video_audio_features = fuse_features(feature_dict=feature_dict)

# Print the shape of the final feature list
print(f'Final feature list shape {normalized_video_audio_features.shape}')

os.chdir(config.feature_folder)
# Save to an npy file
np.save(
    f'normalized_train_audio{config.audio_flag}_depth{config.depth_flag}_video{config.video_flag}_features.npy',
    normalized_video_audio_features
)

np.save(
    f'train_audio{config.audio_flag}_depth{config.depth_flag}_video{config.video_flag}_labels.npy',
    train_labels
)


# Load the array from the npy file
loaded_features_array = np.load(
    f'normalized_train_audio{config.audio_flag}_depth{config.depth_flag}_video{config.video_flag}_features.npy',
    allow_pickle=True
)

# Compare arrays
are_equal = np.array_equal(loaded_features_array, normalized_video_audio_features)

# Print the result
print(f"Arrays are equal: {are_equal}")
os.chdir(config.src_folder)