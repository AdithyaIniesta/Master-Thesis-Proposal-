# Operating system
import os  
# Module for serializing and deserializing Python objects
import pickle  
from typing import List, Tuple

# NumPy library
import numpy as np  

# Configuration file
import config

import pandas as pd

from pre_process import reduce_dimensionality_jl, get_coreset_idx

os.chdir(config.feature_folder)
# Load the array from the npy file
normalized_video_audio_features = np.load(
    f'normalized_train_audio{config.audio_flag}_depth{config.depth_flag}_'
    f'video{config.video_flag}_features.npy',
    allow_pickle=True
)
for i in np.arange(config.corest_idx, len(normalized_video_audio_features) + 5, 10):
    if config.enable_coreset:
        coreset_indices = get_coreset_idx(normalized_video_audio_features, n = i)
        normalized_video_audio_features \
            = normalized_video_audio_features[coreset_indices]
        print(normalized_video_audio_features.shape)
        
    os.chdir(config.src_folder)

    # Initialize and train the Isolation forest model
    model = config.model
    print(f"Training {config.model_name}")
    model.fit(normalized_video_audio_features)

    # Change the current working directory to the model folder
    os.chdir(config.model_folder)
    model_filename = (
        f'svm_model_audio{config.audio_flag}_'
        f'depth{config.depth_flag}_'
        f'video{config.video_flag}_{i}.pkl'
    )
    # Print a message indicating the progress of saving the model
    with open(model_filename, 'wb') as model_file:
        # Use pickle to save the trained iforest model
        pickle.dump(model, model_file)

    # Print a message indicating the completion of saving the model
    print("Saving model file in the model directory ")

    # Change the current working directory back to the source folder
    os.chdir(config.src_folder)
