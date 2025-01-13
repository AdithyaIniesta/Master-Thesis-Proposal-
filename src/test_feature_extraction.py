# External Libraries
import numpy as np
import pickle
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
import matplotlib.pyplot as plt

# Importing batches from config
from config import normal_batches, absolute_anomaly_batches

# Feature extraction modules
from feature_extraction import extract_combined_features, extract_audio_features

# Importing video and audio feature extractors
from model import video_feature_extractor, audio_feature_extractor

# Pre-processing modules
from pre_process import process_batch, scale_features, fuse_features

# Importing configuration module
import config

# Change the current working directory to the model folder
os.chdir(config.model_folder)

# Load the SVM model using pickle
with open(config.model_filename, 'rb') as model_file:
    loaded_svm_model = pickle.load(model_file)

# Change the current working directory back to the source folder
os.chdir(config.src_folder)

# Process normal batches
normal_audio_features, normal_depth_features, normal_video_features, \
normal_meta_data, normal_labels_list = process_batch(
    config.normal_batches,
    config.normal,
    'start_end_time.npy'
)


print('Contamination dataset')
# Process normal batches
anomaly_audio_features, anomaly_depth_features, anomaly_video_features, \
    anomaly_meta_data, anomaly_labels_list = process_batch(
    config.absolute_anomaly_batches,
    config.anomaly,
    'start_end_time.npy'
)

normal_audio_features.extend(anomaly_audio_features)
normal_depth_features.extend(anomaly_depth_features)
normal_video_features.extend(anomaly_video_features)
normal_meta_data.extend(anomaly_meta_data)
normal_labels_list.extend(anomaly_labels_list)

test_audio_features = np.array(normal_audio_features)
test_depth_features = np.array(normal_depth_features)
test_video_features = np.array(normal_video_features)

test_labels = np.array(normal_labels_list)

print(f'Audio features shape {test_audio_features.shape}')
print(f'Depth features shape {test_depth_features.shape}')
print(f'Video features shape {test_video_features.shape}')
print(f'Number of features {test_labels.shape[0]}')

# Fit the scaler on your data and transform it
normalized_test_audio_features = scale_features(test_audio_features)
normalized_test_depth_features = scale_features(test_depth_features)
normalized_test_video_features = scale_features(test_video_features)


# Testing data
feature_dict = {
    'audio': normalized_test_audio_features,
    'depth': normalized_test_depth_features,
    'video': normalized_test_video_features
}

normalized_test_audio_depth_video_features = fuse_features(feature_dict=feature_dict)
print(f'Fused features shape: {normalized_test_audio_depth_video_features.shape}')

os.chdir(config.feature_folder)
# Save to an npy file
np.save(
    f'normalized_test_audio{config.audio_flag}_'
    f'depth{config.depth_flag}_'
    f'video{config.video_flag}_features.npy',
    normalized_test_audio_depth_video_features
)

np.save(
    f'test_audio{config.audio_flag}_'
    f'depth{config.depth_flag}_'
    f'video{config.video_flag}_labels.npy', 
    test_labels
)
os.chdir(config.src_folder)