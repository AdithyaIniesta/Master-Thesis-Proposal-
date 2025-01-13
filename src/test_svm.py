# Standard Libraries
# Operating system module
import os          
# Module for serializing and deserializing Python objects
import pickle      

# External Libraries
# NumPy library
import numpy as np       
# Matplotlib for plotting           
import matplotlib.pyplot as plt    
# ROC curve metrics  
from sklearn.metrics import (
    roc_curve, 
    auc,
    precision_recall_curve, 
    f1_score)

# Importing batches from config
from config import (
    normal_batches, 
    absolute_anomaly_batches
)
# Feature extraction modules               
from feature_extraction import (                               
    extract_combined_features,
    extract_audio_features
)
# Importing video and audio feature extractors
from model import (                     
    video_feature_extractor,
    audio_feature_extractor
)
# Pre-processing modules
from pre_process import (                
    process_batch,
    scale_features,
    fuse_features, 
    reduce_dimensionality_jl, 
    get_coreset_idx
)
# Importing configuration module
import config                           
from anomalib.models.components import KCenterGreedy

os.chdir(config.model_folder)
# Step 3: Load the SVM model using pickle
print(config.model_filename)
with open(config.model_filename , 'rb') as model_file:
    loaded_svm_model = pickle.load(model_file)

os.chdir(config.feature_folder)

normalized_test_audio_depth_video_features = np.load(
    f'normalized_test_audio{config.audio_flag}_depth{config.depth_flag}_video{config.video_flag}_features.npy',
    allow_pickle=True
)


print(f"Test features {normalized_test_audio_depth_video_features.shape}")
test_labels = np.load(
    f'test_audio{config.audio_flag}_depth{config.depth_flag}_video{config.video_flag}_labels.npy',
    allow_pickle=True
)

os.chdir(config.src_folder)

decision_values = loaded_svm_model.decision_function(normalized_test_audio_depth_video_features)
# Calculate precision, recall, and F1 score for different thresholds
precisions, recalls, thresholds = precision_recall_curve(test_labels, decision_values)
# f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
# print(f'F1 score: {np.max(f1_scores)}')

# Check for division by zero
zero_division_check = (precisions + recalls) == 0
f1_scores = np.where(zero_division_check, 0, 2 * (precisions * recalls) / (precisions + recalls))

# Handle NaN values
f1_scores[np.isnan(f1_scores)] = 0
# Identify the optimal threshold that maximizes F1 score
optimal_threshold_index = np.argmax(f1_scores)

# Find the optimal threshold that maximizes F1 score
optimal_threshold = thresholds[optimal_threshold_index]
print(f'Optimal threshold for Maximum F1 score: {optimal_threshold}')

# Apply the optimal threshold to get binary predictions
binary_predictions = (decision_values > optimal_threshold).astype(int)

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(test_labels, decision_values)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, \
         label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Anomaly Detection')
plt.legend(loc='lower right')
os.chdir(config.plots_folder)
plt.savefig(config.roc_plot)
print(f'Saving roc plot in "{config.plots_folder}"')
os.chdir(config.src_folder)



















"""
for batch in normal_batches:
    # Extract combined features and
    # labels for the current batch.
    batch_video_features, \
    batch_audio_features, \
    batches_labels_list, \
    batch_meta_data = extract_combined_features(
        batch,
        video_feature_extractor,
        audio_feature_extractor, 
        time_stamp_file_extension = 'start_end_time.npy'
    )
    
    # Iterate through individual
    # features and labels in the batch.
    for video_feature, audio_feature, label, meta_data in zip(
        batch_video_features,
        batch_audio_features,
        batches_labels_list,
        batch_meta_data 
    ):
        # Append features and labels to
        # the training dataset.
        test_video_features.append(video_feature)
        test_audio_features.append(audio_feature)
        test_labels_list.append(1)
        test_meta_data.append(meta_data)


for batch in absolute_anomaly_batches:
    # Extract combined features and
    # labels for the current batch.
    batch_video_features, \
    batch_audio_features, \
    batches_labels_list, \
    batch_meta_data = extract_combined_features(
        batch,
        video_feature_extractor,
        audio_feature_extractor, 
        time_stamp_file_extension = 'start_end_time.npy'
    )
    
    # Iterate through individual
    # features and labels in the batch.
    for video_feature, audio_feature, label, meta_data in zip(
        batch_video_features,
        batch_audio_features,
        batches_labels_list,
        batch_meta_data 
    ):
        # Append features and labels to
        # the training dataset.
        test_video_features.append(video_feature)
        test_audio_features.append(audio_feature)
        test_labels_list.append(0)
        test_meta_data.append(meta_data)

"""

"""
# Function to process a batch and append features and labels to testing dataset
def process_batch(batch: List[Any], 
                  label_value: int, 
                  test_video_features: 
                  List[Any],
                  test_audio_features: List[Any], 
                  test_labels_list: List[Any], 
                  time_stamp_file_extension: str):
    for batch_item in batch:
        # Extract combined features and labels for the current batch.
        batch_video_features, batch_audio_features, \
        batches_labels_list, batch_meta_data = \
            extract_combined_features(batch_item, 
                                      video_feature_extractor, 
                                      audio_feature_extractor,
                                      time_stamp_file_extension = time_stamp_file_extension)

        # Iterate through individual features and labels in the batch.
        for video_feature, audio_feature, \
            _, meta_data in zip(
                batch_video_features, 
                batch_audio_features, 
                batches_labels_list, 
                batch_meta_data
        ):
            # Append features and labels to the testing dataset.
            test_video_features.append(video_feature)
            test_audio_features.append(audio_feature)
            test_labels_list.append(label_value)
            test_meta_data.append(meta_data)
"""

"""
# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(test_labels, mapped_predictions)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
os.chdir(config.model_folder)
plt.savefig('roc')
"""