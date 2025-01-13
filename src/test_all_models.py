# Standard Libraries
# Operating system module
import os
# Module for serializing and deserializing Python objects
import pickle
# NumPy library
import numpy as np
# Matplotlib for plotting
import matplotlib.pyplot as plt
# ROC curve metrics
from sklearn.metrics import roc_curve, auc, precision_recall_curve,\
      f1_score, average_precision_score
from tabulate import tabulate

# Importing batches from config
from config import normal_batches, absolute_anomaly_batches
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
    fuse_features
)
# Importing configuration module
import config
import re

# Initialize variables
audio = None
video = None
depth = None

# Define regular expression patterns
audio_pattern = re.compile(r'audio(\d+)')
video_pattern = re.compile(r'video(\d+)')
depth_pattern = re.compile(r'depth(\d+)')

# List all files in the folder
file_list = os.listdir(config.model_folder)

# Filter only the pickle files
pickle_files = [file for file in file_list if file.endswith('.pkl')]

# Load each pickle file
loaded_models = []

# Process normal batches
normal_audio_features, normal_video_features, \
    normal_meta_data, normal_labels_list = process_batch(normal_batches,
                                                         config.normal,
                                                         'start_end_time.npy')

# Process normal batches
anomaly_audio_features, anomaly_video_features, \
    anomaly_meta_data, anomaly_labels_list = process_batch(absolute_anomaly_batches,
                                                           config.anomaly,
                                                           'start_end_time.npy')

normal_audio_features.extend(anomaly_audio_features)
normal_video_features.extend(anomaly_video_features)
normal_meta_data.extend(anomaly_meta_data)
normal_labels_list.extend(anomaly_labels_list)

test_audio_features = np.array(normal_audio_features)
test_video_features = np.array(normal_video_features)

test_labels = np.array(normal_labels_list)
# print(f'True labels: {test_labels}')

print(f'Audio features shape {test_audio_features.shape}')
print(f'Video features shape {test_video_features.shape}')
print(f'Number of features {test_labels.shape[0]}')

# Fit the scaler on your data and transform it
normalized_test_video_features = scale_features(test_video_features)
normalized_test_audio_features = scale_features(test_audio_features)

# Testing data
feature_dict = {'audio': normalized_test_audio_features,
                'depth': None,
                'video': normalized_test_video_features}

normalized_video_audio_features = fuse_features(feature_dict=feature_dict)
print(f'Fused features shape: {normalized_video_audio_features.shape}')

# Lists to store tpr and fpr for each model
all_tpr = []
all_fpr = []
 # List to store legend labels
legend_labels = [] 
# Initialize lists to store precision, recall, and F1 score for each model
all_precisions = []
all_recalls = []
all_f1_scores = []
all_decision_scores = []
all_pr_legends = []
# Initialize a list to store model information for the table
table_data = []

# Define a list of colors for each ROC curve
colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'brown']

for i, pickle_file in enumerate(pickle_files):
    file_path = os.path.join(config.model_folder, pickle_file)

    # Use regular expressions to extract values
    audio_match = audio_pattern.search(file_path)
    depth_match = depth_pattern.search(file_path)
    video_match = video_pattern.search(file_path)

    # Assign values if matches are found
    if audio_match:
        config.audio_flag = int(audio_match.group(1))
    if depth_match:
        config.depth_flag = int(depth_match.group(1))
    if video_match:
        config.video_flag = int(video_match.group(1))

    # Testing data
    feature_dict = {
        'audio': normalized_test_audio_features,
        'depth': None,
        'video': normalized_test_video_features
    }
    normalized_video_audio_features = fuse_features(feature_dict=feature_dict)

    with open(file_path, 'rb') as model_file:
        loaded_svm_model = pickle.load(model_file)

        decision_values = loaded_svm_model.decision_function(normalized_video_audio_features)

        # Calculate precision, recall, and F1 score for different thresholds
        precisions, recalls, thresholds = precision_recall_curve(test_labels, decision_values)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

        print(f'F1 score: {np.max(f1_scores)}')
        print(f'Recall: {recalls[np.argmax(f1_scores)]}')
        print(f'Precisions: {precisions[np.argmax(f1_scores)]}')

        # Find the optimal threshold that maximizes F1 score
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        print(f'Optimal threshold for Maximum F1 score: {optimal_threshold}')

        # Apply the optimal threshold to get binary predictions
        binary_predictions = (decision_values > optimal_threshold).astype(int)

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(test_labels, decision_values)
        all_fpr.append(fpr)
        all_tpr.append(tpr)

        # Save precision, recall, and F1 score for the current model
        f1_score_max = np.max(f1_scores)
        f1_score_max_index = np.argmax(f1_score_max)

        # Append precision-recall values to the lists
        all_precisions.append(precisions)
        all_recalls.append(recalls)
        all_decision_scores.append(decision_values)

        # Create legend label with config information
        legend_label = (
            f'Model {i + 1} (Area ROC = {auc(fpr, tpr):.2f})\n'
            f'Model {i + 1} (Area PR = {auc(recalls, precisions):.2f})\n',
            f'Audio: {config.audio_flag}, Depth: {config.depth_flag}, Video: {config.video_flag}'
        )
        legend_labels.append(legend_label)

        all_pr_legends.append(
            f'Audio: {config.audio_flag}, Depth: {config.depth_flag}, Video: {config.video_flag}'
        )

        model_info = legend_label

        # Add model information to the table_data list
        table_data.append([
            model_info,
            f'{auc(fpr, tpr):.2f}',
            f'{np.max(f1_scores):.2%}'
        ])

plt.figure(figsize=(12, 12))

# Plot precision-recall curves for all models with different colors and legends
for i, (precisions, recalls) in enumerate(zip(all_precisions, all_recalls)):
    plt.plot(
        recalls, precisions, lw=2, color=colors[i],
        label=f'{all_pr_legends[i]} Area PR = {auc(recalls, precisions):.2f})'
    )

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Anomaly Detection')
plt.legend(loc='upper right', fontsize='small')
plt.grid(True)

os.chdir(config.plots_folder)
plt.savefig('combined_auc_pr')

# Create a table using tabulate
table_headers = ["Model Information", "AUC", "F1 Score"]
table = tabulate(table_data, headers=table_headers, tablefmt="grid")

# Apply color formatting to the table
colored_table = tabulate(
    table_data, table_headers, tablefmt="fancy_grid",
    numalign="center", stralign="center",
    colalign=("center", "center", "center")
)

# Print or save the table
print(table)
print(colored_table)

# If you want to save the table to a file
with open("model_metrics_table.txt", "w") as file:
    file.write(colored_table)

# Plot ROC curves for all models with different colors and legends
for i, (fpr, tpr) in enumerate(zip(all_fpr, all_tpr)):
    plt.plot(fpr, tpr, lw=2, color=colors[i])

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Anomaly Detection')
plt.legend(legend_labels, loc='lower right', fontsize='small')

# os.chdir(config.plots_folder)
plt.savefig('combined_roc_auc')
print(f'Saving roc plot in "{config.plots_folder}"')
os.chdir(config.src_folder)

