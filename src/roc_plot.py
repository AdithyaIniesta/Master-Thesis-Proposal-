import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from tabulate import tabulate
import config
import re

# Initialize variables
audio, video, depth = None, None, None

# Define regular expression patterns
audio_pattern = re.compile(r'audio(\d+)')
video_pattern = re.compile(r'video(\d+)')
depth_pattern = re.compile(r'depth(\d+)')

# List all files in the folder
file_list = os.listdir(config.model_folder)

# Filter only the pickle files
pickle_files = [file for file in file_list if file.endswith('.pkl')]

# Lists to store fpr, tpr, and AUC for each model
all_fpr, all_tpr, all_auc = [], [], []
# List to store legend labels
legend_labels = []
# Initialize a list to store model information for the table
table_data = []

# Define a list of colors for each ROC curve
colors = ['red', 'green', 'blue', 'orange', 'purple', 'violet', 'brown']

# Variables to store information about the best F-score model
best_f1_model_index = -1
best_f1_score = -1

for i, pickle_file in enumerate(pickle_files):
    file_path = os.path.join(config.model_folder, pickle_file)

    audio_match = audio_pattern.search(file_path)
    depth_match = depth_pattern.search(file_path)
    video_match = video_pattern.search(file_path)

    if audio_match:
        config.audio_flag = int(audio_match.group(1))
    if depth_match:
        config.depth_flag = int(depth_match.group(1))
    if video_match:
        config.video_flag = int(video_match.group(1))

    os.chdir(config.feature_dir)
    features_file = f'normalized_test_audio{config.audio_flag}_depth{config.depth_flag}_video{config.video_flag}_features.npy'
    normalized_video_audio_features = np.load(features_file, allow_pickle=True)

    test_labels = np.load(
        f'test_audio{config.audio_flag}_depth{config.depth_flag}_video{config.video_flag}_labels.npy',
        allow_pickle=True
    )
    os.chdir(config.src_folder)

    with open(file_path, 'rb') as model_file:
        loaded_svm_model = pickle.load(model_file)
        decision_values = loaded_svm_model.decision_function(normalized_video_audio_features)

        fpr, tpr, thresholds = roc_curve(test_labels, decision_values)
        roc_auc = auc(fpr, tpr)

        precisions, recalls, thresholds = precision_recall_curve(test_labels, decision_values)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

        zero_division_check = (precisions + recalls) == 0
        f1_scores = np.where(zero_division_check, 0, 2 * (precisions * recalls) / (precisions + recalls))

        # Handle NaN values
        f1_scores[np.isnan(f1_scores)] = 0
        # Identify the optimal threshold that maximizes F1 score
        optimal_threshold_index = np.argmax(f1_scores)

        # Find the optimal threshold that maximizes F1 score
        optimal_threshold = thresholds[optimal_threshold_index]

        # optimal_threshold = thresholds[np.argmax(f1_scores)]
        binary_predictions = (decision_values > optimal_threshold).astype(int)

        f1_score_max = np.max(f1_scores)
        f1_score_max_index = np.argmax(f1_scores)

        if f1_score_max > best_f1_score:
            best_f1_score = f1_score_max
            best_f1_model_index = i

        # Store fpr, tpr, and AUC for each model
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_auc.append(roc_auc)
        file_path_ = os.path.splitext(os.path.basename(file_path))[0]

        legend_label = (
            f'Model {file_path_.split("_")[0]} (Area ROC = {roc_auc:.2f})\n',
            f'Audio: {config.audio_flag}, Depth: {config.depth_flag}, Video: {config.video_flag}'
        )
        legend_labels.append(legend_label)

        model_info = legend_label
        table_data.append([model_info, f'{roc_auc:.2f}', f'{f1_score_max:.2%}'])

plt.figure(figsize=(10, 10))

# Plot ROC curves for all models with different colors and legends
for i, (fpr, tpr) in enumerate(zip(all_fpr, all_tpr)):
    if i == best_f1_model_index:
        plt.plot(
            fpr, tpr, lw=5, color=colors[i], linestyle='--',  # Highlight the best F1 score curve with a thicker and dashed line
            label=f'{legend_labels[i][0]} {legend_labels[i][1]} Area ROC = {all_auc[i]:.2f} (Best F1 Score)'
        )
    else:
        plt.plot(
            fpr, tpr, lw=2, color=colors[i],
            label=f'{legend_labels[i][0]} {legend_labels[i][1]} Area ROC = {all_auc[i]:.2f}'
        )

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Use the threshold from precision-recall curve corresponding to the best F1 score
optimal_threshold = thresholds[np.argmax(f1_scores)]
plt.axvline(x=optimal_threshold, color='black', linestyle=':', lw=2)  # Add a dotted vertical line at the best F1 score point

# Increase the size of the legend
plt.legend(loc='lower right', fontsize='large')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Anomaly Detection')

# Set custom axis limits for magnification
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])

plt.legend(loc='lower right', fontsize='small')
plt.grid(True)

os.chdir(config.plots_folder)
plt.savefig('combined_roc_auc_magnified_highlighted')  # Save the magnified and highlighted ROC plot

table_headers = ["Model Information", "AUC", "F1 Score"]
table = tabulate(table_data, headers=table_headers, tablefmt="grid")

print(table)

with open("model_metrics_table.txt", "w") as file:
    file.write(table)