import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from tabulate import tabulate
from config import normal_batches, absolute_anomaly_batches
from feature_extraction import extract_combined_features
from model import video_feature_extractor, audio_feature_extractor
from pre_process import process_batch, scale_features, fuse_features
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

# Load each pickle file
loaded_models = []

# Lists to store precision, recall, and F1 score for each model
all_precisions, all_recalls, all_decision_scores = [], [], []
# List to store legend labels
legend_labels = []
# Initialize a list to store model information for the table
table_data = []

# Define a list of colors for each PR curve
colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'brown'] * 10

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
        f1_score_max_index = np.argmax(f1_score_max)

        all_precisions.append(precisions)
        all_recalls.append(recalls)
        all_decision_scores.append(decision_values)

        file_path_ = os.path.splitext(os.path.basename(file_path))[0]
        
        legend_label = (
            f'Model {file_path_.split("_")[0]} (Area PR = {auc(recalls, precisions):.2f})\n',
            f'Audio: {config.audio_flag}, Depth: {config.depth_flag}, Video: {config.video_flag}'
        )
        legend_labels.append(legend_label)

        model_info = legend_label
        table_data.append([model_info, f'{auc(recalls, precisions):.2f}', f'{np.max(f1_scores):.2%}'])

plt.figure(figsize=(10, 10))
best_auc_model_index = np.argmax([auc(recalls, precisions) for recalls, precisions in zip(all_recalls, all_precisions)])

for i, (precisions, recalls) in enumerate(zip(all_precisions, all_recalls)):
    if i == best_auc_model_index:
        plt.plot(
            recalls, precisions, lw=8, linestyle='--', color=colors[i],  # Highlight the best AUC curve with a thicker and dashed line
            label=f'{legend_labels[i][0]} {legend_labels[i][1]} Area PR = {auc(recalls, precisions):.2f} (Best AUC)'
        )
    else:
        plt.plot(
            recalls, precisions, lw=2, color=colors[i],
            label=f'{legend_labels[i][0]} {legend_labels[i][1]} Area PR = {auc(recalls, precisions):.2f}'
        )

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Anomaly Detection')
plt.legend(loc='lower right', fontsize='small')  # Adjust the legend position
plt.grid(True)

os.chdir(config.plots_folder)
plt.savefig('combined_auc_pr')

table_headers = ["Model Information", "AUC", "F1 Score"]
table = tabulate(table_data, headers=table_headers, tablefmt="grid")

print(table)

with open("model_metrics_table.txt", "w") as file:
    file.write(table)