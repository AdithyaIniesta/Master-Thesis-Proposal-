import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import config
import os 
from pre_process import get_coreset_idx

def plot_tsne(video_features, modality,
              save_path='tsne_plot.png'):
    """
    Perform 2D t-SNE on video features and save the plot.

    Parameters:
    - video_features (numpy array): Video feature data.
    - save_path (str): Path to save the plot (default: 'tsne_plot.png').
    """
    # Perform t-SNE on video features
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    video_features_2d = tsne.fit_transform(video_features)

    # Create a scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(video_features_2d[:, 0], video_features_2d[:, 1], \
                c='b', marker='*', s=50)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.title(f'2D t-SNE of {modality} Features')

    # Save the 2D t-SNE plot
    plt.savefig(save_path)
    plt.close()

def plot_box(video_features, modality, save_path='box_plot.png'):
    """
    Create a box plot of video features and save the plot.

    Parameters:
    - video_features (numpy array): Video feature data.
    - save_path (str): Path to save the plot (default: 'box_plot.png').
    """
    # Create a box plot
    plt.figure(figsize=(8, 6))
    plt.boxplot(video_features)
    plt.xlabel('Feature Dimension')
    plt.ylabel('Feature Value')
    plt.title(f'Box Plot of {modality} Features')

    # Save the box plot
    plt.savefig(save_path)
    plt.close()

def plot_and_save_2d_tsne(audio_features, video_features, depth_features, save_path,
                          a_marker='o', v_marker='*', d_marker='^',
                          a_color='r', v_color='g', d_color='b'):
    """
    Plot 2D t-SNE of audio, video, and depth features and save the plot.

    Parameters:
    - audio_features (numpy array): Audio feature data.
    - video_features (numpy array): Video feature data.
    - depth_features (numpy array): Depth feature data.
    - save_path (str): Path to save the plot.
    - a_marker (str): Marker for audio features (default: 'o').
    - v_marker (str): Marker for video features (default: 's').
    - d_marker (str): Marker for depth features (default: '^').
    - a_color (str): Color for audio features (default: 'r').
    - v_color (str): Color for video features (default: 'g').
    - d_color (str): Color for depth features (default: 'b').
    """
    # Perform 2D t-SNE on audio, video, and depth features
    af_2d = TSNE(n_components=2, perplexity=10, random_state=42).fit_transform(audio_features)
    vf_2d = TSNE(n_components=2, perplexity=10, random_state=42).fit_transform(video_features)
    df_2d = TSNE(n_components=2, perplexity=10, random_state=42).fit_transform(depth_features)

    # Scatter plot for audio features
    plt.scatter(af_2d[:, 0], af_2d[:, 1], c=a_color, marker=a_marker, label='Audio')

    # Scatter plot for video features
    plt.scatter(vf_2d[:, 0], vf_2d[:, 1], c=v_color, marker=v_marker, label='Video')

    # Scatter plot for depth features
    plt.scatter(df_2d[:, 0], df_2d[:, 1], c=d_color, marker=d_marker, label='Fused')

    # Customize plot
    plt.xlabel('Dim 1'), plt.ylabel('Dim 2')
    plt.title('2D t-SNE of Audio, Video, and Depth Features'), plt.legend()
    plt.legend()
    # Save the plot
    plt.savefig(save_path)

from mpl_toolkits.mplot3d import Axes3D

def plot_and_save_3d_tsne(audio_features, video_features, depth_features, save_path,
                          a_marker='o', v_marker='*', d_marker='^',
                          a_color='r', v_color='g', d_color='b'):
    """
    Plot 3D t-SNE of audio, video, and depth features and save the plot.

    Parameters:
    - audio_features (numpy array): Audio feature data.
    - video_features (numpy array): Video feature data.
    - depth_features (numpy array): Depth feature data.
    - save_path (str): Path to save the plot.
    - a_marker (str): Marker for audio features (default: 'o').
    - v_marker (str): Marker for video features (default: '*').
    - d_marker (str): Marker for depth features (default: '^').
    - a_color (str): Color for audio features (default: 'r').
    - v_color (str): Color for video features (default: 'g').
    - d_color (str): Color for depth features (default: 'b').
    """
    # Perform 3D t-SNE on audio, video, and depth features
    af_3d = TSNE(n_components=3, perplexity=10, random_state=42).fit_transform(audio_features)
    vf_3d = TSNE(n_components=3, perplexity=10, random_state=42).fit_transform(video_features)
    df_3d = TSNE(n_components=3, perplexity=10, random_state=42).fit_transform(depth_features)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for audio features
    ax.scatter(af_3d[:, 0], af_3d[:, 1], af_3d[:, 2], c=a_color, marker=a_marker, label='Audio')

    # Scatter plot for video features
    ax.scatter(vf_3d[:, 0], vf_3d[:, 1], vf_3d[:, 2], c=v_color, marker=v_marker, label='Video')

    # Scatter plot for depth features
    ax.scatter(df_3d[:, 0], df_3d[:, 1], df_3d[:, 2], c=d_color, marker=d_marker, label='Fused')

    # Customize plot
    ax.set_xlabel('Dim 1'), ax.set_ylabel('Dim 2'), ax.set_zlabel('Dim 3')
    plt.title('3D t-SNE of Audio, Video, and Depth Features'), plt.legend()
    
    # Save the plot
    plt.savefig(save_path)

def plot_2d_tsne(normal_features, anomaly_features):
    # Generate labels for normal and anomaly features
    normal_labels = np.zeros(normal_features.shape[0])
    anomaly_labels = np.ones(anomaly_features.shape[0])

    # Perform t-SNE on normal features
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    normal_features_tsne = tsne.fit_transform(normal_features)

    # Plot normal features
    plt.scatter(
        normal_features_tsne[:, 0],
        normal_features_tsne[:, 1],
        c='blue',
        label='Normal',
        alpha=0.7
    )

    # Perform t-SNE on anomaly features
    anomaly_features_tsne = tsne.fit_transform(anomaly_features)

    # Plot anomaly features
    plt.scatter(
        anomaly_features_tsne[:, 0],
        anomaly_features_tsne[:, 1],
        c='orange',
        label='Anomaly',
        alpha=0.7
    )

    plt.title('2D t-SNE Plot of Normal and Anomaly Features')
    plt.legend()

    # Set labels for x and y axes
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    os.chdir(config.plots_folder)
    plt.savefig(f'coreset_audio{config.audio_flag}_depth{config.depth_flag}_video{config.video_flag}')
    os.chdir(config.src_folder)
    plt.show()


def plot_3d_tsne(normal_features, anomaly_features):
    # Generate labels for normal and anomaly features
    normal_labels = np.zeros(normal_features.shape[0])
    anomaly_labels = np.ones(anomaly_features.shape[0])

    # Combine normal and anomaly features
    combined_features = np.vstack([normal_features, anomaly_features])
    combined_labels = np.hstack([normal_labels, anomaly_labels])

    # Perform t-SNE on combined features
    tsne = TSNE(n_components=3, perplexity=5, random_state=42)
    combined_features_tsne = tsne.fit_transform(combined_features)

    # Plot the results in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        combined_features_tsne[:, 0],
        combined_features_tsne[:, 1],
        combined_features_tsne[:, 2],
        c=combined_labels,
        cmap='viridis',
        alpha=0.7
    )

    ax.set_title('3D t-SNE Plot of Normal and Anomaly Features')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.legend(['Normal', 'Anomaly'])
    fig.colorbar(scatter)

    os.chdir(config.plots_folder)
    plt.savefig("test_normal_anomaly_features_3D.png")
    os.chdir(config.src_folder)

os.chdir(config.feature_folder)
# Load the array from the npy file
normalized_train_audio_depth_video_features = np.load(
    f'normalized_train_audio{config.audio_flag}_depth{config.depth_flag}_video{config.video_flag}_features.npy',
    allow_pickle=True
)

# Example usage
normalized_test_audio_depth_video_features = np.load(
    f'normalized_test_audio{config.audio_flag}_depth{config.depth_flag}_video{config.video_flag}_features.npy',
    allow_pickle=True
)


test_lables = np.load(
    f'test_audio{config.audio_flag}_depth{config.depth_flag}_video{config.video_flag}_labels.npy', 
    allow_pickle=True
)
print(normalized_test_audio_depth_video_features.shape)

normal_features = normalized_test_audio_depth_video_features[np.where(test_lables == 1)]
normalized_audio_depth_video_features = np.vstack((normalized_train_audio_depth_video_features, normal_features))
anomaly_features = normalized_test_audio_depth_video_features[np.where(test_lables == 0)]
# plot_tsne(normalized_test_video_audio_features, "all")
# plot_2d_tsne(normalized_audio_depth_video_features , anomaly_features)

coreset_indices = get_coreset_idx(normalized_train_audio_depth_video_features, n = 100)
print(normalized_train_audio_depth_video_features[coreset_indices].shape, normalized_train_audio_depth_video_features.shape)
plot_2d_tsne(normalized_train_audio_depth_video_features, \
             normalized_train_audio_depth_video_features[coreset_indices])

# print(normalized_video_audio_features.shape)
# print(anomaly_features.shape)
