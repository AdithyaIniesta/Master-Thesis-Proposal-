import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def plot_tsne(high_dim_features, low_dim_features, save_path=None):
    """
    Perform t-SNE dimensionality reduction on the input features and create a scatter plot.

    Parameters:
    - high_dim_features (numpy.ndarray): High-dimensional input features.
    - low_dim_features (numpy.ndarray): Low-dimensional input features.
    - save_path (str, optional): If provided, the plot will be saved at the specified path.

    Returns:
    None
    """
    # Perform t-SNE dimensionality reduction for both high and low-dimensional features
    tsne = TSNE(n_components=2, random_state=42)
    embedded_high_dim = tsne.fit_transform(high_dim_features)
    embedded_low_dim = tsne.fit_transform(low_dim_features)

    # Plot the t-SNE features
    plt.figure(figsize=(12, 6))

    # Plot High-dimensional Features
    plt.subplot(1, 2, 1)
    plt.scatter(embedded_high_dim[:, 0], embedded_high_dim[:, 1], alpha=0.5)
    plt.title('t-SNE Plot of High-dimensional Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    # Plot Low-dimensional Features
    plt.subplot(1, 2, 2)
    plt.scatter(embedded_low_dim[:, 0], embedded_low_dim[:, 1], alpha=0.5)
    plt.title('t-SNE Plot of Low-dimensional Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='png')
        print(f'T-SNE plot saved to {save_path}')
    else:
        plt.show()

def save_joint_density_plot(audio_features, video_features, save_path='joint_density_plot.png'):
    """
    Create and save a joint density plot for low-dimensional audio features and high-dimensional video features.

    Parameters:
    - audio_features (numpy.ndarray): Low-dimensional audio features (2D array).
    - video_features (numpy.ndarray): High-dimensional video features (2D or 3D array).
    - save_path (str): Path to save the plot. Default is 'joint_density_plot.png'.

    Returns:
    - None
    """
    
    # Create a joint density plot
    plt.figure(figsize=(8, 6))
    
    # Plot Univariate Distributions (Marginal Distributions)
    sns.histplot(audio_features[:, 0], color='blue', kde=True, label='Audio Feature 1', stat='density')
    sns.histplot(audio_features[:, 1], color='blue', kde=True, label='Audio Feature 2', stat='density')

    if video_features.shape[1] == 3:
        sns.histplot(video_features[:, 0], color='green', kde=True, label='Video Feature 1', stat='density')
        sns.histplot(video_features[:, 1], color='green', kde=True, label='Video Feature 2', stat='density')
    else:
        sns.histplot(video_features[:, 0], color='green', kde=True, label='Video Feature', stat='density')

    # Plot Joint Density
    sns.kdeplot(
        x=audio_features[:, 0], 
        y=audio_features[:, 1], 
        cmap='Blues', 
        fill=True, 
        label='Joint Density'
    )

    plt.title('Joint Density Plot with Univariate Distributions')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.legend()

    # Save the plot
    plt.savefig(save_path)
    print(f'Plot saved at {save_path}')

# # Example usage for the t-SNE plot
# high_dim_features = np.random.rand(100, 128)
# low_dim_features = np.random.rand(100, 10)
# plot_tsne(high_dim_features, low_dim_features, save_path='tsne_plot_comparison.png')

# # Example usage for the joint density plot
# audio_features_example = np.random.randn(100, 2)
# video_features_example = np.random.randn(100, 3)
# save_joint_density_plot(audio_features_example, video_features_example, save_path='joint_density_plot.png')


import numpy as np
import matplotlib.pyplot as plt

# Values for y
# values_y =  [85.71, 78.79, 84.62, 96.00, 87.67, 93.67]
values_y = [89.74, 90.91, 90.91, 92.11, 92.11, 93.33]


# values_z =  [87.32, 88.00, 88.89, 89.16, 86.75, 93.15]
values_z = [89.47, 90.67, 89.47, 88.31, 90.67, 89.47]

y = np.array(values_y)

# Values for z

z = np.array(values_z)

# Indices for x
x = np.arange(50, 101, 10)

# Plot x vs y with label "svm + coreset" and markers
plt.plot(x, y, label="SVM + coreset", marker='o')

# Plot x vs z with label "svm + uniform" and markers
plt.plot(x, z, label="SVM + uniform", marker='o')
# Set y-axis limits
plt.ylim(0, 100)
# Set plot details
plt.xlabel("Num of Train Samples")
plt.ylabel("F1 Score %")
plt.title("Line Plots of Num of Train Samples vs F1 Score")
plt.legend()  # Show legend with labels
# Show the plot
plt.savefig("coreset")