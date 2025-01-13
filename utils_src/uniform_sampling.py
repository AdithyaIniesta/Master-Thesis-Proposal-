import matplotlib.pyplot as plt
import numpy as np

# Number of frames in the video
total_frames = 20

# Number of frames to uniformly sample
num_frames_to_sample = 8

# Generate random 1D coordinates for each frame
np.random.seed(42)
frame_coordinates = np.random.rand(total_frames)

# Generate indices for uniform sampling
sampled_frame_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)
print(sampled_frame_indices)
# Create a line plot
plt.figure(figsize=(10, 1))  # Set the figure height to 1 to only display along the x-axis

# Plot all frames
plt.plot(range(total_frames), frame_coordinates, marker='o', linestyle='', color='blue', label='All Frames')

# Highlight sampled frames
plt.scatter(sampled_frame_indices, np.zeros_like(sampled_frame_indices), color='red', label='Sampled Frames')

# Remove y-axis
plt.yticks([])

# Set labels and title
plt.xlabel('Frame Index')
plt.title('Uniform Sampling of Video Frames Along X-Axis')

# Add a legend
plt.legend()

# Save the plot as an image file (e.g., PNG)
plt.savefig('1d_video_frames_plot.png')

# Show the plot
plt.show()
