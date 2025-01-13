# Module for handling warnings
import warnings  

# Ignore the specific UserWarning from torchvision.io
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torchvision.io"
)

# PyTorch library for neural networks
import torch 
import torch.nn as nn  
import torch.optim as optim

# Models and model weights for video classification
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights  

# Import the ResNet-34 model from torchvision
from torchvision.models import resnet18, ResNet18_Weights

# Import your custom configuration and data loaders
import config

# Get the device from the configuration
device = config.device

# Create a ResNet-34 model with pretrained weights
audio_feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
audio_feature_extractor = nn.Sequential(
    *list(audio_feature_extractor.children())[:-1]
)
# Move the model to the specified device (e.g., GPU or CPU)
audio_feature_extractor = audio_feature_extractor.to(device)

audio_feature_extractor.eval()

# Load the pre-trained R2Plus1D-18 model for audio feature extraction
video_feature_extractor = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
video_feature_extractor = nn.Sequential(
    *list(video_feature_extractor.children())[:-1]
)
# Move the model to the specified device (e.g., GPU or CPU)
video_feature_extractor = video_feature_extractor.to(device)
# Set the model to evaluation mode
video_feature_extractor.eval()

# Create a ResNet-34 model with pretrained weights
depth_feature_extractor = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
depth_feature_extractor = nn.Sequential(
    *list(depth_feature_extractor.children())[:-1]
)
# Move the model to the specified device (e.g., GPU or CPU)
depth_feature_extractor = depth_feature_extractor.to(device)
depth_feature_extractor.eval()



















"""
# Remove the softmax layer from the model for feature extraction
video_feature_extractor = nn.Sequential(
    *list(video_feature_extractor.children())[:-1]
)

# Define the loss function (MultiMarginLoss)
loss_fn = nn.MultiMarginLoss()

# Get the learning rate from the configuration
learning_rate = config.learning_rate

# Define the optimizer (Adam) for training the model
optimizer = optim.Adam(
    audio_feature_extractor.parameters(),
    lr=learning_rate
)

Create a Sequential model for feature extraction 
(excluding the last softmax layer)
audio_feature_extractor = torch.nn.Sequential(
    *list(audio_feature_extractor.children())[:-1]
)
"""
