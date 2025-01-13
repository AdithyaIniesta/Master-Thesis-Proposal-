# Operating system functions
import os  
# Warning handling
import warnings  
# Ignore the specific UserWarning from torchvision.io
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.io"
)

import json

# Type hints for function arguments and return values
from typing import List, Tuple  

import config
# Audio processing library
import librosa  

# Progress bar for iterations
from tqdm import tqdm

# PyTorch library
import torch  

# NumPy library for numerical operations
import numpy as np  

# Image module from the PIL library
from PIL import Image  

# PyTorch vision library for image processing
import torchvision  

# Specific transformations for image processing
from torchvision.transforms import Compose, CenterCrop, \
    Resize, ToTensor, Normalize  

# Interpolation mode for image resizing
from torchvision.transforms.functional import InterpolationMode, to_pil_image 

# Define the parameters for image transformations
# Crop size for center cropping
crop_size = [112, 112]  
# Resize dimensions after cropping
resize_size = [128, 171]  
# Mean values for normalization
mean = [0.43216, 0.394666, 0.37645]  
 # Standard deviation values for normalization
std = [0.22803, 0.22145, 0.216989] 
# Interpolation mode for resizing
interpolation = InterpolationMode.BILINEAR  

# Define the video classification transformation pipeline
video_classification_transform = Compose([
    # Center crop the image
    CenterCrop(crop_size),  
     # Resize the image
    Resize(resize_size, interpolation=interpolation), 
    # Convert the image to a PyTorch tensor
    ToTensor(),  
    # Normalize the tensor
    Normalize(mean=mean, std=std),  
])

depth_transform = Compose([
    # Convert the image to a PyTorch tensor
    ToTensor(), 
    # Center crop the image
    CenterCrop(crop_size),  
     # Resize the image
    Resize(resize_size, interpolation=interpolation , antialias=True),  
    # Normalize the tensor
    Normalize(mean=mean, std=std),  
])

def spec_to_image(spec, eps=1e-6):
    """
    Convert a spectrogram to an image.

    Args:
        spec (numpy.ndarray): The input spectrogram.
        eps (float, optional): A small constant to prevent division by zero.

    Returns:
        numpy.ndarray: The converted image.
    """
    # Compute mean and standard deviation
    mean = spec.mean()
    std = spec.std()

    # Normalize the spectrogram
    spec_norm = (spec - mean) / (std + eps)

    # Scale the spectrogram to the range [0, 255]
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)

    # Convert to unsigned integer type
    spec_scaled = spec_scaled.astype(np.uint8)

    return spec_scaled

def get_melspectrogram_db(
    file_path,
    start_time, 
    end_time,
    sr=None,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    fmin=20,
    fmax=8300,
    top_db=80,
):
    """
    Compute the mel spectrogram in decibels from an audio file.

    Args:
        file_path (str): Path to the audio file.
        sr (int or None, optional): Sample rate. 
        If None, it will be automatically determined.
        n_fft (int, optional): Number of FFT points.
        hop_length (int, optional): Hop length for spectrogram computation.
        n_mels (int, optional): Number of mel bins.
        fmin (float, optional): Minimum frequency for mel filterbanks.
        fmax (float, optional): Maximum frequency for mel filterbanks.
        top_db (float, optional): Threshold for dynamic range compression.

    Returns:
        numpy.ndarray: The mel spectrogram in decibels.
    """
    # Load the audio file
    
    wav, sr = librosa.load(file_path, sr=sr)
    # Convert timestamps to sample indices
    start_index = int(start_time * sr)
    end_index = int(end_time * sr)
    # Clip the audio between the specified timestamps
    wav = wav[start_index:end_index]
    # Compute the mel spectrogram
    spec = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )

    # Convert to decibels
    spec_db = spec_to_image(librosa.power_to_db(spec, top_db=top_db))
    return spec_db


def extract_audio_features(
    # Path to the parent folder with audio files.
    parent_folder: str,
    # Audio feature extractor model.
    audio_feature_extractor: torch.nn.Module,
) -> List[np.ndarray]:
    
    audio_features = []

    # Loop through subfolders (videos) in the parent folder
    for dir in tqdm(sorted(os.listdir(parent_folder))):
        # Construct path to RGB frames.
        sub_dirs_path = os.path.join(parent_folder, dir)

        sub_dirs = sorted(os.listdir(sub_dirs_path))

        # Find the first file with the specified suffix in sub_dirs
        time_stamp_file = next(
            filter(lambda file: file.endswith("start_end_time_new.npy"), sub_dirs),
            None)
        
        # Construct the full path for time_stamp_file
        time_stamp_file = os.path.join(sub_dirs_path, time_stamp_file)

        # Load time stamps from the file
        time_stamps = np.load(time_stamp_file, allow_pickle=True)

        # Extract start and end times
        start_bag_time = time_stamps[0]['start_time']
        start_pour_time = time_stamps[0]['time'] - start_bag_time
        end_pour_time = time_stamps[1]['time'] - start_bag_time

        # Find the first audio file in the subfolder
        audio_file = next(
            filter(lambda file: file.endswith(".wav"), sub_dirs),
            None
        )

        # Construct the full path for the audio file
        audio_file = os.path.join(sub_dirs_path, audio_file)


        # Get Mel spectrogram as a NumPy array
        mel_spectrogram = get_melspectrogram_db(audio_file,\
                                                start_pour_time,\
                                                end_pour_time)
        
        # Convert to PyTorch tensor and move to CPU
        mel_spectrogram = torch.tensor(mel_spectrogram)\
            .to('cpu', dtype=torch.float32)
        
        # Reshape mel_spectrogram to [1, 1, 128, 431]
        # mel_spectrogram = mel_spectrogram.reshape(1, *mel_spectrogram.shape)

        # Perform inference using the audio feature extractor
        extracted_features = audio_feature_extractor(mel_spectrogram)\
            .cpu().detach().numpy().ravel()

        audio_features.append(extracted_features)
    
    return audio_features

def _extract_combined_features(
    # Path to the parent folder with RGB frames and audio files.
    parent_folder: str,    
    # Video feature extractor model.
    video_feature_extractor: torch.nn.Module,
    # Audio feature extractor model.
    audio_feature_extractor: torch.nn.Module,
    # Number of frames to uniformly sample from each video.
    num_frames_to_sample: int = config.num_frames_to_sample,
    # Transformation for each frame.
    video_transform: torchvision.transforms.Compose = (
        video_classification_transform
    ), 
    time_stamp_file_extension: str = "start_end_time_new.npy",
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Process video frames and audio files to extract features using
    pre-trained models.

    Parameters:
    - parent_folder (str): Path to the parent folder
      containing subfolders with RGB frames and audio files.
    - num_frames_to_sample (int): Number of frames to
      uniformly sample from each video.
    - video_feature_extractor (torch.nn.Module): Pre-trained neural
      network model for video.
    - video_transform (torchvision.transforms.Compose):
      Transformation to apply to each video frame.
    - audio_model (torch.nn.Module): Pre-trained neural
      network model for audio.
    - audio_feature_extractor (torch.nn.Module):
      Audio feature extractor model.

    Returns:
    - video_features_list (list): List to store processed
      video features.
    - audio_features_list (list): List to store processed
      audio features.
    - labels_list (list): List to store labels (0
      in this case).
    """
    # List to store processed features for each video
    video_features_list = []

    # List to store processed features for each audio file
    audio_features_list = []

    meta_data_list = []

    # Loop through subfolders (videos) in the parent folder
    for dir in tqdm(sorted(os.listdir(parent_folder))):
        # Construct path to RGB frames and audio files.
        sub_dirs_path = os.path.join(parent_folder, dir)

        sub_dirs = sorted(os.listdir(sub_dirs_path))

        meta_data_file = next(
            filter(lambda file: file.endswith("generic_task.json"), sub_dirs),
            None)
        
        meta_data_file = os.path.join(sub_dirs_path, meta_data_file)

        with open(meta_data_file, 'r') as file:
            meta_data = json.load(file)
        # Extract 'Initial level' from the data
        initial_level = meta_data['config']['Initial level'][0]
        meta_data_list.append(initial_level)

        # Find the first file with the specified suffix in sub_dirs
        time_stamp_file = next(
            filter(lambda file: file.endswith(time_stamp_file_extension), sub_dirs),
            None)

        # Construct the full path for time_stamp_file
        time_stamp_file = os.path.join(sub_dirs_path, time_stamp_file)

        # Load time stamps from the file
        time_stamps = np.load(time_stamp_file, allow_pickle=True)

        # Extract start and end times
        start_bag_time = time_stamps[0]['start_time']
        start_pour_time = time_stamps[0]['time'] - start_bag_time
        end_pour_time = time_stamps[1]['time'] - start_bag_time

        # Find the first audio file in the subfolder
        audio_file = next(
            filter(lambda file: file.endswith(".wav"), sub_dirs),
            None
        )

        # Construct the full path for the audio file
        audio_file = os.path.join(sub_dirs_path, audio_file)

        # Get Mel spectrogram as a NumPy array
        mel_spectrogram = get_melspectrogram_db(audio_file,\
                                                start_pour_time,\
                                                end_pour_time)
        
        # Convert to PyTorch tensor and move to CPU
        mel_spectrogram = torch.tensor(mel_spectrogram)\
            .to('cpu', dtype=torch.float32).repeat(1, 3, 1, 1)
        print(mel_spectrogram.shape)
        # mel_spectrogram = mel_spectrogram.unsqueeze(0).permute(0,2,1,3,4)

        # Reshape mel_spectrogram to [1, 3, 128, 431]
        # mel_spectrogram = mel_spectrogram.reshape(1, *mel_spectrogram.shape)

        # Perform inference using the audio feature extractor
        extracted_audio_features = audio_feature_extractor(mel_spectrogram)\
            .cpu().detach().numpy().ravel()

        audio_features_list.append(extracted_audio_features)

        # Calculate video duration
        video_duration = end_pour_time - start_bag_time

        # Find the first file with the specified suffix in sub_dirs
        image_stamp_file = next(
            filter(lambda file: file.endswith("timestamps.npy"), sub_dirs),
            None)
        
        # Construct the full path for image_stamp_file
        image_stamp_file = os.path.join(sub_dirs_path, image_stamp_file)

        # Load image time stamps from the file and adjust for start_bag_time
        image_time_stamps = np.load(image_stamp_file, allow_pickle=True)
        image_time_stamps = image_time_stamps - start_bag_time

        # Create a boolean mask for the values between the bounds
        mask = (image_time_stamps >= start_pour_time) & \
            (image_time_stamps <= end_pour_time)
        
        indices = np.where(mask)[0]

        # Filter image time stamps based on the boolean mask
        image_time_stamps = image_time_stamps[mask]

        rgb_dir = os.path.join(sub_dirs_path, 'rgb')
    
        # List to store processed frames for each video
        processed_frames = []

        sorted_files = sorted(os.listdir(rgb_dir))
        sorted_files = [sorted_files[frame_idx] for frame_idx in indices]

        # List to store uniformly sampled frame indices
        sampled_frame_indices = np.linspace(
            0, len(sorted_files) - 1,
            num_frames_to_sample,
            dtype=int
        )
        
        # Loop through frames in the video
        for frame_idx in sampled_frame_indices:
            frame_name = sorted_files[frame_idx]
            frame_path = os.path.join(rgb_dir, frame_name)
            # Open and convert to RGB format.
            image = Image.open(
                frame_path).convert("RGB") 
            
            # Apply the transformation to the image
            input_tensor = video_transform(image)  
            processed_frames.append(input_tensor)

        # Stack the processed frames to create the input batch
        input_batch = torch.stack(
            processed_frames)  
        # Add a batch dimension
        input_tensor_permuted = input_batch.unsqueeze(0)
        input_tensor_permuted = input_tensor_permuted.permute(0,2,1,3,4)

        # Extract features using the pre-trained model.
        with torch.no_grad():
            features = video_feature_extractor(
                input_tensor_permuted).squeeze().numpy()  

        # Horizontally stack features
        # features_reshaped = np.hstack((features,))
        video_features_list.append(features)

    return video_features_list, audio_features_list, \
          meta_data_list

def extract_combined_features(
    parent_folder: str,
    audio_feature_extractor: torch.nn.Module,
    depth_feature_extractor: torch.nn.Module,
    video_feature_extractor: torch.nn.Module,
    time_stamp_file_extension: str,
    num_frames_to_sample: int = config.num_frames_to_sample,
    video_transform: torchvision.transforms.Compose = video_classification_transform,
    depth_transform: torchvision.transforms.Compose = depth_transform, 
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Extracts combined features from video, audio, and depth modalities.

    Parameters:
    - parent_folder (str): Path to the parent folder containing subfolders with data.
    - video_feature_extractor (torch.nn.Module): Pre-trained neural network model for video.
    - audio_feature_extractor (torch.nn.Module): Pre-trained neural network model for audio.
    - depth_feature_extractor (torch.nn.Module): Pre-trained neural network model for depth.
    - num_frames_to_sample (int): Number of frames to uniformly sample from each video.
    - video_transform (transforms.Compose): Transformation to apply to each video frame.
    - time_stamp_file_extension (str): Extension of the timestamp file.

    Returns:
    - video_features_list (list): List to store processed video features.
    - audio_features_list (list): List to store processed audio features.
    - depth_features_list (list): List to store processed depth features.
    - meta_data_list (list): List to store metadata.
    """
    video_features_list = []
    audio_features_list = []
    depth_features_list = []
    meta_data_list = []

    # Loop through subfolders (videos) in the parent folder
    for dir in tqdm(sorted(os.listdir(parent_folder))):
        sub_dirs_path = os.path.join(parent_folder, dir)
        sub_dirs = sorted(os.listdir(sub_dirs_path))

        # Load metadata from generic_task.json file
        meta_data_file = next(
            filter(lambda file: file.endswith("generic_task.json"), sub_dirs),
            None
        )
        meta_data_file = os.path.join(sub_dirs_path, meta_data_file)
        with open(meta_data_file, 'r') as file:
            meta_data = json.load(file)
        initial_level = meta_data['config']['Initial level'][0]
        meta_data_list.append(initial_level)
        
        # Load time stamps from the file
        time_stamp_file = next(
            filter(lambda file: file.endswith(time_stamp_file_extension), sub_dirs),
            None
        )
        time_stamp_file = os.path.join(sub_dirs_path, time_stamp_file)
        time_stamps = np.load(time_stamp_file, allow_pickle=True)

        start_bag_time = time_stamps[0]['start_time']
        start_pour_time = time_stamps[0]['time'] - start_bag_time
        end_pour_time = time_stamps[1]['time'] - start_bag_time

        # Load audio file and extract mel spectrogram
        audio_file = next(
            filter(lambda file: file.endswith(".wav"), sub_dirs),
            None
        )
        audio_file = os.path.join(sub_dirs_path, audio_file)
        mel_spectrogram = get_melspectrogram_db(audio_file, start_pour_time, end_pour_time)
        mel_spectrogram = torch.tensor(mel_spectrogram)\
            .to('cpu', dtype=torch.float32).repeat(1, 3, 1, 1)

        # Extract audio features
        extracted_audio_features = audio_feature_extractor(mel_spectrogram)\
            .cpu().detach().numpy().ravel()
        audio_features_list.append(extracted_audio_features)

        video_duration = end_pour_time - start_bag_time

        # Load image time stamps
        image_stamp_file = next(
            filter(lambda file: file.endswith("timestamps.npy"), sub_dirs),
            None
        )
        image_stamp_file = os.path.join(sub_dirs_path, image_stamp_file)
        image_time_stamps = np.load(image_stamp_file, allow_pickle=True)
        image_time_stamps = image_time_stamps - start_bag_time

        # Create a boolean mask for the values between the bounds
        mask = (image_time_stamps >= start_pour_time) & \
            (image_time_stamps <= end_pour_time)
        indices = np.where(mask)[0]
        image_time_stamps = image_time_stamps[mask]

        # Extract video features
        rgb_dir = os.path.join(sub_dirs_path, 'rgb')
        processed_frames = []
        sorted_files = sorted(os.listdir(rgb_dir))
        sorted_files = [sorted_files[frame_idx] for frame_idx in indices]

        sampled_frame_indices = np.linspace(
            0, len(sorted_files) - 1,
            num_frames_to_sample,
            dtype=int
        )
        
        for frame_idx in sampled_frame_indices:
            frame_name = sorted_files[frame_idx]
            frame_path = os.path.join(rgb_dir, frame_name)
            image = Image.open(frame_path).convert("RGB") 
            input_tensor = video_transform(image)  
            processed_frames.append(input_tensor)

        input_batch = torch.stack(processed_frames)  
        input_tensor_permuted = input_batch.unsqueeze(0)
        input_tensor_permuted = input_tensor_permuted.permute(0, 2, 1, 3, 4)

        with torch.no_grad():
            features = video_feature_extractor(input_tensor_permuted).squeeze().numpy()  
        video_features_list.append(features)

        # Extract depth features
        depth_dir = os.path.join(sub_dirs_path, 'depth')
        processed_frames = []
        sorted_files = sorted(os.listdir(depth_dir))
        sorted_files = [sorted_files[frame_idx] for frame_idx in indices]

        sampled_frame_indices = np.linspace(
            0, len(sorted_files) - 1,
            num_frames_to_sample,
            dtype=int
        )
        
        for frame_idx in sampled_frame_indices:
            frame_name = sorted_files[frame_idx]
            frame_path = os.path.join(depth_dir, frame_name)
            image = np.load(frame_path, allow_pickle=True).astype(np.float32)
            input_tensor = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            input_tensor = depth_transform(input_tensor)
            processed_frames.append(input_tensor)

        input_batch = torch.stack(processed_frames)  
        input_tensor_permuted = input_batch.unsqueeze(0)
        input_tensor_permuted = input_tensor_permuted.permute(0, 2, 1, 3, 4)

        with torch.no_grad():
            features = depth_feature_extractor(input_tensor_permuted).squeeze().numpy()  
        depth_features_list.append(features)

    return audio_features_list, depth_features_list, video_features_list, meta_data_list


def extract_depth_features(
    parent_folder: str,
    depth_feature_extractor: torch.nn.Module,
    num_frames_to_sample: int = config.num_frames_to_sample,
    depth_transform: torchvision.transforms.Compose = video_classification_transform,
    time_stamp_file_extension: str = "start_end_time_new.npy",
) -> List[np.ndarray]:
    """
    Process video frames and audio files to extract features using
    pre-trained models.

    Parameters:
    - parent_folder (str): Path to the parent folder containing subfolders with RGB frames and audio files.
    - depth_feature_extractor (torch.nn.Module): Pre-trained neural network model for depth.
    - num_frames_to_sample (int): Number of frames to uniformly sample from each video.
    - video_transform (transforms.Compose): Transformation to apply to each video frame.
    - time_stamp_file_extension (str): Extension for the timestamp file.

    Returns:
    - depth_features_list (List[np.ndarray]): List to store processed depth features.
    """
    # List to store processed features for each video
    depth_features_list = []

    # Loop through subfolders (videos) in the parent folder
    for dir in tqdm(sorted(os.listdir(parent_folder))):
        # Construct path to RGB frames and audio files.
        sub_dirs_path = os.path.join(parent_folder, dir)
        sub_dirs = sorted(os.listdir(sub_dirs_path))

        # Find the first file with the specified suffix in sub_dirs
        time_stamp_file = next(
            filter(lambda file: file.endswith(time_stamp_file_extension), sub_dirs),
            None
        )

        # Construct the full path for time_stamp_file
        time_stamp_file = os.path.join(sub_dirs_path, time_stamp_file)

        # Load time stamps from the file
        time_stamps = np.load(time_stamp_file, allow_pickle=True)

        # Extract start and end times
        start_bag_time = time_stamps[0]['start_time']
        start_pour_time = time_stamps[0]['time'] - start_bag_time
        end_pour_time = time_stamps[1]['time'] - start_bag_time

        # Find the first file with the specified suffix in sub_dirs
        image_stamp_file = next(
            filter(lambda file: file.endswith("timestamps.npy"), sub_dirs),
            None
        )

        # Construct the full path for image_stamp_file
        image_stamp_file = os.path.join(sub_dirs_path, image_stamp_file)

        # Load image time stamps from the file and adjust for start_bag_time
        image_time_stamps = np.load(image_stamp_file, allow_pickle=True)
        image_time_stamps = image_time_stamps - start_bag_time

        # Create a boolean mask for the values between the bounds
        mask = (image_time_stamps >= start_pour_time) & \
            (image_time_stamps <= end_pour_time)

        indices = np.where(mask)[0]

        # Filter image time stamps based on the boolean mask
        image_time_stamps = image_time_stamps[mask]

        depth_dir = os.path.join(sub_dirs_path, 'depth')

        # List to store processed frames for each video
        processed_frames = []

        sorted_files = sorted(os.listdir(depth_dir))
        sorted_files = [sorted_files[frame_idx] for frame_idx in indices]

        # List to store uniformly sampled frame indices
        sampled_frame_indices = np.linspace(
            0, len(sorted_files) - 1,
            num_frames_to_sample,
            dtype=int
        )

        # Loop through frames in the video
        for frame_idx in sampled_frame_indices:
            frame_name = sorted_files[frame_idx]
            frame_path = os.path.join(depth_dir, frame_name)

            # Open depth image
            image = np.load(frame_path, allow_pickle=True).astype(np.float32)
            input_tensor = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            input_tensor = depth_transform(input_tensor)
            # Apply the transformation to the image
            processed_frames.append(input_tensor)

        # Stack the processed frames to create the input batch
        input_batch = torch.stack(
            processed_frames)
        # Add a batch dimension
        input_tensor_permuted = input_batch.unsqueeze(0)
        input_tensor_permuted = input_tensor_permuted.permute(0, 2, 1, 3, 4)
        # Extract features using the pre-trained model.
        with torch.no_grad():
            features = depth_feature_extractor(
                input_tensor_permuted).squeeze().numpy()

        # Horizontally stack features
        depth_features_list.append(features)

    return depth_features_list
