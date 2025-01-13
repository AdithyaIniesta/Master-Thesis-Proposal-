from typing import List, Any
import numpy as np 
from sklearn.preprocessing import (
    LabelEncoder, 
    OneHotEncoder, 
    StandardScaler
)

import config

from feature_extraction import (  
    extract_combined_features,
)
from model import (  
    audio_feature_extractor,
    depth_feature_extractor, 
    video_feature_extractor,
)

from sklearn.random_projection import GaussianRandomProjection

# Function to process a batch and append features and labels to testing dataset
def process_batch(batch: List[Any], 
                  label_value: int, 
                  time_stamp_file_extension: str):
    
    audio_features_list = []
    depth_feature_list = []
    video_features_list = []
    labels_list = []
    meta_data_list = []

    # Iterate through each item in the batch
    for batch_item in batch:
        # Extract combined features and labels for the current batch.
        # Extract video features and labels for the current batch.
        batch_audio_features_list, batch_depth_features_list, \
        batch_video_features_list, batch_meta_data_list = extract_combined_features(
        batch_item,
        audio_feature_extractor,
        video_feature_extractor,
        depth_feature_extractor,
        time_stamp_file_extension 
    )
    
        # Iterate through individual features and labels in the batch.
        for audio_feature, video_feature, \
            depth_feature, meta_data in zip(
                batch_audio_features_list, 
                batch_depth_features_list,  
                batch_video_features_list, 
                batch_meta_data_list
        ):
            # Append features and labels to the testing dataset.
            
            audio_features_list.append(audio_feature)
            depth_feature_list.append(depth_feature)
            video_features_list.append(video_feature)
            labels_list.append(label_value)
            meta_data_list.append(meta_data)

    return audio_features_list, depth_feature_list, video_features_list,\
            meta_data_list, labels_list

def scale_features(features: List[Any]) -> List[Any]:
    """
    Standardize the input features using the provided scaler.

    Parameters:
    - features (List[Any]): List of input features.
    - scaler (StandardScaler): Scaler to standardize the features.

    Returns:
    - List[Any]: Standardized features.
    """
    scaler = StandardScaler()
    features_array = np.array(features)
    normalized_features = scaler.fit_transform(features_array)
    return normalized_features

def fuse_features(feature_dict) -> np.ndarray:
    # Initialize an empty list for selected features
    selected_features = []  

    # Fuse audio modality if the flag is set
    if config.audio_flag:
        print("Audio modality")
        selected_features.append(reduce_dimensionality_jl(feature_dict['audio']))

    # Fuse depth modality if the flag is set
    if config.depth_flag:
        print("Depth modality")
        selected_features.append(reduce_dimensionality_jl(feature_dict['depth']))

    # Fuse video modality if the flag is set
    if config.video_flag:
        print("Video modality")
        selected_features.append(reduce_dimensionality_jl(feature_dict['video']))

    # Check if at least one feature is selected
    if not any([
        config.audio_flag,
        config.video_flag,
        config.depth_flag
    ]):
        raise ValueError("At least one feature should be selected.")

    # Concatenate selected features
    return np.concatenate(selected_features, axis=1)

def reduce_dimensionality_jl(M, d_star = 400, disable_dl = config.disable_dl):
    """
    Reduce the dimensionality of matrix M using Johnson-Lindenstrauss lemma.

    Parameters:
    - M (numpy.ndarray): Input matrix of shape (m, d).
    - d_star (int): Target dimensionality.

    Returns:
    - numpy.ndarray: Matrix with reduced dimensionality of shape (m, d_star).
    """
    m, d = M.shape

    # Create a GaussianRandomProjection model
    jl_model = GaussianRandomProjection(n_components=d_star)

    # Fit and transform the input matrix
    if disable_dl:
        print(f'Reduced feature list {M_reduced.shape}')
        M_reduced = jl_model.fit_transform(M)
        return M_reduced
    else:
        return M 

def get_coreset_idx(
    z_lib: np.ndarray,
    n: int = 50,
    eps: float = 0.90,
    float16: bool = True,
) -> np.ndarray:
    """
    Adapted version of coreset subsampling for NumPy arrays.

    Parameters:
    - z_lib (np.ndarray): Input NumPy array.
    - n (int): Number of tensors to keep in the coreset.
    - eps (float): Epsilon value for random projection.
    - float16 (bool): Use 16-bit float precision.

    Returns:
    - coreset_idx (np.ndarray): Array of indices for the coreset.
    """
    print("Beginning of the coreset subsampling reduction...")

    print(f"Fitting random projections. Start dim = {z_lib.shape}.")
    try:
        # A random projection is performed on the tensor_list
        # transformer = np.random.randn(z_lib.shape[1], int(eps * z_lib.shape[1]))
        # z_lib = np.dot(z_lib, transformer)
        # print(f"Done. Transformed dim = {z_lib.shape}.")
        pass
    except ValueError:
        print("   Error: could not project vectors. Please increase `eps`.")

    select_idx = 0
    last_item = z_lib[select_idx:select_idx + 1]
    coreset_idx = [select_idx]
    min_distances = np.linalg.norm(z_lib - last_item, axis=1, keepdims=True)

    if float16:
        last_item = last_item.astype(np.float16)
        z_lib = z_lib.astype(np.float16)
        min_distances = min_distances.astype(np.float16)

    # Iteration from 0 to n-1, the number of tensors that will be kept in the subsampling
    for i in range(n - 1):

        # The variables are updated based on the distances calculations
        # Broadcasting step
        distances = np.linalg.norm(z_lib - last_item, axis=1, keepdims=True)
        # Iterative step
        min_distances = np.minimum(distances, min_distances)
        # Selection step
        select_idx = np.argmax(min_distances)

        # Bookkeeping
        last_item = z_lib[select_idx:select_idx + 1]
        min_distances[select_idx] = 0
        coreset_idx.append(select_idx)

    print("End of the coreset subsampling reduction")
    return np.array(coreset_idx)
