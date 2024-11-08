import numpy as np

def apply_feature_masking(model, data, feature_index):
    # Ensure data has the correct 3D shape and enough features to mask
    if data.ndim != 3:
        raise ValueError(f"Input data has incompatible shape: {data.shape}, expected a 3D array.")
    if data.shape[2] <= feature_index:
        raise ValueError(f"Feature index {feature_index} is out of bounds for input data with shape {data.shape}.")

    # Copy data to avoid modifying the original testX
    masked_data = data.copy()
    
    # Set the specified feature to zero across all time steps (masking)
    masked_data[:, :, feature_index] = 0
    
    return masked_data
