import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error

def feature_masking(model, data, feature_indices, model_type, true_labels, noise_std=0.1):
    """Applies masking to the specified features by replacing them with more disruptive noise."""
    masked_data = data.copy()
    for feature_idx in feature_indices:
        if feature_idx >= data.shape[2]:
            raise ValueError(f"Feature index {feature_idx} is out of bounds for input data with shape {data.shape}.")
        
        # Add uniform noise
        original_values = data[:, :, feature_idx].flatten()
        noise = np.random.uniform(low=np.min(original_values), high=np.max(original_values), size=original_values.shape)
        masked_data[:, :, feature_idx] = noise.reshape(data.shape[0], data.shape[1])
    
    masked_data_reshaped = masked_data.reshape(masked_data.shape[0], -1)  # Flatten for LightGBM, XGBoost, etc.
    predictions = model.predict(masked_data_reshaped)
    rmse = np.sqrt(mean_squared_error(true_labels, predictions))
    return rmse, model
