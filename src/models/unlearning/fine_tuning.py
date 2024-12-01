import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

def fine_tuning(model, trainX, trainY, testX, testY, feature_columns, model_type, noise_std=0.01):
    """
    Fine-tunes the model after masking multiple features at once, optionally adding noise.
    Args:
        model: The trained model (LightGBM, XGBoost, etc.)
        trainX, trainY: Training data.
        testX, testY: Test data.
        feature_columns: List of feature column indices to mask.
        model_type: The type of model ('lightgbm', 'xgboost', etc.)
        noise_std: Standard deviation of Gaussian noise added to the masked features.
        
    Returns:
        Fine-tuned RMSE and the fine-tuned model.
    """
    # Create a copy of the data to avoid modifying the original
    masked_trainX = trainX.copy()
    masked_testX = testX.copy()

    # Masking multiple columns at once and adding noise
    for col in feature_columns:
        # Generate noise with the same shape as the masked column
        train_noise = np.random.normal(loc=0, scale=noise_std, size=masked_trainX[:, :, col].shape)
        test_noise = np.random.normal(loc=0, scale=noise_std, size=masked_testX[:, :, col].shape)

        # Apply masking and add noise
        masked_trainX[:, :, col] = train_noise
        masked_testX[:, :, col] = test_noise
    
    # Flatten the data for LightGBM and XGBoost
    masked_trainX_reshaped = masked_trainX.reshape(masked_trainX.shape[0], -1)  # Flatten to 2D
    masked_testX_reshaped = masked_testX.reshape(masked_testX.shape[0], -1)  # Flatten to 2D
    
    # Fine-tune the model with the masked and noisy data
    if model_type in ["lightgbm", "catboost"]:
        fine_tuned_predictions = model.predict(masked_testX_reshaped)
    elif model_type == "xgboost":
        fine_tuned_predictions = model.predict(xgb.DMatrix(masked_testX_reshaped))
    
    # Calculate RMSE for the fine-tuned model
    fine_tuned_rmse = np.sqrt(mean_squared_error(testY, fine_tuned_predictions))
    
    return fine_tuned_rmse, model
