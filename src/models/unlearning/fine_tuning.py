import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

def fine_tuning(model, trainX, trainY, testX, testY, feature_columns, model_type, noise_std=0.2):
    """Fine-tunes the model after masking multiple features with stronger noise."""
    masked_trainX = trainX.copy()
    masked_testX = testX.copy()

    for col in feature_columns:
        train_noise = np.random.normal(loc=0, scale=noise_std, size=masked_trainX[:, :, col].shape)
        test_noise = np.random.normal(loc=0, scale=noise_std, size=masked_testX[:, :, col].shape)

        masked_trainX[:, :, col] = train_noise
        masked_testX[:, :, col] = test_noise
    
    masked_trainX_reshaped = masked_trainX.reshape(masked_trainX.shape[0], -1)
    masked_testX_reshaped = masked_testX.reshape(masked_testX.shape[0], -1)
    
    if model_type in ["lightgbm", "catboost"]:
        fine_tuned_predictions = model.predict(masked_testX_reshaped)
    elif model_type == "xgboost":
        fine_tuned_predictions = model.predict(xgb.DMatrix(masked_testX_reshaped))
    
    fine_tuned_rmse = np.sqrt(mean_squared_error(testY, fine_tuned_predictions))
    
    return fine_tuned_rmse, model
