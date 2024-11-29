import lightgbm as lgb
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error

def fine_tune_model(model, trainX, trainY, testX, testY, feature_index, model_type):
    # Create a copy of the data to avoid modifying the original data
    masked_trainX = trainX.copy()
    masked_testX = testX.copy()
    
    # Set the specified feature to zero in both the training and testing data
    masked_trainX[:, :, feature_index] = 0
    masked_testX[:, :, feature_index] = 0
    
    # For LightGBM and XGBoost, flatten the data (3D to 2D)
    if model_type in ["lightgbm", "xgboost"]:
        masked_trainX_reshaped = masked_trainX.reshape(masked_trainX.shape[0], -1)  # Flatten to 2D
        masked_testX_reshaped = masked_testX.reshape(masked_testX.shape[0], -1)  # Flatten to 2D
        
        # Make predictions using the fine-tuned model
        if model_type == "lightgbm":
            masked_trainX_flat = masked_trainX.reshape(masked_trainX.shape[0], -1)  # Flatten to 2D
            masked_testX_flat = masked_testX.reshape(masked_testX.shape[0], -1)  # Flatten to 2D
            fine_tuned_predictions = model.predict(masked_testX_flat)

        elif model_type == "xgboost":
            fine_tuned_predictions = model.predict(xgb.DMatrix(masked_testX_reshaped))
    else:
        # For LSTM and other model types, just use the regular numpy array (3D input)
        fine_tuned_predictions = model.predict(masked_testX)
    
    # Calculate RMSE for the fine-tuned model
    fine_tuned_rmse = np.sqrt(mean_squared_error(testY, fine_tuned_predictions))
    
    #return fine_tuned_predictions, fine_tuned_rmse
    return fine_tuned_rmse, model
