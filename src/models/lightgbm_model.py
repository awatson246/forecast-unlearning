import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error
import math

def train_lightgbm(trainX, trainY, testX, testY):
    """Trains a LightGBM model and calculates RMSE on test data."""
    
    # Initialize and train the LightGBM model
    model = lgb.LGBMRegressor()
    model.fit(trainX, trainY)
    
    # Predict on the test set
    predictions = model.predict(testX)
    
    # Calculate RMSE
    rmse = math.sqrt(root_mean_squared_error(testY, predictions))
    
    return rmse, model
