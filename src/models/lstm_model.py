import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import root_mean_squared_error
import math

def train_lstm(train, test, input_shape):
    """Builds, trains, and evaluates the LSTM model."""
    
    # Split train and test datasets into features (X) and targets (Y)
    trainX, trainY = train
    testX, testY = test
    
    # Reshape trainX and testX for LSTM [samples, time steps, features]
    trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], trainX.shape[2]))
    testX = testX.reshape((testX.shape[0], testX.shape[1], testX.shape[2]))
    
    # Build the LSTM model
    model = Sequential([
        LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])),
        Dense(1)  # Assuming you're predicting a single value
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)
    
    # Make predictions
    test_predict = model.predict(testX)

    # Calculate RMSE
    rmse = math.sqrt(root_mean_squared_error(testY, test_predict))
    
    return rmse, model, testX, testY
