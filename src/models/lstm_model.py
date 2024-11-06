import math
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_lstm(train, test, input_shape, look_back=1):
    """Builds, trains, and evaluates the LSTM model."""
    
    # Unpack train and test datasets into features (X) and targets (Y)
    trainX, trainY = train
    testX, testY = test
    
    # Build the LSTM model
    model = Sequential([
        LSTM(25, input_shape=(trainX.shape[1], trainX.shape[2])),
        Dense(1)  # Assuming you're predicting a single value
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(trainX, trainY, epochs=25, batch_size=1, verbose=2)
    
    # Make predictions
    test_predict = model.predict(testX)

    # Calculate RMSE
    rmse = math.sqrt(mean_squared_error(testY, test_predict))
    
    return rmse, model, testX, testY
