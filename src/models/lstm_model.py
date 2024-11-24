import math
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def train_lstm(train, test, input_shape, look_back=1):
    """Builds, trains, and evaluates the LSTM model."""
    
    # Unpack train and test datasets into features (X) and targets (Y)
    trainX, trainY = train
    testX, testY = test
    
    # Build the LSTM model
    model = Sequential([
        LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(25),
        Dropout(0.2),
        Dense(1),
    ])

    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(trainX, trainY, epochs=100, batch_size=64, verbose=2, callbacks=early_stop)
    
    # Make predictions
    test_predict = model.predict(testX)

    # Calculate RMSE
    rmse = math.sqrt(mean_squared_error(testY, test_predict))
    
    return rmse, model, testX, testY
