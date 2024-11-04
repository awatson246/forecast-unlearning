from src.utils.menu import display_menu, get_user_choice, display_model_menu, get_model_choice
from src.data_loading import load_data
from src.data_preprocessing import preprocess_data, create_dataset
from src.models.lstm_model import train_lstm
from src.models.lightgbm_model import train_lightgbm
from src.utils.evaluation import permutation_importance

def main():
    # Display menu and get user choices
    display_menu()
    dataset_choice = get_user_choice()
    df, settings = load_data(dataset_choice)
    
    display_model_menu()
    model_choice = get_model_choice()

    # Preprocess data
    df_processed = preprocess_data(df, settings)

    # Train-test split
    train_size = int(len(df_processed) * 0.8)
    train_data, test_data = df_processed[:train_size], df_processed[train_size:]

    # Prepare datasets with a look-back window
    look_back = 3
    trainX, trainY = create_dataset(train_data, settings['target_column'], look_back)
    testX, testY = create_dataset(test_data, settings['target_column'], look_back)

    if model_choice == '1':
        print("Training LSTM model...")
        input_shape = trainX.shape[1:]
        rmse, model, _, _ = train_lstm((trainX, trainY), (testX, testY), input_shape)
        model_type = "lstm"
    elif model_choice == '2':
        print("Training LightGBM model...")
        # Flatten the input for LightGBM (2D shape)
        trainX_flat = trainX.reshape(trainX.shape[0], -1)
        testX_flat = testX.reshape(testX.shape[0], -1)
        rmse, model = train_lightgbm(trainX_flat, trainY, testX_flat, testY)
        model_type = "lightgbm"
    elif model_choice == '3':
        print("Sorry! This feature isn't ready for the world yet...")
    else:
        print("Invalid choice. Please select a valid model.")
        return

    # Display RMSE
    print(f"Model RMSE: {rmse}")

    # Calculate location-based feature importance if available
    location_columns = settings.get('location_columns', [])
    feature_indices = [train_data.columns.get_loc(col) for col in location_columns if col in train_data.columns]

    if feature_indices:
        # Determine model input format for feature importance calculation
        X_test_model = testX if model_type == "lstm" else testX_flat
        location_importance = permutation_importance(model, X_test_model, testY, feature_indices, look_back, model_type)
        print("Location-based feature importance calculated.")
    else:
        print("No location columns specified for feature importance calculation.")

if __name__ == "__main__":
    main()