from src.utils.menu import display_menu, get_user_choice, display_model_menu, get_model_choice
from src.data_loading import load_data
from src.data_preprocessing import preprocess_data, create_dataset
from src.models.lstm_model import train_lstm
from src.models.lightgbm_model import train_lightgbm
from src.models.xgboost_model import train_xgboost
from src.utils.evaluation import evaluate_model, permutation_importance, display_feature_importance

def main():
    display_menu()
    dataset_choice = get_user_choice()
    df, settings = load_data(dataset_choice)
    
    display_model_menu()
    model_choice = get_model_choice()

    # Preprocess data
    df_processed = preprocess_data(df, settings)

    # Train test split
    train_size = int(len(df_processed) * 0.8)
    train_data = df_processed[:train_size]
    test_data = df_processed[train_size:]

    # Create datasets
    train = create_dataset(train_data, settings['target_column'], look_back=3)
    test = create_dataset(test_data, settings['target_column'], look_back=3)

    if model_choice == '1':
        print("Training LSTM model...")
        rmse, model, testX, testY = train_lstm(train, test, settings['target_column'], look_back=3)
        model_type = "lstm"  # Specify model type for permutation importance
    elif model_choice == '2':
        print("Training LightGBM model...")
        rmse, model = train_lightgbm(train, test, settings['target_column'])
        model_type = "lightgbm"
        testY = test[settings['target_column']].values  # Extract target for LightGBM
    else:
        print("Invalid choice. Please select a valid model.")
        return

    # Display RMSE
    print(f"Model RMSE: {rmse}")

    # Evaluate location importance
    location_columns = settings['location_columns']
    feature_indices = [train.columns.get_loc(col) for col in location_columns if col in train.columns]
    
    # Get the model input format (3D for LSTM, 2D for LightGBM)
    if model_type == "lstm":
        X_test_model = testX  # Reshaped 3D array for LSTM
    else:
        X_test_model = test.drop(settings['target_column'], axis=1)  # 2D array for LightGBM
    
    location_importance = permutation_importance(model, X_test_model, testY, feature_indices, look_back=3, model_type=model_type)

if __name__ == "__main__":
    main()
