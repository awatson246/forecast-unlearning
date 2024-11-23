import pandas as pd
import numpy as np
from src.utils.menu import display_menu, get_user_choice, display_model_menu, get_model_choice, display_unlearning_menu, get_unlearning_choice
from src.data_loading import load_data
from src.data_preprocessing import DataPreprocessor
from src.models.lstm_model import train_lstm
from src.models.lightgbm_model import train_lightgbm
from src.models.xgboost_model import train_xgboost
from src.utils.evaluation import permutation_importance, evaluate_unlearning
from src.utils.loading_animation import loading_animation
from src.models.unlearning import feature_masking, fine_tuning, full_retraining
import threading
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error


def main():

    # Prepare outputs for metrics
    metrics_summary = {
        "Original RMSE": None,
        "Feature Masking RMSE": None,
        "Fine-Tuned RMSE": None,
        "Fully Retrained RMSE": None
    }

    # Display menu and get user choices
    display_menu()
    dataset_choice = get_user_choice()
    df, settings = load_data(dataset_choice)
    
    display_model_menu()
    model_choice = get_model_choice()

    # Initialize the preprocessor
    preprocessor = DataPreprocessor(settings)

    # Preprocess data
    df_processed = preprocessor.preprocess_data(df)

    # Train-test split
    train_size = int(len(df_processed) * 0.8)
    train_data, test_data = df_processed[:train_size], df_processed[train_size:]

    # Prepare datasets with a look-back window
    look_back = 10
    trainX, trainY = preprocessor.create_dataset(train_data, settings['target_column'], look_back)
    testX, testY = preprocessor.create_dataset(test_data, settings['target_column'], look_back)

    bun = input("()()...? (y/n)")

    if model_choice == '1':
        print("Training LSTM model...")
        if bun == 'y': 
            # Start bunny animation in a separate thread
            bunny_thread = threading.Thread(target=loading_animation)
            bunny_thread.daemon = True 
            bunny_thread.start()
            
        input_shape = trainX.shape[1:]
        rmse, model, _, _ = train_lstm((trainX, trainY), (testX, testY), input_shape)
        model_type = "lstm"
    elif model_choice == '2':
        print("Training LightGBM model...")
        if bun == 'y': 
            # Start bunny animation in a separate thread
            bunny_thread = threading.Thread(target=loading_animation)
            bunny_thread.start()
        
        # Flatten the input for LightGBM (2D shape)
        trainX_flat = trainX.reshape(trainX.shape[0], -1)
        testX_flat = testX.reshape(testX.shape[0], -1)
        rmse, model = train_lightgbm(trainX_flat, trainY, testX_flat, testY)
        model_type = "lightgbm"
    elif model_choice == '3':
        # Flatten the input for LightGBM (2D shape)
        trainX_flat = trainX.reshape(trainX.shape[0], -1)
        testX_flat = testX.reshape(testX.shape[0], -1)
        print("Training XGboost model...")
        rmse, model = train_xgboost((trainX_flat, trainY), (testX_flat, testY))
        model_type = "xgboost"
    else:
        print("Invalid choice. Please select a valid model.")
        return

    # Display RMSE
    print(f"Model RMSE: {rmse}")
    metrics_summary["Original RMSE"] = rmse

    # Get feature importance for full model
    sorted_importance, most_important_feature = permutation_importance(
        model,
        testX,  # Now it's a 2D DataFrame with the correct shape and column names
        testY,
        df.columns,
        look_back=3, 
        model_type=model_type
    )

    # Sequential Unlearning
    feature_index = df.columns.get_loc(most_important_feature)
    
    # Apply Feature Masking
    print(f"\nMasking {most_important_feature}...")
    feature_masking_rmse = feature_masking.apply_feature_masking(model, testX, feature_index, model_type, testY)
    metrics_summary["Feature Masking RMSE"] = feature_masking_rmse
    
    # Apply Fine-Tuning
    print("Fine Tuning...")
    fine_tuned_rmse = fine_tuning.fine_tune_model(model, trainX, trainY, testX, testY, feature_index, model_type)
    metrics_summary["Fine-Tuned RMSE"] = fine_tuned_rmse
    
    # Full Retraining
    print("Retraining the model...")
    retrained_rmse = full_retraining.full_retraining(model, trainX, trainY, testX, testY, feature_index, model_type)
    metrics_summary["Fully Retrained RMSE"] = retrained_rmse

    # Pretty print metrics summary
    print(f"\nUnlearned Feature: {most_important_feature}")
    for method, rmse in metrics_summary.items():
        print(f"{method}: {rmse:.4f}")

    print("No errors this time!")


if __name__ == "__main__":
    main()