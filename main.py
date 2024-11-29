import pandas as pd
import numpy as np
from src.utils.menu import display_menu, get_user_choice, display_model_menu, get_model_choice
from src.data_loading import load_data
from src.data_preprocessing import DataPreprocessor
from src.models.lstm_model import train_lstm
from src.models.lightgbm_model import train_lightgbm
from src.models.xgboost_model import train_xgboost
from src.utils.evaluation import permutation_importance, feature_sensitivity_analysis
from src.utils.loading_animation import loading_animation
from src.models.unlearning import feature_masking, fine_tuning, full_retraining
import threading
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error


def main():

    # Prepare outputs for metrics
    metrics_summary = {
        "Stage": ["Original", "Feature Masking", "Fine-Tuned", "Fully Retrained"],
        "RMSE": [None, None, None, None],

    }
    sensitivity_summary = {
        "Stage": ["Original", "Feature Masking", "Fine-Tuned", "Fully Retrained"],
        "Sensitivity": [None, None, None, None],
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
    

    # Capture  RMSE for original model
    metrics_summary["RMSE"][0] = rmse

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

    print("Baseline Sensitivity Analysis: ")
    baseline_sensitivity, sensitivity_scores = feature_sensitivity_analysis(model, testX, testY, feature_index, model_type, dataset_choice)
    sensitivity_summary["Sensitivity"][0] = baseline_sensitivity
    

    # Apply Feature Masking
    print(f"\nMasking {most_important_feature}...")
    feature_masking_rmse, feature_masking_model = feature_masking.apply_feature_masking(model, testX, feature_index, model_type, testY)
    metrics_summary["RMSE"][1] = feature_masking_rmse

    print("Masked Sensitivity Analysis: ")
    masked_sensitivity, sensitivity_scores = feature_sensitivity_analysis(feature_masking_model, testX, testY, feature_index, model_type, dataset_choice)
    sensitivity_summary["Sensitivity"][1] = masked_sensitivity
    

    # Apply Fine-Tuning
    print("Fine Tuning...")
    fine_tuned_rmse, fine_tuning_model = fine_tuning.fine_tune_model(model, trainX, trainY, testX, testY, feature_index, model_type)
    metrics_summary["RMSE"][2] = fine_tuned_rmse

    print("Fine Tuned Sensitivity Analysis: ")
    fineTune_sensitivity, sensitivity_scores = feature_sensitivity_analysis(fine_tuning_model, testX, testY, feature_index, model_type, dataset_choice)
    sensitivity_summary["Sensitivity"][2] = fineTune_sensitivity
    
    # Full Retraining
    print("Retraining the model...")
    retrained_rmse , retrained_model = full_retraining.full_retraining(model, trainX, trainY, testX, testY, feature_index, model_type)
    metrics_summary["RMSE"][3] = retrained_rmse

    print("Retrained Model Sensitivity Analysis: ")
    retrained_sensitivity, sensitivity_scores = feature_sensitivity_analysis(retrained_model, testX, testY, feature_index, model_type, dataset_choice)
    sensitivity_summary["Sensitivity"][3] = retrained_sensitivity

    print(f"\nMost important feature for unlearning: {most_important_feature}")


    # Print metrics summary
    print("\nMetrics Summary:")
    metrics_df = pd.DataFrame(metrics_summary)
    print(metrics_df.to_string(index=False))

    print("Sensitivity Summary:")
    sensitivity_df = pd.DataFrame(sensitivity_summary)
    print(sensitivity_df.to_string(index=False))

    print("No errors this time!")


if __name__ == "__main__":
    main()