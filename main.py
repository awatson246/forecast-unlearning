import pandas as pd
from src.utils.menu import display_menu, get_user_choice, display_model_menu, get_model_choice
from src.data_loading import load_data
from src.data_preprocessing import DataPreprocessor
from src.models.lightgbm_model import train_lightgbm
from src.models.xgboost_model import train_xgboost
from src.models.catboost_model import train_catboost
from src.utils.loading_animation import loading_animation
from src.models.unlearning import full_retraining, pruning
import threading
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main():
    # Prepare outputs for metrics
    metrics_summary = {
        "Stage": ["Original", "Fully Retrained", "Basic Pruning", "Pruning Repace w/ Avg", "Pruning w/ Weight Adjustment"],
        "RMSE": [None, None, None, None, None],
    }
    feature_importance_summary = {
        "Stage": ["Original", "Fully Retrained", "Basic Pruning", "Pruning Repace w/ Avg", "Pruning w/ Weight Adjustment"], 
        "Feature Importance": [None, None, None, None, None]
    }

    # Display menu and get user choices
    display_menu()
    dataset_choice = get_user_choice()
    df, settings = load_data(dataset_choice)
    
    display_model_menu()
    model_choice = get_model_choice()

    bun = input("()()...? (y/n)")

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

    feature_names = df.columns

    # Specify the columns to unlearn (e.g., "City", "Country")
    location_columns = settings.get("location_columns", [])

    # Convert column names to indices
    masked_indices = [df.columns.get_loc(col) for col in location_columns]

    if model_choice == '1':
        print("Training LightGBM model...")
        if bun == 'y': 
            # Start bunny animation in a separate thread
            bunny_thread = threading.Thread(target=loading_animation)
            bunny_thread.daemon = True 
            bunny_thread.start()
            
        rmse, model, sorted_importances = train_lightgbm(trainX, trainY, testX, testY, feature_names)
        model_type = "lightgbm"
    elif model_choice == '2':
        print("Training XGBoost model...")
        if bun == 'y': 
            # Start bunny animation in a separate thread
            bunny_thread = threading.Thread(target=loading_animation)
            bunny_thread.start()
        
        rmse, model, sorted_importances = train_xgboost(trainX, trainY, testX, testY, feature_names)
        model_type = "xgboost"
    elif model_choice == '3':
        print("Training CatBoost model...")
        if bun == 'y': 
            # Start bunny animation in a separate thread
            bunny_thread = threading.Thread(target=loading_animation)
            bunny_thread.start()
        
        rmse, model, sorted_importances = train_catboost(trainX, trainY, testX, testY, feature_names)
        model_type = "catboost"
    else:
        print("Invalid choice. Please select a valid model.")
        return

    # Capture RMSE for original model
    metrics_summary["RMSE"][0] = rmse    
    feature_importance_summary["Feature Importance"][0] = sorted_importances

    # Full Retraining
    print("Retraining the model...")
    retrained_rmse, retrained_model, retrained_sorted_importances = full_retraining.full_retraining(model, trainX, trainY, testX, testY, masked_indices, model_type, feature_names)
    metrics_summary["RMSE"][1] = retrained_rmse
    feature_importance_summary["Feature Importance"][1] = retrained_sorted_importances

    # Pruning
    print("Trimming some trees (don't worry, they'll grow back)...")
    pruning_types = ["basic", "average", "weighted"]

    summary_iter = 2

    for pruning_type in pruning_types: 
        if model_type == "lightgbm":
            print(f"Applying pruning strategy: {pruning_type}")
            pruned_model, pruned_rmse, pruned_sorted_importance = pruning.prune_lightgbm_trees(model, masked_indices, feature_names, testX, testY, pruning_type)
        elif model_type == "xgboost":
            # pruned_model, pruned_rmse, pruned_sorted_importance = pruning.prune_xgboost_trees(model, location_columns, feature_names, trainX, trainY, testX, testY)
            pruned_rmse = None
            pruned_sorted_importance = None
        elif model_type == "catboost":
            print("Pruning not avaliable for catboost ^^")
            pruned_rmse = None
            pruned_sorted_importance = None
        metrics_summary["RMSE"][summary_iter] = pruned_rmse
        feature_importance_summary["Feature Importance"][summary_iter] = pruned_sorted_importance
        summary_iter += 1

    # Print metrics summary
    print(f"\nMasked columns: {location_columns}")
    print("\nMetrics Summary:")
    metrics_df = pd.DataFrame(metrics_summary)
    print(metrics_df.to_string(index=False))

    # Create a dictionary to store the feature importance data
    feature_importance_data = {}

    # Loop through stages and their respective importance data
    for stage, importance in zip(feature_importance_summary["Stage"], feature_importance_summary["Feature Importance"]):
        if isinstance(importance, dict):
            # If it's a dictionary, store the feature importance values
            for feature, score in importance.items():
                if feature not in feature_importance_data:
                    feature_importance_data[feature] = {}
                feature_importance_data[feature][stage] = score
        elif isinstance(importance, list):
            for item in importance:
                if isinstance(item, tuple):
                    feature, score = item
                    if feature not in feature_importance_data:
                        feature_importance_data[feature] = {}
                    feature_importance_data[feature][stage] = score
        elif importance is None:
            print(f"No feature importance data available for stage: {stage}")

    # Create a pandas DataFrame from the feature importance data
    importance_df = pd.DataFrame.from_dict(feature_importance_data, orient='index')

    # Print the table
    print("\nFeature Importance Table:")
    print(importance_df.to_string())

    print("No errors this time!")

if __name__ == "__main__":
    main()