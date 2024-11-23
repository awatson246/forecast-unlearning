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

    sorted_importance, most_important_feature = permutation_importance(
        model,
        testX,  # Now it's a 2D DataFrame with the correct shape and column names
        testY,
        df.columns,
        look_back=3, 
        model_type=model_type
    )

    while True: 
        # Select and set the unlearning model
        display_unlearning_menu()
        unlearning_choice = get_unlearning_choice()

        if unlearning_choice == 4:
            break;

        # Prepare the feature_index for feature masking or layer freezing, if needed
        feature_names = df_processed.columns.tolist()
        feature_index = feature_names.index(most_important_feature)

        if unlearning_choice == '1':
            # Feature masking: apply to data and keep the original model
            unlearning_type = "feture_masking"
            unlearned_X = feature_masking.apply_feature_masking(model, testX, feature_index)

            if model_type == "lightgbm" or "xgboost":
                # Reshape unlearned_X to 2D for LightGBM
                unlearned_X = unlearned_X.reshape(unlearned_X.shape[0], -1)
                if model_type == "xgboost":
                    unlearned_X = xgb.DMatrix(data=unlearned_X)
            elif model_type == "lstm":
                # Keep unlearned_X in 3D for LSTM
                pass

            # Run prediction on the masked data
            y_pred_unlearned = model.predict(unlearned_X)

            # Evaluate unlearning by calculating RMSE for original vs. unlearned predictions
            initial_rmse, unlearned_rmse = evaluate_unlearning(model, testX, testY, unlearned_X, model_type, unlearning_type)


        elif unlearning_choice == '2':
            # Layer Freezing: Here you may not need feature_index, so we pass only the model and X
            unlearning_type = "layer_freezing"
            num_layers_to_freeze = 2  # or any number you choose
            #unlearned_model = layer_freezing.apply_layer_freezing(model, num_layers_to_freeze, trainX, trainY)
            unlearned_model = layer_freezing.apply_layer_freezing(model, num_layers_to_freeze, trainX, trainY)
            unlearned_preds = unlearned_model.predict(testX)
            unlearned_rmse = root_mean_squared_error(testY, unlearned_preds)
            print(f"Unlearned RMSE after layer freezing: {unlearned_rmse}")
            #initial_rmse, unlearned_rmse = evaluate_unlearning(model, testX, testY, unlearned_model, model_type, unlearning_type)

        elif unlearning_choice == '3':
            # Knowledge Distillation: Again, pass only model and X (depending on your function signature)
            unlearning_choice = "knowledge_dist"
            unlearned_model = lambda model, X, *args: knowledge_distillation.knowledge_distill(model, X, *args)

            initial_rmse, unlearned_rmse = evaluate_unlearning(model, testX, testY, unlearned_model, model_type, unlearning_type)

        else:
            print("Invalid choice. Please select a valid model.")
            return

        print(f"Full model RMSE: {initial_rmse}")
        print(f"Unlearning model RMSE: {unlearned_rmse}")
        print("No errors this time!")


if __name__ == "__main__":
    main()