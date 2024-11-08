from src.utils.menu import display_menu, get_user_choice, display_model_menu, get_model_choice, display_unlearning_menu, get_unlearning_choice
from src.data_loading import load_data
from src.data_preprocessing import DataPreprocessor
from src.models.lstm_model import train_lstm
from src.models.lightgbm_model import train_lightgbm
from src.utils.evaluation import permutation_importance, evaluate_unlearning
from src.utils.loading_animation import loading_animation
from src.models.unlearning import feature_masking, layer_freezing, knowledge_distillation

import threading

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
        print("Sorry! This feature isn't ready for the world yet...")
    else:
        print("Invalid choice. Please select a valid model.")
        return

    # Display RMSE
    print(f"Model RMSE: {rmse}")

    # Get feature importance and identify the most important feature for unlearning
    sorted_importance, most_important_feature = permutation_importance(
        model, 
        testX, 
        testY, 
        train_data.columns,
        look_back, 
        model_type
    )

    # Select and set the unlearning model
    display_unlearning_menu()
    unlearning_choice = get_unlearning_choice()

    # Prepare the feature_index for feature masking or layer freezing, if needed
    feature_names = df_processed.columns.tolist()
    feature_index = feature_names.index(most_important_feature)

    if unlearning_choice == '1':
        # Feature masking: apply to data and keep the original model
        unlearned_X = feature_masking.apply_feature_masking(model, testX, feature_index)

        # Print the shape immediately after calling apply_feature_masking
        print(f"Shape of unlearned_X after feature masking: {unlearned_X.shape}")

        # Run prediction on the masked data
        y_pred_unlearned = model.predict(unlearned_X)

        # Evaluate unlearning by calculating RMSE for original vs. unlearned predictions
        initial_rmse, unlearned_rmse = evaluate_unlearning(model, testX, testY, unlearned_X, model_type)


    elif unlearning_choice == '2':
        # Layer Freezing: Here you may not need feature_index, so we pass only the model and X
        #unlearned_model = lambda model, X, *args: layer_freezing.apply_layer_freezing(model, X, *args)
        #initial_rmse, unlearned_rmse = evaluate_unlearning(model, testX, testY, unlearned_model)
        print("Sorry! This feature isn't ready for the world yet...")

    elif unlearning_choice == '3':
        # Knowledge Distillation: Again, pass only model and X (depending on your function signature)
        #unlearned_model = lambda model, X, *args: knowledge_distillation.knowledge_distill(model, X, *args)
        #initial_rmse, unlearned_rmse = evaluate_unlearning(model, testX, testY, unlearned_model)
        print("Sorry! This feature isn't ready for the world yet...")

    else:
        print("Invalid choice. Please select a valid model.")
        return

    print(f"Full model RMSE: {initial_rmse}")
    print(f"Unlearning model RMSE: {unlearned_rmse}")
    print("No errors this time!")


if __name__ == "__main__":
    main()