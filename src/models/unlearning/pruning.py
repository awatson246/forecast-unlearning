import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error
import xgboost as xgb

def prune_lightgbm_trees(model, masked_features, feature_names, trainX, trainY, testX, testY):
    """
    Prune trees in a LightGBM model based on reliance on specified masked features.
    
    Args:
        model: Trained LightGBM model.
        masked_features: List of feature names to prune.
        trainX, trainY: Training data.
        testX, testY: Testing data.
        
    Returns:
        pruned_model: Pruned LightGBM model.
        rmse: RMSE of the pruned model.
        feature_importance: Feature importance after pruning.
    """
    # Dump the model structure
    tree_info = model.booster_.dump_model()["tree_info"]

    # Identify trees to retain (exclude trees with splits on masked features)
    retained_trees = []
    for tree in tree_info:
        tree_structure = tree["tree_structure"]

        def check_split(split, masked_features):
            """Recursively checks if a tree split is based on a masked feature."""
            if "split_feature" in split:
                return split["split_feature"] in masked_features
            if "left_tree" in split and "right_tree" in split:
                return check_split(split["left_tree"], masked_features) or check_split(split["right_tree"], masked_features)
            return False

        # If the tree doesn't rely on any of the masked features, keep it
        if not check_split(tree_structure, masked_features):
            retained_trees.append(tree)

    # Rebuild the model with only the retained trees
    model_str = model.booster_.model_to_string()  # Get the model string representation
    model_dict = lgb.Booster(model_str=model_str)
    
    # We can specify the number of trees when re-loading the model
    pruned_model = lgb.Booster(model_str=model_dict.model_to_string())  # Initialize with pruned trees
    
    # Recalculate the number of trees in the pruned model
    pruned_model.num_trees = len(retained_trees)
    print(f"Number of retained trees: {len(retained_trees)}")

    # Flatten the test data for prediction (ensure it is 2D)
    testX_reshaped = testX.reshape(testX.shape[0], -1)  # Flatten to 2D
    
    # Evaluate the pruned model
    predictions = pruned_model.predict(testX_reshaped)
    rmse = np.sqrt(mean_squared_error(testY, predictions))

    # Get feature importance after pruning
    feature_importance = pruned_model.feature_importance(importance_type="gain")

    # Map the flattened importances back to the original feature names
    grouped_importances = {name: 0 for name in feature_names}
    for i, importance in enumerate(feature_importance):
        original_feature_idx = i % len(feature_names)
        original_feature_name = feature_names[original_feature_idx]
        grouped_importances[original_feature_name] += importance

    # Normalize by the number of time steps (adjust as needed)
    look_back = 10  # Adjust this if the look-back period changes
    grouped_importances = {k: v / look_back for k, v in grouped_importances.items()}

    # Sort by importance (descending order)
    sorted_importances = sorted(grouped_importances.items(), key=lambda x: x[1], reverse=True)

    return pruned_model, rmse, sorted_importances

def prune_xgboost_trees(model, masked_features, feature_names, trainX, trainY, testX, testY):
    """
    Prune trees in an XGBoost model based on reliance on specified masked features.
    
    Args:
        model: Trained XGBoost model.
        masked_features: List of feature names to prune.
        feature_names: List of original feature names.
        trainX, trainY: Training data.
        testX, testY: Testing data.
        
    Returns:
        pruned_model: Pruned XGBoost model.
        rmse: RMSE of the pruned model.
        feature_importance: Feature importance after pruning.
    """
    # Reshape trainX to 2D arrays (flatten time steps and features)
    trainX = trainX.reshape(-1, trainX.shape[-1])  # (20831 * 10, 9)
    testX = testX.reshape(-1, testX.shape[-1])  
    trainY = np.tile(trainY, trainX.shape[0] // len(trainY))  # Repeat the labels for each time step
    testY = np.tile(testY, testX.shape[0] // len(testY))  # Repeat the labels for each time step

    # Create DMatrix objects for training and testing data
    dtrain = xgb.DMatrix(trainX, label=trainY)
    dtest = xgb.DMatrix(testX, label=testY)
    
    # Dump the model structure (retrieve all trees and stats)
    tree_info = model.get_dump(with_stats=True)

    # Create a feature-to-index mapping
    feature_to_index = {feature: i for i, feature in enumerate(feature_names)}

    # Identify trees to retain (exclude trees with splits on masked features)
    retained_trees = []
    for tree in tree_info:
        prune_tree = False
        for line in tree.splitlines():
            # Look for masked feature names in the tree dump (e.g., f0, f1, f2...)
            for feature in masked_features:
                feature_index = feature_to_index.get(feature, None)
                if feature_index is not None and f"f{feature_index}" in line:
                    prune_tree = True
                    break
            if prune_tree:
                break
        if not prune_tree:
            retained_trees.append(tree)

    print(f"Number of retained trees: {len(retained_trees)}")
    
    # Extract parameters directly from the model's booster and prepare for retraining
    params = model.attributes()  # This will return a dictionary of parameters

    # Create a new model using the retained trees
    pruned_model = xgb.train(
        params=params,  # Get the model's parameters
        dtrain=dtrain,  # Use the same training data
        num_boost_round=len(retained_trees),  # Set the number of trees to the length of retained trees
        evals=[(dtest, 'eval')],  # Set validation data
        early_stopping_rounds=None,  # No early stopping, we use only retained trees
        verbose_eval=False  # Suppress output during training
    )

    # Evaluate the pruned model on the test data
    predictions = pruned_model.predict(dtest)
    rmse = np.sqrt(mean_squared_error(testY, predictions))

    # Get feature importance after pruning
    feature_importance = pruned_model.get_score(importance_type="gain")
    
    # Map the importances back to the original feature names
    grouped_importances = {name: 0 for name in feature_names}
    for feature, importance in feature_importance.items():
        # Extract feature index from the dump (e.g., f0, f1, f2...)
        feature_index = int(feature[1:])  # This extracts the index number from f0, f1, ...
        
        # Ensure the index is within bounds of the feature_names list
        if feature_index < len(feature_names):
            original_feature_name = feature_names[feature_index]  # Get the original feature name
            grouped_importances[original_feature_name] += importance
        else:
            print(f"Warning: feature index {feature_index} out of range for feature names.")
    
    # Sort by importance (descending order)
    sorted_importances = sorted(grouped_importances.items(), key=lambda x: x[1], reverse=True)
    print(f"Sorted feature importances: {sorted_importances}")

    return pruned_model, rmse, sorted_importances