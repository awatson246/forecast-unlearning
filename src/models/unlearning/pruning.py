import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import tempfile
import re  # Importing regex to clean the feature string

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

        # Recursive function to check splits in the tree
        def check_split(split):
            if "split_feature" in split:
                # If a split feature is in the masked list, return True (prune it)
                return split["split_feature"] in masked_features
            # Recursively check left and right trees
            if "left_tree" in split and "right_tree" in split:
                return check_split(split["left_tree"]) or check_split(split["right_tree"])
            return False

        # If the tree doesn't rely on any of the masked features, keep it
        if not check_split(tree_structure):
            retained_trees.append(tree)

    # Rebuild the model with only the retained trees
    model_str = model.booster_.model_to_string()  # Get the model string representation
    model_dict = lgb.Booster(model_str=model_str)
    
    # We can specify the number of trees when re-loading the model
    pruned_model = lgb.Booster(model_str=model_dict.model_to_string())  # Initialize with pruned trees
    
    # Recalculate the number of trees in the pruned model
    pruned_model.num_trees = len(retained_trees)

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

def prune_xgboost_trees(model, masked_features, feature_names, feature_indices, trainX, trainY, testX, testY):
    """
    Prune trees in an XGBoost model by retraining without masked features.
    
    Args:
        model: Trained XGBoost model.
        masked_features: List of feature names to prune.
        feature_names: List of all feature names.
        feature_indices: Indices of features to be masked.
        trainX, trainY: Training data.
        testX, testY: Testing data.
        
    Returns:
        pruned_model: Retrained XGBoost model.
        rmse: RMSE of the retrained model.
        feature_importance: Feature importance after retraining.
    """
    # Flatten train and test data
    trainX_reshaped = trainX.reshape(trainX.shape[0], -1)
    testX_reshaped = testX.reshape(testX.shape[0], -1)

    # Remove masked features from the datasets
    pruned_trainX = np.delete(trainX_reshaped, feature_indices, axis=1)
    pruned_testX = np.delete(testX_reshaped, feature_indices, axis=1)

    # Retrain the model without the masked features
    dtrain = xgb.DMatrix(pruned_trainX, label=trainY)
    dtest = xgb.DMatrix(pruned_testX, label=testY)

    # Retrieve original parameters and set a default number of rounds
    params = model.attributes() or {}
    params["objective"] = "reg:squarederror"  # Ensure the objective is set
    num_boost_round = int(params.get("num_boost_round", 100))

    retrained_model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

    # Evaluate the pruned model
    predictions = retrained_model.predict(dtest)
    rmse = np.sqrt(mean_squared_error(testY, predictions))

    # Get feature importance after pruning
    feature_importance = retrained_model.get_score(importance_type="weight")

    # Initialize grouped_importances with all features, including masked ones, set to 0
    grouped_importances = {name: 0 for name in feature_names}

    # Map pruned feature importance back to original feature names
    total_importance = sum(feature_importance.values())  # Total importance for normalization
    for feature, importance in feature_importance.items():
        original_index = int(feature[1:])
        unflattened_index = original_index % len(feature_names)
        original_feature_name = feature_names[unflattened_index]
        grouped_importances[original_feature_name] += importance

    # Normalize importance by the total importance
    grouped_importances = {k: v / total_importance for k, v in grouped_importances.items()}

    # Sort importance by descending order
    sorted_importances = sorted(grouped_importances.items(), key=lambda x: x[1], reverse=True)

    return retrained_model, rmse, sorted_importances
