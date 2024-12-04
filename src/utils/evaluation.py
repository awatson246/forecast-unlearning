def compute_feature_importance(model, X, feature_names, model_type, look_back=10):
    """
    Compute feature importance based on the model type and handle differences in outputs.
    
    Args:
        model: The trained model.
        X: Input data (optional, included for compatibility with permutation importance).
        feature_names: List of feature names.
        model_type: Type of model ('lightgbm', 'xgboost', 'catboost').
        look_back: Number of time steps for time-series data (used for grouping).
        
    Returns:
        sorted_importances: List of tuples (feature_name, importance) sorted by importance.
    """
    if model_type == "lightgbm":
        # Retrieve feature importance values for LightGBM
        importance_values = model.booster_.feature_importance(importance_type="gain")
    elif model_type == "xgboost":
        # Retrieve feature importance values for XGBoost
        raw_importance = model.get_score(importance_type="gain")
        # Map feature indices (f0, f1, ...) to a list
        max_index = len(feature_names) * look_back
        importance_values = [raw_importance.get(f"f{i}", 0) for i in range(max_index)]
    elif model_type == "catboost":
        # Retrieve feature importance values for CatBoost
        importance_values = model.get_feature_importance()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Initialize grouped feature importances
    grouped_importances = {name: 0 for name in feature_names}
    
    # Map importances to their original feature names, accounting for time steps
    for i, importance in enumerate(importance_values):
        original_feature_idx = i % len(feature_names)
        original_feature_name = feature_names[original_feature_idx]
        grouped_importances[original_feature_name] += importance

    # Normalize grouped importances by the number of time steps (look_back)
    #grouped_importances = {k: v / look_back for k, v in grouped_importances.items()}

    # Sort features by importance scores in descending order
    sorted_importances = sorted(grouped_importances.items(), key=lambda x: x[1], reverse=True)

    return sorted_importances
