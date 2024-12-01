def compute_feature_importance(model, X, feature_names, model_type):
    """
    Compute feature importance based on the model type.
    
    Args:
        model: The trained model.
        X: Input data (not used for built-in feature importance but kept for consistency).
        feature_names: List of feature names.
        model_type: Type of model ('lightgbm', 'xgboost', 'catboost').
        
    Returns:
        importance: Dictionary mapping feature names to importance scores.
    """
    if model_type == "lightgbm":
        # Get feature importances from LightGBM
        importance_values = model.booster_.feature_importance(importance_type="gain")
    elif model_type == "xgboost":
        # Get feature importances from XGBoost
        raw_importance = model.get_score(importance_type="gain")
        importance_values = [raw_importance.get(f"f{i}", 0) for i in range(len(feature_names))]
    elif model_type == "catboost":
        # Get feature importances from CatBoost
        importance_values = model.get_feature_importance()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Group importance values if using time-series data (normalize by time steps)
    grouped_importances = {name: 0 for name in feature_names}
    for i, importance in enumerate(importance_values):
        original_feature_idx = i % len(feature_names)
        original_feature_name = feature_names[original_feature_idx]
        grouped_importances[original_feature_name] += importance

    # Normalize by the number of time steps (if applicable)
    look_back = 10
    grouped_importances = {k: v / look_back for k, v in grouped_importances.items()}

    # Sort the features based on their importance scores in descending order
    sorted_importances = sorted(grouped_importances.items(), key=lambda x: x[1], reverse=True)

    return sorted_importances
