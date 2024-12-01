def get_parameters(model_type, dataset_id):
    """
    Returns dataset-specific parameters.
    """
    if model_type == "xgboost":
            params = {
                1: {},  # Dataset 1
                2: {"max_depth": 50, "eta": 0.005, "gamma": 0, "colsample_bytree": 0.8, "lambda": 0.5, "alpha": 0.01, "num_boost_round": 500, "early_stopping_rounds": 50},  # Dataset 2
                3: {} # Dataset 3
            }
    # Default parameters if dataset_id not in params
    return params.get(dataset_id, {"learning_rate": 0.001, "batch_size": 32, "epochs": 50})
