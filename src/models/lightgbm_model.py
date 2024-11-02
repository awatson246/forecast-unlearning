import lightgbm as lgb
import pandas as pd
from sklearn.metrics import root_mean_squared_error

def train_lightgbm(train, test, target_column):
    """
    Train and evaluate a LightGBM model with early stopping.
    
    Args:
        train: Training DataFrame.
        test: Testing DataFrame.
        target_column: The column to predict.

    Returns:
        rmse: Root Mean Squared Error of the model on the test set.
        model: Trained LightGBM model.
    """
    # Prepare the train and test data
    X_train = train.drop(target_column, axis=1)
    y_train = train[target_column]
    X_test = test.drop(target_column, axis=1)
    y_test = test[target_column]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Define model parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.9
    }

    # Train the LightGBM model with early stopping using a callback
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(10)  # Log every 10 rounds for verbosity
        ]
    )

    # Make predictions and calculate RMSE
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"LightGBM Model RMSE: {rmse}")

    return rmse, model
