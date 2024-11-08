import numpy as np

def apply_feature_masking(model, X, feature_index):
    """Apply feature masking by zeroing out the feature column in input data."""
    X_masked = X.copy()
    X_masked[:, feature_index] = 0
    return model.predict(X_masked)
