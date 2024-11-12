import numpy as np
from tensorflow.python.keras.models import clone_model

def knowledge_distill(original_model, X, y, temperature=2.0):
    """Create a new model that mimics the behavior of the original model using softened probabilities."""
    # Clone the original model architecture
    distilled_model = clone_model(original_model)
    distilled_model.compile(optimizer='adam', loss='mse')
    
    # Generate "soft targets"
    y_soft = original_model.predict(X) / temperature
    distilled_model.fit(X, y_soft, epochs=10)
    return distilled_model
