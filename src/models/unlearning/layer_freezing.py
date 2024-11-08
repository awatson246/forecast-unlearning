from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

def apply_layer_freezing(model, num_layers_to_freeze, X, y, epochs=5):
    """Freeze specific layers in the model and fine-tune on the remaining data."""
    for layer in model.layers[:num_layers_to_freeze]:
        layer.trainable = False
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs)
    return model
