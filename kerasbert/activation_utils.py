import tensorflow as tf
import numpy as np


def gelu(x):
    """Gaussian Error Linear Unit

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation

    Returns:
        `x` with the GELU activation applied
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def get_activation(identifier):
    if callable(identifier):
        return identifier
    elif (isinstance(identifier, str) and
          identifier == 'gelu'):
        return gelu
    return tf.keras.activations.get(identifier)


def serialize_activation(activation):
    if (hasattr(activation, '__name__') and
            activation.__name__ == 'gelu'):
        return 'gelu'
    return tf.keras.activations.serialize(activation)
