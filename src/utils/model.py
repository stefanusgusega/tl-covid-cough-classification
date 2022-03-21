"""
Model util functions that should not be in one class.
"""

import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from src.model.resnet50 import ResNet50Model
from src.utils.randomize import set_random_seed


def hyperparameter_tune_resnet_model(
    initial_model: ResNet50Model, first_dense_units, second_dense_units, learning_rates
) -> tf.keras.Model:
    """
    Hyperparameters should contain these:
    * first_dense_units
    * second_dense_units
    * learning_rates
    * epochs
    * batch_size
    """
    set_random_seed()

    # Build model
    model = tf.keras.Sequential()

    model.add(
        tf.keras.applications.resnet50.ResNet50(
            include_top=initial_model.include_resnet_top,
            weights=initial_model.initial_weights,
            input_shape=initial_model.input_shape,
        )
    )

    model.add(tf.keras.layers.AveragePooling2D())
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(first_dense_units, activation="relu"))
    model.add(tf.keras.layers.Dense(second_dense_units, activation="relu"))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rates),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC()],
    )
    return model


def evaluate_model(model, x, y):
    # If instance of original Keras Model
    if isinstance(model, tf.keras.Model):
        return model.evaluate(x, y)[1]

    # If instance of Keras Classifier wrapper
    if isinstance(model, KerasClassifier):
        return model.score(x, y)

    # If none of the above conditions, raise Exception
    raise Exception(
        "The model should be an instance of either this two: tensorflow.keras.Model"
        "or scikeras.wrappers.KerasClassifier"
    )
