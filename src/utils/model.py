"""
Model util functions that should not be in one class.
"""
import math
import os
import tensorflow as tf
from src.model.resnet50 import ResNet50Model, resnet50_block
from src.utils.chore import generate_now_datetime
from src.utils.randomize import set_random_seed


def hyperparameter_tune_resnet_model(
    initial_model: ResNet50Model, first_dense_units, second_dense_units, learning_rates
) -> tf.keras.Model:
    """
    Hyperparameters should contain these:
    * first_dense_units
    * second_dense_units
    * learning_rates
    * batch_size
    """
    set_random_seed()

    # Build model
    input_tensor = tf.keras.layers.Input(shape=initial_model.input_shape)

    # ResNet50 block
    model = resnet50_block(input_tensor=input_tensor)

    model = tf.keras.layers.AveragePooling2D(name="avg_pool")(model)
    model = tf.keras.layers.Flatten()(model)

    # FC layers
    model = tf.keras.layers.Dense(first_dense_units, activation="relu")(model)
    model = tf.keras.layers.Dense(second_dense_units, activation="relu")(model)
    model = tf.keras.layers.Dense(2, activation="softmax")(model)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=model, name="hp_tuning")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rates),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC()],
    )
    return model


def generate_tensorboard_callback(log_dir: str):
    """
    Generate a TensorBoard callback for analysis and visualization needs.
    TensorBoard source : https://www.tensorflow.org/tensorboard/graphs
    """
    specified_log_dir = os.path.join(log_dir, generate_now_datetime())

    return tf.keras.callbacks.TensorBoard(log_dir=specified_log_dir)


def lr_step_decay(epoch, _):
    initial_learning_rate = 5e-3
    drop_rate = 0.5
    epochs_drop = 10.0

    return initial_learning_rate * math.pow(drop_rate, math.floor(epoch / epochs_drop))
