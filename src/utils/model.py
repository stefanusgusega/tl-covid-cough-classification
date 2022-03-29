"""
Model util functions that should not be in one class.
"""
import os
import tensorflow as tf
from src.model.resnet50 import ResNet50Model
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

    # TODO: Soon will be deleted, because it will be included in ResNet builder from scratch
    model.add(tf.keras.layers.AveragePooling2D())
    model.add(tf.keras.layers.Flatten())

    # FC layers
    model.add(tf.keras.layers.Dense(first_dense_units, activation="relu"))
    model.add(tf.keras.layers.Dense(second_dense_units, activation="relu"))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))

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


def identity_block(input_tensor, middle_kernel_size, filters, stage, block):
    """
    Implementation of the identity block of ResNet

    Arguments:
    * input_tensor -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    * middle_kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    * filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    * stage -- integer, used to name the layers, depending on their position in the network
    * block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    * model -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # Define name basis
    conv_name_base = f"res{stage}{block}_branch"
    bn_name_base = f"bn{stage}{block}_branch"

    # First component of main path
    model = tf.keras.layers.Conv2D(
        filters=filters[0],
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        name=f"{conv_name_base}2c",
        kernel_initializer="glorot_uniform",
    )(input_tensor)
    model = tf.keras.layers.BatchNormalization(axis=3, name=f"{bn_name_base}2a")(model)
    model = tf.keras.layers.Activation("relu")(model)

    # Second component of main path
    model = tf.keras.layers.Conv2D(
        filters=filters[1],
        kernel_size=(middle_kernel_size, middle_kernel_size),
        strides=(1, 1),
        padding="same",
        name=f"{conv_name_base}2b",
        kernel_initializer="glorot_uniform",
    )(model)
    model = tf.keras.layers.BatchNormalization(axis=3, name=f"{bn_name_base}2b")(model)
    model = tf.keras.layers.Activation("relu")(model)

    # Third component of main path
    model = tf.keras.layers.Conv2D(
        filters=filters[2],
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        name=f"{conv_name_base}2c",
        kernel_initializer="glorot_uniform",
    )(model)
    model = tf.keras.layers.BatchNormalization(axis=3, name=f"{bn_name_base}2c")(model)

    # Final step: Add shortcut value, which is input tensor, to main path, and pass it through a RELU activation
    model = tf.keras.layers.Add()([model, input_tensor])
    model = tf.keras.layers.Activation("relu")(model)

    return model
