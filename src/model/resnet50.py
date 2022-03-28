"""
ResNet50Model class.
"""
from typing import Tuple
import tensorflow as tf
from src.model.base import BaseModel


class ResNet50Model(BaseModel):
    """
    Main ResNet50 model for COVID-19 classification
    """

    def __init__(
        self,
        input_shape: Tuple,
        include_resnet_top: bool = False,
        initial_weights=None,
    ) -> None:
        # Override the BaseModel constructor
        super().__init__(input_shape=input_shape, initial_weights=initial_weights)

        # Construct
        self.include_resnet_top = include_resnet_top
        self.model_type = "resnet50"

    def build_model(self, metrics=None, n_classes: int = 2):
        # TODO : Should construct resnet from scratch.
        # Source: https://www.kaggle.com/code/sandy1112/create-and-train-resnet50-from-scratch
        if metrics is None:
            metrics = [tf.keras.metrics.AUC()]

        self.model = tf.keras.Sequential()

        # ResNet block
        self.model.add(
            tf.keras.applications.resnet50.ResNet50(
                include_top=self.include_resnet_top,
                weights=self.initial_weights,
                input_shape=self.input_shape,
            )
        )

        # The top layer of ResNet
        self.model.add(tf.keras.layers.AveragePooling2D())
        self.model.add(tf.keras.layers.Flatten())

        # The fully connected layers
        self.model.add(tf.keras.layers.Dense(512, activation="relu"))
        self.model.add(tf.keras.layers.Dense(32, activation="relu"))
        self.model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=metrics,
        )

        return self.model

    # TODO : define identity block, conv block, and then integrate to ResNet
    def identity_block(self):
        ...
