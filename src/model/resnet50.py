"""
ResNet50Model class.
"""
from typing import Tuple
import tensorflow as tf
from src.model.base import BaseModel
from src.model.builder import resnet50_block


class ResNet50Model(BaseModel):
    """
    Main ResNet50 model for COVID-19 classification
    """

    def __init__(
        self,
        input_shape: Tuple,
        initial_weights=None,
    ) -> None:
        # Override the BaseModel constructor
        super().__init__(input_shape=input_shape, initial_weights=initial_weights)

        # Construct
        self.model_type = "resnet50"

    def build_model(self, metrics=None, n_classes: int = 2):
        if metrics is None:
            metrics = [tf.keras.metrics.AUC(), tf.keras.metrics.Accuracy()]

        input_tensor = tf.keras.layers.Input(shape=self.input_shape)

        # ResNet50 block
        model = resnet50_block(input_tensor=input_tensor)

        # The top layer of ResNet
        model = tf.keras.layers.AveragePooling2D(name="avg_pool")(model)
        model = tf.keras.layers.Flatten()(model)

        # The fully connected layers
        model = tf.keras.layers.Dense(512, activation="relu")(model)
        model = tf.keras.layers.Dropout(rate=0.2)(model)
        model = tf.keras.layers.Dense(32, activation="relu")(model)
        model = tf.keras.layers.Dropout(rate=0.2)(model)
        model = tf.keras.layers.Dense(n_classes, activation="softmax")(model)

        model = tf.keras.models.Model(
            inputs=input_tensor, outputs=model, name="ResNet50"
        )

        model.compile(
            # optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=metrics,
        )

        # Save to attribute
        self.model = model

        # print(self.model.summary())

        return model
