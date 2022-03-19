from typing import Tuple
import tensorflow as tf
from model.base import BaseModel

HYPERPARAMETER_KEYS = [
    "first_dense_units",
    "second_dense_units",
    "learning_rate",
    "batch_size",
    "epochs",
]


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
        super(ResNet50Model, self).__init__(
            input_shape=input_shape, initial_weights=initial_weights
        )

        # Construct
        self.include_resnet_top = include_resnet_top
        self.model_type = "resnet50"

    def build_model(self, metrics=[tf.keras.metrics.AUC()]):
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
        self.model.add(tf.keras.layers.Dense(2, activation="softmax"))

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=metrics,
        )

        return self.model

    def hyperparameter_tune_model(
        self, first_dense_units, second_dense_units, learning_rate
    ) -> tf.keras.Model:
        """
        Hyperparameters should contain these:
        * first_dense_units
        * second_dense_units
        * learning_rate
        * epochs
        * batch_size
        """
        super().hyperparameter_tune_model()

        # Check if hyperparameters dictionary is complete
        # if list(hyperparameters.keys()) == HYPERPARAMETER_KEYS:
        #     raise Exception(
        #         f"The hyperparameters is not complete. Should state this keys: {set(HYPERPARAMETER_KEYS) - set(hyperparameters.keys())}"
        #     )

        # Build model
        self.model = tf.keras.Sequential()

        self.model.add(
            tf.keras.applications.resnet50.ResNet50(
                include_top=self.include_resnet_top,
                weights=self.initial_weights,
                input_shape=self.input_shape,
            )
        )

        self.model.add(tf.keras.layers.AveragePooling2D())
        self.model.add(tf.keras.layers.Flatten())

        self.model.add(tf.keras.layers.Dense(first_dense_units, activation="relu"))
        self.model.add(tf.keras.layers.Dense(second_dense_units, activation="relu"))
        self.model.add(tf.keras.layers.Dense(2, activation="softmax"))

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC()],
        )

        return self.model
