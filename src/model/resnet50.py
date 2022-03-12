from typing import Tuple
import tensorflow as tf
from keras.applications.resnet import ResNet50
from keras.models import Sequential
import keras
from model.base import BaseModel
import numpy as np

tf.random.set_seed(42)


class ResNet50Model(BaseModel):
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

    def build_model(self, metrics=[tf.keras.metrics.AUC()]):
        # TODO : consider to take this function to constructor instead
        self.model = Sequential()

        self.model.add(
            ResNet50(
                include_top=self.include_resnet_top,
                weights=self.initial_weights,
                input_shape=self.input_shape,
            )
        )

        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(8, activation="relu"))
        self.model.add(keras.layers.Dense(1, activation="sigmoid"))

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=metrics,
        )

    def fit(
        self,
        datas: np.ndarray,
        labels: np.ndarray,
        validation_datas: Tuple,
        epochs: int = 100,
        batch_size: int = None,
    ):
        return self.model.fit(
            x=datas,
            y=labels,
            validation_data=validation_datas,
            epochs=epochs,
            batch_size=batch_size,
        )
