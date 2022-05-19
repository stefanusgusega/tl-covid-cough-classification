"""
ResNet50 Pretrain Model class
"""
from typing import Tuple
import tensorflow as tf
from src.model.base import BaseModel
from src.model.builder import resnet50_block


# ! In progress
class ResNet50PretrainModel(BaseModel):
    """
    Pretrain ResNet50 Model for cough classification
    """

    def __init__(
        self, input_shape: Tuple, include_resnet_top: bool = False, initial_weights=None
    ) -> None:
        # Override the BaseModel constructor
        super().__init__(input_shape, initial_weights)

        # Construct
        self.include_resnet_top = include_resnet_top
        self.model_type = "pretrain-resnet50"

    def build_model(self, metrics=None, n_classes: int = 3):
        if metrics is None:
            metrics = ["accuracy"]

        input_tensor = tf.keras.layers.Input(shape=self.input_shape)

        # ResNet50 block
        model = resnet50_block(input_tensor=input_tensor)

        # The top layer of ResNet
        model = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model)
        # model = tf.keras.layers.Dropout(rate=0.1)(model)
        # model = tf.keras.layers.Flatten()(model)

        # The fully connected layers
        model = tf.keras.layers.Dense(512, activation="relu")(model)
        # model = tf.keras.layers.Dropout(rate=0.2)(model)
        # model = tf.keras.layers.Activation("relu")(model)
        # model = tf.keras.layers.Dense(256, activation="relu")(model)
        # model = tf.keras.layers.Dropout(rate=0.2)(model)
        # # model = tf.keras.layers.Activation("relu")(model)

        # model = tf.keras.layers.Dense(128, activation="relu")(model)
        # model = tf.keras.layers.Dropout(rate=0.2)(model)
        # model = tf.keras.layers.Activation("relu")(model)

        model = tf.keras.layers.Dense(64, activation="relu")(model)
        # model = tf.keras.layers.Activation("relu")(model)

        model = tf.keras.layers.Dense(n_classes, activation="softmax")(model)

        model = tf.keras.models.Model(
            inputs=input_tensor, outputs=model, name="ResNet50"
        )

        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=1e-3, decay_steps=1000, decay_rate=0.95
        # )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            # optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
            # loss=Flooding(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=metrics,
        )

        # Save to attribute
        self.model = model

        print(self.model.summary())

        return model
