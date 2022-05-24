"""
ResNet50 Transfer Learning Model class
"""

from typing import Tuple
import tensorflow as tf
from src.model.base import BaseModel
from src.model.builder import convolutional_block, identity_block

AVAILABLE_TRANSFER_LEARNING_MODES = ["weight_init", "feat_ext"]


class TransferLearningModel(BaseModel):
    """
    Transfer Learning Model using pretrained weights
    """

    def __init__(
        self,
        input_shape: Tuple,
        mode: str,
        pretrained_model_path: str = None,
        open_layer: int = 0,
        initial_weights=None,
    ) -> None:
        super().__init__(input_shape, initial_weights)

        if mode not in AVAILABLE_TRANSFER_LEARNING_MODES:
            raise Exception(
                f"The mode should be one of these: {AVAILABLE_TRANSFER_LEARNING_MODES}"
                f"Found: {mode}."
            )
        self.model_type = "transfer-resnet50"
        self.pretrained_model_path = pretrained_model_path
        self.mode = mode
        self.open_layer = open_layer

    def build_model(self, metrics=None, n_classes: int = 2):
        if metrics is None:
            metrics = [tf.keras.metrics.AUC(), "accuracy"]

        # Load pretrained model
        loaded_model = tf.keras.models.load_model(self.pretrained_model_path)

        # Initiate new fully connected layers
        if self.mode == "weight_init":
            # If weight init, keep old 512 dense
            new_model = loaded_model.layers[-3].output

        else:
            # If feat ext, initiate new layers but based on open layer
            new_model = self.initiate_new_layers(loaded_model=loaded_model)

        new_model = tf.keras.layers.Dense(32, activation="relu", name="new_32")(
            new_model
        )
        new_model = tf.keras.layers.Dense(1, activation="sigmoid", name="new_output")(
            new_model
        )

        new_model = tf.keras.models.Model(inputs=loaded_model.input, outputs=new_model)

        # Open all
        if self.open_layer != 0:
            print("Deactivating trainable...")
            self.deactivate_trainable(new_model)

        new_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=metrics,
        )

        print(new_model.summary())
        return new_model

    def initiate_new_layers(self, loaded_model):
        assert self.mode == "feat_ext", "The mode should be feature extractor"

        if self.open_layer == 0:
            last_layer_index = -4

        elif self.open_layer == 1:
            last_layer_index = -99

        elif self.open_layer == 2:
            last_layer_index = -36

        new_model = loaded_model.layers[last_layer_index].output

        # init first FC layer
        if self.open_layer == 0:
            new_model = tf.keras.layers.Dense(512, activation="relu", name="new_512")(
                new_model
            )

        elif self.open_layer == 1:
            # conv4
            new_model = convolutional_block(
                new_model,
                middle_kernel_size=3,
                filters=[256, 256, 1024],
                stage=4,
                block="new_a",
                stride=2,
            )
            # Remaining identity blocks
            for block in ["b", "c", "d", "e", "f"]:
                new_model = identity_block(
                    new_model,
                    middle_kernel_size=3,
                    filters=[256, 256, 1024],
                    stage=4,
                    block=f"new_{block}",
                )
                print(f"New Block 4{block} initialized.")
            print("New Conv4 Layer initialized.")

            # conv5
            new_model = convolutional_block(
                new_model,
                middle_kernel_size=3,
                filters=[512, 512, 2048],
                stage=5,
                block="new_a",
                stride=2,
            )
            # Remaining identity blocks
            for block in ["b", "c"]:
                new_model = identity_block(
                    new_model,
                    middle_kernel_size=3,
                    filters=[512, 512, 2048],
                    stage=5,
                    block=f"new_{block}",
                )
                print(f"New Block 5{block} initialized.")
            print("New Conv5 Layer initialized.")

            new_model = tf.keras.layers.GlobalAveragePooling2D(name="new_avg_pool")(
                new_model
            )
            new_model = tf.keras.layers.Dense(512, activation="relu")(new_model)

        elif self.open_layer == 2:
            # conv5
            new_model = convolutional_block(
                new_model,
                middle_kernel_size=3,
                filters=[512, 512, 2048],
                stage=5,
                block="new_a",
                stride=2,
            )
            # Remaining identity blocks
            for block in ["b", "c"]:
                new_model = identity_block(
                    new_model,
                    middle_kernel_size=3,
                    filters=[512, 512, 2048],
                    stage=5,
                    block=f"new_{block}",
                )
                print(f"New Block 5{block} initialized.")
            print("New Conv5 Layer initialized.")

            new_model = tf.keras.layers.GlobalAveragePooling2D(name="new_avg_pool")(
                new_model
            )
            new_model = tf.keras.layers.Dense(512, activation="relu")(new_model)

        return new_model

    def deactivate_trainable(self, model):
        # Open layer 0 == 'retrain all' (weight_init), freeze only not FC' (feat_ext)
        # Open layer 1 == 'retrain start from conv4' (weight_init), 'freeze until conv4' (feat_ext)
        # Open layer 2 == 'retrain start from conv5' (weight_init), 'freeze until conv5' (feat_ext)

        # TODO : add condition when open layer 0, because it is needed to freeze only not FC
        assert (
            self.open_layer != 0
        ), f"Cannot deactivate trainable, because open_layer == {self.open_layer}"

        if self.open_layer == 1:
            last_layer_index = -98
        elif self.open_layer == 2:
            last_layer_index = -36  # Activate conv4 and so on

        # Iterate the layers
        for layer in model.layers[:last_layer_index]:
            layer.trainable = False

        # If not discarding 512 dense layer from pretrained
        # last_layer_index = -3 if self.discard_512 else -2
        # last_layer_index = -4 if self.discard_512 else -3
        # last_layer_index = -36  # Activate conv5 and so on
