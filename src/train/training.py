"""
Trainer module.
"""
import numpy as np
from icecream import ic
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from src.model import ResNet50Model
from src.utils.preprocess import encode_label, expand_mel_spec
from src.utils.model import (
    generate_tensorboard_callback,
)

AVAILABLE_MODELS = ["resnet50"]


class Trainer:
    """
    This class is used for training and hyperparameter model tuning
    """

    def __init__(
        self,
        audio_datas: np.ndarray,
        audio_labels: np.ndarray,
        model_type: str = "resnet50",
        model_args: dict = None,
        tensorboard_log_dir: str = None,
    ) -> None:
        # If not one of available models, throw an exception
        if model_type not in AVAILABLE_MODELS:
            raise Exception(
                f"The model type should be one of these: {', '.join(AVAILABLE_MODELS)}. "
                f"Found: {model_type}."
            )

        self.x_full = audio_datas
        self.y_full = audio_labels

        # Init array to save metrics and losses
        self.test_metrics_arr = []
        # self.val_metrics_arr = []
        self.test_losses_arr = []
        # self.val_losses_arr = []
        # self.train_accuracy_arr = []
        self.test_accuracy_arr = []

        # Init callbacks array
        self.callbacks_arr = []

        self.model_type = model_type
        self.model_args = model_args

        self.using_tensorboard = False

        if tensorboard_log_dir is not None:
            self.using_tensorboard = True

        self.tensorboard_log_dir = tensorboard_log_dir

    def train(
        self,
        n_splits: int = 5,
        epochs: int = 100,
        batch_size: int = None,
    ):
        """
        Train the model
        """

        # The outer is to split between data and test
        outer_skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for outer_idx, (folds_index, test_index) in enumerate(
            outer_skf.split(self.x_full, self.y_full)
        ):
            # Shuffle the index produced
            np.random.shuffle(folds_index)
            np.random.shuffle(test_index)

            x_folds, x_test = (
                self.x_full[folds_index],
                self.x_full[test_index],
            )
            y_folds, y_test = (
                self.y_full[folds_index],
                self.y_full[test_index],
            )

            # If the model is a resnet-50, the input should be expanded
            # Because ResNet expects a 3D shape
            if self.model_type == "resnet50":
                # TODO: Should check whether using mel spec or mfcc
                x_folds = expand_mel_spec(x_folds)
                x_test = expand_mel_spec(x_test)

            # Apply one hot encoding
            y_folds = encode_label(y_folds, "COVID-19")
            y_test = encode_label(y_test, "COVID-19")

            ic(folds_index[:50])
            ic(test_index[:50])

            model = self.generate_model().build_model()

            # Training for this fold
            print(f"Training for fold {outer_idx+1}/{n_splits}...")

            # Always reset the TensorBoard callback
            # whenever using TensorBoard
            additional_callbacks = []
            if self.using_tensorboard:
                tb_callback = generate_tensorboard_callback(self.tensorboard_log_dir)
                additional_callbacks.append(tb_callback)

            model.fit(
                x_folds,
                y_folds,
                validation_data=(x_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[*self.callbacks_arr, *additional_callbacks],
            )

            # Evaluate model for outer loop
            print(f"Evaluating model fold {outer_idx + 1}/{n_splits}...")
            # Accuracy from Keras seems inaccurate
            loss, auc, acc = model.evaluate(x_test, y_test)

            # Save the values for outer loop
            self.test_metrics_arr.append(auc)
            self.test_losses_arr.append(loss)

            # Save accuracy
            self.test_accuracy_arr.append(acc)

        print(f"AUC-ROC average: {np.mean(self.test_metrics_arr)}")
        print(f"Accuracy average: {np.mean(self.test_accuracy_arr)}")
        print(f"Loss average: {np.mean(self.test_losses_arr)}")

    def generate_model(self):
        """
        Generate the model based on model type with using model arguments
        """
        if self.model_type == "resnet50":
            model = ResNet50Model(**self.model_args)

        return model

    def set_checkpoint_callback(self, checkpoint_path: str, save_weights_only=True):
        """
        Set the checkpoint callback for fitting the model
        """

        # Append checkpoint callback
        self.callbacks_arr.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, save_weights_only=save_weights_only
            )
        )

    def set_early_stopping_callback(self):
        """
        Set the early stopping callback for fitting the model
        """
        self.callbacks_arr.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss"))
