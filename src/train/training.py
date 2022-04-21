"""
Trainer module.
"""
import os
import warnings
import numpy as np
from icecream import ic
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from src.model import ResNet50Model
from src.utils.chore import generate_now_datetime
from src.utils.preprocess import encode_label, expand_mel_spec
from src.utils.model import (
    generate_checkpoint_callback,
    generate_tensorboard_callback,
    # lr_step_decay,
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
        log_dir: dict,
        model_type: str = "resnet50",
        model_args: dict = None,
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

        # Init array to save models for each fold
        self.models = []

        # Init callbacks array with learning rate scheduler
        self.callbacks_arr = [tf.keras.callbacks.ReduceLROnPlateau(verbose=1)]

        self.model_type = model_type
        self.model_args = model_args

        self.log_dir = log_dir

        self.using_tensorboard = False

    def train(
        self,
        n_splits: int = 5,
        epochs: int = 100,
        batch_size: int = None,
    ):
        """
        Train the model
        """

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for idx, (folds_index, test_index) in enumerate(
            skf.split(self.x_full, self.y_full)
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
                x_folds = expand_mel_spec(x_folds)
                x_test = expand_mel_spec(x_test)

            # Apply one hot encoding
            y_folds = encode_label(y_folds, "COVID-19")
            y_test = encode_label(y_test, "COVID-19")

            ic(np.unique(y_folds, return_counts=True))
            ic(np.unique(y_test, return_counts=True))
            ic(folds_index[:10])
            ic(test_index[:10])

            model = self.generate_model().build_model()

            # Training for this fold
            print(f"Training for fold {idx+1}/{n_splits}...")

            # Init dynamic callbacks list. Dynamic means should take different actions
            # every fold. For example, checkpoint for each fold, tensorboard for each fold.
            # Init with checkpoint callback and tensorboard callback
            dynamic_callbacks = [
                generate_tensorboard_callback(self.log_dir["tensorboard"]),
                generate_checkpoint_callback(self.log_dir["checkpoint"]),
            ]

            # if idx != 4:
            #     continue

            model.fit(
                x_folds,
                y_folds,
                validation_data=(x_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[*self.callbacks_arr, *dynamic_callbacks],
                shuffle=True,
                verbose=2,
            )

            # Evaluate model for outer loop
            print(f"Evaluating model fold {idx + 1}/{n_splits}...")
            # Accuracy from Keras seems inaccurate
            loss, auc, acc = model.evaluate(x_test, y_test)

            # Save the values for outer loop
            self.test_metrics_arr.append(auc)
            self.test_losses_arr.append(loss)

            # Save accuracy
            self.test_accuracy_arr.append(acc)

            # Last save the model
            self.models.append(model)

            # Limit to only once
            # break

        print(f"AUC-ROC average: {np.mean(self.test_metrics_arr)}")
        print(f"Accuracy average: {np.mean(self.test_accuracy_arr)}")
        print(f"Loss average: {np.mean(self.test_losses_arr)}")
        print(f"AUC-ROC std: {np.std(self.test_metrics_arr)}")
        print(f"Accuracy std: {np.std(self.test_accuracy_arr)}")
        print(f"Loss std: {np.std(self.test_losses_arr)}")

    def generate_model(self):
        """
        Generate the model based on model type with using model arguments
        """
        if self.model_type == "resnet50":
            model = ResNet50Model(**self.model_args)

        return model

    def set_early_stopping_callback(self):
        """
        Set the early stopping callback for fitting the model
        """
        self.callbacks_arr.append(
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
        )

    def save_models(self, path_to_save: str, save_format: str = "tf"):
        """
        Save the models for each fold.
        """
        if save_format not in ["tf", "h5"]:
            warnings.warn("```save_format``` unavailable. Defaults to None.")
            save_format = None

        # Prefix for the file name
        prefix = f"model_{generate_now_datetime()}"

        for idx, model in enumerate(self.models):
            print(f"Saving model for fold {idx+1}... ({idx+1}/{len(self.models)})")
            # Define the filename
            model_filename = os.path.join(path_to_save, f"{prefix}_{idx+1}")
            # Save the model
            model.save(model_filename, save_format=save_format)
            print(
                f"Model for fold {idx+1} saved at {model_filename} in {save_format} format."
            )
