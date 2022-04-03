"""
Trainer module.
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from src.model import ResNet50Model
from src.utils.chore import generate_now_datetime
from src.utils.preprocess import encode_label, expand_mel_spec
from src.utils.model import (
    generate_tensorboard_callback,
    hyperparameter_tune_resnet_model,
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
        hyperparameter_tuning_args: dict = None,
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
        self.train_accuracy_arr = []
        self.test_accuracy_arr = []

        # Init callbacks array
        self.callbacks_arr = []

        self.model_type = model_type
        self.model_args = model_args

        self.hyperparameter_tuning_args = hyperparameter_tuning_args
        self.using_tensorboard = False

        if tensorboard_log_dir is not None:
            self.using_tensorboard = True

        self.tensorboard_log_dir = tensorboard_log_dir

    def train(
        self,
        hp_model_tuning_folder: str = None,
        n_splits: int = 5,
        epochs: int = 100,
        batch_size: int = None,
    ):
        """
        Train the model and save the hyperparameter model
        """

        if hp_model_tuning_folder is not None:
            if self.hyperparameter_tuning_args is None:
                raise Exception(
                    "Please specify the hyperparameter tuning arguments at the constructor."
                )

        # The outer is to split between data and test
        outer_skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # The center is to split between train and vals
        # center_skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        # inner_skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for outer_idx, (folds_index, test_index) in enumerate(
            outer_skf.split(self.x_full, self.y_full)
        ):
            x_folds, x_test = self.x_full[folds_index], self.x_full[test_index]
            y_folds, y_test = self.y_full[folds_index], self.y_full[test_index]

            # If the model is a resnet-50, the input should be expanded
            # Because ResNet expects a 3D shape
            if self.model_type == "resnet50":
                # TODO: Should check whether using mel spec or mfcc
                x_folds = expand_mel_spec(x_folds)
                x_test = expand_mel_spec(x_test)

            # Apply one hot encoding
            y_folds = encode_label(y_folds, "COVID-19")
            y_test = encode_label(y_test, "COVID-19")

            print(y_folds)
            # break

            # TODO : Hyperparameter tuning
            # ! This is still tuning in classifier level
            # If not intended to do tuning, then directly generate model and build
            if self.hyperparameter_tuning_args is None:
                model = self.generate_model().build_model()
            # Else, hyperparameter tune it
            else:
                model = self.hyperparameter_tune(x_folds, y_folds)
                # Save the optimized model
                self.save_optimum_hyperparameter_model(hp_model_tuning_folder, model)

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

            # Get accuracy
            y_train_pred = np.argmax(model.predict(x_folds), axis=1)
            y_train_acc = accuracy_score(
                y_pred=y_train_pred, y_true=np.argmax(y_folds, axis=1)
            )
            y_test_pred = np.argmax(model.predict(x_test), axis=1)
            y_test_acc = accuracy_score(
                y_pred=y_test_pred, y_true=np.argmax(y_test, axis=1)
            )

            print(f"Training accuracy: {y_train_acc}")
            print(f"Test accuracy: {y_test_acc}")
            print(f"Test accuracy based on keras metrics: {acc}")

            # Save the values for outer loop
            self.test_metrics_arr.append(auc)
            self.test_losses_arr.append(loss)

            # Save accuracy
            self.train_accuracy_arr.append(y_train_acc)
            self.test_accuracy_arr.append(acc)

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

    def hyperparameter_tune(
        self, datas: np.ndarray, labels: np.ndarray, n_splits: int = 5
    ) -> tf.keras.Model:
        """
        This is located in deepest loop of nested cross validation.
        Should return the best model.
        """
        initial_model = self.generate_model()

        model = KerasClassifier(
            model=hyperparameter_tune_resnet_model,
            optimizer="adam",
            # This is kwargs
            first_dense_units=[],
            second_dense_units=[],
            learning_rates=[],
            initial_model=initial_model,
        )

        grid = GridSearchCV(
            estimator=model,
            param_grid=self.hyperparameter_tuning_args,
            # scoring="roc_auc",
            cv=n_splits,
            verbose=1,
        )
        grid_result = grid.fit(datas, labels)

        return grid_result.best_estimator_.model_

    # TODO : save as h5 file, maybe add parameter to choose whether save as h5 or not.
    def save_optimum_hyperparameter_model(self, folder: str, model: tf.keras.Model):
        """
        Save the optimum hyperparameter model for specified traning folds in specified folder.
        """
        if self.hyperparameter_tuning_args is not None:
            folder_name = os.path.join(folder, generate_now_datetime())
            print("Saving the optimum hyperparameter model...")
            model.save(folder_name)
            print(f"Optimum hyperparameter model saved at {folder_name}.")
