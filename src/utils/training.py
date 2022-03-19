import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from model.base import BaseModel
from model.resnet50 import ResNet50Model
from utils.preprocess import encode_label, expand_mel_spec
import tensorflow as tf
from scikeras.wrappers import KerasClassifier


AVAILABLE_MODELS = ["resnet50"]


class Trainer:
    def __init__(
        self,
        audio_datas: np.ndarray,
        audio_labels: np.ndarray,
        model_type: str = "resnet50",
        model_args: dict = None,
        test_size: float = 0.8,
        hyperparameter_tuning_args: dict = None,
    ) -> None:
        # If not one of available models, throw an exception
        if model_type not in AVAILABLE_MODELS:
            raise Exception(
                f"The model type should be one of these: {', '.join(AVAILABLE_MODELS)}. Found: {model_type}"
            )

        self.X_full = audio_datas
        self.y_full = audio_labels

        # TODO : Soon will be removed, because cross validation should apply for both outer and inner loop
        print("Splitting dataset to ready to train and test...")
        # Split here in stratified fashion
        self.X, self.X_test, self.y, self.y_test = train_test_split(
            self.X_full,
            self.y_full,
            test_size=test_size,
            stratify=self.y_full,
            random_state=42,
        )
        print("Dataset splitted.")

        # Init array to save metrics and losses
        self.test_metrics_arr = []
        self.val_metrics_arr = []
        self.test_losses_arr = []
        self.val_losses_arr = []

        # Init callbacks array
        self.callbacks_arr = []

        self.model_type = model_type
        self.model_args = model_args

        self.hyperparameter_tuning_args = hyperparameter_tuning_args

    def train(self, n_splits: int = 5, epochs: int = 100, batch_size: int = None):
        # The outer is to split between data and test
        outer_skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # The center is to split between train and vals
        # center_skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        # inner_skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for outer_idx, (folds_index, test_index) in enumerate(
            outer_skf.split(self.X_full, self.y_full)
        ):
            X_folds, X_test = self.X_full[folds_index], self.X_full[test_index]
            y_folds, y_test = self.y_full[folds_index], self.y_full[test_index]

            # If the model is a resnet-50, the input should be expanded
            # Because ResNet expects a 3D shape
            if self.model_type == "resnet50":
                # TODO: Should check whether using mel spec or mfcc
                X_folds = expand_mel_spec(X_folds)
                X_test = expand_mel_spec(X_test)

            # Apply one hot encoding
            y_folds = encode_label(y_folds, "COVID-19")
            y_test = encode_label(y_test, "COVID-19")

            # TODO : Hyperparameter tuning
            # ! This is still tuning in classifier level
            # If not intended to do tuning, then directly generate model and build
            if self.hyperparameter_tuning_args is None:
                model = self.generate_model().build_model()
            # Else, hyperparameter tune it
            else:
                model = self.hyperparameter_tune(X_folds, y_folds)

            # Training for this fold
            # TODO : Use the optimum hyperparamter to train.
            print(f"Training for fold {outer_idx+1}/{n_splits}...")
            model.fit(
                x=X_folds,
                y=y_folds,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=self.callbacks_arr,
            )

            # Evaluate model for outer loop
            print(f"Evaluating model fold {outer_idx + 1}/{n_splits}...")
            loss, metric = model.evaluate(X_test, y_test)

            # Save the values for outer loop
            self.test_metrics_arr.append(metric)
            self.test_losses_arr.append(loss)

    # ! Temporarily unused
    def stratified_k_fold_cross_validation(
        self, n_splits: int = 5, epochs: int = 100, batch_size: int = None
    ):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Outer loop
        for idx, (train_index, val_index) in enumerate(skf.split(self.X, self.y)):
            # 80 : 20 --> 80 will be stratified 5 fold cross validated
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y[train_index], self.y[val_index]

            # If the model is a resnet-50, the input should be expanded
            # Because ResNet expects a 3D shape
            if self.model_type == "resnet50":
                # TODO: Should check whether using mel spec or mfcc
                X_train = expand_mel_spec(X_train)
                X_val = expand_mel_spec(X_val)

            # Apply one hot encoding
            y_train = encode_label(y_train, "COVID-19")
            y_val = encode_label(y_val, "COVID-19")

            if self.hyperparameter_tuning_args is not None:
                self.hyperparameter_tune(
                    n_splits=5,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=self.callbacks_arr,
                )

            model = self.generate_model().build_model()

            print(f"Training for fold {idx+1}/{n_splits}...")
            model.fit(
                x=X_train,
                y=y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=self.callbacks_arr,
            )

            print(f"Evaluating model fold {idx + 1}/{n_splits}...")
            loss, metric = model.evaluate(X_val, y_val)

            self.metrics_arr.append(metric)
            self.losses_arr.append(loss)

    def generate_model(self) -> BaseModel:
        if self.model_type == "resnet50":
            model = ResNet50Model(**self.model_args)

        return model

    def set_checkpoint_callback(self, checkpoint_path: str, save_weights_only=True):
        # Append checkpoint callback
        self.callbacks_arr.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, save_weights_only=save_weights_only
            )
        )

    def set_early_stopping_callback(self):
        self.callbacks_arr.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss"))

    def hyperparameter_tune(
        self, datas: np.ndarray, labels: np.ndarray, n_splits: int = 5
    ) -> BaseModel:
        """
        This is the located in deepest loop of nested cross validation.
        Should return the best model.
        """
        initial_model = self.generate_model()

        model = KerasClassifier(
            build_fn=initial_model.hyperparameter_tune_model,
            verbose=0,
            first_dense_units=[],
            second_dense_units=[],
            learning_rate=[],
        )

        grid = GridSearchCV(
            estimator=model,
            param_grid=self.hyperparameter_tuning_args,
            cv=n_splits,
        )
        grid_result = grid.fit(datas, labels)

        return grid_result.best_estimator_
