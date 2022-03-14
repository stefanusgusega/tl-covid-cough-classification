import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from model.resnet50 import ResNet50Model
from utils.preprocess import encode_label, expand_mel_spec
import tensorflow as tf

AVAILABLE_MODELS = ["resnet50"]


class Trainer:
    def __init__(
        self,
        audio_datas: np.ndarray,
        audio_labels: np.ndarray,
        model_type: str = "resnet50",
        model_args: dict = None,
        test_size: float = 0.8,
    ) -> None:

        # TODO : the model resets its randomness when construct.
        # So it needs to make a new model whenever starts to train
        # Idea : init model only with string, then build a model based on the name
        # The param of the model passed on constructor
        # Need a method to build new model

        # If not one of available models, throw an exception
        if model_type not in AVAILABLE_MODELS:
            raise Exception(
                f"The model type should be one of these: {', '.join(AVAILABLE_MODELS)}. Found: {model_type}"
            )

        self.X_full = audio_datas
        self.y_full = audio_labels

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
        self.metrics_arr = []
        self.losses_arr = []

        # Init callbacks array
        self.callbacks_arr = []

        self.model_type = model_type
        self.model_args = model_args

    def stratified_k_fold_cross_validation(
        self, n_splits: int = 5, epochs: int = 100, batch_size: int = None
    ):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

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

            model = self.generate_model()

            print(f"Training for fold {idx+1}/{n_splits}...")
            model.fit(
                datas=X_train,
                labels=y_train,
                validation_datas=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=self.callbacks_arr,
            )

            loss, metric = model.evaluate(X_val, y_val)

            self.metrics_arr.append(metric)
            self.losses_arr.append(loss)

    def generate_model(self):
        if self.model_type == "resnet50":
            return ResNet50Model(**self.model_args)

    def set_checkpoint_callback(self, checkpoint_path: str, save_weights_only=True):
        # Append checkpoint callback
        self.callbacks_arr.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, save_weights_only=save_weights_only
            )
        )
