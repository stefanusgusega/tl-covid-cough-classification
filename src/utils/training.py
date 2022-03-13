# TODO: implement sttratified k-fold validation
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from model.base import BaseModel
from utils.preprocess import encode_label, expand_mel_spec
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)


class Trainer:
    def __init__(
        self,
        audio_datas: np.ndarray,
        audio_labels: np.ndarray,
        model: BaseModel,
        test_size: float = 0.8,
    ) -> None:

        if model.model_type == "base":
            raise Exception("The model cannot be a base model.")

        self.X_full = audio_datas

        # print("Encoding labels...")
        # Label encode the audio labels first
        # label_encoder = OrdinalEncoder(categories=[["healthy", "COVID-19"]])
        self.y_full = audio_labels
        # self.y_full = label_encoder.fit_transform(audio_labels)
        # print("Labels encoded.")

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

        # Init model attribute
        self.model = model

        # Build model
        self.model.build_model()

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
            if self.model.model_type == "resnet50":
                # TODO: Should check whether using mel spec or mfcc
                X_train = expand_mel_spec(X_train)
                X_val = expand_mel_spec(X_val)

            # Apply one hot encoding
            y_train = encode_label(y_train, "COVID-19")
            y_val = encode_label(y_val, "COVID-19")

            print(f"Training for fold {idx+1}/{n_splits}...")
            self.model.fit(
                datas=X_train,
                labels=y_train,
                validation_datas=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
            )

            loss, metric = self.model.evaluate(X_val, y_val)

            self.metrics_arr.append(metric)
            self.losses_arr.append(loss)
