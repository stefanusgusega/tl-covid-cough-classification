# TODO: implement sttratified k-fold validation
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from model.resnet50 import ResNet50Model
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)


class Trainer:
    def __init__(
        self, audio_datas: np.ndarray, audio_labels: np.ndarray, test_size: float = 0.8
    ) -> None:
        self.X_full = audio_datas

        print("Encoding labels...")
        # Label encode the audio labels first
        label_encoder = OrdinalEncoder(categories=[["healthy", "COVID-19"]])
        self.y_full = label_encoder.fit_transform(audio_labels)
        print("Labels encoded.")

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

        # Init array to save metrics
        self.metrics_arr = []

    def stratified_k_fold_cross_validation(
        self, n_splits: int = 5, epochs: int = 100, batch_size: int = None
    ):
        # Reset the counter of the global random seed
        tf.random.set_seed(42)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for idx, (train_index, val_index) in enumerate(skf.split(self.X, self.y)):
            # TODO: do training
            # 80 : 20 --> 80 will be stratified 5 fold cross validated

            # X_train, X_test = audio_datas[train_index], audio_datas[test_index]
            # y_train, y_test = audio_labels[train_index], audio_labels[test_index]
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y[train_index], self.y[val_index]

            # Get dimension of each audio data (mel spectrogram)
            # TODO : ResNet50 expects the input_shape to be 3D
            X_train = self.expand_mel_spec(X_train)
            X_val = self.expand_mel_spec(X_val)

            input_shape = X_train.shape[1:]

            model = ResNet50Model(input_shape=input_shape)
            model.build_model()

            print(f"Training for fold {idx+1}...")
            model.fit(
                datas=X_train,
                labels=y_train,
                validation_datas=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
            )

            _, metric = model.evaluate(X_val, y_val)

            self.metrics_arr.append(metric)

        # self.X = self.expand_mel_spec(self.X)
        # model = ResNet50Model(input_shape = self.X.shape[1:])
        # model.build_model()

        # fit_params = {
        #     'epochs': epochs,
        #     'batch_size': batch_size,
        #     'validation_datas': ()
        # }

        # self.metrics_arr = cross_val_score(estimator=model, X=self.X, y=self.y, )

    def expand_mel_spec(self, audio_datas: np.ndarray):
        new_mel_specs = []

        for mel_spec in audio_datas:
            new_mel_spec = np.expand_dims(mel_spec, -1)
            new_mel_specs.append(new_mel_spec)

        return np.array(new_mel_specs)
