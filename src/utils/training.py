# TODO: implement sttratified k-fold validation
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder


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
            self.X_full, self.y_full, test_size=test_size, stratify=self.y
        )
        print("Dataset splitted.")

    def stratified_k_fold_cross_validation(self, n_splits: int = 5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for idx, (train_index, val_index) in enumerate(skf.split(self.X, self.y)):
            ...
            # TODO: do training
            # 80 : 20 --> 80 will be stratified 5 fold cross validated

            # X_train, X_test = audio_datas[train_index], audio_datas[test_index]
            # y_train, y_test = audio_labels[train_index], audio_labels[test_index]
