# TODO: implement sttratified k-fold validation
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder


def stratified_k_fold_cross_validation(
    audio_datas: np.ndarray, audio_labels: np.ndarray, n_splits: int = 5
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Label encode the audio labels
    label_encoder = OrdinalEncoder(categories=[["healthy", "COVID-19"]])
    labels_encoded = label_encoder.fit_transform(audio_labels)

    for idx, (train_index, test_index) in enumerate(
        skf.split(audio_datas, labels_encoded)
    ):
        ...
        # TODO: do training
        # 80 : 20 --> 80 will be stratified 5 fold cross validated

        # X_train, X_test = audio_datas[train_index], audio_datas[test_index]
        # y_train, y_test = audio_labels[train_index], audio_labels[test_index]
