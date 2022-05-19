"""
Trainer module.
"""
import os
from icecream import ic
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from src.model import ResNet50Model
from src.model.pretrain import ResNet50PretrainModel
from src.utils.chore import save_obj_to_pkl
from src.utils.preprocess import FeatureExtractor, encode_label, expand_mel_spec
from src.utils.model import (
    generate_tensorboard_callback,
    # lr_step_decay,
)

AVAILABLE_MODELS = ["resnet50", "pretrain-resnet50"]


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
        self.test_auc_arr = []
        # self.val_metrics_arr = []
        self.test_losses_arr = []
        # self.val_losses_arr = []
        # self.train_accuracy_arr = []
        self.test_accuracy_arr = []
        self.test_f1_arr = []

        # Init array to save models for each fold
        self.models = []

        # Init callbacks array with learning rate scheduler
        self.callbacks_arr = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", verbose=1)
        ]

        self.model_type = model_type

        self.log_dir = log_dir

        self.using_tensorboard = False

    def cross_validation(
        self,
        n_splits: int = 5,
        epochs: int = 100,
        batch_size: int = None,
        feature_parameter: dict = None,
    ):
        """
        Perform cross validation to validate the hyperparameter of the model
        """

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for idx, (folds_index, test_index) in enumerate(
            skf.split(self.x_full, self.y_full)
        ):
            # Train for the specified fold
            # if idx > 1:
            #     print(f"Skipping fold {idx+1}...")
            #     continue

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

            feature_extractor = FeatureExtractor(backup_every_stage=False)
            print(f"Extracting features for training data of fold {idx+1}/{n_splits}")
            x_folds, y_folds = feature_extractor.run(
                aggregated_data=x_folds, aggregated_labels=y_folds, **feature_parameter
            )

            test_feat_ext = FeatureExtractor(
                backup_every_stage=False,
                offset=feature_extractor.offset,
                for_training=False,
            )
            print(f"Extracting features for testing data of fold {idx+1}/{n_splits}")
            x_test, y_test = test_feat_ext.run(
                aggregated_data=x_test, aggregated_labels=y_test, **feature_parameter
            )

            # If the model is a resnet-50, the input should be expanded
            # Because ResNet expects a 3D shape
            if self.model_type in ["resnet50", "pretrain-resnet50"]:
                x_folds = expand_mel_spec(x_folds)
                x_test = expand_mel_spec(x_test)

            # Apply one hot encoding
            y_folds = encode_label(y_folds, "COVID-19")
            y_test = encode_label(y_test, "COVID-19")

            ic(np.unique(y_folds, return_counts=True))
            ic(np.unique(y_test, return_counts=True))
            # ic(folds_index[:10])
            # ic(test_index[:10])

            model = self.generate_model(
                model_args=dict(input_shape=(x_folds.shape[1], x_folds.shape[2], 1))
            ).build_model()

            # Training for this fold
            print(f"Training for fold {idx+1}/{n_splits}...")

            # Init dynamic callbacks list. Dynamic means should take different actions
            # every fold. For example, checkpoint for each fold, tensorboard for each fold.
            # Init with checkpoint callback and tensorboard callback
            dynamic_callbacks = [
                generate_tensorboard_callback(self.log_dir["tensorboard"]),
            ]

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
            self.evaluate_fold(
                fold_num=idx + 1,
                n_splits=n_splits,
                model=model,
                x_test=x_test,
                y_test=y_test,
            )

            # Draw ROC-AUC curve and then save it
            # print("Drawing ROC-AUC curve...")
            # draw_roc(
            #     model=model,
            #     x_test=x_test,
            #     y_test=y_test,
            #     plot_name=f"baseline_crossval_{idx+1}",
            # )

            # Limit to only once
            # break

        print(f"AUC-ROC average: {np.mean(self.test_auc_arr)}")
        print(f"Accuracy average: {np.mean(self.test_accuracy_arr)}")
        print(f"F1 average: {np.mean(self.test_f1_arr)}")
        print(f"Loss average: {np.mean(self.test_losses_arr)}")
        print(f"AUC-ROC std: {np.std(self.test_auc_arr)}")
        print(f"Accuracy std: {np.std(self.test_accuracy_arr)}")
        print(f"F1 std: {np.std(self.test_f1_arr)}")
        print(f"Loss std: {np.std(self.test_losses_arr)}")

    def train(
        self,
        epochs: int,
        batch_size: int,
        model_filepath: str = None,
        feature_extractor_filepath: str = None,
        feature_parameter: dict = None,
    ):
        assert len(self.x_full) == len(
            self.y_full
        ), "x and y of training data have inequal length."

        train_indexes = np.arange(len(self.y_full))

        # Shuffling the indexes
        np.random.shuffle(train_indexes)

        # Preprocess the training data
        feature_extractor = FeatureExtractor(backup_every_stage=False)
        x_train, y_train = feature_extractor.run(
            aggregated_data=self.x_full,
            aggregated_labels=self.y_full,
            **feature_parameter,
        )

        # If the model is a resnet-50, the input should be expanded
        # Because ResNet expects a 3D shape
        if self.model_type in ["resnet50", "pretrain-resnet50"]:
            x_train = expand_mel_spec(x_train)

        # Apply one hot encoding
        y_train = encode_label(y_train, "COVID-19")

        model = self.generate_model(
            model_args=dict(input_shape=(x_train.shape[1], x_train.shape[2], 1))
        ).build_model()

        ic(np.unique(y_train, return_counts=True))

        # Start training
        print("Training...")

        dynamic_callbacks = [
            generate_tensorboard_callback(self.log_dir["tensorboard"]),
        ]

        model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[*self.callbacks_arr, *dynamic_callbacks],
            shuffle=True,
            verbose=2,
        )

        # Save model
        print(f"Saving model {os.path.basename(model_filepath)}...")
        model.save(model_filepath)
        print("Model saved.")

        # Save feature extractor instance
        print(
            f"Saving feature extractor {os.path.basename(feature_extractor_filepath)}..."
        )
        save_obj_to_pkl(feature_extractor, feature_extractor_filepath)
        print("Feature extractor saved.")

        # Return feature extractor instance and model instance
        return feature_extractor, model

    def evaluate_fold(self, fold_num: int, n_splits: int, model, x_test, y_test):
        print(f"Evaluating model fold {fold_num}/{n_splits}...")

        loss, auc, acc = model.evaluate(x_test, y_test)

        # Do prediction
        y_proba = model.predict(x_test)

        # Make label to 0 and 1
        y_pred = np.where(y_proba >= 0.5, 1, 0)

        # Save the values for outer loop
        self.test_auc_arr.append(auc)
        self.test_losses_arr.append(loss)

        # Save accuracy
        self.test_accuracy_arr.append(acc)

        # Save F1 score
        f_one = f1_score(y_true=y_test, y_pred=y_pred)
        self.test_f1_arr.append(f_one)
        print(f"F1 score fold {fold_num}: {f_one}")

        print(classification_report(y_true=y_test, y_pred=y_pred))
        print(confusion_matrix(y_true=y_test, y_pred=y_pred))

    def generate_model(self, model_args):
        """
        Generate the model based on model type with using model arguments
        """
        if self.model_type == "resnet50":
            model = ResNet50Model(**model_args)

        return model


class Pretrainer(Trainer):
    """
    The class aimed for pretraining
    """

    def __init__(
        self,
        audio_datas: np.ndarray,
        audio_labels: np.ndarray,
        log_dir: dict,
        model_type: str = "resnet50",
    ) -> None:
        super().__init__(audio_datas, audio_labels, log_dir, model_type)
        self.test_f1_0 = []
        self.test_f1_2 = []

    def cross_validation(
        self,
        n_splits: int = 5,
        epochs: int = 100,
        batch_size: int = None,
        feature_parameter: dict = None,
    ):
        """
        Perform cross validation to validate the hyperparameter of the pretrained model
        """

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for idx, (folds_index, test_index) in enumerate(
            skf.split(self.x_full, self.y_full)
        ):
            # Train for the specified fold
            # if idx < 4:
            #     print(f"Skipping fold {idx+1}...")
            #     continue

            # Shuffle the index produced
            np.random.shuffle(folds_index)
            np.random.shuffle(test_index)

            ic(folds_index)
            ic(test_index)

            x_folds, x_test = (
                self.x_full[folds_index],
                self.x_full[test_index],
            )
            y_folds, y_test = (
                self.y_full[folds_index],
                self.y_full[test_index],
            )

            feature_extractor = FeatureExtractor(backup_every_stage=False)
            print(f"Extracting features for training data of fold {idx+1}/{n_splits}")
            x_folds, y_folds = feature_extractor.run(
                aggregated_data=x_folds, aggregated_labels=y_folds, **feature_parameter
            )

            test_feat_ext = FeatureExtractor(
                backup_every_stage=False,
                offset=feature_extractor.offset,
                for_training=False,
            )
            print(f"Extracting features for testing data of fold {idx+1}/{n_splits}")
            x_test, y_test = test_feat_ext.run(
                aggregated_data=x_test, aggregated_labels=y_test, **feature_parameter
            )

            # If the model is a resnet-50, the input should be expanded
            # Because ResNet expects a 3D shape
            if self.model_type in ["resnet50", "pretrain-resnet50"]:
                x_folds = expand_mel_spec(x_folds)
                x_test = expand_mel_spec(x_test)

            # Apply one hot encoding
            y_folds = encode_label(y_folds, pos_label="cough")
            y_test = encode_label(y_test, pos_label="cough")

            print(f"Data for training fold {idx+1}")
            ic(y_folds)
            print(np.unique(y_folds, return_counts=True))

            print(f"Data for validation fold {idx+1}")
            ic(y_test)
            print(np.unique(y_test, return_counts=True))

            # ic(np.unique(y_folds, return_counts=True))
            # ic(np.unique(y_test, return_counts=True))
            # ic(folds_index[:10])
            # ic(test_index[:10])

            model = self.generate_model(
                model_args=dict(input_shape=(x_folds.shape[1], x_folds.shape[2], 1))
            ).build_model(n_classes=3)

            # Training for this fold
            print(f"Training for fold {idx+1}/{n_splits}...")

            # Init dynamic callbacks list. Dynamic means should take different actions
            # every fold. For example, checkpoint for each fold, tensorboard for each fold.
            # Init with checkpoint callback and tensorboard callback
            dynamic_callbacks = [
                generate_tensorboard_callback(self.log_dir["tensorboard"]),
            ]

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
            self.evaluate_fold(
                fold_num=idx + 1,
                n_splits=n_splits,
                model=model,
                x_test=x_test,
                y_test=y_test,
            )

            # Draw ROC-AUC curve and then save it
            # print("Drawing ROC-AUC curve...")
            # draw_roc(
            #     model=model,
            #     x_test=x_test,
            #     y_test=y_test,
            #     plot_name=f"baseline_crossval_{idx+1}",
            # )

            # Limit to only once
            # break

        print(f"Accuracy average: {np.mean(self.test_accuracy_arr)}")
        print(f"F1 class 0 average: {np.mean(self.test_f1_0)}")
        print(f"F1 class 1 average: {np.mean(self.test_f1_arr)}")
        print(f"F1 class 2 average: {np.mean(self.test_f1_2)}")

        print(f"Loss average: {np.mean(self.test_losses_arr)}")
        print(f"Accuracy std: {np.std(self.test_accuracy_arr)}")
        print(f"F1 class 0 std: {np.std(self.test_f1_0)}")
        print(f"F1 class 1 std: {np.std(self.test_f1_arr)}")
        print(f"F1 class 2 std: {np.std(self.test_f1_2)}")

        print(f"Loss std: {np.std(self.test_losses_arr)}")

    def train(
        self,
        epochs: int,
        batch_size: int,
        model_filepath: str = None,
        feature_extractor_filepath: str = None,
        feature_parameter: dict = None,
    ):
        assert len(self.x_full) == len(
            self.y_full
        ), "x and y of training data have inequal length."

        train_indexes = np.arange(len(self.y_full))

        # Shuffling the indexes
        np.random.shuffle(train_indexes)

        # Preprocess the training data
        feature_extractor = FeatureExtractor(backup_every_stage=False)
        x_train, y_train = feature_extractor.run(
            aggregated_data=self.x_full,
            aggregated_labels=self.y_full,
            **feature_parameter,
        )

        # If the model is a resnet-50, the input should be expanded
        # Because ResNet expects a 3D shape
        if self.model_type in ["resnet50", "pretrain-resnet50"]:
            x_train = expand_mel_spec(x_train)

        # Apply one hot encoding
        y_train = encode_label(y_train, "cough")

        model = self.generate_model(
            model_args=dict(input_shape=(x_train.shape[1], x_train.shape[2], 1))
        ).build_model()

        ic(np.unique(y_train, return_counts=True))

        # Start training
        print("Training...")

        dynamic_callbacks = [
            generate_tensorboard_callback(self.log_dir["tensorboard"]),
        ]

        model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[*self.callbacks_arr, *dynamic_callbacks],
            shuffle=True,
            verbose=2,
        )

        # Save model
        print(f"Saving model {os.path.basename(model_filepath)}...")
        model.save(model_filepath)
        print("Model saved.")

        # Save feature extractor instance
        print(
            f"Saving feature extractor {os.path.basename(feature_extractor_filepath)}..."
        )
        save_obj_to_pkl(feature_extractor, feature_extractor_filepath)
        print("Feature extractor saved.")

        # Return feature extractor instance and model instance
        return feature_extractor, model

    def evaluate_fold(self, fold_num: int, n_splits: int, model, x_test, y_test):
        print(f"Evaluating model fold {fold_num}/{n_splits}...")

        loss, acc = model.evaluate(x_test, y_test)

        # Do prediction
        y_softmax = model.predict(x_test)

        # Make label to 0, 1, and 2
        y_pred = np.argmax(y_softmax, axis=1)
        y_test = np.argmax(y_test, axis=1)

        # Save the values for outer loop
        self.test_losses_arr.append(loss)

        # Save accuracy
        self.test_accuracy_arr.append(acc)

        # Save F1 score
        f_ones = f1_score(y_true=y_test, y_pred=y_pred, average=None)
        self.test_f1_0.append(f_ones[0])
        self.test_f1_arr.append(f_ones[1])
        self.test_f1_2.append(f_ones[2])

        print(f"F1 score fold {fold_num} for class 0: {f_ones[0]}")
        print(f"F1 score fold {fold_num} for cough class (class 1): {f_ones[1]}")
        print(f"F1 score fold {fold_num} for class 2: {f_ones[2]}")
        print(classification_report(y_true=y_test, y_pred=y_pred))
        print(confusion_matrix(y_true=y_test, y_pred=y_pred))

    def generate_model(self, model_args):
        """
        Generate the model based on model type with using model arguments
        """
        model = ResNet50PretrainModel(**model_args)

        return model
