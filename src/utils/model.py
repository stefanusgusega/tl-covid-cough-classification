"""
Model util functions that should not be in one class.
"""
import math
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow as tf
from src.utils.chore import generate_now_datetime


def generate_tensorboard_callback(log_dir: str):
    """
    Generate a TensorBoard callback for analysis and visualization needs.
    TensorBoard source : https://www.tensorflow.org/tensorboard/graphs
    """
    specified_log_dir = os.path.join(log_dir, generate_now_datetime())

    return tf.keras.callbacks.TensorBoard(log_dir=specified_log_dir)


def generate_checkpoint_callback(parent_dir: str):
    """
    Generate a checkpoint callback.
    For details : https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
    """
    specified_ckpt_dir = os.path.join(parent_dir, generate_now_datetime())

    return tf.keras.callbacks.ModelCheckpoint(
        filepath=specified_ckpt_dir, save_best_only=True
    )


def draw_roc(
    model,
    x_test,
    y_test,
    plot_name: str,
):
    # Generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_test))]

    y_pred_proba = model.predict(x_test)
    fpr, tpr, _ = roc_curve(y_true=y_test, y_score=y_pred_proba)
    ns_fpr, ns_tpr, _ = roc_curve(y_true=y_test, y_score=ns_probs)
    auc = roc_auc_score(y_true=y_test, y_score=y_pred_proba)

    # Do plotting
    plt.plot(fpr, tpr, label=f"AUC={auc}")
    plt.plot(ns_fpr, ns_tpr, label="No Skill AUC=0.5")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc=4)

    # Save the plot
    plt.savefig(os.path.join("dumps\\plots", f"{plot_name}.png"), transparent=True)

    plt.show()


def lr_step_decay(epoch, _):
    initial_learning_rate = 5e-3
    drop_rate = 0.5
    epochs_drop = 10.0

    return initial_learning_rate * math.pow(drop_rate, math.floor(epoch / epochs_drop))
