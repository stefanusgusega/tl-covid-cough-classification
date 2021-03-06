"""
Randomizer util functions.
"""

import os
import random
import numpy as np
import tensorflow as tf


def set_random_seed(seed_value: int = 42):
    # Workaround for
    os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"] = "True"

    # Configures TensorFlow ops to run deterministically
    tf.config.experimental.enable_op_determinism()

    # Set the PYTHONHASHSEED environment variable at a fixed value
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    # Set the python built-in pseudorandom number generator
    random.seed(seed_value)

    # Set the numpy pseudorandom number generator
    np.random.seed(seed_value)

    # Set the tensorflow pseudorandom number generator
    tf.random.set_seed(seed_value)
