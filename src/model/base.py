from typing import Tuple


class BaseModel:
    def __init__(self, input_shape: Tuple, initial_weights=None) -> None:
        # Check whether the input shape of the model 2D
        if len(input_shape) != 2:
            raise Exception(f"The input shape should be 2D. Current : {input_shape}D.")
        self.input_shape = input_shape
        self.initial_weights = initial_weights

    def fit(self, datas, labels):
        ...

    def print_summary(self):
        self.model.summary()
