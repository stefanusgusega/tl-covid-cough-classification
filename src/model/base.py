from abc import abstractmethod
from typing import Tuple


class BaseModel:
    def __init__(self, input_shape: Tuple, initial_weights=None) -> None:
        # Check whether the input shape of the model 2D
        if len(input_shape) != 3:
            raise Exception(
                f"The input shape should be 3D. Current : {len(input_shape)}D."
            )

        self.input_shape = input_shape
        self.initial_weights = initial_weights
        self.model = None
        self.model_type = "base"

    @abstractmethod
    def build_model(self):
        ...

    @abstractmethod
    def fit(self, datas, labels):
        ...

    def evaluate(self, datas, labels):
        scores = self.model.evaluate(datas, labels)
        return scores

    def print_summary(self):
        self.model.summary()
