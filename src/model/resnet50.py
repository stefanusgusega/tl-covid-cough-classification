from typing import Tuple
import tensorflow as tf
from keras.applications.resnet import ResNet50
from keras.models import Sequential
import keras

from model.base import BaseModel

tf.random.set_seed(42)


class ResNet50Model(BaseModel):
    def __init__(
        self,
        input_shape: Tuple,
        include_resnet_top: bool = False,
        initial_weights=None,
    ) -> None:
        self.include_resnet_top = include_resnet_top
        self.initial_weights = initial_weights

    def build_model(self):
        self.model = Sequential()

        resnet50 = ResNet50(
            include_top=self.include_resnet_top,
            weights=self.initial_weights,
            input_shape=self.input_shape,
        )
        resnet50_out = resnet50.output
        resnet50_out = keras.layers.Flatten()(resnet50_out)

        self.model.add(resnet50_out)

    def fit(self, datas, labels):
        self.model.fit(datas, labels)
