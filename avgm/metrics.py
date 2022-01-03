from typing import Literal, Optional, Union

import torch
from typing_extensions import TypeAlias

InputType: TypeAlias = Union[Literal["scores"], Literal["classes"]]

class Accuracy(object):
    def __init__(self, input_type: Optional[InputType] = "scores"):
        assert input_type in ("scores", "classes")
        self.input_type = input_type
        self.count = 0
        self.correct = 0

    def update(self, y_hat: torch.Tensor, y: torch.Tensor) -> float:
        if self.input_type == "scores":
            y_hat = y_hat.argmax(axis=1)
        for pred, actual in zip(y_hat, y):
            pred, actual = pred.item(), actual.item()
            self.count += 1
            self.correct += 1 if pred == actual else 0
        return ("acc", self.correct / self.count)

    def reset(self) -> None:
        self.count = 0
        self.correct = 0


class AccuracyThresshold(object):
    def __init__(self, thresshold, input_type: Optional[InputType] = "scores"):
        assert input_type in ("scores", "classes")
        self.input_type = input_type
        self.count = 0
        self.correct = 0
        self.thresshold = thresshold

    def update(self, y_hat: torch.Tensor, y: torch.Tensor):
        if self.input_type == "scores":
            y_hat = y_hat.argmax(axis=1)
        for pred, actual in zip(y_hat, y):
            pred, actual = pred.item(), actual.item()
            self.count += 1
            self.correct += 1 if abs(pred - actual) <= self.thresshold else 0
        return "acc{0}".format(self.thresshold), self.correct / self.count

    def reset(self) -> None:
        self.count = 0
        self.correct = 0


class MAE(object):
    def __init__(self, input_type: Optional[InputType] = "scores"):
        assert input_type in ("scores", "classes")
        self.input_type = input_type
        self.count = 0
        self.sum_error = 0

    def update(self, y_hat: torch.Tensor, y: torch.Tensor):
        if self.input_type == "scores":
            y_hat = y_hat.argmax(axis=1)
        for pred, actual in zip(y_hat, y):
            pred, actual = pred.item(), actual.item()
            self.count += 1
            self.sum_error += abs(pred - actual)
        return "MAE", self.sum_error / self.count

    def reset(self) -> None:
        self.count = 0
        self.sum_error = 0
