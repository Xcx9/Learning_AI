from math import log

class Loss:
    def __init__(self):
        pass

    def loss(self, target: float, prediction: float) -> float:
        raise NotImplementedError("Метод должен быть реализован в дочернем классе")

    def derivative(self, target, prediction: float) -> float:
        raise NotImplementedError("Метод должен быть реализован в дочернем классе")


class MSE(Loss):
    def loss(self, target: float, prediction: float) -> float:
        return (prediction - target) ** 2

    def derivative(self, target, prediction: float) -> float:
        return 2 * (prediction - target)

class CROSS_ENTROPY(Loss):
    def __init__(self):
        super().__init__()
        self.p = 0

    def loss(self, target, prediction: float) -> float:
        eps = 1e-12
        self.p = max(eps, min(1 - eps, prediction))
        return -(target * log(self.p) + (1 - target) * log(1 - self.p))

    def derivative(self, target, prediction: float) -> float:
        eps = 1e-12
        p = max(eps, min(1 - eps, prediction))
        return -(target/p) + ((1 - target)/(1 - p))
