import math
from math import e, tanh


class Activation:
    def __init__(self):
        self.last_z = []

    def forward(self, z):
        pass

    def backward(self, grad_output):
        pass

    def derivative(self, z):
        pass


class ReLu(Activation):
    def __init__(self):
        super().__init__()
        self.last_z = []

    def derivative(self, z: float) :
        return 1 if z > 0 else 0

    def forward(self, z: int | float | list[float]):
        self.last_z = z
        if isinstance(z, (int, float)):
            return max(0, z)
        if isinstance(z, list):
            return [max(0, z) for z in z]
        return None

    def backward(self, grad_output):
        if isinstance(self.last_z, (int, float)):
            derivative = self.derivative(self.last_z)
            return derivative * grad_output
        if isinstance(self.last_z, list):
            return [
                self.derivative(z) * grad
                for z, grad in zip(self.last_z, grad_output)
            ]
        return None

# Добавь в Activations.py
class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.last_z = []

    def derivative(self, z: float):
        return 1 if z > 0 else self.alpha  # ← не 0, а маленькое число

    def forward(self, z):
        self.last_z = z
        if isinstance(z, (int, float)):
            return z if z > 0 else self.alpha * z
        if isinstance(z, list):
            return [v if v > 0 else self.alpha * v for v in z]

    def backward(self, grad_output):
        if isinstance(self.last_z, (int, float)):
            return self.derivative(self.last_z) * grad_output
        if isinstance(self.last_z, list):
            return [self.derivative(z) * g for z, g in zip(self.last_z, grad_output)]

class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
        self.last_z = []

    def sigma(self, z):
        return 1/(1 + e**(-z))

    def forward(self, z):
        self.last_z = z
        if isinstance(z, (int, float)):
            return self.sigma(z)
        if isinstance(z, list):
            return [self.sigma(v) for v in z]

    def backward(self, grad_output):
        if isinstance(self.last_z, (int, float)):
            return self.derivative(self.last_z) * grad_output
        if isinstance(self.last_z, list):
            return [self.derivative(z) * g for z, g in zip(self.last_z, grad_output)]

    def derivative(self, z):
        return self.sigma(z) * (1 - self.sigma(z))


class Tanh(Activation):
    def __init__(self):
        super().__init__()
        self.last_z = []

    def forward(self, z):
        self.last_z = z
        if isinstance(z, (int, float)):
            return tanh(z)
        if isinstance(z, list):
            return [tanh(v) for v in z]

    def backward(self, grad_output):
        if isinstance(self.last_z, (int, float)):
            return self.derivative(self.last_z) * grad_output
        if isinstance(self.last_z, list):
            return [self.derivative(z) * g for z, g in zip(self.last_z, grad_output)]

    def derivative(self, z):
        return 1 - tanh(z) ** 2
