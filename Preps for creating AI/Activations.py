class Activation:
    def forward(self, x):
        pass

    def backward(self, grad_output):
        pass

    def derivative(self, x):
        pass


class ReLu(Activation):
    def __init__(self):
        self.last_z = []

    def derivative(self, x: float) :
        return 1 if x > 0 else 0

    def forward(self, x: int | float | list[float]):
        self.last_z = x
        if isinstance(x, (int, float)):
            return max(0, x)
        if isinstance(x, list):
            return [max(0, x) for x in x]
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
        self.alpha = alpha
        self.last_z = []

    def derivative(self, x: float):
        return 1 if x > 0 else self.alpha  # ← не 0, а маленькое число

    def forward(self, x):
        self.last_z = x
        if isinstance(x, (int, float)):
            return x if x > 0 else self.alpha * x
        if isinstance(x, list):
            return [v if v > 0 else self.alpha * v for v in x]

    def backward(self, grad_output):
        if isinstance(self.last_z, (int, float)):
            return self.derivative(self.last_z) * grad_output
        if isinstance(self.last_z, list):
            return [self.derivative(z) * g for z, g in zip(self.last_z, grad_output)]

