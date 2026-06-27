import random
from Loss import *

class Neuron:
    def __init__(self):
        self.last_z = 0
        self.last_input = []
        self.weights = []
        self.b = 0

    def forward(self, x: list[float|int]) -> float:
        """
        The Thinking Function of a Neuron
        Response Based on Trained Data
        :param x: [x1,x2, ...]
        :returns output: float
        """
        output = 0
        if len(self.weights) == 0:
            self.weights = [random.uniform(-0.25, 0.25) for _ in range(len(x))]
        if len(self.weights) == len(x):
            output = sum([self.weights[i] * x[i] for i in range(len(self.weights))]) + self.b
            self.last_z = output
            self.last_input = x
            return output
        elif len(self.weights) > len(x): raise ValueError("Not enough inputs")
        else: raise ValueError("Too much inputs")

    def backward(self, grad_output: float, lr) -> list:
        grad_input = [grad_output * weight for weight in self.weights]
        self.weights = [self.weights[i] - lr * grad_output * self.last_input[i] for i in range(len(self.weights))] # Обновляем веса
        self.b -= lr * grad_output
        return grad_input
