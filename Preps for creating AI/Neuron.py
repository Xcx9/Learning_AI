import math
import random
from Optimizer import *

class Neuron:
    def __init__(self, initialization: str = "uniform", fan_out: int = 1):
        """Choose uniform, He normal, or Xavier normal weight initialization.

        fan_out is the number of neurons in this neuron's layer.
        Weights are initialized once, when the input size becomes known.
        """
        initialization = initialization.lower()
        if initialization not in ("uniform", "he", "xavier"):
            raise ValueError("initialization must be 'uniform', 'he', or 'xavier'")
        if not isinstance(fan_out, int) or isinstance(fan_out, bool) or fan_out < 1:
            raise ValueError("fan_out must be a positive integer")
        self.initialization = initialization
        self.fan_out = fan_out
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
        print(x)
        if len(self.weights) == 0:
            fan_in = len(x)
            if fan_in == 0:
                raise ValueError("At least one input is required")
            if self.initialization == "uniform":
                self.weights = [random.uniform(-0.25, 0.25) for _ in range(fan_in)]
            else:
                # He: Var(w) = 2 / fan_in; Xavier: Var(w) = 2 / (fan_in + fan_out).
                variance = 2 / (fan_in if self.initialization == "he" else fan_in + self.fan_out)
                std = math.sqrt(variance)
                self.weights = [random.gauss(0, std) for _ in range(fan_in)]
        if len(self.weights) == len(x):
            output = sum([self.weights[i] * x[i] for i in range(len(self.weights))]) + self.b
            self.last_z = output
            self.last_input = x
            return output
        elif len(self.weights) > len(x): raise ValueError("Not enough inputs")
        else: raise ValueError("Too much inputs")

    def backward(self, grad_output: float, optimizer: Optimizer) -> list:
        grad_input = [grad_output * weight for weight in self.weights]
        grad_b = grad_output
        grad_weights = [grad_output * self.last_input[i] for i in range(len(self.weights))]
        self.weights, self.b = optimizer.step(self.weights, self.b, grad_b, grad_weights)
        return grad_input
