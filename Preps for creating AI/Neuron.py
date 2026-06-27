import random

class Neuron:
    def __init__(self):
        self.last_z = 0
        self.last_input = []
        self.weights = []
        self.b = 0

    def think(self, x: list[float|int]) -> float:
        """
        The Thinking Function of a Neuron
        Response Based on Trained Data
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

    def mse(self, y, ans):
        """
        Function for Calculating the Error
        """
        return (y - ans) ** 2

    def backwards(self, grad_output: float, lr) -> list:
        grad_input = []
        for i in range(len(self.weights)):
            grad_input.append(grad_output * self.weights[i])
            self.weights[i] -= lr * grad_output * self.last_input[i]
        self.b -= lr * grad_output
        return grad_input
