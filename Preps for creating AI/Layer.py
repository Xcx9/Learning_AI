from Neuron import *
from Activations import *

class Layer:
    def __init__(self, neurons:int = 1, activation: Activation | None = None, optimizer: Optimizer = SGD(), initialization: str = "uniform"):
        self.optimizer = optimizer
        self.last_output = []
        self.activation = activation
        fan_out = len(range(neurons)) or 1
        self.neurons = [Neuron(initialization=initialization, fan_out=fan_out) for _ in range(fan_out)]

    def backward(self, grad_output):
        # dL_dw = grad_output * self.last_input[neuron]
        if isinstance(grad_output, (float, int)):
            grad_output = [grad_output] * len(self.neurons)  # раздаём каждому нейрону

        grad_input = []
        if len(self.neurons) == len(grad_output):
            if self.activation is not None:
                grad_output = self.activation.backward(grad_output)
            for neuron, grad in zip(self.neurons, grad_output):
                grad_input.append(neuron.backward(grad, self.optimizer))
        else:
            raise ValueError("The number of neurons does not match the number of gradients.")
        output = []
        for coord in range(len(grad_input[0])):
            coord_sum = 0
            for i in range(len(grad_input)):
                coord_sum += grad_input[i][coord]
            output.append(coord_sum)
        return output

    def forward(self, x: list[float | int]):
        self.last_output = [neuron.forward(x) for neuron in self.neurons] # z
        return self.activation.forward(self.last_output) if self.activation is not None else self.last_output
