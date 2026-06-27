from Neuron import *
from Activations import *

class Layer:
    def __init__(self, neurons:int = 1, activation: Activation | None = None):
        self.last_output = []
        self.activation = activation
        self.neurons = [Neuron() for _ in range(neurons)] or [Neuron()]

    def backward(self, grad_output, lr: float = 0.001):
        # dL_dw = grad_output * self.last_input[neuron]
        if isinstance(grad_output, (float, int)):
            grad_output = [grad_output] * len(self.neurons)  # ← раздаём каждому нейрону

        grad_input = []
        for neuron, grad in zip(self.neurons, grad_output):
            grad_input.append(neuron.backward(grad, lr))
        output = []
        for coord in range(len(grad_input[0])):
            coord_sum = 0
            for i in range(len(grad_input)):
                coord_sum += grad_input[i][coord]
            output.append(coord_sum)
        return self.activation.backward(output) if self.activation is not None else output

    def forward(self, x):
        self.last_output = [neuron.forward(x) for neuron in self.neurons] # z
        return self.activation.forward(self.last_output) if self.activation is not None else self.last_output
