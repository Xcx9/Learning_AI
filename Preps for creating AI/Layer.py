from Neuron import *

class Layer:
    def __init__(self, neurons:int = None):
        self.last_output = []
        self.neurons = [Neuron() for _ in range(neurons)] or [Neuron()]

    def train(self, data, lr, epochs):
        return [neuron.learn(data, lr, epochs) for neuron in self.neurons]

    def backwards(self, ans, grad_output, lr: float = 0.001):
        grad_input = []
        # dL_dw = grad_output * self.last_input[neuron]
        for neuron in range(len(self.neurons)):
            grad_input.append(self.neurons[neuron].backwards(grad_output, lr)) # Слой просит каждый нейрон просчитать градиенты и веса
        output = []
        for coord in range(len(grad_input[0])):
            coord_sum = 0
            for i in range(len(grad_input)):
                coord_sum += grad_input[i][coord]
            output.append(coord_sum)
        return output


    def forward(self, x):
        self.last_output = [neuron.think(x) for neuron in self.neurons] # z
        return self.last_output
