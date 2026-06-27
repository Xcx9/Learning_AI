from Layer import Layer
from Activations import *

def relu(xs):
    return [max(0, x) for x in xs]


class Network:
    def __init__(self, layers: list[Layer], activation: Activation):
        self.layers = layers
        self.activation = activation

    def train(self, data):
        x = data
        for layer in self.layers:
            x = layer.train(x, 0.0001, 1000)

    def loss(self, y, ans):
        return (y - ans) ** 2

    def loss_derivative(self, y, ans):
        return (y - ans) * 2

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer.forward(x)
            x = self.activation.forward(x)
        x = self.layers[-1].forward(x)
        return x

l1 = Layer(4)
l2 = Layer(2)
l3 = Layer(1)

net = Network([l1, l2, l3], ReLu())

data_think = [[x] for x in range(-10, 10)]

print(net.forward([6]))
