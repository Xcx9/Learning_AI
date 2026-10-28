from Layer import *
from Activations import *
from Loss import *
from Graph_Visualizer import *
from Optimizer import *

class Network:
    def __init__(self, layers: list[Layer], loss: Loss = MSE(), show_graph: bool = False):
        self.loss = loss
        self.layers = layers
        self.activations = []
        self.visualizer = InteractiveVisualizer(500) if show_graph else None

    def backward(self, data: list, epochs: int = 1):
        """
        Function for learning Network
        :param epochs -> num of iterations for learning
        :param data: [(x1, x2, ...), target]
        :return: Nothing
        """

        for epoch in range(epochs):
            total_loss = 0
            for x, target in data:
                prediction = self.forward(x)
                total_loss += self.loss.loss(target, prediction[0])
                grad = self.loss.derivative(target, prediction[0])
                for layer in self.layers[::-1]:
                        grad = layer.backward(grad)
            avg_loss = total_loss / len(data)

            if self.visualizer:
                self.visualizer.update(epoch, avg_loss)

        if self.visualizer:
            self.visualizer.finish()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer.forward(x)
        x = self.layers[-1].forward(x)
        return x



