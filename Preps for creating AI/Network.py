from Layer import *
from Activations import *
from Loss import *
from Graph_Visualizer import *
from Optimizer import *

class Network:
    def __init__(self, layers: list[Layer], show_graph: bool = False):
        self.loss = Loss()
        self.layers = layers
        self.activations = []
        self.visualizer = InteractiveVisualizer(500) if show_graph else None

    def backward(self, data: list, epochs: int = 1):
        """
        Function for learning Network
        :param epochs -> num of iterations for learning
        :param data: [(x1, x2, ...), ans]
        :return: Nothing
        """

        for epoch in range(epochs):
            total_loss = 0
            for x, ans in data:
                prediction = self.forward(x)
                total_loss += self.loss.mse(prediction[0], ans)
                # print((x, ans), prediction, self.loss.mse_derivative(prediction[0], ans))
                grad = self.loss.mse_derivative(prediction[0], ans)
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



