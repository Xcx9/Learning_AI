from Layer import Layer
from Activations import *
from Loss import Loss
from Graph_Visualizer import InteractiveVisualizer

class Network:
    def __init__(self, layers: list[Layer], show_graph: bool = False):
        self.loss = Loss()
        self.layers = layers
        self.activations = []
        self.visualizer = InteractiveVisualizer(500) if show_graph else None

    def backward(self, data: list, lr: float = 0.0001, epochs: int = 1):
        """
        Function for learning Network
        :param epochs -> num of iterations for learning
        :param lr -> learning rate
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
                        grad = layer.backward(grad, lr)
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


data_learn = [[[x], x**2] for x in range(-10, 11)]
data_think = [[x] for x in range(-10, 10)]

l1 = Layer(16, activation=ReLu())
l2 = Layer(16, ReLu())
l3 = Layer(1, activation=None)  # выход без активации

net = Network([l1, l2, l3])


# net.forward(data_think)
net.backward(data_learn, 0.00001, 8000)
for x in [-10, -5, 0, 5, 10]:
    print(f"x={x}, pred={net.forward([x])}, expected={x**2}")
