import random

class Neuron:
    def __init__(self):
        self.w = 0
        self.x = 0
        self.b = 0

    def create(self, x, bias):
        self.w = random.uniform(-0.25, 0.25)
        self.x = x
        self.b = bias

    def answer(self):
        return self.w * self.x + self.b


