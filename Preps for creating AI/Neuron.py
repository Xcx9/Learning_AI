import random

class Neuron:
    def __init__(self):
        self.y = None
        self.w = random.uniform(-0.25, 0.25)
        self.x = 0
        self.b = 0

    def think(self, x, bias):
        self.x = x
        self.b = bias
        self.y = self.w * self.x + self.b
        return self.y

    def learn(self, ans):
        while abs(self.y - ans) > 0.2:
            if self.y > ans:
                self.w -= 0.1
            else:
                self.w += 0.1
            self.y = self.w * self.x + self.b
        while abs(self.y - ans) > 0.05:
            if self.y > ans:
                self.w -= 0.02
            else:
                self.w += 0.02
            self.y = self.w * self.x + self.b





