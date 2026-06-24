import random

class Neuron:
    def __init__(self):
        self.w = random.uniform(-0.25, 0.25)
        self.b = 0

    def think(self, x: float|int) -> float|int:
        """
        The Thinking Function of a Neuron
        Response Based on Trained Data
        """
        return self.w * x + self.b

    def mse(self, y, ans):
        """
        Function for Calculating the Error
        """
        return (y - ans) ** 2

    # Обучение
    def learn(self, dataset: list[tuple[float, float]], lr: float, epochs: int):
        """
        Training function: the neural network searches for patterns in the database
        dataset -> [x, ans]
        x - input parameter
        ans - expected value
        lr - learning_rate (step size, i.e., how quickly we converge)
        epochs - number of training iterations
        """
        for epoch in range(epochs):
            loss = 0
            for x, ans in dataset:
                dw = 2 * x * (self.w * x + self.b - ans) # dL / dw (How quickly does L change when w changes)
                db = 2 * (self.w * x + self.b - ans) # dL / db (How quickly does L change when b changes)
                self.w -= lr * dw # weight update
                self.b -= lr * db # bias update
                loss += self.mse(self.think(x), ans)
            print('loss', loss / len(dataset))
        return f'w: {self.w}\nb: {self.b}'

data = [(x, x*2-5) for x in range(-50, 50)]

random.shuffle(data)
print(data)

n1 = Neuron()

print(n1.learn(data, 0.00001, 10000))

print(n1.think(6))
