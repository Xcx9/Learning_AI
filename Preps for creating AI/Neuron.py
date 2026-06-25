import random

class Neuron:
    def __init__(self):
        self.weights = []
        self.b = 0

    def think(self, x: list[float|int]) -> float:
        """
        The Thinking Function of a Neuron
        Response Based on Trained Data
        """
        output = 0
        if len(self.weights) == len(x):
            for i in range(len(x)):
                output += self.weights[i] * x[i]
            return output + self.b
        elif len(self.weights) > len(x): raise "Not enough inputs"
        else: raise "Too much inputs"


    def mse(self, y, ans):
        """
        Function for Calculating the Error
        """
        return (y - ans) ** 2

    def learn(self, dataset: list[tuple[list[float], float]], lr: float, epochs: int):
        """
        Training function: the neural network searches for patterns in the database
        dataset -> [([x1, x2, x3...], ans), ...]
        x_i - input parameter
        ans - expected value
        lr - learning_rate (step size, i.e., how quickly we converge)
        epochs - number of training iterations
        """
        self.weights = [random.uniform(-0.25, 0.25) for _ in range(len(dataset[0][0]))]

        for epoch in range(epochs):
            loss = 0
            for i in range(len(dataset)):
                dw = []
                x = dataset[i][0]
                ans = dataset[i][1]
                for x_i in range(len(self.weights)):
                    y = self.think(x)
                    dw.append(2 * x[x_i] * (y - ans)) # dL / dw (How quickly does L change when w_i changes)
                for x_i in range(len(x)):
                    self.weights[x_i] -= lr * dw[x_i] # weights update
                y = self.think(x)
                db = 2 * (y - ans)  # dL / db (How quickly does L change when b changes)
                self.b -= lr * db # bias update
                loss += self.mse(y, ans)
            print('loss', loss / len(dataset))
        return f'w: {self.weights}\nb: {self.b}'

data1 = [([x], x*2-5) for x in range(-50, 50)]
data2 = [
    ([1, 2], 2*1 + 3*2 + 5),
    ([2, 4], 2*2 + 3*4 + 5),
    ([3, 1], 2*3 + 3*1 + 5),
]

random.shuffle(data1)
random.shuffle(data2)

n1 = Neuron()
n2 = Neuron()

def generate_dataset(size: int):
    dataset = []

    for _ in range(size):
        x1 = random.randint(-50, 50)
        x2 = random.randint(-50, 50)
        x3 = random.randint(-50, 50)

        y = 2 * x1 + 3 * x2 - 4 * x3 + 7

        dataset.append(([x1, x2, x3], y))

    return dataset

data = generate_dataset(1000)

n3 = Neuron()
print(n3.learn(data, 0.0001, 1000))
