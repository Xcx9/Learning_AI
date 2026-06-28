class Optimizer:
    def __init__(self, lr:float):
        pass

    def step(self, weights, b, grad_b, grad_weights):
        pass

class SGD(Optimizer):
    def __init__(self, lr:float = 0.0001):
        super().__init__(lr)
        self.lr = lr

    def step(self, weights, b, grad_b, grad_weights):
        weights = [weights[i] - self.lr * grad_weights[i] for i in range(len(weights))] # Обновляем веса
        b -= self.lr * grad_b
        return weights, b


class Momentum(Optimizer):
    def __init__(self, lr:float = 0.0001):
        super().__init__(lr)
        self.lr = lr

    def step(self, weights, b, grad_b, grad_weights):
        weights = [weights[i] - self.lr * grad_weights[i] for i in range(len(weights))]
        b -= self.lr * grad_b
        return weights, b


class Adam(Optimizer):
    def __init__(self, lr:float = 0.0001):
        super().__init__(lr)
        self.lr = lr

    def step(self, weights, b, grad_b, grad_weights):
        weights = [weights[i] - self.lr * grad_weights[i] for i in range(len(weights))]
        b -= self.lr * grad_b
        return weights, b
