class Loss:
    def __init__(self):
        pass

    def mse(self, y: float, ans: float) -> float:
        return (y - ans) ** 2

    def cross_enthropy(self):
        return 0

    def mse_derivative(self, y, ans):
        return 2 * (y - ans)