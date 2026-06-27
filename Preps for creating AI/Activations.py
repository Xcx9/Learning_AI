class Activation:
    def forward(self, x):
        pass

    def derivative(self, x):
        pass


class ReLu(Activation):
    def __init__(self):
        pass

    def forward(self, x: int | float | list[float]):
        if isinstance(x, (int, float)):
            return max(0, x)
        if isinstance(x, list):
            return [max(0, x) for x in x]
        return None

    def derivative(self, x: float):
        return 1 if x > 0 else 0

