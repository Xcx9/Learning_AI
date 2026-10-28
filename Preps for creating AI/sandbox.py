from Network import *

data_learn = [
    [[0], 0],
    [[1], 1],
]

data_think = [x for x in range(-5, 21)]

# for x in data_think:
#     data_learn.append([x, x*2])
#     data_learn.append([x, x * 2])

lr = 0.1

# l1 = Layer(16, activation=LeakyReLU(), optimizer=SGD(lr))
# l2 = Layer(16, LeakyReLU(), SGD(lr))
# l3 = Layer(8, ReLu(), SGD(lr))
l4 = Layer(1, activation=Sigmoid(), optimizer=SGD(lr))  # выход без активации

net = Network([l4], loss=CROSS_ENTROPY())


# net.forward(data_think)
net.backward(data_learn, 1000)
# print(net.layers[0].neurons[0].weights, net.layers[0].neurons[0].b)
print("Вход 0:", net.forward([0]))
print("Вход 1:", net.forward([1]))
# for x in data_think:
#     pred = net.forward([x / 10.0])[0] * 100.0
#     print(f"x={x}, pred={pred:.2f}, expected={x**2}")