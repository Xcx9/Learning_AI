from Network import *

data_learn = [[[1], 2], [[0], 0]]
data_think = [x for x in range(-5, 21)]

# for x in data_think:
#     data_learn.append([x, x*2])
#     data_learn.append([x, x * 2])

lr = 0.01

# l1 = Layer(16, activation=LeakyReLU(), optimizer=SGD(lr))
# l2 = Layer(16, LeakyReLU(), SGD(lr))
# l3 = Layer(8, ReLu(), SGD(lr))
l4 = Layer(1, activation=None)  # выход без активации

net = Network([l4])


# net.forward(data_think)
net.backward(data_learn, 3000)
print(net.layers[0].neurons[0].weights, net.layers[0].neurons[0].b)
# for x in data_think:
#     pred = net.forward([x / 10.0])[0] * 100.0
#     print(f"x={x}, pred={pred:.2f}, expected={x**2}")