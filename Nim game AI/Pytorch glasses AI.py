import random
import torch
import torch.nn as nn
from torch import optim
from collections import deque


class NimNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.net(x)


def get_action(model, sticks, epsilon=0.1):
    valid = [1] if sticks == 1 else [1, 2]  # допустимые ходы

    if random.random() < epsilon:
        return random.choice(valid)

    sticks_t = torch.FloatTensor([[sticks]])
    q_values = model(sticks_t)[0]

    # выбираем лучший из ДОПУСТИМЫХ ходов
    best = max(valid, key=lambda a: q_values[a - 1].item())
    return best

def train_step(model, optimizer, loss_fn, sticks, action, reward):
    sticks = torch.FloatTensor([[sticks]])
    q_values = model(sticks)
    target = q_values.clone().detach()
    target[0][action - 1] = reward
    loss = loss_fn(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# def play_episode(model):
#     history = []
#     sticks = 11
#
#     while sticks > 0:
#         # ход сети
#         ...
#         # считаем награду для сети
#         ...
#         # ход противника (случайный)
#         ...
#
#     return history

def play_episode(model):
    history = []
    sticks = 11
    curr_player = random.choice([0, 1])
    reward = 0
    while sticks > 0:
        action = 0
        match curr_player:
            case 0:
                action = get_action(model, sticks)
                sticks -= action

                # считаем награду сразу после хода
                if sticks == 0:
                    history.append((sticks + action, action, -1))
                    break
                elif sticks % 3 == 0:
                    reward = 1  # оставил кратное 3 — хорошо
                else:
                    reward = -0.1  # нейтральный ход
                curr_player = 1 - curr_player

                history.append((sticks + action, action, reward))  # sticks до хода
            case 1:
                enemy_action = random.choice([1, 2] if sticks > 1 else [1])
                sticks -= enemy_action
                curr_player = 1 - curr_player

                if sticks <= 0 and history:
                    last_sticks, last_action, _ = history[-1]  # берём последний ход сети
                    history[-1] = (last_sticks, last_action, 1)  # меняем награду на +1
                    break

    return history


def train(model, optimizer, fn_loss, epochs, batch_size=32):
    for epoch in range(epochs):
        history = play_episode(model)

        # ШАГ 1: сохраняем всю партию в память
        # reward = -1
        for sticks, action, reward in history:
            memory.append((sticks, action, reward))

        # ШАГ 2: учимся на случайной выборке из памяти
        if len(memory) < batch_size:
            continue  # пропускаем пока памяти мало

        batch = random.sample(memory, batch_size)
        for sticks, action, reward in batch:
            train_step(model, optimizer, fn_loss, sticks, action, reward)


model = NimNet()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
memory = deque(maxlen=10000)  # хранит последние 10000 ходов

train(model, optimizer, loss_fn, epochs=10000)

for sticks in range(1, 12):
    action = get_action(model, sticks)
    print(f"Палочек: {sticks} → сеть берёт: {action}")


