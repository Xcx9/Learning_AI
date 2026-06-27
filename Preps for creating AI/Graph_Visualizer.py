import matplotlib.pyplot as plt

class InteractiveVisualizer:
    def __init__(self, update_every: int = 100):
        self.update_every = update_every
        self.losses = []
        self.epochs = []

        plt.ion()  # интерактивный режим — график обновляется во время обучения
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.line, = self.ax.plot([], [], color='steelblue')

    def update(self, epoch: int, loss: float):
        self.losses.append(loss)
        self.epochs.append(epoch)

        if epoch % self.update_every == 0:
            self.line.set_data(self.epochs, self.losses)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def finish(self):
        plt.ioff()
        plt.show()
