import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


class DataAnimator:
    def __init__(self, data, num_classes, file_name):
        self.data = data
        self.num_classes = num_classes
        self.file_name = file_name

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True)
        self.ax.set_xlabel("x1")
        self.ax.set_ylabel("x2")

        self.pt = [self.ax.plot([], [], 'o') for _ in range(num_classes)]

    def init(self):
        for c in range(self.num_classes):
            self.pt[c].set_data([], [])

        return self.pt,

    def animate(self, i):
        # ax.set_title("Epoch %3d - Mean Acc: %.4f - Mean KLD: %.4f" % (i + 1, mean_met[i], mean_kld[i]))
        for c in range(self.num_classes):
            self.pt[c].set_data(self.data[i][c][:, 0], self.data[i][c][:, 1])

        return self.pt,

    def animate_data(self, n_frames, min_x=0, max_x=0, min_y=0, max_y=0):
        # ax.set_title("Epoch %3d - Mean Acc: %.4f - Mean KLD: %.4f" % (1, mean_met[0], mean_kld[0]))

        if min_x != 0 and max_x != 0 and min_y != 0 and max_y != 0:
            self.ax.set_xlim(1.1 * min_x, 1.1 * max_x)
            self.ax.set_ylim(1.1 * min_y, 1.1 * max_y)

        self.ax.legend([self.pt[k] for k in range(self.num_classes)],
                       ['class ' + str(k) for k in range(self.num_classes)], loc='upper left')

        ani = FuncAnimation(self.fig, self.animate, frames=n_frames, init_func=self.init,
                            interval=200, blit=True, repeat=False)

        ani.save(self.file_name, dpi=300, writer=PillowWriter(fps=10))
