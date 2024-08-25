import numpy as np
import matplotlib.pyplot as plt
from datagenerator import AbstractDataGenerator


class DataGenerator(AbstractDataGenerator):
    def __init__(self, sigma=1, n=200, dim=2):
        super().__init__(sigma=sigma)
        self.n = n
        self.dim = 2

    def generate_x(self):
        self.x = np.random.uniform(low=-10, high=10, size=(self.n, self.dim))
        return self.x

    def generate_data(self):
        self.generate_x()

    def show_data(self, show_plot=True):
        fig, ax = plt.subplots()
        ax.scatter(self.x[:, 0], self.x[:, 1])
        plt.show()


class Kmeans:
    def __init__(self, x, K=5):
        self.x = x
        self.n = len(self.x)
        self.d = len(self.x[0])

    def run_clustering(self, n_repetitions=10):
        # affect random classes to each data point

        # get centroids
        # get the nearest centroid for each point
        # affect new classes to each point
        pass


if __name__ == "__main__":
    gen = DataGenerator()
    gen.generate_data()
    gen.show_data()
    model = Kmeans(x=gen.x)
