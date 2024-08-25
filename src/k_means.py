import numpy as np
import matplotlib.pyplot as plt
from datagenerator import AbstractDataGenerator


class UniformDataGenerator(AbstractDataGenerator):
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


class KNormalPacketsDataGenerator(UniformDataGenerator):
    def __init__(self, sigma=1, n=200, dim=2, K=4):
        super().__init__(sigma=sigma)
        self.K = K

    def generate_x(self):
        centroids = np.random.uniform(low=-10, high=10, size=(self.K, self.dim))
        total = self.n
        point_list = []
        for k in range(self.K):
            nb_of_points = total // self.K
            total += nb_of_points
            points = np.random.normal(
                scale=self.sigma, loc=centroids[k], size=(nb_of_points, self.dim)
            )
            point_list.append(points)
        self.x = np.concat(point_list, axis=0)
        return self.x


class Kmeans:
    def __init__(self, x, K=5):
        self.x = x
        self.n = len(self.x)
        self.d = len(self.x[0])
        self.K = K

    def get_centroids(self, x, classes):
        centroids = []
        for i_class in range(self.K):
            i_x = np.where(classes == i_class)[0]
            res_x = x[i_x]
            x_k_mean = np.mean(res_x, axis=0)
            centroids.append(x_k_mean)
        return centroids

    def run_clustering(self, n_repetitions=1):
        # affect random classes to each data point
        classes = np.random.randint(low=0, high=self.K, size=(self.n, 1))
        x = np.copy(self.x)

        for k in range(n_repetitions):
            # get centroids
            centroids = self.get_centroids(x=x, classes=classes)

            # get the nearest centroid for each point
            big_x = np.concat([x for _ in range(self.K)], axis=0)
            big_centroid_list = []
            for i in range(self.K):
                centroidi = tuple([centroids[i] for _ in range(self.n)])
                centroidi = np.stack(centroidi, axis=0)
                big_centroid_list.append(centroidi)
            big_centroids = np.concat(big_centroid_list, axis=0)
            norms = np.sum(np.power((big_x - big_centroids), 2), axis=1).reshape(
                (-1, 1)
            )
            norm1 = [norms[k * self.n : (k + 1) * self.n] for k in range(self.K)]
            norms = np.concat(norm1, axis=1)
            min_norms = np.min(norms, axis=1).reshape((-1, 1))

            # affect new classes to each point
            new_classes = np.where(norms == min_norms)[1].reshape((-1, 1))
            classes = np.copy(new_classes)

        self.classes = classes
        return classes

    def show_plot(self):
        fig, ax = plt.subplots()
        centroids = []
        for i_class in range(self.K):
            i_x = np.where(self.classes == i_class)[0]
            res_x = self.x[i_x]
            ax.scatter(res_x[:, 0], res_x[:, 1])
        plt.show()


def run_uniform_clustering():
    gen = UniformDataGenerator(n=1000)
    gen.generate_data()
    # gen.show_data()
    model = Kmeans(x=gen.x, K=5)
    model.run_clustering(n_repetitions=5)
    model.show_plot()


def run_normal_clustering():
    gen = KNormalPacketsDataGenerator(n=1000, K=5, sigma=3)
    gen.generate_data()
    # gen.show_data()
    model = Kmeans(x=gen.x, K=gen.K)
    model.run_clustering(n_repetitions=5)
    model.show_plot()


if __name__ == "__main__":
    # run_uniform_clustering()
    run_normal_clustering()
