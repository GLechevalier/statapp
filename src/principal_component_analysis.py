import numpy as np
import matplotlib.pyplot as plt
from datagenerator import AbstractDataGenerator


class DataGenerator(AbstractDataGenerator):
    def __init__(self, n=1000, sigma=1, h_dim=3, l_dim=2):
        super().__init__(sigma=sigma)
        self.h_dim = h_dim
        self.l_dim = l_dim
        self.n = n

        self.hyperplane = self.generate_hyperplane()

    def generate_hyperplane(self):
        mu = np.random.uniform(low=-10, high=10, size=(self.h_dim, 1))
        A = np.random.uniform(low=-5, high=5, size=(self.h_dim, self.l_dim))
        return {"mu": mu, "A": A}

    def generate_x(self):
        z = np.random.uniform(low=-10, high=10, size=(self.l_dim, self.n))
        X = (
            self.hyperplane["A"] @ z
            + self.hyperplane["mu"]
            + np.random.normal(scale=self.sigma, size=(self.h_dim, self.n))
        )
        self.X = X
        return X

    def generate_data(self):
        self.generate_x()

    def show_data(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(self.X[0], self.X[1], self.X[2])
        plt.show()


def run_PCA():
    gen = DataGenerator(n=300, sigma=10)
    gen.generate_data()
    print(gen.X)
    gen.show_data()


if __name__ == "__main__":
    run_PCA()
