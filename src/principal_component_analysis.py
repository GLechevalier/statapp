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

    def set_X(self, X):
        self.X = X

    def show_data(self, show_plot=True):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(self.X[0], self.X[1], self.X[2])
        if show_plot:
            plt.show()
        return fig, ax


class PCA:
    def __init__(self, X):
        self.X = X

    def compute_mean(self):
        return np.mean(self.X, axis=1).reshape((len(self.X), 1))

    def run_svd(self, Xcentered, q):
        U, S, Vt = np.linalg.svd(Xcentered)
        # Sort the singular values and get the sorting indices
        sorted_indices = np.argsort(S)[::-1]

        # Sort the singular values in descending order
        S_sorted = S[sorted_indices]

        # Reorder the columns of U and rows of Vt
        U_sorted = U[:, sorted_indices]
        Vt_sorted = Vt[sorted_indices, :]
        Vsorted = Vt_sorted.transpose()
        return Vsorted[:, :q]

    def find_hyperplane(self):
        mean = self.compute_mean()
        self.mean = mean
        Xcent = (self.X - mean).transpose()
        Vq = self.run_svd(Xcentered=Xcent, q=2)
        self.Vq = Vq

    def reduce_data(self):
        z = self.Vq.transpose() @ (self.X - self.mean)
        x_hat = self.mean + self.Vq @ z
        self.x_hat = x_hat
        return x_hat

    def show_plot(self):
        temp = DataGenerator()
        temp.set_X(self.X)
        fig, ax = temp.show_data(show_plot=False)
        x = np.linspace(-100, 100, 2)
        y = np.linspace(-100, 100, 2)
        X, Y = np.meshgrid(x, y)
        X = X.reshape((-1, 1))
        Y = Y.reshape((-1, 1))
        XY = np.concat((X, Y), axis=1)
        X_hat = self.mean + self.Vq @ XY.transpose()
        ax.plot_surface(
            X=X_hat[0].reshape((2, -1)),
            Y=X_hat[1].reshape((2, -1)),
            Z=X_hat[2].reshape((2, -1)),
            color="green",
            alpha=0.5,
        )
        plt.show()


def run_PCA():
    gen = DataGenerator(n=300, sigma=5)
    gen.generate_data()
    # gen.show_data()
    model = PCA(X=gen.X)
    model.find_hyperplane()
    model.reduce_data()
    model.show_plot()


if __name__ == "__main__":
    run_PCA()
