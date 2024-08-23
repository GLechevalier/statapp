import matplotlib.pyplot as plt
import numpy as np


class DataGenerator:
    def __init__(self, dimension=2, n_points=100, sigma=1, beta_tild=None, delta0=0.5):
        self.dim = dimension
        self.n = n_points
        self.sigma = sigma
        if not beta_tild:
            self.beta_tild = self.init_beta_tild()
        self.delta0 = delta0

    def logit(self, x):
        return np.log(x / (1 - x))

    def init_beta_tild(self):
        self.beta_tild = np.random.uniform(low=-10, high=10, size=(self.dim + 1, 1))
        print(self.beta_tild)
        return self.beta_tild

    def generate_x(self, min_x=-10, max_x=10):
        x = np.random.uniform(low=min_x, high=max_x, size=(self.n, self.dim))
        self.x = np.concat((np.ones((self.n, 1)), x), axis=1)
        return self.x

    def generate_y(self):
        arr = self.x @ self.beta_tild + np.random.normal(
            0, scale=self.sigma, size=(self.n, 1)
        )
        self.y = np.where(arr > self.logit(self.delta0), 1, 0)
        return self.y

    def generate_data(self):
        self.generate_x()
        self.generate_y()

    def show_data(self):
        x = self.x
        y = self.y.transpose().reshape(self.n)
        indices1 = np.where(y == 1)
        indices0 = np.where(y == 0)
        x1 = x[indices1].transpose()
        x0 = x[indices0].transpose()
        plt.scatter(x0[1], x0[2], color="blue")
        plt.scatter(x1[1], x1[2], color="red")
        plt.show()


class LogisticRegression:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x  # size (n,1)
        self.y = y  # size (n,1)
        self.dim = len(x[0]) - 1

    def loss(self, y, y_prime):
        indices = np.where(y != y_prime)
        return len(indices[0])

    def loss_beta(self, beta_tild):
        y = self.y.reshape((-1,))
        y_prime = (self.x @ beta_tild).reshape((-1,))
        return self.loss(y=y, y_prime=y_prime)

    def loss_beta_p(self, beta_tild):
        y = self.y.reshape(-1)
        y_prime = (self.x @ beta_tild).reshape(-1)
        indices = np.where(y != y_prime)[0]
        new_x = self.x[:]
        new_x[indices, :] = 0
        result = np.sum(new_x, axis=0).reshape((-1, 1))
        print(result)
        return result


if __name__ == "__main__":
    gen = DataGenerator(dimension=2, sigma=1)
    gen.generate_data()
    # gen.show_data()
    true_beta_tild = gen.beta_tild
    model = LogisticRegression(x=gen.x, y=gen.y)
    model.loss_beta_p(beta_tild=np.zeros((3, 1)))
