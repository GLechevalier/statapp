import matplotlib.pyplot as plt
import numpy as np


class DataGenerator:
    def __init__(self, p, n, sigma):
        self.p = p
        self.n = n
        self.sigma = sigma
        self.beta = self.init_beta()

    def init_beta(self):
        self.beta = np.random.uniform(low=-10, high=10, size=(self.p + 1, 1))
        return self.beta

    def generate_x(self, min_x=-100, max_x=100):
        x = np.random.uniform(low=min_x, high=max_x, size=(self.n, 1))
        self.x = x
        return self.x

    def get_X(self, p_model=None, x=None):
        if x == None:
            x = self.x
        if p_model == None:
            p_model = self.p
        X = np.ones(shape=(self.n, 1))
        old_value = X
        for i in range(p_model):
            new_value = old_value * x
            X = np.concat((X, new_value), axis=1)
            old_value = new_value
        self.X = X
        return X

    def generate_y(self, X=None):
        if X == None:
            X = self.get_X()
        Y = X @ self.beta + np.random.normal(scale=self.sigma**2, size=(self.n, 1))
        self.Y = Y
        return Y

    def generate_data(self):
        self.generate_x()
        self.generate_y()

    def show_data(self):
        plt.scatter(self.x, self.Y)
        plt.show()


class LinearRegression:
    def __init__(self, x: np.array, y: np.array):
        self.x = x  # size (n,1)
        self.y = y  # size (n,1)

    def get_X(self, p_model):
        self.X = DataGenerator(p=1, n=1, sigma=0).get_X(p_model=p_model, x=self.x)
        return self.X

    def calculate_beta(self):
        Xt = np.transpose(self.X)
        XtX = Xt * self.X
        beta_hat = np.linalg.inv(a=XtX) * Xt * self.y
        return beta_hat

    def infer_y_hat(self):
        y_hat = 0


if __name__ == "__main__":
    p = 1
    n = 100
    sigma = 10
    gen = DataGenerator(p=p, n=n, sigma=sigma)
    gen.generate_data()
    gen.show_data()
