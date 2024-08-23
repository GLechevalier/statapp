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

    def generate_x(self, min_x=-10, max_x=10):
        x = np.random.uniform(low=min_x, high=max_x, size=(self.n, 1))
        self.x = x
        return self.x

    def get_X(self, p_model=None, x=None):
        if not isinstance(x, np.ndarray):
            x = self.x
        if p_model == None:
            p_model = self.p
        X = np.ones(shape=(len(x), 1))
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
        x = self.x
        self.X = DataGenerator(p=1, n=1, sigma=0).get_X(p_model=p_model, x=x)
        return self.X

    def calculate_beta(self):
        Xt = np.transpose(self.X)
        XtX = Xt @ self.X
        beta_hat = np.linalg.inv(a=XtX) @ Xt @ self.y
        self.beta_hat = beta_hat
        return beta_hat

    def infer_y_hat(self):
        y_hat = self.X @ self.beta_hat
        self.y_hat = y_hat
        return y_hat

    def show_plot(self):
        unordered_x = self.x[:].reshape((-1))
        unordered_y = self.y[:].reshape((-1))
        unordered_y_hat = self.y_hat[:].reshape((-1))

        # Step 1: Get the indices that would sort array1
        sorted_indices = np.argsort(unordered_x)

        # Step 2: Use the sorted indices to sort both arrays
        ordered_x = unordered_x[sorted_indices]
        ordered_y = unordered_y[sorted_indices]
        ordered_y_hat = unordered_y_hat[sorted_indices]

        plt.scatter(ordered_x, ordered_y)
        plt.plot(ordered_x, ordered_y_hat, color="red")
        plt.show()

    def run(self, p_model):
        self.get_X(p_model=p_model)
        self.calculate_beta()
        self.infer_y_hat()
        self.show_plot()


if __name__ == "__main__":
    p = 7
    n = 100
    sigma = 2000
    gen = DataGenerator(p=p, n=n, sigma=sigma)
    gen.generate_data()
    model = LinearRegression(x=gen.x, y=gen.Y)
    model.run(p_model=p)
