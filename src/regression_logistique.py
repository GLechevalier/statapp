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
        self.n = len(self.x)
        self.dim = len(x[0]) - 1

    def loss(self, y, y_prime):
        indices = np.where(y != y_prime)
        return len(indices[0])

    def loss_beta(self, beta_tild):
        y = self.y.reshape((-1,))
        y_prime = (self.x @ beta_tild).reshape((-1,))
        return self.loss(y=y, y_prime=y_prime)

    def likelihood(self, beta_tild):
        # produit (i 1->n) pi^yi*(1-pi)^(1-yi)
        # pi = sigmoid(beta_tild*x)
        x = self.x @ beta_tild
        px = self.sigmoid(x)
        y = self.y
        px_pow_y = np.power(px, y)
        one_minus_y = np.ones((self.n, 1)) - y
        one_minus_px = np.ones((self.n, 1)) - px
        one_minus_px_pow_one_minus_y = np.power(one_minus_px, one_minus_y)
        before_concat = np.multiply(px_pow_y, one_minus_px_pow_one_minus_y)
        return np.prod(before_concat)

    def sigmoid(self, x):
        ex = np.exp(x)
        ex1 = np.ones((len(x), 1)) + ex
        return np.divide(ex, ex1)

    def der_sigmoid(self, x):
        ex1 = np.ones((len(x), 1)) + np.exp(x)
        return np.divide(self.sigmoid(x), ex1)

    def log_likelihood(self, beta_tild):
        x = self.x @ beta_tild
        ex = np.exp(x)
        ex1 = np.ones((self.n, 1)) + ex
        lnex1 = np.log(ex1)
        yx = np.multiply(self.y, x)
        before_concat = yx - lnex1
        return np.sum(before_concat)

    def grad_log_likelihood(self, beta_tild):
        x = self.x
        xbeta = x @ beta_tild
        xy = np.multiply(x, self.y)
        ex = self.sigmoid(xbeta)
        lnder = np.multiply(x, ex)
        before_concat = (xy - lnder).transpose()
        return np.sum(before_concat, axis=1).reshape((self.dim + 1, 1))

    def hess_log_likelihood(self, beta_tild):
        xbeta = self.x @ beta_tild
        sigmoid_der_xbeta = self.der_sigmoid(xbeta)
        base = np.multiply(self.x, sigmoid_der_xbeta)
        dim_list = []
        for i in range(len(self.x[0])):
            dimi = np.multiply(self.x[:, i].reshape((self.n, 1)), base)
            dim_list.append(dimi)
        before_concat = np.stack(dim_list, axis=2)
        return np.sum(before_concat, axis=0)

    def compute_step_vector(self, beta_tild):
        grad_log = self.grad_log_likelihood(beta_tild=beta_tild)
        hess_log = self.hess_log_likelihood(beta_tild=beta_tild)
        hess_inv = np.linalg.inv(hess_log)
        return hess_inv @ grad_log


if __name__ == "__main__":
    gen = DataGenerator(dimension=2, sigma=1)
    gen.generate_data()
    # gen.show_data()
    true_beta_tild = gen.beta_tild
    model = LogisticRegression(x=gen.x, y=gen.y)
    beta_one = np.zeros((3, 1))
    print(model.log_likelihood(beta_tild=true_beta_tild))
    print(model.log_likelihood(beta_tild=beta_one))
    print(model.compute_step_vector(beta_tild=true_beta_tild))
    print(model.compute_step_vector(beta_tild=beta_one))
