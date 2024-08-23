import matplotlib.pyplot as plt
import numpy as np


class DataGenerator:
    def __init__(self, p, n, sigma, beta=None):
        self.p = p
        self.n = n
        self.sigma = sigma
        if not beta:
            self.beta = self.init_beta()
        else:
            self.beta = beta

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
        Y = X @ self.beta + np.random.normal(scale=self.sigma, size=(self.n, 1))
        self.Y = Y
        return Y

    def generate_data(self):
        self.generate_x()
        self.generate_y()

    def show_data(self):
        plt.scatter(self.x, self.Y)
        plt.show()


class DataGeneratorDeterministicX(DataGenerator):
    def __init__(self, p, n, sigma, beta=None):
        super().__init__(p, n, sigma, beta)

    def generate_x(self, min_x=-10, max_x=10):
        self.x = np.array(
            [min_x + (max_x - min_x) * i / self.n for i in range(self.n)]
        ).reshape((-1, 1))
        return self.x


class LinearRegression:
    def __init__(self, x: np.array, y: np.array, p_model):
        self.x = x  # size (n,1)
        self.y = y  # size (n,1)
        self.p_model = p_model

    def get_X(self, x=None):
        if not isinstance(x, np.ndarray):
            x = self.x
        self.X = DataGenerator(p=1, n=1, sigma=0).get_X(p_model=self.p_model, x=x)
        return self.X

    def calculate_beta(self):
        Xt = np.transpose(self.X)
        XtX = Xt @ self.X
        self.XtX = XtX
        beta_hat = np.linalg.inv(a=XtX) @ Xt @ self.y
        self.beta_hat = beta_hat
        return beta_hat

    def infer_y_hat(self, x=None):
        if not isinstance(x, np.ndarray):
            x = self.x
        X = self.get_X(x=x)
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

    def run(self, show_plot=True):
        self.get_X()
        self.calculate_beta()
        self.infer_y_hat()
        if show_plot:
            self.show_plot()

    def get_risk(self, x=None, y=None):
        if not isinstance(x, np.ndarray):
            x = self.x
        y_hat = self.infer_y_hat(x=x)
        if not isinstance(y, np.ndarray):
            y = self.y
        n = len(x)

        return np.linalg.norm(y - y_hat) ** 2 / n

    def get_sigma_hat(self):
        self.sigma_hat = np.sqrt(self.get_risk())
        return self.sigma_hat

    def get_SCR(self, beta, x=None, y=None):
        default_beta = self.beta_hat
        self.beta_hat = beta
        result = self.get_risk(x=x, y=y)
        self.beta_hat = default_beta
        return result


class LoopTester:
    def __init__(self, beta=[[2], [1], [2]], sigma=1, n=1000):
        self.beta = beta
        self.sigma = sigma
        self.n = n
        self.p = len(self.beta) - 1

    def run_loop(self, n_times, show_output=True):
        beta_hat_list = []
        sigma_hat_list = []
        for i in range(n_times):
            gen = DataGeneratorDeterministicX(
                p=self.p, n=self.n, sigma=self.sigma, beta=self.beta
            )
            gen.generate_data()
            model = LinearRegression(x=gen.x, y=gen.Y, p_model=self.p)
            model.run(show_plot=False)
            beta_hat_list.append(model.beta_hat)
            sigma_hat_list.append(np.array([model.get_sigma_hat()]))

        XtX = model.XtX
        true_sigma_cov = self.sigma**2 * np.linalg.inv(XtX)

        sigma_hat_list = np.concat(sigma_hat_list)
        sigma_hat_mean = sigma_hat_list.mean()

        beta_hat_list = np.concat(beta_hat_list, axis=1)
        beta_hat_mean = beta_hat_list.mean(axis=1)
        beta_hat_cov = np.cov(beta_hat_list, bias=False)

        self.beta_hat_list = beta_hat_list
        self.beta_hat_mean = beta_hat_mean

        if show_output:
            print("expected beta = ", self.beta)
            print("beta hat mean = ", beta_hat_mean)
            print("expected variance = ", self.sigma)
            print("sigma hat mean = ", sigma_hat_mean)

            data = beta_hat_list.transpose()

            # Create histograms
            plt.figure(figsize=(5 * len(data[0]), 5))

            for i in range(len(data[0])):
                # Extract the i variables
                variable_i = data[:, i]
                # Histogram for the first variable
                plt.subplot(1, len(data[0]), i + 1)
                bin_number = n_times // 30
                # Create a histogram and extract the counts
                counts, bins = np.histogram(variable_i, bins=bin_number)
                # Find the maximum count in any bin
                max_count = np.max(counts)

                plt.hist(variable_i, bins=bin_number, color="blue", alpha=0.7)
                plt.plot([self.beta[i], self.beta[i]], [0, max_count], color="red")
                plt.title(f"Histogram of Variable {i+1}")
                plt.xlabel("Value")
                plt.ylabel("Frequency")

            # Show the plot
            plt.tight_layout()
            plt.show()


def run_regression():
    p = 1
    n = 100
    sigma = 1
    gen = DataGenerator(p=p, n=n, sigma=sigma, beta=[[2], [1]])
    gen.generate_data()
    model = LinearRegression(x=gen.x, y=gen.Y, p_model=p)
    model.run()
    print("risk = ", model.get_risk())
    print("true beta = ", gen.beta)
    print("beta hat = ", model.beta_hat)


def run_loop_tester(n):
    loop_tester = LoopTester(beta=[[1]], sigma=1)
    loop_tester.run_loop(n_times=n)


if __name__ == "__main__":
    # run_regression()
    run_loop_tester(n=5000)
