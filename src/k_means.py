import numpy as np
import matplotlib.pyplot as plt
from datagenerator import AbstractDataGenerator


class DataGenerator(AbstractDataGenerator):
    def __init__(self, sigma, n=200, dim=2):
        super().__init__(sigma=sigma)
        self.n = n
        self.dim = 2

    def generate_x(self):
        self.x = np.random.uniform(low=-10, high=10, size=(self.n, self.dim))
        return self.x
