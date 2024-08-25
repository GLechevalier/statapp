import numpy as np
import matplotlib.pyplot as plt
from datagenerator import AbstractDataGenerator


class DataGenerator(AbstractDataGenerator):
    def __init__(self, sigma, h_dim=3, l_dim=2):
        super().__init__(sigma=sigma)
        self.h_dim = h_dim
        self.l_dim = l_dim
        return
