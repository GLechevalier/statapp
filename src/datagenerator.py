class AbstractDataGenerator:
    def __init__(self, sigma) -> None:
        self.sigma = sigma

    def generate_x(self):
        raise NotImplementedError()

    def generate_data(self):
        raise NotImplementedError()

    def show_data(self):
        raise NotImplementedError()
