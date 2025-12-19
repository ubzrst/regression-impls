from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

class Model:
    def __init__(
            self,
            data: NDArray[np.float64],
            targets: NDArray[np.float64],
            weights: NDArray[np.float64]
    ) -> None:
        self.data = data
        self.targets = targets
        self.weights = weights


    def predict(self) -> NDArray[np.float64]:
        return np.dot(self.data, self.weights)


    def calc_cost(self) -> float:
        # J = (1/n) * sum_(i=0)^(n) (y'_i - y_i)**2
        n: int = self.data.shape[0]
        squared_errors: NDArray[np.float64] = (self.predict() - self.targets)**2
        return np.sum(squared_errors)/(2*n)


    def calc_gradient(self) -> NDArray[np.float64]:
        # dJ/dw = (2/n) * sum_(i=0)^(n) (y'_i - y_i)(x_i_j)
        n: int = self.data.shape[0]
        error_product: NDArray[np.float64] = self.data * (self.predict() - self.targets)[:, np.newaxis]
        return np.sum(error_product, axis=0)/n
        

    def fit(self, alpha: float, epochs: int, repeat_every: int = 1) -> None:
        hist = [(self.calc_cost(), self.weights)]
        for i in range(epochs):
            self.weights = self.weights - alpha * self.calc_gradient()
            if i % repeat_every == 0 or i == 0 or i == epochs - 1:
                hist.append((self.calc_cost(), self.weights.copy()))
        self.hist = hist


data: NDArray[np.float64] = np.array([1,2,3,4,5,6,7,8,9,10])
targets: NDArray[np.float64] = np.array([25,35,45,55,65,75,85,95,105,115])
data = np.hstack((data[:, np.newaxis], np.ones((10,1), dtype=np.float64)))
weights: NDArray[np.float64] = np.array([0, 0])
model: Model = Model(data, targets, weights)
model.fit(1e-2, 1000, 10)
print(model.predict())
