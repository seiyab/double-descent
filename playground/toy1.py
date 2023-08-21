from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Self

import numpy as np
from scipy import linalg


def main():
    np.random.seed(0)
    a, b = np.random.randn(2)
    f = Quad(a, b, 0)
    x = np.random.uniform(size=20)
    y = f(x) + np.random.randn(20) * 0.2
    r = LinearRegressor.fit(x, y, PolynomialBasis(2))
    print(f(x))
    print(y)
    print(r(x))
    ...

    # pred = a.T @ z
    # diff = (a.T @ z).T - y
    # loss = ((a.T @ z).T - y).T @ ((a.T @ z).T - y)
    # loss = (z.T @ a - y).T @ (z.T @ a - y)
    # dlda = 2 * z @ (z.T @ a - y)
    # z @ y = z @ z.T @ a
    # (z @ z.T).inv @ z @ y = a


class Func(ABC):
    @abstractmethod
    def __call__(self, x):
        ...


@dataclass
class Quad(Func):
    a: float
    b: float
    c: float

    def __call__(self, x):
        return self.a * (x**2) + self.b * x + self.c


class Basis(ABC):
    @abstractmethod
    def __call__(self, x) -> np.ndarray:
        ...

    @abstractmethod
    def dim(self) -> int:
        ...


@dataclass
class PolynomialBasis(Basis):
    d: int

    def __call__(self, x) -> np.ndarray:
        return np.array([x**k for k in range(0, self.d + 1)])

    def dim(self) -> int:
        return self.d


@dataclass
class LinearRegressor:
    w: np.ndarray
    b: Basis

    @staticmethod
    def fit(x: np.ndarray, y: np.ndarray, b: Basis) -> Self:
        feat = np.array([b(x_i) for x_i in x])
        w = linalg.inv(feat.T @ feat) @ feat.T @ y
        return LinearRegressor(w=w, b=b)

    def __call__(self, x: np.ndarray) -> float:
        return self.w @ self.b(x)


main()
