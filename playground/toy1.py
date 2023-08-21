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

    x_eval = np.random.uniform(size=20)
    y_eval = f(x_eval)

    for d in range(1, 21):
        r = LinearRegressor.fit(x, y, PolynomialBasis(d))
        print(
            loss(x, y, r),
            loss(x_eval, y_eval, r),
        )


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
        return self.d + 1


@dataclass
class LinearRegressor:
    w: np.ndarray
    b: Basis

    @staticmethod
    def fit(x: np.ndarray, y: np.ndarray, b: Basis) -> Self:
        w, *_ = linalg.lstsq(b(x).T, y)
        return LinearRegressor(w=w, b=b)

    def __call__(self, x: np.ndarray) -> float:
        return self.w @ self.b(x)


def loss(x: np.ndarray, y: np.ndarray, f) -> float:
    pred = f(x)
    diff = y - pred
    return diff @ diff.T


main()
