from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
from scipy import linalg


def main():
    np.random.seed(0)
    a, b = np.random.randn(2)
    f = Quad(a, b, 0)
    x = np.random.uniform(size=20)
    y = f(x) + np.random.randn(20) * 0.01
    z = np.array([x**2, x, np.ones(x.shape)])
    w = linalg.inv(z @ z.T) @ z @ y
    p = Quad(*w)
    print(f(x))
    print(y)
    print()
    print(a, b)
    print(*w)
    print(z.shape)
    print(p(x))
    print(w.T @ z)
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


main()
