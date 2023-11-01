import numpy as np
from typing import List

def simple_linear_regression(x: List[int], y: List[int]):
    m = len(x)
    w = 1.0
    b = 1.0
    alpha = 0.001
    max_steps = 10000
    min_stepsize = 0.000001

    for step in range(max_steps):
        w_derivative = 0.0
        b_derivative = 0.0

        for i in range(m):
            w_derivative += ((x[i] * w + b) - y[i]) * x[i]
            b_derivative += ((x[i] * w + b) - y[i])

        w_derivative /= m
        w_derivative *= alpha
        b_derivative /= m
        b_derivative *= alpha

        if abs(w_derivative) < min_stepsize and abs(b_derivative) < min_stepsize:
            break

        w -= w_derivative
        b -= b_derivative

    return w, b

# w = [[-1,-1,0],[0,1,2]]
def simple_linear_predict(x: List[int], w: float, b: float):
    y_new = []
    for i in range(len(x)):
        y_new.append(w * x[i] + b)
    return y_new

x = np.array([5, 15, 25, 35, 45, 55])
y = np.array([5, 15, 25, 35, 45, 55])
y = np.array([5, 20, 14, 32, 22, 38])

w, b = simple_linear_regression(x, y)

print(w, b)

y_new = simple_linear_predict(x, w, b)

print(y_new)
