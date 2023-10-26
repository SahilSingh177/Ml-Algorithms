import sympy
from typing import Any


class GradientDescent:
    def __init__(self, func: Any, start_point: int, end_point: int):
        self.func = func
        self.x_derivative = func.diff(x)
        self.y_derivative = func.diff(y)
        self.start_point = start_point
        self.end_point = end_point

    def findMin(self):
        # Set learning factor
        alpha = 0.1
        # Set maxsteps
        max_steps = 10000

        a = self.start_point
        b = self.end_point
        # Loop till you find min or maxSteps is exhausted
        step = 0
        while step < max_steps:
            x_deriv = self.x_derivative.subs({x: a, y: b}).doit()
            y_deriv = self.y_derivative.subs({x: a, y: b}).doit()
            if x_deriv == 0 and y_deriv == 0:
                break
            a = a - (x_deriv*alpha)
            b = b - (y_deriv*alpha)
            step += 1
        print(a, b)


x = sympy.Symbol('x')
y = sympy.Symbol('y')
a = 1  # Starting x-coordinate
b = 1  # Starting y-coordinate
a = GradientDescent((y+1)**2+(x)**2, a, b)
a.findMin()
