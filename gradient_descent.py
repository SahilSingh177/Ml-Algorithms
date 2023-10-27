import sympy
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        max_steps = 1000
        # Storing the results in a variable for plotting
        history = []
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
            history.append((a,b,self.func.subs({x:a,y:b})))
        print(a, b)
        return a,b,self.func,history

x = sympy.Symbol('x')
y = sympy.Symbol('y')
a = 1  # Starting x-coordinate
b = 1  # Starting y-coordinate
a = GradientDescent((y+1)**2+(x)**2, a, b)
x_opt,y_opt,f_opt,history = a.findMin()

# Plotting
x_range = np.arange(-10,10,0.1)
y_range = np.arange(-10,10,0.1)
X,Y = np.meshgrid(x_range,y_range)
Z = (Y+1)**2+(X)**2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='coolwarm')
ax.scatter(*zip(*history), c='r', marker='o')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
plt.show()