# https://realpython.com/linear-regression-in-python/

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
x = np.array([5,15,25,35,45,55]).reshape((-1,1))
y = np.array([5,20,14,32,22,38])

# .fit calculates the variable b1 and b1, using the existing input and output, x and y as the arguments. Returns self i.e the model itself.
model = LinearRegression().fit(x,y)

# Coefficient of determination
r_sq = model.score(x,y)
print(f"Coefficient of determination : {r_sq}")

# Model intercept and model coefficient i.e w and b

print(f"Model intercept : {model.intercept_}")
print(f"Model coefficient : {model.coef_}") # coef_ gives an array.

y_pred = model.predict(x)
print(f"Predicted output: \n {y_pred}")

# R-squared
print(f"R-squared value: {model.score(x,y)*100}")

# Mean-absolute, Mean-squared and Root mean square error
mean_abs_error = metrics.mean_absolute_error(y_pred,y)
mean_sq_error = metrics.mean_squared_error(y_pred,y)
root_mean_square_error = np.sqrt(mean_sq_error)
print(f"Mean Absolute error: {mean_abs_error}")
print(f"Mean Square error: {mean_sq_error}")
print(f"Root mean square error: {root_mean_square_error}")
