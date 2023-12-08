from keras.models import Sequential
from keras.layers import Dense
import numpy as np

model = Sequential([
    Dense(units=3,activation='sigmoid'),
    Dense(units=1,activation='sigmoid'),
]) # Taking two layers and joining them
x = np.array([[200.0,17.0],
              [120.0,5.0],
              [425.0,20.0],
              [212.0,18.0]
            ])
y = np.array([[1],[0],[0],[1]]) # Expected Output
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x,y,epochs = 10)
x_new = np.array([[300.0, 15.0]])

# Make predictions
predictions = model.predict(x_new)
print("Predictions:", predictions)