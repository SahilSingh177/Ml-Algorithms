from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 28 * 28)).astype('float32') / 255
x_test = x_test.reshape((10000, 28 * 28)).astype('float32') / 255

# Convert labels to one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Create a Sequential model
model = Sequential([
    Dense(units=25, activation='relu', input_shape=(28 * 28,)),
    Dense(units=15, activation='relu'),
    Dense(units=10, activation='softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=4, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_acc}')

# Make predictions on some test data
predictions = model.predict(x_test[:5])

# Display the predicted and true labels
for i in range(5):
    print(f"Predicted: {predictions[i].argmax() + 1}, True: {y_test[i].argmax() + 1}")
