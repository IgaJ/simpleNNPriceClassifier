import random

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

x_train = []  # Price of item
y_train = []  # Classification of the item as cheap (T) or expensive (D)

x_test = []
y_test = []

for i in range(10):
    randNum = random.randint(0, 10)
    response = input(f"Czy {randNum}zł za 500g pieczywa to drogo czy tanio? (D, T)")
    x_train.append([randNum])

    if response == "D":
        y_train.append([1, 0])  # Classification as expensive [1,0] 1 = expensive, 0 = cheap
    elif response == "T":
        y_train.append([0, 1])  # Classification as cheap

x_train = np.array(x_train)
y_train = np.array(y_train)

# first layer is used to process our input data (price of item).
# second layer is 2 units, used for classification if price is expensive or cheap
model = keras.Sequential([
    keras.layers.Dense(1, input_shape=[1]),
    keras.layers.Dense(2, activation="softmax")
])

# compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# train the model
model.fit(x_train, y_train, epochs=600, verbose=1)

for i in range(128):
    randNum = random.randint(0, 10)
    prediction=model.predict(np.array([randNum]), verbose=0)
    x_test.append(randNum)

    if prediction[0][0] > prediction[0][1]:
        y_test.append(1)
        print(f"{randNum} wysoka")
    else:
        y_test.append(0)
        print(f"{randNum} niska")

# Graph predictions of Neural Network
plt.scatter(x_test, y_test, c=y_test)
plt.xlabel("cena pieczywa")
plt.ylabel("niska (0) lub wysoka (1)")
plt.show()

del x_train
del y_train




