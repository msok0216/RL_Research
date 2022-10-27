import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from quickstart import *

"""
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(x_train)
print("AAAAAAAA")
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

"""
input_shape = (28, 28, 1)

spreadsheet = spread()
x_train = []
x_test = []
y_train = []
y_test = []
for x in range(9):
    for y in range(9):
        if y == x:
            temp_state, temp_action = get_sheet(spreadsheet, SPREADSHEET_ID, y + 1, "P1", "P2")
            x_test.append(temp_state)
            y_test.append(temp_action)
        else:
            temp_state, temp_action = get_sheet(spreadsheet, SPREADSHEET_ID, y + 1, "P1", "P2")
            x_train.append(temp_state)
            y_train.append(temp_action)
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    #x_train = x_train.reshape(60000, 784).astype("float32") / 255
    #x_test = x_test.reshape(10000, 784).astype("float32") / 255

    model = keras.Sequential(
    [
        #keras.Input(shape=input_shape)
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        #layers.Dense(num_classes, activation="softmax"),
    ]
    )

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.RMSprop(),
        metrics=["accuracy"],
    )

    history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
"""
x_train.clear()
x_test.clear()
y_train.clear()
y_test.clear()
for x in range(9):
    for y in range(9):
        if y == x:
            temp_state, temp_action = get_sheet(spreadsheet, SPREADSHEET_ID, y + 1, "P2", "P1")
            x_test.append(temp_state)
            y_test.append(temp_action)
        else:
            temp_state, temp_action = get_sheet(spreadsheet, SPREADSHEET_ID, y + 1, "P2", "P1")
            x_train.append(temp_state)
            y_train.append(temp_action)

x_train.clear()
x_test.clear()
y_train.clear()
y_test.clear()
for x in range(14):
    for y in range(14):
        if y == x:
            temp_state, temp_action = get_sheet(spreadsheet, SPREADSHEET_ID, y + 1, "P1", "P2")
            x_test.append(temp_state)
            y_test.append(temp_action)
        else:
            temp_state, temp_action = get_sheet(spreadsheet, SPREADSHEET_ID, y + 1, "P1", "P2")
            x_train.append(temp_state)
            y_train.append(temp_action)

x_train.clear()
x_test.clear()
y_train.clear()
y_test.clear()
for x in range(14):
    for y in range(14):
        if y == x:
            temp_state, temp_action = getSheet(spreadsheet, SPREADSHEET_ID, y + 1, "P2", "P1")
            x_test.append(temp_state)
            y_test.append(temp_action)
        else:
            temp_state, temp_action = getSheet(spreadsheet, SPREADSHEET_ID, y + 1, "P2", "P1")
            x_train.append(temp_state)
            y_train.append(temp_action)
"""
    