import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder 
from keras.utils import to_categorical


from keras.models import Sequential
from keras.layers import *

from setup import main

data = main()

# x = data.drop(["IDE Interactions", "Timestamp Start", "Timestamp End", "Time Elapsed", "Time Between Last DA", "Speaker"], axis = 1)
# x = data[['Speaker','IDE State','Dialogue State','Intent', 'Delivery','Action','Tone']]
# x = data[['Combined', 'Intent']]
x = data['Dialogue State']
# y = x[x['Speaker'] == 'P2']
# x = x[x['Speaker'] == 'P1']
# x.astype({'Combined' : 'category', 'Intent': 'category'}).dtypes

total_states = []
for i in range(1, 48):
    total_states.append("S" + str(i))

# print(total_states)
mapping = {}




y = x.loc[(x.index % 2 == 1)]
x = x.loc[(x.index % 2 == 0)]

for i in range(1,48):
    mapping[total_states[i-1]] = i
x = np.array(x)
y = np.array(y)
x = x[1:]
print(x)
y = y[1:]
x = [mapping[i] for i in x]
y = [mapping[i] for i in y]
x_encode = to_categorical(x, dtype ="uint8")
y_encode = to_categorical(y, dtype ="uint8")


print(x_encode.shape, y_encode.shape)

panda_x = pd.get_dummies(x)
panda_y = pd.get_dummies(y)


# x = x.iloc[1:]
# y = y.iloc[1:]
lb = LabelEncoder()
x_temp = lb.fit_transform(x)
y_temp = lb.fit_transform(y)

# y = lb.fit_transform(y)
# print(y.dtypes)
# input_shape = (,2)
model = Sequential(
    [
        # keras.layers.CategoryEncoding(num_tokens=2, output_)
        keras.layers.Input(shape=(1,), name='input_1'),
        keras.layers.Dense(10, activation="relu"),
        # keras.layers.Dense(25, activation="relu"),
        # keras.layers.Dense(10, activation="relu"),
    ]
    )
# model.add(Dense(142, activation='relu'))


# layer = tf.keras.layers.CategoryEncoding(
#     num_tokens=None, output_mode="one_hot"
# )

# layerd
# model.add(Activation('sigmoid'))
model.compile(optimizer='rmsprop', loss='mse')
# model.fit(np.array(x), np.array(y))

model.fit(x_temp, y_temp, batch_size=10, epochs=14)
# model.fit(x_temp, y_temp)
# model.fit(x_temp, y_temp)
# model.fit(x_temp, y_temp)

test_scores = model.evaluate(x_temp, y_temp, verbose=2)
print("Test loss:", type(test_scores))
print("Test accuracy:", test_scores)
# test_scores = model.evaluate(x, y, verbose=2)
# print(test_scores)
# def getStates():
#     state = data['Combined'].where(data["Speaker" == "P1"]).tolist()
#     actions = data['Combined'].where(data["Speaker" == "P2"]).tolist()