from data_processing import *
from sklearn import preprocessing as sklearn_preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

scaler = sklearn_preprocessing.MinMaxScaler()
ordinal = sklearn_preprocessing.OrdinalEncoder()
one_hot = sklearn_preprocessing.OneHotEncoder()
lb = sklearn_preprocessing.LabelEncoder()
data = processed_sheet()


lssu = lssu_data(data)
ldsu = ldsu_data(data)
prev = prev_data()



def kfold(data):
    dataset = []
    for i in range(len(data)):
        training = data[:i] + data[i+1:]
        print(len(training))
        x_training = []
        y_training = []
        for j in range(len(training)):
            x_training.extend(training[j][0])
            y_training.extend(training[j][1])

        x_test = data[i][0]
        y_test = data[i][1]
        dataset.append({'x_training': x_training, 'y_training': y_training, 'x_test': x_test, 'y_test': y_test})
    return dataset

# dataset = kfold(lssu)



x_test = ldsu[0][0]
y_test = [ x[2] for x in ldsu[0][1]]
x_train = []
y_train = []

for i in range(1, len(ldsu)):
    x_train.extend(ldsu[i][0])
    y_train.extend(ldsu[i][1])

y_train = [x[2] for x in y_train]
x_train = ordinal.fit_transform(x_train)
y_train = lb.fit_transform(y_train)
print(y_train)
print(len(x_test), len(y_test))
# x_test = ordinal.transform(x_test)
x_test = ordinal.fit_transform(x_test)
y_test = lb.fit_transform(y_test)

tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
print(tree.score(x_test, y_test))
# lb.fit_transform(y_train)
# lb.transform(y_test)

# apply MinMaxScaler to see if it improves the performance





# import tensorflow
# from tensorflow import keras
# from keras import Sequential
# from keras.layers import *
# import numpy as np


# model = Sequential(
#     [
#         # keras.layers.CategoryEncoding(num_tokens=2, output_)
#         # keras.layers.Input(shape=(1,), name='input_1'),
#         keras.layers.Dense(10, activation="relu"),
#         keras.layers.Dense(1, activation="sigmoid"),
#         # keras.layers.Dense(10, activation="relu"),
#     ]
# )

# model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

# model.fit(x_train, y_train, batch_size=10, epochs=14)
# print(model.evaluate(x_test, y_test))