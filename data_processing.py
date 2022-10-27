from setup import get_data
import pandas as pd

list = get_data()


def preprocessing(idx):
    first_speaker = []
    second_speaker = []

    curr_speaker = ' '
    is_firstSpeaker = False
    temp = list[idx][['Combined', 'IDE State', 'Dialogue State', 'Intent', 'Delivery', 'Action', 'Who', 'Stage', 'Tone']].values
    # temp = list[idx].values
    # print(temp)
    for row in temp:
        if row[4] != curr_speaker:
            curr_speaker = row[4]
            is_firstSpeaker = not is_firstSpeaker
            if is_firstSpeaker:
                first_speaker.append([])
            else:
                second_speaker.append([])
        if is_firstSpeaker: first_speaker[-1].append(row)
        else: second_speaker[-1].append(row)
    
    # print(first_speaker)
    return first_speaker,second_speaker

def prev_utterance():
    data = []
    for i in list:
        input = []
        output = []
        row = i.values
        for j in range(len(row)-1):
            input.append(row[j])
            output.append(row[j+1])
        
        data.append((input, output))
    return data



def last_different_speaker_utterance(first, second):
    i = 0
    input = []
    output = []
    while i < len(second):
        input.append(first[i][-1])
        output.append(second[i][0])
        if i + 1 < len(first):
            input.append(second[i][-1])
            output.append(first[i][0])
        i+=1
    return input, output



def last_same_speaker_utterance(first, second):
    input = []
    output = []
    for i in range(len(first)-1):
        input.append(first[i])
        output.append(first[i+1])
    for i in range(len(second)-1):
        input.append(second[i])
        output.append(second[i+1])
    return input, output


# print(len(prev_x), len(prev_y))
# print(len(last_diff_x), len(last_diff_y))
# print(len(last_same_x), len(last_same_y))


from sklearn import preprocessing as sklearn_preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

scaler = sklearn_preprocessing.MinMaxScaler()
lb = sklearn_preprocessing.LabelEncoder()
one_hot = sklearn_preprocessing.OneHotEncoder()
ordinal = sklearn_preprocessing.OrdinalEncoder()


# preprocessed_data = []
# for i in range(len(list)):




def processed_sheet():
    processed_data =[]
    for i in range(len(list)):
        if i != 1:
            s1, s2 = preprocessing(i)
            processed_data.append((s1,s2))
    return processed_data

def lssu_data(processed_data):
    data = []
    for tuple in processed_data:
        input, output = last_same_speaker_utterance(tuple[0], tuple[1])
        data.append((input,output))
    return data

def ldsu_data(processed_data):
    data = []
    for tuple in processed_data:
        input, output = last_different_speaker_utterance(tuple[0], tuple[1])
        data.append((input, output))
    return data


for i in range(9):
    if (i == 1): continue
    preprocessing(i)
# preprocessing(1)


# last_diff_x, last_diff_y = last_different_speaker_utterance(first, second)
# test_last_diff_x, test_last_diff_y = last_different_speaker_utterance(first_test, second_test)
# last_same_x, last_same_y = last_same_speaker_utterance(first, second)

# test = [x[26:] for x in last_diff_x]
# test_y = [x[26] for x in last_diff_y]

# lb = sklearn_preprocessing.OneHotEncoder()
# lb.fit_transform(test)
# # lb.transform(test)
# y_lb = sklearn_preprocessing.LabelEncoder()
# y_lb.fit_transform(test_y)
# # lb.transform(test_y)

# temp = []
# for i in range(len(test)):
#     for j in range(len(test[i])):
#         if test[i][j] is None:
#             test[i][j] = ""


# x_train, x_test, y_train, y_test = label_encoding(test, test, test_y, test_y)

# tree = DecisionTreeClassifier()
# tree.fit(x_train, y_train)
# print(tree.score(x_test, y_test))

# print(prev_x[0])
# last_diff_x = [lb.fit_transform(x) for x in last_diff_x]
# # last_diff_x = scaler.fit_transform(last_diff_x)
# # print(last_diff_x)


# last_diff_y = [y[25] for y in last_diff_y]
# last_diff_y = lb.fit_transform(last_diff_y)

# last_diff_y = last_diff_y.reshape(-1,1)
# # print(last_diff_y)

# # last_diff_y = [lb.fit_transform(y) for y in last_diff_y]
# # last_diff_y = scaler.fit_transform(last_diff_y)
# # print(last_diff_y)

# test_first, test_second = preprocessing(0)
# test_last_diff_x, test_last_diff_y = last_different_speaker_utterance(test_first, test_second) 

# test_last_diff_x = [lb.fit_transform(x) for x in test_last_diff_x]
# # test_last_diff_x = scaler.fit_transform(test_last_diff_x)


# test_last_diff_y = [y[25] for y in test_last_diff_y]
# test_last_diff_y = lb.fit_transform(test_last_diff_y)
# test_last_diff_y = test_last_diff_y.reshape(-1,1)

# tree = DecisionTreeClassifier()
# tree.fit(last_diff_x, last_diff_y)
# print(tree.score(test_last_diff_x, test_last_diff_y))


# reg = LogisticRegression(solver='lbfgs', max_iter=1000)
# reg.fit(last_diff_x, last_diff_y)
# print(reg.score(test_last_diff_x, test_last_diff_y))


# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3,2), random_state=1)
# # clf.fit(last_diff_x, last_diff_y)
# # print(clf.score(test_last_diff_x, test_last_diff_y))

# clf.fit(test_last_diff_x, test_last_diff_y)
# print(clf.score(last_diff_x, last_diff_y))

# def decision_tree():




# def naive_bayes():



# def logistic_regression():


# def svm():





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

# model.compile(optimizer='rmsprop', loss='mse')

# n = np.array(test)
# model.fit(np.array(test), np.array(test_y), batch_size=10, epochs=14)