from setup import get_data, get_additional_data
import pandas as pd

list = get_data()


# Replace IDE States
def replace_idestate(df):
    df.replace("D3", "D1", inplace=True)
    df.replace("D22", "D1", inplace=True)

    df.replace("D28", "D5", inplace=True)

    for i in range(7,17):
        s = "D" + str(i)
        df.replace(s, "D6", inplace=True)
    df.replace("D24", "D6", inplace=True)
    df.replace("D26", "D6", inplace=True)
    df.replace("D30", "D6", inplace=True)
    df.replace("D33", "D6", inplace=True)
    df.replace("D35", "D6", inplace=True)
    df.replace("D36", "D6", inplace=True)
    for i in range(41,46):
        s = "D" + str(i)
        df.replace(s, "D6", inplace=True)
    df.replace("D47", "D6", inplace=True)
    df.replace("D21", "D19", inplace=True)
    df.replace("D27", "D23", inplace=True)
    for i in range(37,41):
        s = "D" + str(i)
        df.replace(s, "D23", inplace=True)
    df.replace("D34", 'D32', inplace=True)
    df.replace("D25", 'D32', inplace=True)
    # print(df)


def preprocessing(idx):
    first_speaker = []
    second_speaker = []

    curr_speaker = ' '
    is_firstSpeaker = False
    temp = list[idx][['Speaker','Combined', 'IDE State', 'Dialogue State', 'Intent', 'Delivery', 'Action', 'Who', 'Stage', 'Tone']].values
    # temp = list[idx].values
    # print(temp)
    for row in temp:
        if row[0] != curr_speaker:
            curr_speaker = row[0]
            is_firstSpeaker = not is_firstSpeaker
            if is_firstSpeaker:
                first_speaker.append([])
            else:
                second_speaker.append([])
        
        for i in range(len(row)):
            if row[i] is None:
                row[i] = ""
        if is_firstSpeaker: first_speaker[-1].append(row[1:])
        else: second_speaker[-1].append(row[1:])
    
    # print(first_speaker)
    return first_speaker,second_speaker

def load_woz_data():
    list = get_additional_data()
    ans = []
    for i in range(len(list)):
        first_speaker = []
        second_speaker = []

        curr_speaker = ' '
        is_firstSpeaker = False
        temp = list[i][['Speaker','Combined', 'IDE State', 'Dialogue State', 'Intent', 'Delivery', 'Action', 'Who', 'Stage', 'Tone']].values
        # temp = list[idx].values
        # print(temp)
        for row in temp:
            if row[0] != curr_speaker:
                curr_speaker = row[0]
                is_firstSpeaker = not is_firstSpeaker
                if is_firstSpeaker:
                    first_speaker.append([])
                else:
                    second_speaker.append([])
            
            for i in range(len(row)):
                if row[i] is None:
                    row[i] = ""
            if is_firstSpeaker: first_speaker[-1].append(row[1:])
            else: second_speaker[-1].append(row[1:])

        # print(first_speaker)
        ans.append((first_speaker, second_speaker))
    return ans


def last_different_speaker_utterance(first, second):
    i = 0
    input = []
    output = []
    while i < len(second):
        input.append(first[i][-1].tolist())
        output.append(second[i][0].tolist())
        if i + 1 < len(first):
            input.append(second[i][-1].tolist())
            output.append(first[i][0].tolist())
        i+=1
    return input, output



def last_same_speaker_utterance(first, second):
    input = []
    output = []
    for i in range(len(first)-1):
        input.append(first[i][-1].tolist())
        output.append(first[i+1][0].tolist())
    for i in range(len(second)-1):
        input.append(second[i][-1].tolist())
        output.append(second[i+1][0].tolist())
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

# first,second = preprocessing(0)
# x, y = last_same_speaker_utterance(first, second)
# print(len(y), len(y[0]))




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

def prev_data():
    data = []

    for i,sheet in enumerate(list):
        if i != 1:
            temp = sheet[['Combined', 'IDE State', 'Dialogue State', 'Intent', 'Delivery', 'Action', 'Who', 'Stage', 'Tone']].values
            input = []
            output = []
            for j, row in enumerate(temp):
                for k in range(len(row)):
                    if row[k] is None:
                        row[k] = ''
                if j % 2 == 0:
                    input.append(row)
                else:
                    output.append(row)        
            if len(input) > len(output):
                input = input[:-1]
            data.append((input, output))
        

    return data


def raw_data():
    for i in range(len(list)):
        replace_idestate(list[i])
    # print(len(list))
    return list

def preprocessed_ide(idx):
    first_speaker = []
    second_speaker = []

    curr_speaker = ' '
    is_firstSpeaker = False
    temp = list[idx][['Speaker','IDE State']].values
    # temp = list[idx].values
    # print(temp)
    for row in temp:
        if row[0] != curr_speaker:
            curr_speaker = row[0]
            is_firstSpeaker = not is_firstSpeaker
            if is_firstSpeaker:
                first_speaker.append([])
            else:
                second_speaker.append([])
        
        # for i in range(len(row)):
        if row[1] is None or len(str(row[1])) < 2:
            row[1] = 0
        if not isinstance(row[1], int):
            if len(row[1]) < 4:
                row[1] = row[1][1:]
            else:
                row[1] = row[1][:3]
                row[1] = row[1].replace('D', '')
                row[1] = row[1].replace('E', '')

        if is_firstSpeaker: first_speaker[-1].append(int(row[1]))
        else: second_speaker[-1].append(int(row[1]))
    
    # print(first_speaker)
    return first_speaker,second_speaker

def preprocessed_dialogue(idx):
    first_speaker = []
    second_speaker = []

    curr_speaker = ' '
    is_firstSpeaker = False
    temp = list[idx][['Speaker','Dialogue State']].values
    # temp = list[idx].values
    # print(temp)
    # print(len(temp))
    for row in temp:
        # print(curr_speaker, row)
        if row[0] != curr_speaker:
            curr_speaker = row[0]
            is_firstSpeaker = not is_firstSpeaker
            if is_firstSpeaker:
                first_speaker.append([])
            else:
                second_speaker.append([])
        
        for i in range(len(row)):
            if row[i] is None:
                row[i] = 0
        if is_firstSpeaker: first_speaker[-1].append(int(row[1][1:]))
        else: second_speaker[-1].append(int(row[1][1:]))
    
    # print(first_speaker)
    return first_speaker,second_speaker

def sort(list):
    input = []
    output = []
    # first speaker = 0, second speaker = 1
    for i in list:
        for j in range(max(len(i[0]), len(i[1]))):
            if j < len(i[1]):
                input.append(i[0][j])
                output.append(i[1][j][0])
            if j+1 < len(i[0]):
                input.append(i[1][j])
                output.append(i[0][j+1][0])
    return input, output