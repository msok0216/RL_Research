from sklearn import preprocessing


def label_encoding(x, y):
    encoder = preprocessing.LabelEncoder()
    x = encoder.fit_transform(x)
    y = encoder.fit_transform(y)
    return x, y


def onehot_encoding(x, y):
    encoder = preprocessing.OneHotEncoder()
    x = encoder.fit_transform(x)
    y = encoder.fit_transform(y)
    return x, y

def onehot_encoding(x, y):
    encoder = preprocessing.OrdinalEncoder()
    x = encoder.fit_transform(x)
    y = encoder.fit_transform(y)
    return x, y

# apply MinMaxScaler to see if it improves the performance