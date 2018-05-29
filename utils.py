import numpy as np
from load_audio import Sound_Data

ITER = [1]


def initialize_sound_data(class_path, features, get_features=False, is_image=True, filter_args=None, sub_classes=None, use_carlos_feat=False):
    s_data = Sound_Data(class_path, funcs_=features)
    if sub_classes:
        # process subclasses
        aux = s_data.classes
        s_data.classes = sub_classes
    if get_features:
        if use_carlos_feat:
            s_data.get_carlos_params()
        else:
            s_data.get_features(filter_args)
    if is_image:
        s_data.load_images()
    elif not get_features:
        s_data.load_features()
    if sub_classes:
        # going back to the originals
        s_data.classes = aux
    return s_data


def extend_one_dimension(data):
    dim = (len(data), 1)
    return np.array(data).reshape(dim)


def get_batch(data, batch_size):
    batch = data[0:batch_size]
    i = batch_size
    while len(batch):
        yield batch, i - batch_size
        batch = data[i: i + batch_size]
        i += batch_size


def scale(X):
    # scale data points within the range [0, 1]
    # minim = np.min(X)
    # return (X - minim) / (np.max(X) - minim) # malisimos resultados escalando con estos datos
    return 1/(1 + np.power(np.e, X))

def k_fold(model, s_data):
    X, _, Y, _ = split_features(s_data, 1)
    k = 5
    l = int(len(X) / k)
    loss, acc = 0, 0
    best_train_x, best_train_y, best_test_x, best_test_y = None, None, None, None
    for i in range(k):
        train_x = X[i * l:(i + 1) * l]
        train_y = Y[i * l:(i + 1) * l]

        test_x = np.concatenate([X[:i * l], X[(i + 1) * l:]])
        test_y = np.concatenate([Y[:i * l], Y[(i + 1) * l:]])

        model.fit(train_x, train_y, epochs=15)
        s_loss, s_acc = model.evaluate(test_x, test_y)
        if s_acc > acc:
            best_train_x = train_x
            best_train_y = train_y
            best_test_x = test_x
            best_test_y = test_y
    return best_train_x, best_test_x, best_train_y, best_test_y


def split_features(s_data, train_split=0.7):
    x_train, x_test, y_train, y_test = s_data.data_partition(train_split)

    trainX = np.array(x_train)
    trainY = np.array(y_train)
    testX = np.array(x_test)
    testY = np.array(y_test)

    return trainX, testX, trainY, testY
