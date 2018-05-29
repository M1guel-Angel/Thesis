from time import time
from utils import initialize_sound_data
from neural_nets import LeNet, CIFAR, DeepNet, DeepNet_WithOut_Dropout
from train_classifier import train_model, load_trainned_model, save_plots
from keras.utils import plot_model
from keras.optimizers import Adam

import numpy as np
from validation import plot_confusion_matrix
from train_classifier import train_model
from utils import split_features


def re_train_trained_model(s_data, is_image=False, extend_dim = True):
    # TESTING trained model
    model_path = "/home/migue/PycharmProjects/keras/modelos y features/meanMFCC en xccoverbl_split adam opt/meanMFCC_model.json"
    weight_path = "/home/migue/PycharmProjects/keras/modelos y features/meanMFCC en xccoverbl_split adam opt/meanMFCC_model_weights.h5"
    model = load_trainned_model(model_path, weight_path)

    opt = Adam()
    loss = "categorical_crossentropy" if len(s_data.classes) > 2 else "binary_crossentropy"
    model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

    if extend_dim:
        s_data.extend_feat_one_dim() # if data is 1D

    trainX, testX, trainY, testY = split_features(s_data)

    loss, acc = model.evaluate(x=testX, y=testY)
    print("ACCURACY:", acc)
    print("LOSS:", loss)

    history = train_model(model, s_data, trainX, testX, trainY, testY, is_image=is_image)
    print("After re-train")
    loss, acc = model.evaluate(x=testX, y=testY)
    print("ACCURACY:", acc)
    print("LOSS:", loss)


if __name__ == '__main__':

    # p = '/home/migue/PycharmProjects/keras/sound data'
    p = '/home/migue/PycharmProjects/keras/murcielagos'
    # p = '/home/migue/Audios/sonidos animales/'
    # p = '/home/migue/Audios/Bat Calls Selection/'
    # p = '/media/migue/5E7A52CD7A52A197/Users/black beard/Desktop/xccoverbl_split/'
    # p = '/media/migue/5E7A52CD7A52A197/Users/black beard/Desktop/xccoverbl_WAV_classes/'
    # features = ['spectrogram']
    # features = ['mfccs']
    features = ['meanMFCC']
    # features = ['all Carlos features']

    # s_data = initialize_sound_data(p, features, get_features=True, is_image=False, filter_args={}) # build and save features
    s_data = initialize_sound_data(p, features, get_features=False, is_image=False, filter_args={})  # load features
    # s_data = initialize_sound_data(p, features, get_features=True, is_image=True, filter_args={}) # save and load images
    # s_data = initialize_sound_data(p, features, get_features=False, is_image=True, filter_args={}) # load images
    # s_data = initialize_sound_data(p, features, get_features=True, is_image=False, filter_args={}, use_carlos_feat=True)


    # Test with images
    # model = LeNet.build(s_data.feat_amount, len(s_data.classes))
    # model = CIFAR.build(s_data.feat_amount, len(s_data.classes))

    # Test with 2D data
    # s_data.extend_feat_one_dim() # if data is 1D
    # model = DeepNet.Conv(s_data.feat_amount, len(s_data.classes))
    # model = DeepNet_WithOut_Dropout.Conv(s_data.feat_amount, len(s_data.classes))

    # Cualquier tipo de datos
    model = DeepNet.NotConv(s_data.feat_amount, len(s_data.classes))

    trainX, testX, trainY, testY = split_features(s_data)

    t1 = time()
    # train with images Adam optimizer
    # history = train_model(model, s_data, trainX, testX, trainY, testY)
    # no train with images, Adam optimizer
    history = train_model(model, s_data, trainX, testX, trainY, testY, is_image=False)
    t2 = time()
    print("Demora : " + str(t2 - t1) + " segundos")
    save_plots(model, s_data, history, testX, testY, "adam")

    # -------------- #
    # train with images SGD (descenso por gradiente) optimizer
    # history = train_model(model, s_data, sgd_opt=True)
    # no train with images, SGD (descenso por gradiente) optimizer
    # history = train_model(model, s_data, is_image=False, sgd_opt=True)
    # save_plots(model, s_data, history, testX, testY, "descenso por gradiente")
    # -------------- #

    # plot_model(model, to_file='model.png')
