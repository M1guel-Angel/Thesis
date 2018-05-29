from keras.models import Sequential
from sklearn.neural_network import BernoulliRBM
import numpy as np

from train_classifier import train_model, autoencoder_train
from neural_nets import Autoencoder
from utils import split_features, scale
from train_classifier import save_plots


def stack_models(models, sound_data, remove_outputs, is_image=False, sgd_opt=False, extend_feat_dim=False):
    if not isinstance(extend_feat_dim, list):
        extend_feat_dim = [extend_feat_dim]*len(models)
    s_data = sound_data.clone()
    histories = []
    for i in range(len(models)-1):
        if extend_feat_dim[i]:
            s_data.extend_feat_one_dim()
        trainX, testX, trainY, testY = split_features(s_data)

        print("Comienza el entranamiento del modelo " + str(i + 1))
        model = models[i]
        h = train_model(model, s_data, trainX, testX, trainY, testY, is_image=is_image, sgd_opt=sgd_opt)
        histories.append(h)

        # Despues de entrenado, eliminar las capas asociadas a la clasificacion (de salida)
        if len(model.layers) < remove_outputs[i]:
            raise TypeError('There are no enough layers in the model.')
        else:
            for aux_var in range(remove_outputs[i]):
                model.pop()

        dic = models[i + 1].layers[0].get_config()
        aux_dim = []
        aux_dim.extend(model.output_shape)
        if extend_feat_dim[i+1]:  # if the next model needs to extend features, do it
            aux_dim.append(1)
        dic['batch_input_shape'] = aux_dim

        aux_model = Sequential()
        aux_model.add(models[i + 1].layers[0].from_config(dic))
        # models[i + 1].layers[0] = models[i + 1].layers[0].from_config(dic)
        for l in models[i + 1].layers[1:]:
            config = l.from_config(l.get_config())
            config.name += str(i + 1)
            aux_model.add(config)
        models[i + 1] = aux_model

        s_data.update_features(model.predict)

    if extend_feat_dim[-1]:
        s_data.extend_feat_one_dim()
    trainX, testX, trainY, testY = split_features(s_data)
    print("Comienza el entranamiento del modelo " + str(len(models)))
    h = train_model(models[-1], s_data, trainX, testX, trainY, testY, is_image=is_image, sgd_opt=sgd_opt)
    histories.append(h)
    for i, h in enumerate(histories):
        n = -1
        print("Model " + str(i+1) + " => loss: " + str(h.history['loss'][n]) + " acc: " + str(h.history['acc'][n])
              + " val_loss: " + str(h.history['val_loss'][n]) + " val_acc: " + str(h.history['val_acc'][n]))

    # saving history
    save_plots(models[-1], s_data, h, testX, testY, "descenso por gradiente")

    return h


def update_feat_with_RBMs(s_data, greedy_pre_train=1):

    data = scale(s_data.get_data())
    print(np.min(data))
    print(np.max(data))
    # Fit and Transform data
    for i in range(greedy_pre_train):
        # Initialize the RBM
        rbm = BernoulliRBM(n_components=90, n_iter=50, learning_rate=0.01, verbose=True)
        rbm.fit(data)
        s_data.update_features(rbm.transform)
        data = s_data.get_data()


def update_feat_with_DBN(s_data, output=400, rbm_num=2):
    # out_dims = np.arange(100, output + 100, int(output/rbm_num))
    out_dims = [60, 30]
    print(out_dims)
    data = scale(s_data.get_data())
    print(np.min(data))
    print(np.max(data))

    for i in range(rbm_num):
        # Initialize the RBM
        rbm = BernoulliRBM(n_components=out_dims[i], n_iter=50, learning_rate=0.01, verbose=True)
        rbm.fit(data)
        s_data.update_features(rbm.transform)
        data = s_data.get_data()


def update_feat_with_Autoencoder(s_data, add_noise_func=None, is_image=False):

    features = s_data.get_data()

    if callable(add_noise_func):
        noisy_features = map(add_noise_func, features)
    else:
        noisy_features = features
    if is_image:
        # noisy_features = np.ravel(np.reshape(noisy_features, ))
        pass

    model = Autoencoder.build(s_data.feat_amount)
    autoencoder_train(model, noisy_features, features)

    # removing last layer to get the new representation
    model.pop()
    s_data.update_features(model.predict)
