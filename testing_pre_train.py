from time import time

from parse_args import ARGS
from utils import initialize_sound_data, ITER
from neural_nets import LeNet, CIFAR, DeepNet, DeepNet_WithOut_Dropout
from pre_trainning import stack_models, update_feat_with_RBMs, update_feat_with_Autoencoder, update_feat_with_DBN


def test_stacked_models(s_data, m_number=3):
    m_number = ARGS.rp
    if ARGS.clf:
        models = []
        if ARGS.features == 'spectrogram':
            models.append(LeNet.build(s_data.feat_amount, len(s_data.classes)))
            extend = [False]
            for i in range(m_number-1):
                models.append(DeepNet.Conv((12, 1,), len(s_data.classes))) # input size is a lie, but it does not matter
                extend.append(True)
            remove_output = [2] * len(models)
            history = stack_models(models, s_data, remove_output, extend_feat_dim=extend)
        else:
            # Modelos Convolucionales 1D sin Drop out
            for i in range(m_number):
                models.append(DeepNet.Conv(s_data.feat_amount, len(s_data.classes)))
            remove_output = [2] * len(models)

            history = stack_models(models, s_data, remove_output, extend_feat_dim=True)
    else:
        # Modelos No convolucionales con Drop out
        models = []
        for i in range(m_number):
            models.append(DeepNet.NotConv(s_data.feat_amount, len(s_data.classes)))
        remove_output = [2]*len(models)
        history = stack_models(models, s_data, remove_output)

    return history


if __name__ == '__main__':

    if ARGS.dataset == 'avisoft':
        p = '/home/migue/PycharmProjects/keras audios/avisoft_birds'
        dataset = 'avisoft'
    elif ARGS.dataset == 'bats':
        p = '/home/migue/PycharmProjects/keras audios/bats'
        dataset = 'bats'
    else:
        # p = '/media/migue/5E7A52CD7A52A197/Users/black beard/Desktop/xccoverbl_WAV_classes/'
        p = '/media/migue/5E7A52CD7A52A197/Users/black beard/Desktop/xccoverbl_split/'
        dataset = 'xccoverbl'
    if ARGS.features == 'meanMFCC':
        features = ['meanMFCC']
    elif ARGS.features == 'spectrogram':
        features = ['spectrogram']
    elif ARGS.features == 'spectral_time':
        features = ['all Carlos features']

    build = "load"
    if ARGS.build:
        build = "build"
        if ARGS.features == 'spectral_time':
            s_data = initialize_sound_data(p, features, get_features=True, is_image=False, filter_args={}, use_carlos_feat=True)
        else:
            s_data = initialize_sound_data(p, features, get_features=True, is_image=False, filter_args={}) # build and save features
    elif ARGS.features != 'spectrogram':
        s_data = initialize_sound_data(p, features, get_features=False, is_image=False, filter_args={})  # load features
    if ARGS.features == 'spectrogram':
        s_data = initialize_sound_data(p, features, get_features=False, is_image=True, filter_args={})  # load images

    clf = "convolucional" if ARGS.clf else "no convolucional"
    print(
        "Training with {0} dataset, {1} {2} features, {3} classifier, with {4} pre-training and {5} stacked networks. ".
        format(dataset, build, ARGS.features, clf, ARGS.pmode, str(ARGS.rp)) +
        "We are going to do {0} iterations".format(str(ARGS.iterations)))

    h_train_acc = []
    h_train_loss = []
    h_test_acc = []
    h_test_loss = []
    h_delay = []
    for i in range(ARGS.iterations):
        print("Comenzando con la iteracion {0}".format(i+1))
        t1 = time()
        if ARGS.pmode == 'rbm':
            update_feat_with_RBMs(s_data, greedy_pre_train=1)
        elif ARGS.pmode == 'dbn':
            update_feat_with_DBN(s_data)
        elif ARGS.pmode == 'autoencoder':
            update_feat_with_Autoencoder(s_data, is_image=ARGS.features == 'spectrogram')

        h = test_stacked_models(s_data, m_number=1)
        t2 = time()
        h_delay.append(t2 - t1)
        # print("Demora : " + str(h_delay[-1]) + " segundos")

        h_train_acc.append(h.history['acc'][-1])
        h_train_loss.append(h.history['loss'][-1])

        h_test_acc.append(h.history['val_acc'][-1])
        h_test_loss.append(h.history['val_loss'][-1])
        ITER[0] += 1

        # Doing this because features could change with pre training
        if ARGS.features != 'spectrogram':
            s_data = initialize_sound_data(p, features, get_features=False, is_image=False,
                                           filter_args={})  # load features
        else:
            s_data = initialize_sound_data(p, features, get_features=False, is_image=True,
                                           filter_args={})  # load images

    if len(h_train_acc):
        print("Accuracy promedio en el entrenamiento: " + str(sum(h_train_acc) / len(h_train_acc)))
        print("Loss promedio en el entrenamiento: " + str(sum(h_train_loss) / len(h_train_loss)))
        print("Accuracy promedio en la validacion: " + str(sum(h_test_acc) / len(h_test_acc)))
        print("Loss promedio en la validacion: " + str(sum(h_test_loss) / len(h_test_loss)))
        print("Demora promedio de una iteracion: " + str(sum(h_delay) / len(h_delay)))
