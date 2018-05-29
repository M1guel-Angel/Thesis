# import the necessary packages
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.convolutional import MaxPooling2D, MaxPooling1D, AveragePooling1D, AveragePooling2D
from keras.layers import LeakyReLU, Input
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense, Dropout
from errors import BadInput, check_input_dimensions


class LeNet:
    @staticmethod
    def build(input_shape, classes):
        check_input_dimensions(input_shape)
        model = Sequential()
        # CONV => RELU => POOL
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=input_shape))

        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # CONV => RELU => POOL
        # for i in range(3):
        model.add(Conv2D(50, (7, 7), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Flatten => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        # a softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model


class CIFAR:
    @staticmethod
    def build(input_shape, classes):
        check_input_dimensions(input_shape)
        # model = Sequential()
        # model.add(Conv2D(32, (3, 3), padding='same',
        #                  input_shape=input_shape))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        # model.add(Flatten())
        # model.add(Dense(512))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(classes))
        # model.add(Activation('softmax'))
        # model.summary()

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model


class DeepNet:
    @staticmethod
    def Conv(input_dim, classes):
        if len(input_dim) == 1:
            input_dim = (input_dim[0], 1)
        check_input_dimensions(input_dim)

        # Esto dio 99% de accuracy con el conjunto de entrenamiento de los murcielagos, con meanMFCCs
        # NO TOCAR

        model = Sequential()
        model.add(Conv1D(15, 5, padding="same",
                         input_shape=input_dim))
        # model.add(Activation("tanh"))
        model.add(LeakyReLU(alpha=0.3))
        # model.add(MaxPooling1D())
        model.add(AveragePooling1D(padding="same"))
        model.add(Conv1D(40, 5, padding="same"))
        model.add(Activation("tanh"))
        model.add(MaxPooling1D())
        model.add(Flatten())
        # model.add(Dense(100))
        # model.add(LeakyReLU(alpha=0.3))
        # model.add(Dropout(0.5))
        model.add(Dense(600))
        model.add(Activation("tanh"))
        # model.add(Dropout(0.5))
        # model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model

    @staticmethod
    def NotConv(input_dim, classes):
        check_input_dimensions(input_dim)

        model = Sequential()
        model.add(Dense(50, input_shape=input_dim))
        model.add(Activation("tanh"))
        # model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(200))
        model.add(Activation("tanh"))
        model.add(Dropout(0.7))
        # for i in range(5):
        model.add(Dense(500))
        model.add(Activation("tanh"))
        model.add(Dropout(0.6))
        model.add(Dense(500))
        model.add(Activation("tanh"))
        # model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model


class DeepNet_WithOut_Dropout:

    @staticmethod
    def Conv(input_dim, classes):
        check_input_dimensions(input_dim)

        model = Sequential()
        model.add(Conv1D(15, 5, padding="same",
                         input_shape=input_dim))
        # model.add(Activation("tanh"))
        model.add(LeakyReLU(alpha=0.3))
        # model.add(MaxPooling1D())
        model.add(AveragePooling1D(padding="same"))
        model.add(Conv1D(40, 5, padding="same"))
        model.add(Activation("tanh"))
        model.add(MaxPooling1D())
        model.add(Flatten())
        # model.add(Dense(100))
        # model.add(LeakyReLU(alpha=0.3))
        # model.add(Dropout(0.5))
        model.add(Dense(600))
        model.add(Activation("tanh"))
        # model.add(Dropout(0.5))
        # model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model

    @staticmethod
    def Not_Conv(input_dim, classes):
        check_input_dimensions(input_dim)

        model = Sequential()
        model.add(Dense(30, input_shape=input_dim))
        # model.add(Activation("tanh"))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(60))
        model.add(Activation("tanh"))
        # model.add(Dropout(0.4))
        # model.add(Dense(100))
        # model.add(LeakyReLU(alpha=0.3))
        # model.add(Dropout(0.5))
        model.add(Dense(100))
        model.add(Activation("tanh"))
        # model.add(Dropout(0.5))
        # model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model

    @staticmethod
    def Conv2D(input_dim, classes):
        check_input_dimensions(input_dim)

        model = Sequential()
        model.add(Conv2D(15, (5, 5), padding="same",
                         input_shape=input_dim))
        # model.add(Activation("tanh"))
        model.add(LeakyReLU(alpha=0.3))
        # model.add(MaxPooling1D())
        model.add(AveragePooling2D(padding="same"))
        model.add(Conv2D(40, (5, 5), padding="same"))
        model.add(Activation("tanh"))
        model.add(MaxPooling2D())
        model.add(Flatten())
        # model.add(Dense(100))
        # model.add(LeakyReLU(alpha=0.3))
        # model.add(Dropout(0.5))
        model.add(Dense(600))
        model.add(Activation("tanh"))
        # model.add(Dropout(0.5))
        # model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model


class Autoencoder:
    @staticmethod
    def build(input_dim):
        if isinstance(input_dim, (list, tuple)):
            if len(input_dim) == 0:
                raise BadInput("There must be one input dimension.")
            if len(input_dim) == 1:
                # feat_input = int(0.7 * input_dim[0])
                feat_input = int(7 * input_dim[0])
            else:
                raise BadInput("Autoencoder input must be a flattened array")
        elif isinstance(input_dim, int):
            input_dim = [input_dim]
            # feat_input = int(0.7 * input_dim[0])
            feat_input = int(7 * input_dim[0])
        else:
            raise BadInput("Input dimensions should be an integer, a list or a tuple")

        model = Sequential()
        model.add(Dense(feat_input, input_shape=input_dim))
        model.add(Activation("tanh"))
        model.add(Dense(input_dim[-1]))

        return model
