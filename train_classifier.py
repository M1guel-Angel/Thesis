from keras.models import model_from_json
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import numpy as np

from validation import plot_history, plot_confusion_matrix
from save_data import load_json_str

EPOCHS = 20
INIT_LR = 1e-3
BATCH_SIZE = 30
VERBOSE = 0
VALIDATION_SPLIT=0.2


def train_model(model, s_data, trainX, testX, trainY, testY, is_image=True, sgd_opt=False):
    """
    :param model: some trainable build in keras neural network
    :param s_data: Sound_Data class object
    :param is_image:  if we are training on an image dataset.
    :param sgd_opt: use gradient optimizer. If False, then Adam opt will be used
    :return: x_train, x_test, y_train, y_test
    """
    print(s_data.feat_amount)
    # print(s_data.classes)

    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) if sgd_opt else Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    loss = "categorical_crossentropy" if len(s_data.classes) > 2 else "binary_crossentropy"
    model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

    if is_image:
        # Para el entrenamiento con imagenes es bueno añadir al conjunto las mismas imagenes con ciertos cambios
        aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
            height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
            horizontal_flip=True, fill_mode="nearest")

        history = model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
            validation_data=(testX, testY),
            # steps_per_epoch=len(trainX)//BATCH_SIZE,
            epochs=EPOCHS, verbose=VERBOSE)
    else:
        history = model.fit(trainX, trainY, epochs=EPOCHS, batch_size=BATCH_SIZE,
                            verbose=VERBOSE, validation_data=(testX, testY))

    return history


def autoencoder_train(model, data_in, data_out, sgd_opt=True):

    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) if sgd_opt else Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    loss = "categorical_crossentropy" if len(data_in) > 2 else "binary_crossentropy"
    model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

    return model.fit(data_in, data_out, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)


def save_plots(model, s_data, history, testX, testY, optimizer_name):

    features = " ".join(s_data.funcs_)
    # save model
    model_json = model.to_json()
    open(features + '_model.json', 'w').write(model_json)
    # And the weights learned by our deep network on the training set
    model.save_weights(features + '_model_weights.h5', overwrite=True)

    with open('info.txt', 'w') as outfile:
        s = "Este modelo se genero usando como caracteristicas " + features + "\n"
        s += str(EPOCHS) + "EPOCHS\n"
        s += str(BATCH_SIZE) + "BATCH_SIZE\n"
        s += optimizer_name + "\n"
        s += "Las clases usadas se listan a continuacion:\n"
        s += "\n".join(s_data.classes)
        outfile.write(s)

    plot_history(history, save_plots)
    predictions = model.predict_classes(testX)
    # print(predictions)
    testY = s_data.set_of_array_to_classes(testY)  # CONVERTING to array of class_number
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(testY, predictions)
    # np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=s_data.classes, title='Confusion matrix, without normalization',
                          save_plot=True)


def load_trainned_model(model_path, weight_path):
    model_json_str = load_json_str(model_path)
    model = model_from_json(model_json_str)
    model.load_weights(weight_path)
    return model