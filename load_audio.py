import os
from os import path
from extract_audio_car import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from essentia.standard import MonoLoader
from PIL import Image

from save_data import save, load
from filters import highpass_filter
from errors import BadInput
from Parameters.example import get_params


class Sound_Data():

    def __init__(self, audios_path, funcs_=FUNCS_.keys()):
        self.classes = os.listdir(audios_path)
        self.class_to_number = {}
        self.number_to_class = {}
        self.features = {}
        self.feat_amount = len(funcs_)
        self.audio_path = audios_path
        self.funcs_ = funcs_
        self.audios_path = None
        if "text" in funcs_:
            self.dictionary = {}

    def get_features(self, filter_args=None):
        # self.classes = ["cat", "Frog"]
        # inp = 1
        # self.classes = [self.classes[inp]]
        for i, animal in enumerate(self.classes):
            current_path = path.join(self.audio_path, animal)
            sounds = os.listdir(current_path)
            self.features[i] = []
            self.class_to_number[animal] = i
            self.number_to_class[i] = animal
            for sound in sounds:
                if not sound.endswith('.wav'): continue
                AUDIO_FILE = path.join(current_path, sound)
                try:
                    audio = MonoLoader(filename=AUDIO_FILE)()
                except:
                    continue
                if filter_args:
                    audio = highpass_filter(audio, 44100, **filter_args)
                feat_ = []
                for f in self.funcs_:
                    aux = do_from_name(f, audio, AUDIO_FILE)
                    # print("{0} dice {1}".format(self.classes[i], aux))
                    # print(len(aux))
                    if f == "text":
                        if aux == -1: continue
                        aux = aux.split()
                        for w in aux:
                            if not self.dictionary.__contains__(w):
                                self.dictionary[w] = len(self.dictionary.keys())
                    feat_.extend(aux)
                if len(feat_) == 0: continue
                self.features[i].append(feat_)
        kaux = self.features[0][0]
        # self.feat_amount = len(self.features[0][0])
        self.feat_amount = np.array(self.features[0][0]).shape
        #joblib.dump(self.features, "features.pkl")
        save(self.features, "features.json")

    def get_carlos_params(self):
        for i, animal in enumerate(self.classes):
            current_path = path.join(self.audio_path, animal)
            sounds = os.listdir(current_path)
            self.features[i] = []
            self.class_to_number[animal] = i
            self.number_to_class[i] = animal
            one_class_json = {}
            for sound in sounds:
                if not sound.endswith('.wav'): continue
                AUDIO_FILE = path.join(current_path, sound)
                print("processing " + AUDIO_FILE)
                feat_ = get_params(AUDIO_FILE)

                print(AUDIO_FILE + " processed")
                one_class_json[sound] = feat_

                if len(feat_) == 0: continue
                self.features[i].append(feat_)
            save(one_class_json, path.join(current_path, animal + "_features.json"))
            print(animal + " features has been json saved")
        self.feat_amount = np.array(self.features[0][0]).shape
        #joblib.dump(self.features, "features.pkl")
        save(self.features, "features.json")

    def load_features(self):
        # self.features = joblib.load("features.pkl")
        aux = load("features.json")
        for i in aux.keys():
            self.features[int(i)] = aux[i]
        self.feat_amount = np.array(self.features[0][0]).shape

    def extend_feat_one_dim(self):
        self.feat_amount = [i for i in self.feat_amount]
        self.feat_amount.append(1)
        for k in self.features.keys():
            l = []
            for feat in self.features[k]:
                l.append(np.array(feat).reshape(self.feat_amount))
            self.features[k] = l

    def load_images(self, spec=True):
        self.audios_path = {}
        for i, animal in enumerate(self.classes):
            current_path = path.join(self.audio_path, animal)
            images = os.listdir(current_path)
            self.features[i] = []
            self.audios_path[i] = []
            for im in images:
                if (not spec and not im.endswith('_mfcc.jpg')) or (spec and not im.endswith('_spec.jpg')):
                    continue
                audio = im.replace("_mfcc.jpg", ".wav")
                audio = audio.replace("_spec.jpg", ".wav")
                audio = path.join(current_path, audio)
                p = path.join(current_path, im)
                imag = Image.open(p)
                imag = imag.resize((28, 28))
                a = np.array(imag)
                # consider them as float and normalize
                a = a.astype('float32')
                a /= 255
                self.features[i].append(a)
                self.audios_path[i].append(audio)
        self.feat_amount = self.features[0][0].shape

    def data_partition(self, percent):
        """
        :param percent: float 0< - <1 representing the percent for separate each class data to train (the rest is for testing)
        :return: data partitioned in X_train, X_test, y_train, y_test
        """
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for an_number in self.features.keys():
            features = self.features[an_number]
            # output = to_categorical([[an_number]]*len(features), 2)#len(self.classes) )
            # x_tr, x_te, y_tr, y_te = train_test_split(features, output, train_size=percent)
            x_tr, x_te = train_test_split(features, train_size=percent)
            y_tr = to_categorical([[an_number]] * len(x_tr), len(self.classes))
            y_te = to_categorical([[an_number]] * len(x_te), len(self.classes))
            X_train.extend(x_tr)
            X_test.extend(x_te)
            y_train.extend(y_tr)
            y_test.extend(y_te)
        #return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
        return X_train, X_test, y_train, y_test

    def get_data(self):
        features = []
        for an_number in self.features.keys():
            features.extend(self.features[an_number])
        return np.array(features)

    def partitionate_data_and_audios_p(self, percent=0.7):
        if self.audios_path:
            X_train = []
            y_train = []
            X_test = []
            y_test = []
            a_train = []
            a_test = []

            for an_number in self.features.keys():
                features = self.features[an_number]
                audios = self.audios_path[an_number]
                x_tr, x_te, audios_train, audios_test = train_test_split(features, audios, train_size=percent)
                y_tr = to_categorical([[an_number]] * len(x_tr), len(self.classes))
                y_te = to_categorical([[an_number]] * len(x_te), len(self.classes))
                X_train.extend(x_tr)
                X_test.extend(x_te)
                y_train.extend(y_tr)
                y_test.extend(y_te)

                a_train.extend(audios_train)
                a_test.extend(audios_test)
            return X_train, X_test, y_train, y_test, a_train, a_test
        else:
            raise BadInput("Not audios_path defined. Use data_partition function instead.")

    def update_features(self, func):
        for i in self.features.keys():
            tata = np.array(list(map(np.array, self.features[i])))
            self.features[i] = list(func(tata))
        self.feat_amount = np.array(self.features[0][0]).shape

    def array_to_class(self, arr):
        """
        :param arr esta en la forma [..., 1, ..., ], Sus valores son cero excepto en la poscion q indica a q clase pertenece
        :return:  la clase a la que pertenece
        """
        arr = list(arr)
        return arr.index(1)

    def set_of_array_to_classes(self, set_arr):
        return [self.array_to_class(i) for i in set_arr]

    def vectorize_text(self):
        dict_size = len(self.dictionary.keys()) # cantidad de palabras
        new_features = {}
        for num_class in self.features.keys():
            a = np.zeros(dict_size)
            for word in self.features[num_class]: # si la palabra esta repetida en la misma sentencia, no se transmite al vectorizar
                a[self.dictionary[word]] = 1
            new_features[num_class] = a
        self.features = new_features
        save(new_features, "words_vector.json")

    def clone(self):
        a = Sound_Data(self.audio_path, self.funcs_)
        a.class_to_number = self.class_to_number
        a.number_to_class = self.number_to_class
        a.features = self.features.copy()
        a.feat_amount = self.feat_amount

        return a
