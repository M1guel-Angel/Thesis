class FeatureError(Exception):

    def __init__(self, classif):
        super(Exception, self).__init__()
        self.classif = classif

    def __str__(self):
        return "None feature extracted for file {0}".format(self.classif)


class BadInput(Exception):

    def __init__(self, message):
        super(Exception, self).__init__()
        self.m = message

    def __str__(self):
        return self.m


def check_input_dimensions(input_dim):
    if isinstance(input_dim, (list, tuple)):
        if len(input_dim) == 0:
            raise BadInput("There must be at least one input dimension.")
    elif not isinstance(input_dim, int):
        raise BadInput("Input dimensions should be an integer, a list or a tuple")

# from keras.models import Sequential
# from keras.layers.core import Dense
# from keras.optimizers import SGD
#
# import numpy as np
#
# model = Sequential()
# model.add(Dense(50, input_shape=(20, )))
# model.add(Dense(5))
#
# print(model.input_shape)
#
# a = Sequential()
# aux = Dense(20, input_shape=(10, ))
# a.add(aux)
#
# dic = a.layers[0].get_config()
# dic['batch_input_shape'] = model.layers[1].output_shape
#
# res = Sequential()
# res.add(a.layers[0].from_config(dic))
# for l in a.layers[1:]:
#     res.add(l)
#
# opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# loss = "binary_crossentropy"
# model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])
#
# model.fit(np.array([[1]*20]), np.array([[2]*5]), epochs=20)
# model.pop()
# print(model.predict(np.array([[1]*20])))
#
# print(res.input_shape)
# print(res.output_shape)
# print("yaaaaaaaa")