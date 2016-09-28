'''
Collection of Neural Network Models
'''

import numpy as np

from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.models import Model

"""
dataset models
"""


class SeedsClassificationDataset():
    @staticmethod
    def load_data(path):
        alldata = np.loadtxt(path)

        dataset = {}
        dataset['data'] = alldata[:, :7]
        dataset['labels'] = to_categorical(
            np.add(alldata[:, 7], [-1]).astype(int)
        )
        return dataset


class Test1RegressionDataset():
    @staticmethod
    def load_data(path):
        alldata = np.loadtxt(path, skiprows=2)

        dataset = {}
        dataset['data'] = alldata[:, :14]
        dataset['labels'] = alldata[:, 14].astype(int)
        return dataset

"""
neural network models
"""


class FullyConnectedNeuralNetwork(object):
    def __init__(self,
                 input_dim,
                 use_2_hidden_layers=False,
                 output_dim=3,
                 output_activation='softmax',
                 loss_layer='categorical_crossentropy'):

        self.loss_layer = loss_layer

        # sequential model form keras lets us add layers as needed
        self.model = Sequential()

        # defines input and first hidden layer
        self.model.add(Dense(5, input_dim=input_dim, activation='relu',
                             init='lecun_uniform'))

        # optional second hidden layer
        if (use_2_hidden_layers):
            self.model.add(Dense(5, activation='relu', init='lecun_uniform'))

        # output layer, its size is the number of classes
        # softmax activation guarantees the result is P(y|x)
        self.model.add(Dense(output_dim, activation=output_activation,
                             init='lecun_uniform'))

    def train(self,
              data,
              labels,
              learning_rate=0.01,
              momentum=0.9,
              validation_split=0.1,
              batch_size=1,
              epochs=50):

        # stochastic gradient descent with paramenters + objective/loss layer
        sgd = SGD(lr=learning_rate, momentum=momentum)
        self.model.compile(loss=self.loss_layer, optimizer=sgd)

        history = self.model.fit(
                       data, labels,
                       nb_epoch=epochs,
                       batch_size=batch_size,
                       shuffle=True,
                       validation_split=validation_split,
                       verbose=False)

        return (epochs, validation_split, learning_rate, momentum,
                history.history['loss'][-1], history.history['val_loss'][-1])

    def save_model(self, model_file):
        self.model.save_weights(model_file)

    def load_model(self, model_file):
        self.model.load_weights(model_file)

    def print_weights(self):
        for layer in self.model.layers:
            print(layer.get_weights())
            print('----------------------')


class ClassificationFullyConnectedNeuralNetwork(FullyConnectedNeuralNetwork):
    def __init__(self,
                 use_2_hidden_layers=False):
        super(ClassificationFullyConnectedNeuralNetwork, self).__init__(
           7,
           output_dim=3,
           output_activation='softmax',
           loss_layer='categorical_crossentropy'
        )


class RegressionFullyConnectedNeuralNetwork(FullyConnectedNeuralNetwork):
    def __init__(self,
                 use_2_hidden_layers=False):
        super(RegressionFullyConnectedNeuralNetwork, self).__init__(
            14,
            output_dim=1,
            output_activation='sigmoid',
            loss_layer='mean_squared_error'
        )


class AdalineNetwork():
    """ implements the simplest possible neural network
        single layer and single output
    """
    def __init__(self, input_size, learning_rate):
        self.weights = np.array([0.0] * input_size)
        self.bias = 0
        self.learning_rate = learning_rate

    def predict(self, data):
        intermediate_result = self.weights*data + self.bias
        return np.sign(np.sum(intermediate_result))

    def train(self, data, label):
        output = self.predict(data)
        error = label - output
        if error != 0:
            delta_w = self.learning_rate*error*data
            self.weights = self.weights + delta_w

    def train_dataset(self, data, labels):
        for x, y in zip(data, labels):
            self.train(x, y)

    def test_dataset(self, data, labels):
        for x, y in zip(data, labels):
            error = y - self.predict(x)
            print("Error: " + str(error))

class FullyConnectedAutoEncoder(object):
    def __init__(self):

        # this is the size of our encoded representations
        encoding_dim = 4  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

        # this is our input placeholder
        input_img = Input(shape=(784,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(encoding_dim, activation='relu')(input_img)
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(784, activation='sigmoid')(encoded)

        # this model maps an input to its reconstruction
        self.model = Model(input=input_img, output=decoded)

         # stochastic gradient descent with paramenters + objective/loss layer
        #sgd = SGD(lr=learning_rate, momentum=momentum)
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy')

    def train(self,
              data,
              batch_size=1,
              epochs=50):

        self.model.fit(
                       data, data,
                       nb_epoch=epochs,
                       batch_size=batch_size,
                       shuffle=True,
                       verbose=False)

        # return (epochs, validation_split, learning_rate, momentum,
        #         history.history['loss'][-1], history.history['val_loss'][-1])

    def save_model(self, model_file):
        self.model.save_weights(model_file)

    def load_model(self, model_file):
        self.model.load_weights(model_file)

    def print_weights(self):
        for layer in self.model.layers:
            print(layer.get_weights())
            print('----------------------')

