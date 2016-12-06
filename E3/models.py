import numpy as np
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.models import Model

from scipy.cluster.vq import vq, kmeans, whiten
from random import randint

"""
Dataset loading
"""
class IrisClassificationDataset():
    @staticmethod
    def load_data(path):
        attributes = np.loadtxt(path, usecols=[0,1,2,3], delimiter=',')
        labels = np.loadtxt(path, usecols=[4], dtype=np.str, delimiter=',')

        dataset = {}
        dataset['data'] = attributes

        # one hot encode the array of *string* labels (keras requires numbers)
        uniques, ids = np.unique(labels, return_inverse=True)
        dataset['labels'] = to_categorical(ids, len(uniques))

        return dataset

"""
Networks
"""
class MLP(object):
    """
    MLP with simple parameters defining each layer
    @param input_dim is the single sata point size
    @param hidden_dim hidden layer dimension, the usual rule of thumb is that
        this parameter should be a number between iput and output dim
    @param output_dim
    """
    def __init__(self,
        input_dim,
        hidden_dim,
        output_dim
    ):

        # sequential model form keras lets us add layers as needed
        self.model = Sequential()

        # defines input and hidden layer
        self.model.add(Dense(hidden_dim, input_dim=input_dim, activation='relu'))

        # output layer, its size is the number of classes
        # softmax activation guarantees the result is P(y|x)
        self.model.add(Dense(output_dim, activation='softmax',
                             init='uniform'))

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
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd)

        history = self.model.fit(
                       data, labels,
                       nb_epoch=epochs,
                       batch_size=batch_size,
                       shuffle=True,
                       validation_split=validation_split
        )

        return {
            'epochs': epochs, 'validation_split': validation_split,
            'learning_rate': learning_rate, 'momentum': momentum,
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def plot(training_loss, validation_loss, resolution=10):

    # five subplots sharing x axis
    fig, axarr = plt.subplots(1, sharex=True, figsize=(9,6))

    # plot loss
    axarr.plot(training_loss[::resolution])
    axarr.plot(validation_loss[::resolution])
    axarr.set_title('Training and Validation Losses')
    axarr.legend(['training loss', 'validation loss'], loc='upper left')

    fig.savefig('./plot.pdf', format='pdf', bbox_inches='tight')

class RBF(object):
    def __init__(self, input_dim, output_dim, n_neurons):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_neurons = n_neurons
        self.weights = np.ones((n_neurons, output_dim), dtype=float)
        self.beta = 0

    def activation(self, x, mi):
      to_be_exp = -self.beta*((x-mi)**2)
      return np.sum(np.exp(to_be_exp))

    def train(self, x, y, epochs=1000, lr=0.001, batch_size=10):

      # calculate clustering
      self.centers, distortion = kmeans(x, self.n_neurons)
      # print self.centers

      # calculate beta given distortion
      self.beta = 1/(2*(distortion**2))
      print self.beta

      for i in range(0, epochs*(x.shape[0]/batch_size)):
        errors = 0
        activations = np.zeros((1, self.n_neurons))
        for batch_i in range(0, batch_size):
          selected = randint(0, x.shape[0]-1)
          x_i = x[selected]
          y_i = y[selected]

          # for each neuron
          for c, center in enumerate(self.centers):

            # radial basis function to
            activations[0, c] = self.activation(x_i, center)

          pred_y = np.dot(activations, self.weights)

          errors += y_i-pred_y
          
        errors = errors/x.shape[0]
        update = lr*np.dot(errors.T,activations)
        # print update
        self.weights = self.weights + update.T
        if i%1000 == 0:
          print "#{} \tError: {:.7f}\tAcc: {}".format(i, (np.sum(errors)), self.evaluate(x, y))

    def evaluate(self, x, y):
      errors = 0
      activations = np.zeros((self.n_neurons))
      count = 0.0
      for x_i, y_i in zip(x, y):

        # for each neuron
        for c, center in enumerate(self.centers):

          # radial basis function to
          activations[c] = self.activation(x_i, center)

        pred_y = np.dot(activations, self.weights)

        count = count + int(np.argmax(pred_y) == np.argmax(y_i))
      return count/float(x.shape[0])


