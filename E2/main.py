from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

# size of encoded representation
encoding_dim = 8

# input placeholder
input_img = Input(shape=(100,))
# encoded representation (bottleneck layer)
encoded = Dense(encoding_dim, activation='relu')(input_img)
# decoded image
decoded = Dense(100, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)

# compile the model ('commit' to it)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# test with identity
x_train = np.identity(10)

x_train.shape = (1, 100,)
print x_train.shape

autoencoder.fit(x_train, x_train,
                nb_epoch=10000,
                batch_size=1,
                shuffle=False)
