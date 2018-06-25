# ***** IMPORTS *****
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# ***** CONSTANTS *****
batch_size = 128
num_classes = 10
epochs = 12
img_rows, img_cols = 28, 28
channels = 1

# ***** LOAD DATA *****
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ***** RESHAPE & NORMALIZE DATA *****
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels).astype('float32') / 255   # [Number of Pictures, Rows, Columns, Colors]
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels).astype('float32')   / 255
input_shape = (img_rows, img_cols, channels)

# ***** ONE HOT ENCODING *****
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# ***** BUILDING THE MODEL*****
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,   # Set Cross Entropy Loss
              optimizer=keras.optimizers.Adadelta(),        # Gradient Descent
              metrics=['accuracy'])                         # Print Accuracy

# ***** TRAIN *****
model.fit(x_train, y_train,                                 # Train Data
          batch_size=batch_size,                            # Batch Size
          epochs=epochs,                                    # Epochs
          verbose=1,                                        # Print Out Information
          validation_data=(x_test, y_test))                 # Test Data
score = model.evaluate(x_test, y_test, verbose=0)
