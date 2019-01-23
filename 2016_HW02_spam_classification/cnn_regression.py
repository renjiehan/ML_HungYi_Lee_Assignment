from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense

import keras
from keras import backend as K

import numpy as np
import pandas as pd
import copy

trainfile = 'spam_data/spam_train.csv'
testfile = 'spam_data/spam_test.csv'

# open raw data
raw_data = pd.read_csv(trainfile, encoding='big5').values
raw_data = raw_data[:, 1:]  # (4001, 58)
raw_data = raw_data.astype(float)

x_train = copy.deepcopy(raw_data)
y_train = x_train[:,57] #(4001, 1)

x_train = np.delete(x_train, -1, 1)
y_train = to_categorical(y_train,2)


print("dim of x_train line 23", np.shape(x_train))
print("dim of y_train line 23", np.shape(y_train))

# construct model
model = Sequential()

model.add( Dense(units=8,input_shape=(57,), activation = 'relu'))
model.add( Dense(units=32, activation = 'relu'))
model.add( Dense(units=2, activation = 'softmax'))

model.compile( loss= 'categorical_crossentropy',
                    optimizer= 'adam',
                    metrics=['accuracy'])

# train and validate
model.fit( x_train, y_train, batch_size= 100, epochs= 30, validation_split= 0.1 )#,



