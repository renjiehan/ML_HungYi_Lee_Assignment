import numpy as np
import os
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model, Model

from keras.utils import to_categorical, multi_gpu_model
from keras.layers import Input, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Dense, Dropout, Lambda
from keras.utils import plot_model, to_categorical
from keras import backend as K
from keras import metrics

def Net1(label_data, label, model_train='', save_model='', validate =0):
    x_train = label_data
    y_train = label
    numTrain = np.shape(x_train)[0]
    
    x_train = np.reshape(x_train, (numTrain, 32, 32, 3))
    y_train = to_categorical(y_train,10)
    
    model = Sequential()
    model.add(Conv2D(192, (5, 5), input_shape=(32, 32, 3)))
    model.add(BatchNormalization(epsilon=1e-03))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(256, (1, 1)))
    model.add(BatchNormalization(epsilon=1e-03))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(384, (1, 1)))
    model.add(BatchNormalization(epsilon=1e-03))
    model.add(Activation('relu'))
    
    model.add(Conv2D(256, (1, 1)))
    model.add(BatchNormalization(epsilon=1e-03))
    model.add(Activation('relu'))
    
    model.add(Conv2D(192, (3, 3)))
    model.add(BatchNormalization(epsilon=1e-03))
    model.add(Activation('relu'))
    
    model.add(Flatten())
    model.add(Dense(512, kernel_initializer='he_normal'))
    model.add(BatchNormalization(epsilon=1e-03))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(10, kernel_initializer='he_normal'))
    model.add(Activation('softmax'))
    
    #parallel_model = multi_gpu_model(model, gpus=2)
    #parallel_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0)
    model_check = ModelCheckpoint(save_model, monitor='val_loss', verbose=1,
                                  save_best_only=True, save_weights_only=False)
    
    if ( model_train and  save_model) :
        print('Load model and save new train model.')
        model = load_model(model_train)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0)
        
        if (validate):
            model.fit( x_train, y_train, batch_size= 25, epochs= 20, validation_split= validate,
                      callbacks=[ model_check, early_stop ])
        else:
            model.fit( x_train, y_train, batch_size= 25, epochs= 20,
                      callbacks=[ model_check, early_stop ])
        model.save(save_model)
        fit_model = load_model(save_model)
    elif model_train:
        print('Load model {}'.format( model_train))
        fit_model = load_model(model_train)
    elif save_model:
        print('Build model and save.')
        model.fit( x_train, y_train, batch_size= 25, epochs= 40, validation_split= validate,
                  callbacks=[ model_check, early_stop ])
                  
        model.save(save_model)
        fit_model = load_model(save_model)
    else:
        print('Build model only.')
        model.fit( x_train, y_train, batch_size= 50, epochs= 30, validation_split= validate )
        fit_model = []
    
    return fit_model


#def Net3(nb_classes, inputs=(32,32,3), file_load_weights=None):
def Net3(label_data, label, model_train='', save_model='', validate =0):
    import keras
    import h5py
    from keras.models import Model
    from keras.layers import Input,Activation,BatchNormalization
    from keras.layers import Flatten, Dropout
    from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D,Dense,ZeroPadding2D
    from keras.layers.advanced_activations import LeakyReLU
    
    def norm_relu(in_layer):
        return Activation('relu')(BatchNormalization(epsilon=1e-03)(in_layer))
    
    x_train = label_data
    y_train = label
    numTrain = np.shape(x_train)[0]
    
    x_train = np.reshape(x_train, (numTrain, 32, 32, 3))
    y_train = to_categorical(y_train,10)
    
    
    
    img_shape = (32,32,3)
    
    model = Sequential()
    model.add(BatchNormalization( epsilon=1e-3, input_shape= img_shape))

    model.add(Conv2D(192, kernel_size= (5, 5), padding='same'))
    model.add(BatchNormalization( epsilon=1e-3))
    model.add(Activation('relu'))
    model.add(Conv2D(160, kernel_size= (1, 1), padding='same'))
    model.add(BatchNormalization( epsilon=1e-3))
    model.add(Activation('relu'))
    model.add(Conv2D(96, kernel_size= (1, 1), padding='same'))
    model.add(BatchNormalization( epsilon=1e-3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D( (2,2), (2,2) ))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, kernel_size= (5, 5), padding='same'))
    model.add(BatchNormalization( epsilon=1e-3))
    model.add(Activation('relu'))
    model.add(Conv2D(192, kernel_size= (1, 1), padding='same'))
    model.add(BatchNormalization( epsilon=1e-3))
    model.add(Activation('relu'))
    model.add(Conv2D(192, kernel_size= (1, 1), padding='same'))
    model.add(BatchNormalization( epsilon=1e-3))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D(padding=(1,1)) )
    model.add(AveragePooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Dropout(0.5))


    model.add(Conv2D(192, kernel_size= (3, 3), padding='same'))
    model.add(BatchNormalization( epsilon=1e-3))
    model.add(Activation('relu'))
    model.add(Conv2D(192, kernel_size= (1, 1), padding='same'))
    model.add(BatchNormalization( epsilon=1e-3))
    model.add(Activation('relu'))
    model.add(Conv2D(10, kernel_size= (1, 1), padding='same'))
    model.add(BatchNormalization( epsilon=1e-3))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((8,8), (1,1)))

    model.add(Flatten())
    model.add(Activation('softmax'))
    

    
    
    
    #if file_load_weights: model.load_weights(file_load_weights)
    
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0)
    model_check = ModelCheckpoint(save_model, monitor='val_loss', verbose=1,
                                  save_best_only=True, save_weights_only=False)

    if ( model_train and  save_model) :
        print('Load model and save new train model.')
    
        model = load_model(model_train)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0)

        if (validate):
            model.fit( x_train, y_train, batch_size= 25, epochs= 20, validation_split= validate,
                      callbacks=[ model_check, early_stop ])
        else:
            model.fit( x_train, y_train, batch_size= 25, epochs= 20,
                      callbacks=[ model_check, early_stop ])
        model.save(save_model)
        fit_model = load_model(save_model)
    elif model_train:
        print('Load model {}'.format( model_train))
        fit_model = load_model(model_train)
    elif save_model:
        print('Build model and save.')
        model.fit( x_train, y_train, batch_size= 25, epochs= 40, validation_split= validate,
                  callbacks=[ model_check, early_stop ])
            
        model.save(save_model)
        fit_model = load_model(save_model)
    else:
        print('Build model only.')
        model.fit( x_train, y_train, batch_size= 100, epochs= 30, validation_split= validate )
        fit_model = []

    return fit_model


#def cnn_autoencoder(nb_classes, inputs=(32,32,3), file_load_weights=None):
def Autoencoder(data, label, model_train='', save_model='', validate =0):
    import keras
    import h5py
    
    
    x_train = data
    y_train = np.asarray(data, dtype='float32')/255.0
    
    print('X size {}'.format(np.shape(x_train)))
    print('Y size {}'.format(np.shape(y_train)))
    
    img_shape = (32,32,3) # 3,32x32
    #img_shape = (3,32,32)
    input_img = Input(shape=img_shape)
    
    #ae_model = Sequential()
    norm = Lambda(lambda x: K.cast(x,dtype='float32')/255.0, input_shape=img_shape)(input_img)
    
    encode = Conv2D(filters=32, kernel_size= (3, 3), padding='same')(norm) # 32x32
    encode = BatchNormalization( epsilon=1e-3)(encode)
    encode = Activation('relu')(encode)
    encode = MaxPooling2D( (2,2), (2,2) )(encode)                    # 16x16
    
    encode = Conv2D(filters=16, kernel_size= (3, 3), padding='same')(encode)
    encode = BatchNormalization( epsilon=1e-3)(encode)
    encode = Activation('relu')(encode)
    encode = MaxPooling2D( (2,2), (2,2) )(encode)                   # 8x8
    
    code = Conv2D(filters=4, kernel_size= (3, 3), padding='same')(encode)
    
    decode = BatchNormalization( epsilon=1e-3)(code)
    decode = Activation('relu')(decode)
    decode = UpSampling2D(size=(2,2))(decode)                       # 8x8
    
    
    decode = Conv2D(filters=16, kernel_size= (3, 3), padding='same')(decode)
    decode = BatchNormalization( epsilon=1e-3)(decode)
    decode = Activation('relu')(decode)
    #ae_model.add(UpSampling2D(size=(2,2)))                       # 16x16
    
    decode = Conv2D(filters=32, kernel_size= (3, 3), padding='same')(decode)
    decode = BatchNormalization( epsilon=1e-3)(decode)
    decode = Activation('relu')(decode)
    decode = UpSampling2D(size=(2,2))(decode)
    
    decode = Conv2D(filters=3, kernel_size= (1, 1), padding='same',
                     activation='sigmoid')(decode) # 3,32x32

    encoder = Model(inputs= input_img, outputs= code)
    encoder.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    ae_model = Model(inputs=input_img, outputs=decode)
    ae_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0)
    model_check = ModelCheckpoint(save_model, monitor='val_loss', verbose=1,
                                  save_best_only=True, save_weights_only=False)
                                  
    classifier = Flatten()(code)
    classifier = Dropout(0.5)(classifier)
    classifier = Dense(1024, activation='relu')(classifier)
    classifier = Dropout(0.5)(classifier)
    classifier = Dense(256, activation='relu')(classifier)
    classifier = Dropout(0.5)(classifier)
    
    classifier = Dense(10, activation='softmax')(classifier)

    ae_dnn = Model(inputs=input_img, outputs=classifier)
    adam2 = keras.optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ae_dnn.compile(loss='categorical_crossentropy', optimizer=adam2, metrics=['accuracy'])
                                  
    if ( model_train and  save_model) :
        print('Load model and save new train model.')
        if os.path.isfile(model_train):
            ae_model = load_model(model_train)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0)
        
        if (validate):
            ae_model.fit( x_train, y_train, batch_size= 25, epochs= 20, validation_split= validate,
                      callbacks=[ model_check, early_stop ])
        else:
            ae_model.fit( x_train, y_train, batch_size= 25, epochs= 20,
                      callbacks=[ model_check, early_stop ])
        ae_model.save(save_model)
        ae_model = load_model(save_model)
    elif model_train:
        print('Load ae_model {}'.format( model_train))
        ae_model = load_model(model_train)
    elif save_model:
        print('Build ae_model and save. File: {}'.format(save_model))
        ae_model.fit( x_train, y_train, batch_size= 100, epochs= 40, validation_split= validate,
                  callbacks=[ model_check, early_stop ])
            
        ae_model.save(save_model)
    else:
        print('Build ae_model only.')
        ae_model.fit( x_train, y_train, batch_size= 100, epochs= 30, validation_split= validate )
        #fit_model = []

    return ae_model, ae_dnn

import utils
def DNN(model_train='', save_model='', validate =0):
    dict = utils.unpickle('cifar_10/data_batch_1')
    test_dict = utils.unpickle('cifar_10/data_batch_2')
    
    LX, LY = utils.data_fromCIFAR10(dict, 10000)
    test_data, test_label = utils.data_fromCIFAR10(test_dict, 10000)
    
    x_train = LX
    x_train = np.reshape(x_train, (np.shape(x_train)[0],32,32,3))
    
    y_train = LY
    y_train = to_categorical(y_train,10)
    
    allData, allLabel = utils.data_fromCIFAR10(dict, 10000)
    
    UX = allData[6000:]
    true_label = allLabel[6000:]
    
    img_shape = (32,32,3)
    input_img = Input(shape=img_shape)
    
    classifier = Flatten()(input_img)
    classifier = Dropout(0.5)(classifier)
    classifier = Dense(1024, activation='relu')(classifier)
    classifier = Dropout(0.5)(classifier)
    classifier = Dense(256, activation='relu')(classifier)
    classifier = Dropout(0.5)(classifier)

    classifier = Dense(10, activation='softmax')(classifier)

    dnn = Model(inputs=input_img, outputs=classifier)
    #adam2 = keras.optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    dnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    dnn.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0)
    model_check = ModelCheckpoint(save_model, monitor='val_loss', verbose=1,
                                  save_best_only=True, save_weights_only=False)

    if ( model_train and  save_model) :
        print('Load model and save new train model.')
        if os.path.isfile(model_train):
            dnn = load_model(model_train)
    
        if (validate):
            dnn.fit( x_train, y_train, batch_size= 100, epochs= 20, validation_split= validate,
                         callbacks=[ model_check, early_stop ])
        else:
            dnn.fit( x_train, y_train, batch_size= 100, epochs= 20,
                         callbacks=[ model_check, early_stop ])
        dnn.save(save_model)
        dnn = load_model(save_model)
    elif model_train:
        print('Load ae_model {}'.format( model_train))
        dnn = load_model(model_train)
    elif save_model:
        print('Build ae_model and save. File: {}'.format(save_model))
        dnn.fit( x_train, y_train, batch_size= 100, epochs= 40, validation_split= validate,
                 callbacks=[ model_check, early_stop ])

        dnn.save(save_model)
    else:
        print('Build ae_model only.')
        dnn.fit( x_train, y_train, batch_size= 100, epochs= 30, validation_split= validate )



    prediction = utils.predict_data(test_data, dnn)
    prediction = np.argmax(prediction, axis=1)
    
    count =0
    for i in range (np.shape(test_label)[0]):
        if (test_label[i] == prediction[i]):
            count += 1



    print('Correct Rate = {}'.format( count/np.shape(test_label)[0]))





    return dnn









