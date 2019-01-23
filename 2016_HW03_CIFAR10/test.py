'''import keras
import h5py
from keras.models import Model
from keras.layers import Input,Activation,BatchNormalization
from keras.layers import Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D,Dense,ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU

img_shape = (32,32,3)

model.add(BatchNormalization( epsilon=1e-3))

model.add(Conv2D(192, kernel_size= (5, 5), input_shape= img_shape, padding='same'))
model.add(BatchNormalization( epsilon=1e-3))
model.add(Activation('relu'))
model.add(Conv2D(160, kernel_size= (1, 1), input_shape= img_shape, padding='same'))
model.add(BatchNormalization( epsilon=1e-3))
model.add(Activation('relu'))
model.add(Conv2D(96, kernel_size= (1, 1), input_shape= img_shape, padding='same'))
model.add(BatchNormalization( epsilon=1e-3))
model.add(Activation('relu'))
model.add(MaxPooling2D( (2,2), (2,2) )
model.add(Dropout(0.5))

model.add(Conv2D(192, kernel_size= (5, 5), input_shape= img_shape, padding='same'))
model.add(BatchNormalization( epsilon=1e-3))
model.add(Activation('relu'))
model.add(Conv2D(192, kernel_size= (1, 1), input_shape= img_shape, padding='same'))
model.add(BatchNormalization( epsilon=1e-3))
model.add(Activation('relu'))
model.add(Conv2D(192, kernel_size= (1, 1), input_shape= img_shape, padding='same'))
model.add(BatchNormalization( epsilon=1e-3))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1,1)) )
model.add(AveragePooling2D(pool_size=(3,3),strides=(2,2)))
model.add(Dropout(0.5))

      
model.add(Conv2D(192, kernel_size= (3, 3), input_shape= img_shape, padding='same'))
model.add(BatchNormalization( epsilon=1e-3))
model.add(Activation('relu'))
model.add(Conv2D(192, kernel_size= (1, 1), input_shape= img_shape, padding='same'))
model.add(BatchNormalization( epsilon=1e-3))
model.add(Activation('relu'))
model.add(Conv2D(10, kernel_size= (1, 1), input_shape= img_shape, padding='same'))
model.add(BatchNormalization( epsilon=1e-3))
model.add(Activation('relu'))
model.add(AveragePooling2D((8,8), (1,1)))

model.add(Flatten())
model.add(Activation('softmax'))
'''




def Autoencoder(label_data, label, model_train='', save_model='', validate =0):
    import keras
    import h5py
    from keras.models import Model
    from keras.layers import Input, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Dense, Dropout, Lambda
    
    
    img_shape = (32,32,3) # 3,32x32
    
    model = Sequential()
    model.add(Lambda(lambda x: K.cast(x,dtype='float32')/255.0,
                     output_shape=img_shape))
    
    model.add(Conv2D(32, kernel_size= (3, 3), padding='same')) # 32x32
    model.add(BatchNormalization( epsilon=1e-3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D( (2,2), (2,2) ))                    # 16x16
    
    model.add(Conv2D(16, kernel_size= (3, 3), padding='same'))
    model.add(BatchNormalization( epsilon=1e-3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D( (2,2), (2,2) ))                   # 8x8
    
    model.add(Conv2D(4, kernel_size= (3, 3), padding='same'))
    model.add(BatchNormalization( epsilon=1e-3))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2,2)))                       # 8x8
    
    
    model.add(Conv2D(16, kernel_size= (3, 3), padding='same'))
    model.add(BatchNormalization( epsilon=1e-3))
    model.add(Activation('relu'))
    #model.add(UpSampling2D(size=(2,2)))                       # 16x16
    
    model.add(Conv2D(32, kernel_size= (3, 3), padding='same'))
    model.add(BatchNormalization( epsilon=1e-3))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2,2)))
    
    model.add(Conv2D(3, kernel_size= (1, 1), padding='same',
                     activation='sigmoid')) # 3,32x32
    
    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    adam2 = keras.optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam2, metrics=['accuracy'])
    
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

























