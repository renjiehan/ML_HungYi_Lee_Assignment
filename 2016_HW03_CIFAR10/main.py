import os
import matplotlib.pyplot as plt
import numpy as np
import copy
import net
import utils
import train_method as tm

from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical


def data_aug(X,Y):
    # mirror
    mirrX = np.flip(X,axis=3)
    newX = np.concatenate((X,mirrX),axis=0)
    newY = np.concatenate((Y,Y),axis=0)
    
    return (newX,newY)

def autoencoder():
    dict = utils.unpickle('cifar_10/data_batch_1')
    test_dict = utils.unpickle('cifar_10/data_batch_2')
    # dict content = [b'batch_label', b'labels', b'data', b'filenames']

    LX, LY = utils.data_fromCIFAR10(dict, 500)
    test_data, test_label = utils.data_fromCIFAR10(test_dict, 10000)
    #LX, LY = utils.data_fromCIFAR10(dict, 5000)
    #print("label data number {}, label number {}".format(np.shape(LX), np.shape(LY)))
    
    
    #model = net.Net1(train_data, train_lable, save_model='test.h5', validate=0.1)
    #model = train(LX, LY, save_model='test.h5', validate=0.1)
    #model = train(LX, LY)
    

    allData, allLabel = utils.data_fromCIFAR10(dict, 10000)
    #UX = UX[1000:]
    #print("unlabel data number line102", np.shape(UX))
   
    UX = allData[6000:]
    true_label = allLabel[6000:]
    
    train_data = np.concatenate((LX, UX), axis=0 )
    train_data, _ = data_aug(train_data, np.ones((train_data.shape[0],1)))
    
    X_normal = np.asarray(train_data,dtype='float32')/255.0
    
    if not os.path.isfile('ae_model.h5'):
        ae_model, ae_dnn = net.Autoencoder(train_data, X_normal,
                                           model_train='ae_model.h5',save_model='ae_model.h5', validate=0.1)
    else:
        ae_model, ae_dnn = net.Autoencoder(train_data, X_normal,
                        model_train='ae_model.h5', validate=0.1)

    '''ae_predict = utils.predict_data(test_data, ae_model)
    ae_predict = np.argmax(ae_predict, axis=1)'''

#print('auto encoder prediction dim = {}'.format(np.shape(ae_predict)))

#plt.plot(np.asarray(ae_predict))
#plt.show()






#ae_model= load_model('ae_model.h5')
    
    #ae_model.summary()
    

    
# CNN part

    for layer in ae_model.layers:
        layer.trainable = False


#ae_dnn.summary()
    x_train, y_train = data_aug(LX, LY)

    y_train = to_categorical(y_train,10)

    iter = 10

    save_model = 'dnn_model.h5'

    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0)
    model_check = ModelCheckpoint(save_model, monitor='val_loss', verbose=1,
                                  save_best_only=True, save_weights_only=False)


    ae_dnn = load_model('dnn_model_1.h5')
    prediction = utils.predict_data(test_data, ae_dnn)
    prediction = np.argmax(prediction, axis=1)

    count =0
    for i in range (np.shape(test_label)[0]):
        if (test_label[i] == prediction[i]):
            count += 1



    print('Correct Rate = {}'.format( count/np.shape(test_label)[0]))


if __name__ == '__main__':
#tm.self_train()
    autoencoder()
#for i in range(10):
#test = net.DNN(model_train='dnn_test.h5', save_model='dnn_test.h5', validate =0.05)





