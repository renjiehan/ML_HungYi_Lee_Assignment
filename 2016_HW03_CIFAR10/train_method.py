import os
import matplotlib.pyplot as plt
import numpy as np
import copy
import net
import utils

from keras.models import load_model

def self_train():
    dict = utils.unpickle('cifar_10/data_batch_1')
    test_dict = utils.unpickle('cifar_10/data_batch_2')
    # dict content = [b'batch_label', b'labels', b'data', b'filenames']

    label_data, label = utils.data_fromCIFAR10(dict, 500)
    test_data, test_label = utils.data_fromCIFAR10(dict, 5000)
    #label_data, label = utils.data_fromCIFAR10(dict, 5000)
    #print("label data number {}, label number {}".format(np.shape(label_data), np.shape(label)))
    
    
    '''
    train_data = []
    train_lable = []

    count = np.zeros(10)
    for i in range(500):
        if(count[0] < 20 and (label[i] == 0) ): count[0] += 1
        elif(count[1] < 20 and (label[i] == 1) ): count[1] += 1
        elif(count[2] < 20 and (label[i] == 2) ): count[2] += 1
        elif(count[3] < 20 and (label[i] == 3) ): count[3] += 1
        elif(count[4] < 20 and (label[i] == 4) ): count[4] += 1
        elif(count[5] < 20 and (label[i] == 5) ): count[5] += 1
        elif(count[6] < 20 and (label[i] == 6) ): count[6] += 1
        elif(count[7] < 20 and (label[i] == 7) ): count[7] += 1
        elif(count[8] < 20 and (label[i] == 8) ): count[8] += 1
        elif(count[9] < 20 and (label[i] == 9) ): count[9] += 1
        else: continue
        train_data.append(label_data[i])
        train_lable.append(label[i])'''



    model = net.Net3(label_data, label, save_model='test.h5', validate=0.1)
    #model = net.Net1(train_data, train_lable, save_model='test.h5', validate=0.1)
    #model = train(label_data, label, save_model='test.h5', validate=0.1)
    #model = train(label_data, label)
    

    allData, allLabel = utils.data_fromCIFAR10(dict, 10000)
    #unlabel_data = unlabel_data[1000:]
    #print("unlabel data number line102", np.shape(unlabel_data))

#unlabel_data = allData[1000:]
#true_label = allLabel[1000:]
    
    unlabel_data = allData[6000:]
    true_label = allLabel[6000:]
    
    predict_maxSet = []
    
    iter = 10
    
    for i in range(iter):
        print('Iteration {}'.format(i+1))
        model = load_model('test.h5')
        predict_raw = utils.predict_data(unlabel_data, model)
        prediction = np.argmax(predict_raw, axis=1)
        #predict_raw = np.argmax(predict_raw, axis=1)
        predict_max = np.max(predict_raw)
        predict_maxSet.append(predict_max)
        
        # test data from another batch, separate from label and unlabel data
        test_raw = utils.predict_data(test_data, model)
        test_raw = np.argmax(test_raw, axis=1)
        test_predict = test_raw.reshape( -1, 1)
        
        count = 0
        index =[]
        aug_data = []
        aug_label = []
        for k in range(predict_raw.shape[0]):
            if( np.max(predict_raw[k]) > (predict_max * 0.995 )):
                count += 1
                aug_data.append(unlabel_data[k])
                aug_label.append(prediction[k])
    
        prediction.reshape( -1, 1)
        
        new_labelData = label_data + aug_data
        new_label = label + aug_label
        
        utils.accy(test_predict, test_label)
        
        model = net.Net3(new_labelData, new_label, 'test.h5', 'test.h5', validate=0.05)
        #model = train(new_labelData, new_label, save_model='test.h5', validate=0.05)
        del model
        
        if( i == iter-1 ):
            utils.accy(test_predict, test_label)

#plt.plot(np.asarray(predict_maxSet))
#plt.show()

#for i in range(np.shape(predict_maxSet)[0]):
#print("Val in predict", predict_maxSet[i])






