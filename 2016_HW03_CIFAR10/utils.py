import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def data_fromCIFAR10(dict, dataNum):
    label_data = []
    label = []
    
    for i in range(dataNum):
        data = dict[b'data'][i]
        data = data.reshape(3,32,32)
        data = data.transpose(1,2,0)
        
        labels = dict[b'labels'][i]
        
        label_data.append(data)
        label.append(labels)
    
    return label_data, label

def predict_data(data, model):
    dataNum = np.shape(data)[0]
    data = np.reshape(data, (dataNum, 32, 32, 3))
    #print('print model line72', model)
    
    if (data.size):
        print('Predict data is processing.')
        prediction = model.predict(data, batch_size= dataNum, verbose=1)
        
        return prediction
    else:
        print('Predict data is absent.')
        return []

def accy(predict_raw, true_label):
    count = 0
    for k in range(np.shape(true_label)[0]):
        if(true_label[k] == predict_raw[k]):
            count += 1

    correct_rate = count / np.shape(true_label)[0]
    print('Correct rate is {}\n\n'.format( correct_rate))
