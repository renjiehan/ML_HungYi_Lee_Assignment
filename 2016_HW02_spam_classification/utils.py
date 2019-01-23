import os
import numpy as np
import copy


def param_init(featNum):
    w = np.random.rand(featNum)
    w = np.reshape(w, (1,featNum))
    b = np.random.rand(1)

    return w, b

def sigmoid(x):
    return 1/ (1+ np.exp(-1*x) )

def predict(data, weight, bias ):
    num_data = data.shape[0]
    featNum = data.shape[1]      # aug_featNum = 162
    result = []
    result = np.array(result).astype(np.float)
    
    for i in range(num_data):
        x = data[i]
        x = np.reshape(x, (1,featNum))
        y = np.dot(weight, x.T) + bias
        y = sigmoid(y)
        
        result = np.append(result, y)
        
    return result

def SGD(data, ground_truth, learning_rate, epochs):
    num_data = np.shape(data)[0]  # total data count 3600
    featNum = np.shape(data)[1]     # aug_featNum = 162
    print('lr_init={}'.format(learning_rate))
    
    w, b = param_init(featNum)
    w_lr = 0
    b_lr = 0
    epsilon = 1e-8

    for epoch in range(epochs):
        for i in range(num_data):
            x = copy.deepcopy(data[ i ])  # (3600, 162)
            y = np.dot(w, x.T) + b
            y = sigmoid(y)
            error = ground_truth[ i ] * y + (1-ground_truth[ i ]) * (1-y)
    
            grad_w = error* x
            grad_b = error
    
            w_lr = w_lr + grad_w**2 + epsilon
            b_lr = b_lr + grad_b**2 + epsilon
    
            w = w - learning_rate/ np.sqrt(w_lr) * grad_w
            b = b - learning_rate/ np.sqrt(b_lr) * grad_b
    
        loss = (np.mean(np.square(error)))
    
        if (epoch+1) % 10 == 0:
            print('epoch:{:3d}\t Loss:{}'.format(epoch+1, np.sqrt(loss)))

    return w, b

def batchGD(data, ground_truth, batch_size, learning_rate, epochs):
    num_data = np.shape(data)[0]  # total data count 3600
    featNum = np.shape(data)[1]     # aug_featNum = 162
    
    print('lr_init={}, batch size={}'.format(learning_rate, batch_size))
    w, b = param_init(featNum)
    epsilon = 1e-8
    
    
    for epoch in range(epochs):
        w_lr = 0
        b_lr = 0
    
        for i in range(0,num_data, batch_size):
            j = i / batch_size
            x = copy.deepcopy(data[ int(j*batch_size) : int((j+1)*batch_size) ])  # (3600, 162)
            y = np.dot(w, x.T) + b
            y = sigmoid(y)
            #error = ground_truth[ int(j*batch_size) : int((j+1)*batch_size) ] - y
            term1 = ground_truth[ int(j*batch_size) : int((j+1)*batch_size) ]
            term1 = term1 * y
            
            term2 = 1- ground_truth[ int(j*batch_size) : int((j+1)*batch_size) ]
            term2 = term2 * ( 1-y )
            
            error = term1 + term2
    
            grad_w = np.dot(error, x)
            grad_b = np.sum(error, axis=1)
    
            w_lr = w_lr + grad_w**2 + epsilon
            b_lr = b_lr + grad_b**2 + epsilon
    
            w = w - learning_rate/ np.sqrt(w_lr) * grad_w
            b = b - learning_rate/ np.sqrt(b_lr) * grad_b
    
        loss = np.sqrt(np.mean(np.square(error)))
    
        if (epoch+1) % 1000 == 0:
            print('epoch:{:3d}\t Loss:{}'.format(epoch+1, loss))

        
    return w, b
