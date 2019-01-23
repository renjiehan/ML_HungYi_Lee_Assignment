
import os
import numpy as np
import pandas as pd
import utils
import copy

trainfile = 'spam_data/spam_train.csv'
testfile = 'spam_data/spam_test.csv'

# open raw data
raw_data = pd.read_csv(trainfile, encoding='big5').values #shape: (4320, 27)
raw_data = raw_data[:, 1:]  # (4001, 58)
raw_data = raw_data.astype(float)

print("dim of raw_data line 16", np.shape(raw_data))

train_data = copy.deepcopy(raw_data)
dup_rawData = copy.deepcopy(train_data)

ground_truth = train_data[:,57] #(4001, 1)
train_data = np.delete(train_data, -1, 1)
print("dim of raw_data line 23", np.shape(train_data))

################# training process, SGD #################

w = []
b = []
    
epochs = 300
learning_rate = 1e-5    # best so far (100, 1e-5), (300, 1000)
    
w,b = utils.SGD(train_data, ground_truth, learning_rate, epochs)

################# training process, mini-batch SG #################
'''
w = []
b = []

epochs = 10000
learning_rate = 1e-7    # best so far (4000, 1e-7),(1000, 1e-7),

w,b = utils.batchGD(train_data, ground_truth, 1000, learning_rate, epochs)
'''

# open test data
test_data = []

#raw_data = pd.read_csv(testfile, header=None, encoding='big5').values #shape: (4320, 11)

test_data = copy.deepcopy(dup_rawData)

# data processing, change to time-param/hour sequence (4320, 9)
#raw_data = raw_data[:, 2:]
#raw_data = raw_data.astype(float)
#test_data = copy.deepcopy(raw_data)

################################### feature scaling ###################################

#mean = np.mean(test_data, axis=0 )
#std = np.std(test_data, axis=0 )
#test_data = (test_data - mean) / (std + 1e-20)


# data processing, change to time-param/day sequence (240,432)
#test_data = np.reshape(raw_data, (-1, featNum*9 ))

# predict
result = utils.predict(train_data, w, b)
#result = utils.predict(test_data, w, b)

output = np.stack(( result, ground_truth))
output = output.T

output = output.astype(int)



count =0
for data in range(0, output.shape[0]):
    if (output[data][0] == output[data][1]):
        count += 1
#print("index\t", data)
print("Correct ratio= {0:.2f} %".format( 100 * count/int(output.shape[0]) ))

np.savetxt('output.csv',output ,delimiter=',')

#result.to_csv('output.csv',index = False)



#for data in output:
#print(data)











