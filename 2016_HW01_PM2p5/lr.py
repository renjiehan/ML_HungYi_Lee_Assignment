import os
import numpy as np
import pandas as pd
import utils
import copy

trainfile = 'data/train.csv'
testfile = 'data/test.csv'

# open raw data
raw_data = pd.read_csv(trainfile, encoding='big5').values #shape: (4320, 27)
raw_data[raw_data == 'NR'] = '0.0'

raw_data = raw_data[:, 3:]  # (4320, 24)
raw_data = raw_data.astype(float)

############ data processing, change to time-param/hour (24*20*12=5760, 18) ##############
##############################  (4320,24) to (5760,18) ###################################
featNum = 18
list1 = []

sz_init_data = raw_data.shape[0]    # total data number (row number= 4320)

for i in range(0, sz_init_data, featNum):
    arr1 = raw_data[i:i+featNum,:].T
    list1.append(arr1)

train_data = copy.deepcopy(list1)
dup_rawData = copy.deepcopy(train_data)

############ data processing, change to time-param/day (12*20=240, 18*24=432) ##############
# data processing, change to time-param/day sequence (240,432)
featNum = 18
hour24 = 24

train_data = np.reshape(train_data, (-1, featNum * hour24 ))
dup_rawData = np.reshape(dup_rawData, (-1, featNum * hour24 ))

# data sampling, 240 days * 15 copies/day, featureNum = 18 * 9 hours, data number= 3600
sz_train_data = train_data.shape[0]

augment_data = []
ground_truth = []
op_ground_truth = []

hourFeat = 9

for days in range(sz_train_data):
    for hour in range(0,hour24-hourFeat):
        augment_data.append(train_data[days][ (hour*featNum) : (hour+hourFeat) * featNum])
        #op_ground_truth.append(train_data[days][ (hour+hourFeat) * featNum + 9])
        ground_truth.append(dup_rawData[days][ (hour+hourFeat) * featNum + 9])

'''
##### only take PM 2.5 feature #####
aug_9feat = []

for j in range(np.shape(augment_data)[0]):
    temp = []
    temp_gr = []
    for i in range(9,aug_featNum, 18):
        temp.append(augment_data[j][i])
    aug_9feat.append( temp)

print("aug new feat", np.shape(aug_9feat))
#print("ground truth", ground_truth)
'''

################################### feature scaling ###################################
mean = np.mean(augment_data, axis=0 )
std = np.std(augment_data, axis=0 )
augment_data = (augment_data - mean) / (std + 1e-20)


################# training process, SGD #################
'''
w = []
b = []

epochs = 100
learning_rate = 1e0    # best so far

w,b = utils.SGD(augment_data, ground_truth, learning_rate, epochs)
'''
################# training process, mini-batch SG #################

w = []
b = []

epochs = 10000
learning_rate = 1e-2    # best so far (100, 1e-3), (3600,1e-2)

w,b = utils.batchGD(augment_data, ground_truth, 3600, learning_rate, epochs)

# open test data
test_data = []

raw_data = pd.read_csv(testfile, header=None, encoding='big5').values #shape: (4320, 11)
raw_data[raw_data == 'NR'] = '0.0'
sz_test_data = raw_data.shape[0] # total data number (4320)

# data processing, change to time-param/hour sequence (4320, 9)
raw_data = raw_data[:, 2:]
raw_data = raw_data.astype(float)
test_data = copy.deepcopy(raw_data)

################################### feature scaling ###################################
'''
mean = np.mean(test_data, axis=0 )
std = np.std(test_data, axis=0 )
test_data = (test_data - mean) / (std + 1e-20)
'''

# data processing, change to time-param/day sequence (240,432)
test_data = np.reshape(raw_data, (-1, featNum*9 ))

# predict
result = utils.predict(augment_data, w, b)
#result = utils.predict(test_data, w, b)
#output = np.stack(( result, ground_truth))
#output = output.T

#output = output.astype(int)


test_ground = 'data/ans.csv'

# open raw data
raw_data = pd.read_csv(test_ground, encoding='big5').values #shape: (4320, 27)
#raw_data[raw_data == 'NR'] = '0.0'

raw_data = raw_data[:, 1:]  # (4320, 24)
raw_data = raw_data.astype(float)

output = np.stack(( result, ground_truth))
output = output.T

'''
count =0
for data in range(0, output.shape[0]):
    if (output[data][0] == output[data][1]):
        count += 1
    #print("index\t", data)
print("Correct ratio= {0:.2f} %".format( 100 * count/int(output.shape[0]) ))
'''

loss = 0
for data in range(0, output.shape[0]):
    error = output[data][0] - output[data][1]
    loss += np.square(error)

loss = loss/ int(output.shape[0])
loss = np.sqrt(loss)

print("loss= {0:.2f} ".format( loss) )

'''
import csv
with open("output.txt", 'w+') as f:
    writer = csv.writer(f)
        #for data in result:
    writer.writerows(result.astype(list))
'''

