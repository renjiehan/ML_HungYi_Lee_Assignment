import os
import numpy as np
import pandas as pd
import utils
import copy

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

trainfile = 'train.csv'
testfile = 'test.csv'
moviefile = 'movies.csv'
usersfile = 'users.csv'

# open raw data
train_data = pd.read_csv(trainfile, engine='python').values

n_sample = 500000
user_train = train_data[:,1]
movie_train= train_data[:,2]
y_train = train_data[:,3]

movie_data = pd.read_csv(moviefile, sep='::', engine='python' ).values
users_data = pd.read_csv(usersfile, sep='::', engine='python' ).values

n_users = np.shape(users_data)[0] +1
n_movie = np.shape(movie_data)[0]
n_movie = movie_data[n_movie-1][0] +1

print('User train data number {}'.format(n_users) )
print('User movie data number {}'.format(n_movie) )

movie_data = np.delete(movie_data, 1, axis=1)
movie_feat, genre_seq, n_genre = utils.genre_tag(movie_data)

movie_id = []

for i in range(np.shape(movie_train)[0]):
    movie_id.append( np.array(np.concatenate((movie_train[i], movie_feat[ movie_train[i] ]), axis=None)))

movie_id = np.array(movie_id, dtype=np.int)
#movie_id.flatten()
#movie_id.reshape((-1,19))



#print('Genre number is {}, movie feature size is {}, {}'.format(n_genre, np.shape(movie_id), movie_id))


#model = utils.mf_model(n_users, n_movie, 120)
#model = utils.nn_model(n_users, n_movie)
model = utils.nn_model(n_users, n_movie, movie_feat)
model.summary()

#input()

MODEL_DIR = './model'
MODEL_WEIGHTS_FILE = 'weights.h5'

MODEL_WEIGHTS_FILE = os.path.join(MODEL_DIR, MODEL_WEIGHTS_FILE)

callbacks = [EarlyStopping('val_rmse', patience=3),
             ModelCheckpoint(MODEL_WEIGHTS_FILE,
                             monitor='val_rmse', verbose=1, save_best_only=True)]

for i in range(300):
    if os.path.exists(MODEL_WEIGHTS_FILE):
        print('Load model {}'.format(MODEL_WEIGHTS_FILE))
        model.load_weights(MODEL_WEIGHTS_FILE)
    print('{} Iteration.'.format(i+1))
    model.fit([user_train, movie_train], y_train, batch_size= 256, epochs= 300, validation_split= 0.1, callbacks=callbacks)
    model.save_weights(MODEL_WEIGHTS_FILE)

#user_train = user_train[:1000]
#movie_train = movie_train[:1000]

#result = model.predict([user_train, movie_train], verbose=1)
#result = np.array(result)

#print('RMSE is {}'.format(utils.rmse(y_train, result)))


