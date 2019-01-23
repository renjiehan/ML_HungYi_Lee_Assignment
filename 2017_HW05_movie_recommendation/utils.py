import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential, load_model, Model
from keras.utils import to_categorical
from keras.layers import Input, Embedding, Activation, Flatten, Conv2D, Dense, Dropout, Dot, Add, Concatenate, Reshape
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.regularizers import l2

def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1., 5.)
    return K.sqrt(K.mean(K.square((y_true - y_pred))))

def genre_tag(data):
    id = data[:,0]
    data = data[:,1]
    sz_data = np.shape(data)[0]
    genre_tag = []
    
    movie_feat = np.zeros((id[-1]+1,30), dtype=np.int)
    
    for i in range(sz_data):
        data[i] = data[i].split('|')
        
        for j in range(np.shape(data[i])[0]):
            try:
                feat = genre_tag.index(data[i][j])
                movie_feat[ id[i] ][feat] = 1
            except:
                genre_tag.append(data[i][j] )
                feat = genre_tag.index(data[i][j])
                movie_feat[ id[i] ][feat] = 1

    #data = np.array(data, dtype=np.int) # fail because different length is list
    num_genre = np.shape(genre_tag)[0]
    movie_feat = movie_feat[:,:num_genre]
    
    return movie_feat, genre_tag, num_genre


def mf_model(n_users, n_movies, latent_dim=15):
    print('\nRun with matrix factorization model\n')
    dropout_rate = 0.1
    user_input = Input(shape=(1,))
    movie_input= Input(shape=(1,))

    user_vec = Embedding(n_users, latent_dim,
                         embeddings_regularizer=l2(1e-5), input_length=1) (user_input)
    user_vec = Reshape((latent_dim,))(user_vec)
    user_vec = Dropout(dropout_rate)(user_vec)
    #user_vec = Flatten()(user_vec)
    movie_vec= Embedding(n_movies, latent_dim,
                         embeddings_regularizer=l2(1e-5), input_length=1) (movie_input)
    movie_vec = Reshape((latent_dim,))(movie_vec)
    movie_vec = Dropout(dropout_rate)(movie_vec)
    #movie_vec = Flatten()(movie_vec)

    print('Complete vector initialization.')
    print('Print singel movie vector {}\n'.format(movie_vec))

    user_bias = Embedding(n_users, 1, embeddings_initializer='zeros',
                          embeddings_regularizer=l2(1e-5), input_length=1) (user_input)
    user_bias = Flatten()(user_bias)
    movie_bias = Embedding(n_movies, 1, embeddings_initializer='zeros',
                           embeddings_regularizer=l2(1e-5), input_length=1) (movie_input)
    movie_bias = Flatten()(movie_bias)

    r_hat = Dot(axes=1)([user_vec, movie_vec])
    r_hat = Add()([user_bias, movie_bias, r_hat])

    model = Model([user_input, movie_input], r_hat)
    model.compile(loss='mse', optimizer='adamax', metrics=[rmse])

    return model

# movie feature = 19
def nn_model(n_users, n_movies, latent_dim=16):
    print('\nRun with NN model\n')
    dropout_rate = 0.1
    user_input = Input(shape=(1,))
    movie_input= Input(shape=(19,))
    
    user_vec = Embedding(n_users, latent_dim) (user_input)
    user_vec = Flatten()(user_vec)
    movie_vec= Embedding(n_movies, latent_dim) (movie_input)
    movie_vec = Flatten()(movie_vec)
    
    merge_vec = Concatenate()([user_vec, movie_vec])
    merge_vec = Dropout(dropout_rate)(merge_vec)
    
    hidden = Dense(256, activation='relu')(merge_vec)
    hidden = Dropout(dropout_rate)(hidden)
    hidden = Dense(128, activation='relu')(hidden)
    hidden = Dropout(dropout_rate)(hidden)
    hidden = Dense(64, activation='relu')(hidden)
    hidden = Dropout(0.15)(hidden)
    hidden = Dense(15,  activation='relu')(hidden)
    hidden = Dropout(0.2)(hidden)
    output = Dense(1, activation='relu')(hidden)
    
    model = Model([user_input, movie_input], output)
    model.compile(loss='mse', optimizer='adam', metrics=[rmse])
    
    return model

# movie feature = 1
def nn_model(n_users, n_movies, embedding_matrix=[], latent_dim=18):
    print('\nRun with NN model\n')
    dropout_rate = 0.1
    user_input = Input(shape=(1,))
    movie_input= Input(shape=(1,))
    
    user_vec = Embedding(n_users, latent_dim) (user_input)
    user_vec = Flatten()(user_vec)
    if  (len(embedding_matrix) ) :
        movie_vec= Embedding(n_movies, latent_dim, weights=[embedding_matrix], trainable=False) (movie_input)
        movie_vec = Flatten()(movie_vec)

    else:
        movie_vec= Embedding(n_movies, latent_dim) (movie_input)
        movie_vec = Flatten()(movie_vec)

    
    merge_vec = Concatenate()([user_vec, movie_vec])
    merge_vec = Dropout(dropout_rate)(merge_vec)
    
    hidden = Dense(324, activation='relu')(merge_vec)
    hidden = Dropout(dropout_rate)(hidden)
    hidden = Dense(162, activation='relu')(hidden)
    hidden = Dropout(dropout_rate)(hidden)
    hidden = Dense(81, activation='relu')(hidden)
    hidden = Dropout(0.15)(hidden)
    hidden = Dense(18,  activation='relu')(hidden)
    hidden = Dropout(0.2)(hidden)
    output = Dense(1, activation='relu')(hidden)
    
    model = Model([user_input, movie_input], output)
    model.compile(loss='mse', optimizer='adam', metrics=[rmse])
    
    return model

