# find maximum genre tags on single film
genre_tag = []
genres = movie_data[:,2]

for i in range(genres.shape[0]):
    genres[i] = genres[i].split('|')
    
    for j in range(np.shape(genres[i])[0]):
        try:
            genres[i][j] = genre_tag.index(genres[i][j]) +1
        except:
            genre_tag.append(genres[i][j] )
            genres[i][j] = genre_tag.index(genres[i][j]) +1




#print('Genres {}'.format(np.shape(genres[0])))
#print('Genres {}'.format(genres) )

