import pandas as pd
import numpy as np
import datetime
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

genome_scores_data = pd.read_csv('genome-scores.csv')
movies_data = pd.read_csv('movies.csv')
ratings_data = pd.read_csv('ratings.csv')

#################################################################
#################################################################
# ml_users

users_df = pd.DataFrame(ratings_data['userId'].unique(), columns=['userId'])

#save users data
users_df.to_csv('graphdb/ml_users.csv', sep='|', header=True, index=False)

#################################################################
#################################################################
# ml_movies

movies_df = movies_data.drop('genres', axis = 1)

#calculate mean of ratings for each movies
agg_rating_avg = ratings_data.groupby(['movieId']).agg({'rating': np.mean}).reset_index()
agg_rating_avg.columns = ['movieId', 'rating_mean']

movies_df = movies_df.merge(agg_rating_avg, left_on='movieId', right_on='movieId', how='left')

#save movies data
movies_df.to_csv('graphdb/ml_movies.csv', sep='|', header=True, index=False)

#################################################################
#################################################################
# ml_genres

genres = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
    "(no genres listed)"]

genres_df = pd.DataFrame(genres, columns=['genres'])

#save genres data
genres_df.to_csv('graphdb/ml_genres.csv', sep='|', header=True, index=False)

#################################################################
#################################################################
# ml_users_movies

users_movies_df = ratings_data.drop('timestamp', axis = 1)

#save users movies data
users_movies_df.to_csv('graphdb/ml_users_movies.csv', sep='|', header=True, index=False)

# ml_movies_genres

movies_genres_df = movies_data.drop('title', axis = 1)

#define a function to split genres field
def get_movie_genres(movieId):
    movie = movies_genres_df[movies_genres_df['movieId']==movieId]
    genres = movie['genres'].tolist()
    df = pd.DataFrame([b for a in [i.split('|') for i in genres] for b in a], columns=['genres'])
    df.insert(loc=0, column='movieId', value=movieId)
    return df

#create empty df
movies_genres=pd.DataFrame(columns=['movieId','genres'])

#dummy variables for checking time 
a1 = [10,100,1000,3000,5000,10000,15000,20000,25000]
b1 = 0

for x in movies_genres_df['movieId'].tolist():
    b1 += 1
    if b1 in a1: print(b1, str(datetime.datetime.now()))
    movies_genres=movies_genres.append(get_movie_genres(x))

#save movies genres data
movies_genres.to_csv('graphdb/ml_movies_genres.csv', sep='|', header=True, index=False)

#################################################################
#################################################################
# ml_users_genres

#join to movies data to get genre information
user_genres_df = ratings_data.merge(movies_data, left_on='movieId', right_on='movieId', how='left')

#drop columns that are not used
user_genres_df.drop(['movieId','rating','timestamp','title'], axis = 1, inplace=True)

#define a funtion to get the most genre, it is based on counts of genre per user
def get_popular_genre(userId):
    user = user_genres_df[user_genres_df['userId']==userId]
    genres = user['genres'].tolist()
    movie_list = [b for a in [i.split('|') for i in genres] for b in a]
    counter = Counter(movie_list)
    return counter.most_common(1)[0][0]

#create empty df
users_genres = pd.DataFrame(columns=['userId','genre'])

#dummy variables for checking time 
a2 = [10,100,1000,5000,10000,25000,50000,75000,100000, 125000]
b2 = 0

for x in user_df['userId'].tolist():
    b2 += 1
    if b2 in a2: print(b2, str(datetime.datetime.now()))
    users_genres=users_genres.append(pd.DataFrame([[x,get_popular_genre(x)]], columns=['userId','genre']))

#save users genres data
users_genres.to_csv('graphdb/ml_users_genres.csv', sep='|', header=True, index=False)


#################################################################
#################################################################
# ml_movies_similarity

#################################################################
# mov_tag_df

scores_pivot = genome_scores_data.pivot_table(index = ["movieId"],columns = ["tagId"],values = "relevance").reset_index()

#join with movies data to get all movieIds 
mov_tag_df = movies_data.merge(scores_pivot, left_on='movieId', right_on='movieId', how='left')

#fill null values and drop columns that are not used
mov_tag_df = mov_tag_df.fillna(0)
mov_tag_df = mov_tag_df.drop(['title','genres'], axis = 1)

#################################################################
# mov_genres_df
mov_genres_df = movies_data.drop('title', axis = 1)

#define function to set genders column if exists or not
def set_genres(genres,col):
    if genres in col.split('|'): return 1
    else: return 0

mov_genres_df["Action"] = mov_genres_df.apply(lambda x: set_genres("Action",x['genres']), axis=1)
mov_genres_df["Adventure"] = mov_genres_df.apply(lambda x: set_genres("Adventure",x['genres']), axis=1)
mov_genres_df["Animation"] = mov_genres_df.apply(lambda x: set_genres("Animation",x['genres']), axis=1)
mov_genres_df["Children"] = mov_genres_df.apply(lambda x: set_genres("Children",x['genres']), axis=1)
mov_genres_df["Comedy"] = mov_genres_df.apply(lambda x: set_genres("Comedy",x['genres']), axis=1)
mov_genres_df["Crime"] = mov_genres_df.apply(lambda x: set_genres("Crime",x['genres']), axis=1)
mov_genres_df["Documentary"] = mov_genres_df.apply(lambda x: set_genres("Documentary",x['genres']), axis=1)
mov_genres_df["Drama"] = mov_genres_df.apply(lambda x: set_genres("Drama",x['genres']), axis=1)
mov_genres_df["Fantasy"] = mov_genres_df.apply(lambda x: set_genres("Fantasy",x['genres']), axis=1)
mov_genres_df["Film-Noir"] = mov_genres_df.apply(lambda x: set_genres("Film-Noir",x['genres']), axis=1)
mov_genres_df["Horror"] = mov_genres_df.apply(lambda x: set_genres("Horror",x['genres']), axis=1)
mov_genres_df["Musical"] = mov_genres_df.apply(lambda x: set_genres("Musical",x['genres']), axis=1)
mov_genres_df["Mystery"] = mov_genres_df.apply(lambda x: set_genres("Mystery",x['genres']), axis=1)
mov_genres_df["Romance"] = mov_genres_df.apply(lambda x: set_genres("Romance",x['genres']), axis=1)
mov_genres_df["Sci-Fi"] = mov_genres_df.apply(lambda x: set_genres("Sci-Fi",x['genres']), axis=1)
mov_genres_df["Thriller"] = mov_genres_df.apply(lambda x: set_genres("Thriller",x['genres']), axis=1)
mov_genres_df["War"] = mov_genres_df.apply(lambda x: set_genres("War",x['genres']), axis=1)
mov_genres_df["Western"] = mov_genres_df.apply(lambda x: set_genres("Western",x['genres']), axis=1)
mov_genres_df["(no genres listed)"] = mov_genres_df.apply(lambda x: set_genres("(no genres listed)",x['genres']), axis=1)

#not need genres anymore
mov_genres_df.drop('genres', axis = 1, inplace=True)

#################################################################
# mov_rating_df

movies = movies_data.drop('genres', axis = 1)

#define function to extract year
def set_year(title):
    year = title.strip()[-5:-1]
    if unicode(year, 'utf-8').isnumeric() == True: return int(year)
    else: return 1800

#add year field
movies['year'] = movies.apply(lambda x: set_year(x['title']), axis=1)

#define function to group years
def set_year_group(year):
    if (year < 1900): return 0
    elif (1900 <= year <= 1975): return 1
    elif (1976 <= year <= 1995): return 2
    elif (1996 <= year <= 2003): return 3
    elif (2004 <= year <= 2009): return 4
    elif (2010 <= year): return 5
    else: return 0

movies['year_group'] = movies.apply(lambda x: set_year_group(x['year']), axis=1)

#no need title and year fields
movies.drop(['title','year'], axis = 1, inplace=True)

#calculate mean and counts of ratings for each movies
agg_movies_rat = ratings_data.groupby(['movieId']).agg({'rating': [np.size, np.mean]}).reset_index()
agg_movies_rat.columns = ['movieId','rating_counts', 'rating_mean']

#define function to group rating counts
def set_rating_group(rating_counts):
    if (rating_counts <= 1): return 0
    elif (2 <= rating_counts <= 10): return 1
    elif (11 <= rating_counts <= 100): return 2
    elif (101 <= rating_counts <= 1000): return 3
    elif (1001 <= rating_counts <= 5000): return 4
    elif (5001 <= rating_counts): return 5
    else: return 0

agg_movies_rat['rating_group'] = agg_movies_rat.apply(lambda x: set_rating_group(x['rating_counts']), axis=1)

#no need rating_counts field
agg_movies_rat.drop('rating_counts', axis = 1, inplace=True)

mov_rating_df = movies.merge(agg_movies_rat, left_on='movieId', right_on='movieId', how='left')
mov_rating_df = mov_rating_df.fillna(0)

#################################################################
# calculate similarity with using cosine similarity function

#before calculate cosine similarity, set movieId field as index
mov_tag_df = mov_tag_df.set_index('movieId')
mov_genres_df = mov_genres_df.set_index('movieId')
mov_rating_df = mov_rating_df.set_index('movieId')

#cosine similarity for mov_tag_df
cos_tag = cosine_similarity(mov_tag_df.values)*0.5

#cosine similarity for mov_tag_df
cos_genres = cosine_similarity(mov_genres_df.values)*0.25

#cosine similarity for mov_tag_df
cos_rating = cosine_similarity(mov_rating_df.values)*0.25

#mix
cos = cos_tag+cos_genres+cos_rating

#################################################################
#create df
cols = mov_tag_df.index.values
inx = mov_tag_df.index
movies_sim = pd.DataFrame(cos, columns=cols, index=inx)

#define function to extract the most 5 similar movies for each movies
def get_similar(movieId):
    df = movies_sim.loc[movies_sim.index == movieId].reset_index(). \
            melt(id_vars='movieId', var_name='sim_moveId', value_name='relevance'). \
            sort_values('relevance', axis=0, ascending=False)[1:6]
    return df 

#create empty df
ml_movies_similarity = pd.DataFrame(columns=['movieId','sim_moveId','relevance'])

#dummy variables for checking time 
a3 = [10,100,1000,5000,10000,20000,30000]
b3 = 0

for x in movies_sim.index.tolist():
    b3 += 1
    if b3 in a3: print(b3, str(datetime.datetime.now()))
    ml_movies_similarity=ml_movies_similarity.append(get_similar(x))

#save users genres data
ml_movies_similarity.to_csv('graphdb/ml_movies_similarity.csv', sep='|', header=True, index=False)


#################################################################
#################################################################
# recommendation

# Data is ready to import graph db. We already calculate similarity of movies so we can create a function to get 5 similar movies

def movie_recommender(movieId):
    df = movies_sim.loc[movies_sim.index == movieId].reset_index(). \
            melt(id_vars='movieId', var_name='sim_moveId', value_name='relevance'). \
            sort_values('relevance', axis=0, ascending=False)[1:6]
    df['sim_moveId'] = df['sim_moveId'].astype(int)
    sim_df = movies_data.merge(df, left_on='movieId', right_on='sim_moveId', how='inner'). \
                sort_values('relevance', axis=0, ascending=False). \
                loc[: , ['movieId_y','title','genres']]. \
                rename(columns={ 'movieId_y': "movieId" })
    return sim_df 

#################################################################

#check movieId 1
print(movies_data[movies_data['movieId'] == 3793])

#get recommendation for Toy Story
print(movie_recommender(1))

#get recommendation for Inception
print(movie_recommender(79132))

#get recommendation for X-Men
print(movie_recommender(3793))

#get recommendation for Lock, Stock & Two Smoking Barrels
print(movie_recommender(2542))

#get recommendation for Casino Royale
print(movie_recommender(49272))

#get recommendation for Hangover Part II
print(movie_recommender(86911))

#get recommendation for Eternal Sunshine of the Spotless Mind
print(movie_recommender(7361))

#get recommendation for Scream 4
print(movie_recommender(86295))