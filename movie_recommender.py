# Import libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

############### Helper functions ##################
def get_title_from_index(index):
    return df[df.index==index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]
#####################################################

# Get the data
df = pd.read_csv('movie_dataset.csv')

# Select features
features = ['keywords', 'cast', 'genres', 'director']

# Combine all selected features
for feature in features:
    df[feature] = df[feature].fillna('')
    

def combine_features(row):
    try:
        return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']
    except:
        print("Error:", row)

df["combined_features"] = df.apply(combine_features, axis=1)

# Find the similarity scores
cv = CountVectorizer()

count_matrix = cv.fit_transform(df['combined_features'])

similarity_scores = cosine_similarity(count_matrix)

movie_user_likes = "Avatar"

# get the row in the matrix
movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(similarity_scores[movie_index]))

# Sort based on the score
sorted_similar_movies = sorted(similar_movies, key=lambda x:x[1], reverse=True)

i = 0
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0]))
    i += 1
    if i == 50:
        break
