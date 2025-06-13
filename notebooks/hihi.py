import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("../data/processed/movie_data_processed.csv", index_col= False)
df["Year"] = df["Year"].fillna("Unknown")

# 1. Simple Recommender


def get_simple_recommendations(df, value, collumn):
    # Lọc các dòng chứa genre
    filtered_df = df[df[collumn].str.contains(value, na=False)]
    
    # Sắp xếp theo trọng số và lấy top 250
    filtered_df = filtered_df.sort_values('wr', ascending=False).head(10)
    
    return filtered_df[['Title', 'Year', 'Genre','imdbVotes', 'imdbRating', collumn]]


# 2. Content based

tf = TfidfVectorizer(analyzer='word', stop_words='english')
tfidf_matrix = tf.fit_transform(df['Plot'])
tfidf_matrix.shape
# def computeCosineSimilarity(word_matrix):
#     cosine_similarity = linear_kernel(word_matrix, word_matrix)
#     return cosine_similarity
def computePearsonCorrelation(word_matrix):
    return np.corrcoef(word_matrix)
indices = pd.Series(df.index, index=df['Title'])
indices
def get_content_based_recommendations(movie_title, similarity_scores):
    # Fetch index of movie based on given title
    movie_idx = indices[movie_title]
    
    # Fetch similarity score of all movies with the given movie
    # Fetch it as a tuple of (index, score)
    similarity_scores = list(enumerate(similarity_scores[movie_idx]))
    
    # Sort the above score
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Pick index and score of 10 most similar movies
    # Skip the 0th index since it is same movie (itself)
    similarity_scores = similarity_scores[1:11]
    
    # Find the indices of these similar movies
    movie_similar_indices = [i[0] for i in similarity_scores]
    
    # Find title of these top movies and return
    return df.iloc[movie_similar_indices][['Title', 'imdbID', 'Year', 'Genre','imdbVotes', 'imdbRating', 'wr']].sort_values(by='wr', ascending=False)
similarity = cosine_similarity(tfidf_matrix)
get_content_based_recommendations('The Dark Knight', similarity)
tags_df = pd.read_csv('../data/processed/tags_processed.csv')
tags_df.info()
tags_grouped = tags_df.groupby('imdbId')['tag'].apply(list).reset_index()
del tags_df
tags_grouped.rename(columns={"imdbId": "imdbID"}, inplace=True)
tags_grouped
df = df.merge(tags_grouped, on='imdbID', how='left')

tag_counts = df.apply(lambda x: pd.Series(x['tag']),axis=1).stack().reset_index(level=1, drop=True)
tag_counts.name = 'tag'
tag_counts = tag_counts.value_counts()
tag_counts[:5]
tag_counts = tag_counts[tag_counts > 1]
def filter_tags(x):
    words = []
    for i in x:
        if i in tag_counts:
            words.append(i)
    return words
df["tag"] = df["tag"].fillna('')

reco_features = ['Title', 'Director', 'Actors', 'tag', 'Genre']
def cleanUpData(data):
    if isinstance(data, list):
        return [str.lower(val.replace(" ", "")) for val in data]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(data, str):
            return str.lower(data.replace(" ", ""))
        else:
            return ''
# Apply data cleanup to reco features
modified_features = ['Director', 'Actors', 'tag', 'Genre']

for feature in modified_features:
    df[feature] = df[feature].apply(cleanUpData)
    

# Chuyển đổi các cột thành danh sách
df['Director'] = df['Director'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
df['Actors'] = df['Actors'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
df['Genre'] = df['Genre'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

def createSoup(data):
    # Loại bỏ trùng lặp trong mỗi danh sách trước khi nối
    director = ' '.join(data['Director'])
    actors = ' '.join(data['Actors'])
    tag = ' '.join(set(data['tag']))  # Dùng set để loại bỏ trùng lặp
    genre = ' '.join(data['Genre'])
    
    # Ghép tất cả lại thành chuỗi duy nhất
    return f"{director} {actors} {tag} {genre}"

createSoup(df.iloc[0,:])
# Create a new feature Soup with mixed data
df['soup'] = df.apply(createSoup, axis=1)

reco_features = ['Title', 'Director', 'Actors', 'tag', 'Genre', 'soup']

# Define a CountVectorizer Object
cntVec = CountVectorizer(stop_words='english')

# Remove NaN from soup with empty strings
df['soup'] = df['soup'].fillna('')

# Construct CountVectorizer matrix by fitting and transforming the data
cntVec_matrix = cntVec.fit_transform(df['soup'])



# Topmost frequently occuring words
words = cntVec.get_feature_names_out()
counts = cntVec_matrix.sum(axis=0).reshape(-1,1).tolist()

word_count = dict(sorted(zip(words, counts), key=lambda x : x[1], reverse=True)[:20])

# Find recommendations based on Cosine Similarity
similarity = cosine_similarity(cntVec_matrix)
get_content_based_recommendations('Spectre', similarity)
# 3. Item-based Collaborative Filtering

df_ratings = pd.read_csv('../data/processed/ratings_processed.csv')

df_ratings = df_ratings.merge(df[['imdbID', 'Title']], on= 'imdbID', how= 'inner')

# Create User-Item interaction matrix
matrix = df_ratings.pivot_table(index='userId', columns='Title', values='rating')

# Free memory
del df_ratings

def get_collaborative_filtering_recommendations(movie):
    
    # Fetch ratings for movie
    movie_user_rating = matrix[movie]

    # Find correlation between movies
    similar_to_movie= matrix.corrwith(movie_user_rating)

    # Getting correlated movies
    corr_movies = pd.DataFrame(similar_to_movie, columns=['Correlation'])
    corr_movies = corr_movies.sort_values(by='Correlation', ascending=False)
    
    corr_movies_indeces = corr_movies[1:11].index
    
    return df[df['Title'].isin(corr_movies_indeces)][['Title', 'imdbID', 'Year', 'Genre','imdbVotes', 'imdbRating', 'wr']].sort_values(by='wr', ascending=False)

# 4. Weighted-Mixed Hybrid

def get_hybrid_recommendations(movie):
    content_based_recommends = get_content_based_recommendations(movie, similarity)
    collaborative_filtering_recommends = get_collaborative_filtering_recommendations(movie)
    
    # Combine 2 recommendations lists
    hybrid_recommends = pd.concat([content_based_recommends, collaborative_filtering_recommends], ignore_index=True) \
                            .sort_values(by='wr', ascending=False) \
                            .drop_duplicates(['Title'],ignore_index= True)
    
    return hybrid_recommends

