import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load data
movies = pd.read_csv('data/movies_metadata.csv')
keywords = pd.read_csv('data/keywords.csv')
credits = pd.read_csv('data/credits.csv')
ratings = pd.read_csv('data/ratings_small.csv')

# For demonstration purposes, limit the number of movies and ratings
movies = movies[:1000]
ratings = ratings[:1000]

# Convert the 'id' column in keywords and credits to a common data type (e.g., str)
keywords['id'] = keywords['id'].astype(str)
credits['id'] = credits['id'].astype(str)

# Merge keywords and credits into the movies DataFrame
movies = movies.merge(keywords, on='id')
movies = movies.merge(credits, on='id')

# Fill NaN values in the 'overview' column with an empty string
movies['overview'] = movies['overview'].fillna('')

# Split the data into training and validation sets
train_movies, val_movies = train_test_split(movies, test_size=0.2, random_state=42)

# Content-based filtering: TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['overview'])

# Create a user-movie rating matrix
user_movie_ratings = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Apply Truncated SVD to reduce dimensionality for content-based filtering
num_components_content = min(len(movies), 1000)  # Set the number of components based on the number of movies
svd_content = TruncatedSVD(n_components=num_components_content)
tfidf_matrix_reduced_content = svd_content.fit_transform(tfidf_matrix)

train_ratings, val_ratings = train_test_split(user_movie_ratings.values, test_size=0.2, random_state=42)

# Apply Truncated SVD to reduce dimensionality for collaborative filtering
num_components_collab = min(user_movie_ratings.shape[1], 1000)  # Set the number of components based on the number of movies
svd_collab = TruncatedSVD(n_components=num_components_collab)
collaborative_matrix = svd_collab.fit_transform(train_ratings)

# Calculate cosine similarity for content-based filtering
cosine_sim_content = linear_kernel(tfidf_matrix_reduced_content, tfidf_matrix_reduced_content)

# Split the user-movie ratings matrix into training and validation sets
train_ratings, val_ratings = train_test_split(user_movie_ratings.values, test_size=0.2, random_state=42)

# Apply Truncated SVD to reduce dimensionality for collaborative filtering
num_components_collab = 10
svd_collab = TruncatedSVD(n_components=num_components_collab)
collaborative_matrix = svd_collab.fit_transform(train_ratings)

# Calculate the cosine similarity for collaborative filtering
collaborative_sim = linear_kernel(collaborative_matrix, collaborative_matrix)

# Function to get hybrid recommendations
def get_hybrid_recommendations(movie_title, num_recommendations=10):
    try:
        # Find the index of the movie in the movies DataFrame
        idx = movies[movies['title'] == movie_title].index[0]

        # Get the content-based and collaborative similarity scores for the movie
        content_sim_scores = cosine_sim_content[idx]
        collaborative_sim_scores = collaborative_sim[idx]

        # Calculate hybrid scores by combining content-based and collaborative filtering
        # Ensure that both similarity scores have the same shape before combining
        min_length = min(len(content_sim_scores), len(collaborative_sim_scores))
        hybrid_scores = (content_sim_scores[:min_length] + collaborative_sim_scores[:min_length]) / 2

        # Get the indices of the top recommended movies
        top_indices = hybrid_scores.argsort()[::-1][:num_recommendations]

        # Get hybrid recommendations
        hybrid_recommendations = movies.iloc[top_indices][['title']]

        return hybrid_recommendations
    except IndexError:
        print(f"Movie '{movie_title}' not found in the dataset.")
        return None

# Example: Get hybrid recommendations for a movie
movie_title = 'Jumanji'
num_recommendations = 10  # Define the number of recommendations

# Ensure that num_recommendations is less than or equal to the number of movies in the validation set
if num_recommendations > val_ratings.shape[1]:
    print("Number of recommendations exceeds the number of movies in the validation set.")
else:
    recommendations = get_hybrid_recommendations(movie_title, num_recommendations)
    if recommendations is not None:
        print(f"Hybrid Recommendations for '{movie_title}':")
        print(recommendations)

        # Evaluate the model's performance using RMSE
        movie_indices = recommendations.index.tolist()

        # Ensure that the number of movie indices does not exceed the size of collaborative_matrix
        if len(movie_indices) == num_recommendations:
            # Find common rows between val_ratings and movie_indices
            common_rows = np.intersect1d(np.arange(val_ratings.shape[0]), movie_indices)

            # Slice both val_ratings and collaborative_matrix using common rows and movie_indices
            val_ratings_sliced = val_ratings[common_rows][:, movie_indices]
            predicted_ratings_sliced = collaborative_matrix[common_rows][:, :num_recommendations]

            # Ensure that val_ratings_sliced and predicted_ratings_sliced have the same shape
            if val_ratings_sliced.shape == predicted_ratings_sliced.shape:
                # Calculate RMSE
                rmse = sqrt(mean_squared_error(val_ratings_sliced, predicted_ratings_sliced))
                print(f"RMSE for recommendations: {rmse}")
            else:
                print("Shapes of val_ratings_sliced and predicted_ratings_sliced do not match.")
        else:
            print("Number of recommendations does not match num_recommendations.")
