import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = pd.read_csv('tmdb_5000_movies.csv')

# Merge the two datasets on 'title'
data = pd.merge(movies, credits, on='title')

# Check for missing values in columns used for recommendation
data.dropna(subset=['overview', 'vote_average'], inplace=True)

# Create a user-item matrix using pivot_table
user_movie_matrix = data.pivot_table(index='title', columns='id', values='vote_average', fill_value=0)

# Calculate cosine similarity between movies
movie_similarity = cosine_similarity(user_movie_matrix)

# Create a DataFrame to store movie similarity
movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# Function to get movie recommendations
def get_movie_recommendations(movie_title, num_recommendations=10):
    similar_movies = movie_similarity_df[movie_title]
    similar_movies = similar_movies.sort_values(ascending=False)
    similar_movies = similar_movies.iloc[1:num_recommendations+1]
    recommended_movies = list(similar_movies.index)
    return recommended_movies

# Example: Get recommendations for a movie
movie_title = 'Titanic'
recommendations = get_movie_recommendations(movie_title)
print(f"Recommendations for '{movie_title}':")
for i, movie in enumerate(recommendations, start=1):
    print(f"{i}. {movie}")

# Split data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2)

# Calculate mean rating for all movies in the training set
mean_rating = train_data.groupby('title')['vote_average'].mean().reset_index()

# Merge the mean rating with the test data
test_data = pd.merge(test_data, mean_rating, on='title', how='left')
test_data.rename(columns={'vote_average_x': 'vote_average', 'vote_average_y': 'mean_rating'}, inplace=True)

# Calculate RMSE
rmse = sqrt(mean_squared_error(test_data['vote_average'].fillna(0), test_data['mean_rating'].fillna(0)))
print(f"RMSE: {rmse}")

# Data Visualization
# Histogram of movie ratings
plt.figure(figsize=(8, 5))
sns.histplot(data['vote_average'], bins=20, kde=True)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Bar plot of movie genres
genres = data['genres'].str.split(',').explode().str.strip()
genre_counts = genres.value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=genre_counts, y=genre_counts.index, palette='viridis')
plt.title('Top 10 Movie Genres')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()

# Box plot of movie budgets
plt.figure(figsize=(8, 5))
sns.boxplot(x=data['budget'], palette='Blues')
plt.title('Distribution of Movie Budgets')
plt.xlabel('Budget')
plt.show()

# Data Cleaning and Handling
# Remove rows with missing values (if needed)
data_cleaned = data.dropna()

# Save the cleaned data to a new CSV file (if needed)
# data_cleaned.to_csv('cleaned_movie_data.csv', index=False)


