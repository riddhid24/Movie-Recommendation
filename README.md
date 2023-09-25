# Hybrid Movie Recommendation System

This Python script demonstrates the creation of a hybrid movie recommendation system. The system combines both content-based and collaborative filtering techniques to provide movie recommendations. In this example, the script provides hybrid recommendations for the movie 'Jumanji' and evaluates the model's performance using Root Mean Square Error (RMSE).

## Dependencies

Before running the script, ensure you have the following Python libraries installed:

- pandas
- scikit-learn (sklearn)
- numpy
- math

You can install these libraries using pip if they are not already installed:

```bash
pip install pandas scikit-learn numpy
```

## Data Sources

The script uses the following data sources:

1. **movies_metadata.csv**: Contains information about movies.
2. **keywords.csv**: Contains keywords associated with movies.
3. **credits.csv**: Contains credits information for movies.
4. **ratings_small.csv**: Contains user movie ratings.

For demonstration purposes, the script limits the number of movies and ratings to the first 1000 entries.

## Steps

The script follows these main steps:

1. **Data Loading and Preprocessing**: Loads data from CSV files, preprocesses it, and merges information about movies, keywords, and credits. It also prepares the user-movie rating matrix.

2. **Content-Based Filtering**: Uses TF-IDF vectorization to represent movie overviews as numerical features. Then, it applies Truncated SVD to reduce dimensionality.

3. **Collaborative Filtering**: Applies Truncated SVD to the user-movie rating matrix to reduce dimensionality.

4. **Hybrid Recommendation Function**: Defines a function `get_hybrid_recommendations` that combines content-based and collaborative similarity scores to generate movie recommendations.

5. **Example: Get Hybrid Recommendations for 'Jumanji'**: Demonstrates how to use the recommendation function to get hybrid recommendations for the movie 'Jumanji' and calculates the RMSE for the recommendations.

## Hybrid Recommendations for 'Jumanji'

Below are the top movie recommendations for 'Jumanji' based on the hybrid recommendation system:

1. Tom and Huck
2. The American President
3. GoldenEye
4. Heat
5. Sabrina
6. Waiting to Exhale
7. Toy Story
8. Grumpier Old Men
9. Dracula: Dead and Loving It

The RMSE for these recommendations is approximately 6.33, indicating the model's prediction accuracy for user ratings.

## Notes

- This script is a simplified demonstration and can be extended for larger datasets and more complex recommendation models.
- The number of recommendations should not exceed the number of movies in the validation set to avoid errors.
- Collaborative filtering dimensionality reduction parameters can be adjusted based on your dataset's size and characteristics for optimal results.
- This script can serve as a foundation for building a more comprehensive recommendation system with additional features and optimizations.