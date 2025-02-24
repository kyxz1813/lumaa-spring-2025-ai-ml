# Hybrid Movie Recommendation System

This project implements a movie recommendation system that employ both TF-IDF and Sentence Transformer embeddings to suggest movies based on a user’s query.

## Dataset: full_movies.csv

- **Source:**  
  Combined datasets from kaggle. 

  - **Title:** The title of the movie.
  - **Plot:** A short plot description.
  - **genres:** A list (or string) of genres.
  - **keywords:** A list (or string) of relevant keywords.
  - **vote_average:** The average vote or rating for the movie.

## Setup

- **Python Version:**  
  Python 3.7 or higher is recommended.

# Install Dependencies:

Install the required packages using:

pip install -r requirements.txt


# Running the Code

To run the recommendation system from the command line, use the following command:

python recommendation_system.py data/full_movies.csv "I love thrilling action movies set in space, with a comedic twist."


