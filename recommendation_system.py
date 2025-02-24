import sys
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_data(df):
    """
    - Convert list-type columns (genres and keywords) to strings.
    - Create a 'combined_text' column by concatenating Plot, genres, and keywords.
    """

    df['genres'] = df['genres'].apply(lambda x: " ".join(x) if isinstance(x, list) else x)
    df['keywords'] = df['keywords'].apply(lambda x: " ".join(x) if isinstance(x, list) else x)
    df['combined_text'] = df['Plot'] + " " + df['genres'] + " " + df['keywords']
    
    return df

def tfidf_function(text):
    """
    Convert movie text into TF-IDF vectors.
    """
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(text)
    return vectorizer, tfidf_matrix

def compute_embeddings(text_list, model):
    """
    Compute dense sentence embeddings for a list of texts using the given model.
    """
    return model.encode(text_list, convert_to_tensor=True)

def recommend_movies(query, vectorizer, tfidf_matrix, model, embeddings, df, weight_tfidf=0.7, weight_st=0.3, top_n=5):
    """
    Given a user query, compute a combined similarity score using both TF-IDF and Sentence Transformer embeddings.
    
    Parameters:
      - query: the user query string.
      - vectorizer, tfidf_matrix: TF-IDF vectorizer and matrix for the combined text.
      - model, embeddings: Sentence Transformer model and precomputed embeddings for the combined text.
      - df: the dataframe containing the movie data.
      - weight_tfidf: weight for the TF-IDF similarity score.
      - weight_st: weight for the Sentence Transformer similarity score.
      - top_n: number of recommendations to return.
      
    Returns:
      A dictionary containing the top recommended movies with Title, combined similarity score, and vote_average.
    """
    # Compute TF-IDF similarity
    query_tfidf = vectorizer.transform([query])
    tfidf_sim = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    
    # Compute Sentence Transformer similarity
    query = model.encode([query], convert_to_tensor=True)
    st_sim = util.cos_sim(query, embeddings)[0]
    st_sim = st_sim.cpu().numpy() if hasattr(st_sim, "cpu") else st_sim
    st_sim = st_sim.flatten()
    
    # Combine the two similarity scores using a weighted sum
    combined_sim = weight_tfidf * tfidf_sim + weight_st * st_sim
    
    # Get indices of the top_n movies
    top_indices = combined_sim.argsort()[-top_n:][::-1]


    recommendations = []
    for idx in top_indices:
        rec = {
            "Title": df.iloc[idx]['Title'],
            "combined_similarity": float(combined_sim[idx]),
            "vote_average": df.iloc[idx]['vote_average']
        }
        recommendations.append(rec)
    
    return recommendations

def main():
    if len(sys.argv) < 3:
        print("Usage: python recommend_combined_no_df.py <path_to_dataset> \"<user query>\"")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    user_query = sys.argv[2]
    
    # Load and preprocess the dataset
    df = pd.read_csv(dataset_path)
    df = preprocess_data(df)
    
    # Build the TF-IDF matrix for the combined text
    vectorizer, tfidf_matrix = tfidf_function(df['combined_text'])
    
    # Load a pre-trained Sentence Transformer model
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Compute Sentence Transformer embeddings for each movie's combined text
    st_embeddings = compute_embeddings(df['combined_text'].tolist(), st_model)
    
    # Get combined recommendations
    recommendations = recommend_movies(user_query, vectorizer, tfidf_matrix, st_model, st_embeddings, df)
    
    # Print recommendations in a user-friendly format
    print("Top recommendations:")
    for rec in recommendations:
        print(f"Title: {rec['Title']} | Similarity: {rec['combined_similarity']:.4f} | Vote Average: {rec['vote_average']}")
    
if __name__ == "__main__":
    main()
