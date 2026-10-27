# Movie Recommendation System



## Project Overview  
This project builds movie recommenders on Kaggle's [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) (TMDB metadata for ~45k movies plus a 100k-rating MovieLens sample). The goal is to compare different types of recommendation engines (popularity, content-based, semantic and collaborative filtering) and see how each one recommends movies.



## Project Highlights  

### Objective  
- Build a ranking system for movies using IMDB's weighted rating formula.
- Develop content-based recommenders using:
   - Movie overviews & taglines (TF-IDF).
   - Cast, director, genres, and keywords (token overlap).
   - Sentence embeddings of overviews, genres and keywords (semantic similarity).
- Implement user-user collaborative filtering to personalize recommendations.

### Concepts & Techniques Used  
- **Movie Ranking System:** 
   - Weighted rating (Bayesian average) based on vote count and average rating, overall and per genre

- **Text Preprocessing & NLP:** 
   - Lowercasing, number normalisation, stopword removal
   - Stemming of keywords
   - TF-IDF and Count Vectorizer for feature extraction

- **Content-Based Filtering:**  
  - *Overview and Tagline* with TF-IDF and cosine similarity (matches exact words)
  - *Metadata*: top-3 cast, director (weighted x3), genres and keywords with a Count Vectorizer
  - *Semantic similarity*: pretrained sentence embeddings (`all-MiniLM-L6-v2`) compared with cosine similarity (matches meaning); vectors are cached to disk
  - Nearest neighbours computed per query instead of a full N x N similarity matrix

- **Collaborative Filtering:**
   - User-user CF: cosine similarity between users' mean-centred rating vectors
   - Predicted rating = user's average + similarity-weighted average of the 30 nearest neighbours' (centred) ratings
   - Personalised recommendations per user from their rating history; users without ratings get none
   - MovieLens ids mapped to TMDB ids with `links_small.csv`

### Tools & Libraries  
- **Python** (Pandas, NumPy, SciPy, Scikit-learn, NLTK)  
- **Sentence-Transformers** (semantic embeddings)  

## How to run

1. Download `movies_metadata.csv`, `credits.csv`, `keywords.csv`, `ratings_small.csv` and `links_small.csv` from [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) into the project folder (or set `DATA_DIR` in the notebook).
2. `pip install -r requirements.txt`
3. Run `Movie-Recommender.ipynb` top to bottom. The first run downloads the embedding model (~90 MB) and embeds all movies (a few minutes on CPU); later runs reuse `movie_embeddings.npz`.

## Next steps
- Quantitative offline evaluation (time-based split, NDCG@10) against popularity baselines
- Combine the semantic, metadata and collaborative signals, then learn the combination with a ranking model
- Item-item CF and matrix factorisation (SVD / ALS)
