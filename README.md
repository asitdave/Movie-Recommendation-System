# Movie Recommendation System



## Project Overview  
This project builds and evaluates movie recommenders on Kaggle's [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) (TMDB metadata for ~45k movies plus a 100k-rating MovieLens sample). The goal is to compare different types of recommendation engines (popularity, content-based, collaborative filtering and a hybrid) and measure how well each one ranks movies a user will like.



## Project Highlights  

### Objective  
- Build a ranking system for movies using IMDB's weighted rating formula.
- Develop content-based recommenders using:
   - Movie overviews & taglines.
   - Cast, director, genres, and keywords.
- Implement collaborative filtering (item-item and SVD) to personalize recommendations.
- Compare all approaches with an offline, time-based evaluation.

### Concepts & Techniques Used  
- **Movie Ranking System:** 
   - Weighted rating (Bayesian average) based on vote count and average rating, overall and per genre

- **Text Preprocessing & NLP:** 
   - Lowercasing, number normalisation, stopword removal
   - Stemming of keywords
   - TF-IDF and Count Vectorizer for feature extraction

- **Content-Based Filtering:**  
  - *Overview and Tagline* with TF-IDF and cosine similarity
  - *Metadata*: top-3 cast, director (weighted x3), genres and keywords with a Count Vectorizer
  - Nearest neighbours computed per query instead of a full N x N similarity matrix

- **Collaborative Filtering:**
   - Item-item CF on mean-centred ratings
   - SVD matrix factorisation (Surprise)
   - Hybrid: reciprocal rank fusion of SVD and metadata content

- **Offline Evaluation:**
   - Per-user time-based split (most recent 20% of each user's ratings held out)
   - Precision@10, Recall@10, Hit rate@10, NDCG@10 with bootstrap confidence intervals
   - Catalogue coverage and popularity bias
   - RMSE / MAE for rating prediction

### Tools & Libraries  
- **Python** (Pandas, NumPy, SciPy, Scikit-learn, NLTK)  
- **Surprise** (SVD-based collaborative filtering)  
- **Matplotlib** (data visualization)  

## How to run

1. Download `movies_metadata.csv`, `credits.csv`, `keywords.csv`, `ratings_small.csv` and `links_small.csv` from [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) into the project folder (or set `DATA_DIR` in the notebook).
2. `pip install -r requirements.txt`
3. Run `Movie-Recommender.ipynb` top to bottom.
