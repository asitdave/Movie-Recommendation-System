# Movie Recommendation System



## Project Overview  
This project builds a Movie Recommendation System using the TMDB 5000 Movie Dataset. The goal is to design different types of recommendation engines that help users discover movies they are likely to enjoy based on ratings, metadata, cast/crew, and user behavior.

The notebook explores ranking methods, content-based filtering, and collaborative filtering to evaluate movie popularity, similarity, and personalization.



## Project Highlights  

### Objective  
- Build a ranking system for movies using IMDB’s weighted rating formula.
- Develop content-based recommenders using:
   - Movie overviews & taglines.
   - Cast, crew, genres, and keywords.
- Implement collaborative filtering to personalize recommendations for users.
- Compare different approaches to understand strengths and limitations.

### Concepts & Techniques Used  
- **Movie Ranking System:** 
   - Weighted rating formula based on vote count and average rating

- **Text Preprocessing & NLP:** 
   - Stopword removal, lemmatization, and stemming
   - TF-IDF and Count Vectorizer for feature extraction

- **Content-Based Filtering:**  
  - Using *Overview and Tagline* for recommendation through TF-IDF and cosine similarity
  - Using *Metadata*, combining cast, crew, director and generes with CountVectorizer 
  - Cosine similarity to identify most similar movies

- **Collaborative Filtering:**
   - User-movie rating matrix
   - Centered ratings to account for user biases
   - Experimentation with Surprise library (SVD-based models planned)

### Tools & Libraries  
- **Python** (Pandas, NumPy, Scikit-learn, NLTP, SciPy)  
- **Matplotlib & Seaborn** (data visualization)  
- **Surprise** (collaborative filtering & SVD-based recommendation)  

---