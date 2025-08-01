import streamlit as st
import pandas as pd
import requests
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

st.set_page_config(layout="wide")  


load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

@st.cache_data(show_spinner=True)
def load_data_and_models():
    movies = pd.read_csv("./movies_cleaned.csv", encoding='ISO-8859-1')
    tfidf_vectorizer_rec = joblib.load('tfidf_vectorizer_Rec.pkl')
    tfidf_matrix_rec = joblib.load('tfidf_matrix_Rec.pkl')
    tfidf_vectorizer_sent = joblib.load('tfidf_vectorizer.pkl')
    logistic_model = joblib.load('logistic_model.pkl')
    cosine_sim = cosine_similarity(tfidf_matrix_rec)
    indices = pd.Series(movies.index, index=movies['Series_Title'].str.lower())
    return movies, tfidf_vectorizer_rec, tfidf_matrix_rec, tfidf_vectorizer_sent, logistic_model, cosine_sim, indices

def fetch_movie_details(title):
    try:
        with st.spinner("Fetching movie details..."):
            url = (
                f"https://api.themoviedb.org/3/search/movie"
                f"?api_key={TMDB_API_KEY}&query={title}&sort_by=release_date.desc"
            )
            res = requests.get(url).json()
            results = res.get('results', [])
            if not results:
                return None
            
            filtered = [m for m in results if m['title'].lower() == title.lower()]
            movie = filtered[0] if filtered else results[0]
            movie_id = movie['id']

            details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=credits"
            details = requests.get(details_url).json()
            return details
    except Exception as e:
        st.error(f"Error fetching movie details: {e}")
    return None

def recommend_movies(title, indices, cosine_sim, movies):
    title = title.lower()
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:7]  
    return [movies.iloc[i[0]]['Series_Title'] for i in sim_scores]

def fetch_movie_poster(tmdb_movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={tmdb_movie_title}"
    res = requests.get(url).json()
    results = res.get('results', [])
    if results:
        poster_path = results[0].get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w300{poster_path}"
    return None

def predict_sentiment(review, tfidf_vectorizer_sent, logistic_model):
    review_vector = tfidf_vectorizer_sent.transform([review])
    pred = logistic_model.predict(review_vector)[0]
    return pred

movies, tfidf_vectorizer_rec, tfidf_matrix_rec, tfidf_vectorizer_sent, logistic_model, cosine_sim, indices = load_data_and_models()

st.title("🎬 CineBuddy")
st.write("Your go-to app to search movies, explore details, get personalized recommendations, and instantly analyze the sentiment of your reviews for a smarter movie experience!")


movie_titles = movies['Series_Title'].unique()
selected_movie = st.selectbox(
    "Select a movie from the list or start typing",
    options=[""] + list(movie_titles),
    index=0,
    help="Start typing to search movies"
)

if selected_movie:
    details = fetch_movie_details(selected_movie)
    if details:
        poster_path = details.get('poster_path')
        poster_url = f"https://image.tmdb.org/t/p/w400{poster_path}" if poster_path else None

        col1, col2, col3 = st.columns([3, 8, 6], gap="large")

        with col1:
            if poster_url:
                st.image(poster_url, use_container_width=True)
            else:
                st.write("No poster available")

        with col2:
            st.header(details.get('title', ''))
            st.write(f"**Release Date:** {details.get('release_date', 'Unknown')}")
            st.write(details.get('overview', 'No overview available.'))

            cast = details.get('credits', {}).get('cast', [])
            if cast:
                cast_names = ", ".join([member['name'] for member in cast[:7]])
                st.write(f"**Cast:** {cast_names}")

            crew = details.get('credits', {}).get('crew', [])
            directors = [member['name'] for member in crew if member['job'] == 'Director']
            if directors:
                st.write(f"**Director(s):** {', '.join(directors)}")

            review = st.text_area("Leave your review here")
            if st.button("Analyze Sentiment"):
                if review.strip():
                    with st.spinner("Analyzing sentiment..."):
                        sentiment = predict_sentiment(review, tfidf_vectorizer_sent, logistic_model)
                        if sentiment == 1:
                            label = "Positive"
                            emoji = "😊"
                        else:
                            label = "Negative"
                            emoji = "😞"
                        st.write(f"You gave a **{label}** review {emoji}")
                else:
                    st.warning("Please enter a review to analyze.")

        with col3:
            st.subheader("Recommended Movies")

            recs = recommend_movies(details['title'], indices, cosine_sim, movies)
            if recs:
                for i in range(0, len(recs), 2):
                    cols = st.columns(2)
                    for j, rec_title in enumerate(recs[i:i+2]):
                        with cols[j]:
                            rec_poster = fetch_movie_poster(rec_title)
                            if rec_poster:
                                st.image(rec_poster, width=150)
                            st.write(rec_title)
            else:
                st.write("No recommendations found.")

    else:
        st.error("Movie not found in TMDB API.")
