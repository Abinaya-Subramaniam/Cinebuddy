# 🎬 CineBuddy

## Live Demo

Check out the live app here: [https://cinebuddy.streamlit.app/](https://cinebuddy.streamlit.app/)

**CineBuddy** is a sleek and smart Streamlit app that lets you:

- Search movies from a curated dataset
- Explore detailed movie info fetched from TMDB API (cast, crew, release date, overview)
- Get personalized movie recommendations based on content similarity
- Analyze the sentiment of your movie reviews instantly using a trained ML model

---

## Features

- **Movie Search**: Select or type a movie title from the curated list.
- **Detailed Info**: See movie posters, overviews, cast and director info fetched in real-time from TMDB API.
- **Personalized Recommendations**: Get 6 similar movies shown with their posters.
- **Sentiment Analysis**: Enter your review and get an instant positive/negative sentiment prediction.
- **Clean UI**: Responsive layout with a nice background image and clear readability.

---

## How to Run Locally

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/cinebuddy.git
   cd cinebuddy
   pip install -r requirements.txt
    ```
   
2. Add your TMDB API key as an environment variable:
 ```bash
   TMDB_API_KEY=your_tmdb_api_key_here
 ```

3. Run the app:
```bash
   streamlit run movie_recommendation.py
 ```
   
