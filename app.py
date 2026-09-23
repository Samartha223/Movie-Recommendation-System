import streamlit as st
import pickle as pkl
import pandas as pd
import requests
import ssl
import urllib3
from requests.adapters import HTTPAdapter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_resource
def load_similarity():
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    return cosine_similarity(vectors)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class SSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)

# One persistent session for the whole app
session = create_session = requests.Session()
session.mount('https://', SSLAdapter())

# Load data once
movies_dict = pkl.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

@st.cache_resource
def load_similarity():
    return pkl.load(open('similarity.pkl', 'rb'))

@st.cache_data
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=6ed11e18d1a2b6610529d686c76f4698&language=en-US"
    try:
        data = session.get(url, timeout=5).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        pass
    return "https://via.placeholder.com/500x750?text=No+Poster"

def recommend(movie):
    similarity = load_similarity()
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(enumerate(distances), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))
    return recommended_movies, recommended_posters

# UI
st.title("Movie Recommendation System")
selected_movie = st.selectbox("Select a movie:", movies['title'].values)

if st.button('Recommend'):
    with st.spinner('Fetching recommendations...'):
        recommended_movies, recommended_posters = recommend(selected_movie)

    st.subheader("Recommended Movies:")
    cols = st.columns(5)
    for col, title, poster in zip(cols, recommended_movies, recommended_posters):
        with col:
            st.image(poster)
            st.caption(title)