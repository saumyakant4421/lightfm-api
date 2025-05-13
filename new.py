import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import logging
import os
import csv
import heapq
from difflib import get_close_matches
from collections import Counter
import re

BASE_DIR = os.path.abspath(os.path.dirname(__file__)) 

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=os.path.join(BASE_DIR, 'new.log'),  # Use BASE_DIR directly
    filemode='a'
)

# File paths
 # Use script's directory
MOVIELENS_MOVIES = os.path.join(BASE_DIR, "movies.csv")
MOVIELENS_RATINGS = os.path.join(BASE_DIR, "ratings.csv")
MOVIELENS_LINKS = os.path.join(BASE_DIR, "links.csv")
TMDB_PRE_2016 = os.path.join(BASE_DIR, "tmdb_movies_data_10k.csv")
TMDB_2016_2024 = os.path.join(BASE_DIR, "tmdb_main.csv")
FEEDBACK_FILE = os.path.join(BASE_DIR, "user_feedback.csv")

# Weights for hybrid scoring
W_SVD = 0.5
W_CONTENT = 0.45
W_POPULARITY = 0.05

# Load and preprocess datasets
def load_datasets(sample_ratings=False, sample_size=1000000):
    """Load and merge MovieLens and TMDb datasets without API."""
    logging.info("Loading datasets...")
    
    try:
        ml_movies = pd.read_csv(MOVIELENS_MOVIES, dtype={'movieId': str})
        logging.info(f"Loaded MovieLens movies: {len(ml_movies)} rows")
    except Exception as e:
        logging.error(f"Error loading {MOVIELENS_MOVIES}: {e}")
        raise
    
    try:
        if sample_ratings:
            ml_ratings = pd.read_csv(MOVIELENS_RATINGS, dtype={'userId': str, 'movieId': str, 'rating': float})
            ml_ratings = ml_ratings.sample(n=min(sample_size, len(ml_ratings)), random_state=42)
        else:
            ml_ratings = pd.read_csv(MOVIELENS_RATINGS, dtype={'userId': str, 'movieId': str, 'rating': float}, usecols=['userId', 'movieId', 'rating'])
        logging.info(f"Loaded MovieLens ratings: {len(ml_ratings)} rows")
    except Exception as e:
        logging.error(f"Error loading {MOVIELENS_RATINGS}: {e}")
        raise
    
    try:
        ml_links = pd.read_csv(MOVIELENS_LINKS, dtype={'movieId': str, 'tmdbId': str})
        logging.info(f"Loaded MovieLens links: {len(ml_links)} rows")
    except Exception as e:
        logging.error(f"Error loading {MOVIELENS_LINKS}: {e}")
        raise
    
    try:
        tmdb_pre = pd.read_csv(TMDB_PRE_2016, dtype={'id': str})
        tmdb_pre.name = 'TMDb pre-2016'
        logging.info(f"Loaded TMDb pre-2016: {len(tmdb_pre)} rows")
    except Exception as e:
        logging.error(f"Error loading {TMDB_PRE_2016}: {e}")
        raise
    
    try:
        tmdb_post = pd.read_csv(TMDB_2016_2024, quoting=csv.QUOTE_NONNUMERIC, dtype={'id': str})
        tmdb_post.name = 'TMDb 2016-2024'
        logging.info(f"Loaded TMDb 2016-2024: {len(tmdb_post)} rows")
    except Exception as e:
        logging.error(f"Error loading {TMDB_2016_2024}: {e}")
        raise
    
    tmdb_pre = tmdb_pre.rename(columns={'original_title': 'title'})
    tmdb_pre = tmdb_pre[['id', 'title', 'budget', 'revenue', 'cast', 'director', 'tagline',
                         'keywords', 'genres', 'release_date', 'runtime', 'vote_count', 'vote_average']]
    
    for df in [tmdb_pre, tmdb_post]:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        invalid_dates = df[df['release_date'].isna()]
        if not invalid_dates.empty:
            logging.warning(f"Found {len(invalid_dates)} invalid release dates in {df.name}")
            df.drop(invalid_dates.index, inplace=True)
        invalid_titles = df[df['title'].str.lower().eq('title') | df['title'].isna()]
        if not invalid_titles.empty:
            logging.warning(f"Found {len(invalid_titles)} invalid titles in {df.name}")
            df.drop(invalid_titles.index, inplace=True)
        df['release_year'] = df['release_date'].dt.year.fillna(0).astype(int)
        df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0)
        df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0)
        df['genres'] = df['genres'].fillna('Unknown')
        df['cast'] = df['cast'].fillna('Unknown')
        df['director'] = df['director'].fillna('Unknown')
        df['keywords'] = df['keywords'].fillna('')
    
    tmdb_all = pd.concat([tmdb_pre, tmdb_post]).drop_duplicates(subset=['id']).reset_index(drop=True)
    logging.info(f"Combined TMDb dataset: {len(tmdb_all)} rows")
    
    tmdb_all = tmdb_all[tmdb_all['vote_count'] > 20].reset_index(drop=True)
    logging.info(f"Filtered TMDb dataset (vote_count > 20): {len(tmdb_all)} rows")
    
    merged_movies = ml_movies.merge(ml_links, on='movieId', how='left')
    merged_movies = merged_movies.merge(tmdb_all, left_on='tmdbId', right_on='id', how='left', suffixes=('_ml', '_tmdb'))
    logging.info(f"Merged dataset: {len(merged_movies)} rows")
    
    return ml_ratings, merged_movies, tmdb_all

# Train SVD model
def train_svd(ratings):
    """Train SVD model on MovieLens ratings."""
    logging.info("Training SVD model...")
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    svd = SVD(n_factors=50, n_epochs=20, random_state=42)
    svd.fit(trainset)
    
    logging.info("SVD model trained")
    return svd, trainset

# Build content-based features
def build_content_features(movies, merged_movies, watched_tmdb_ids, top_k=50):
    """Create TF-IDF vectors for all movies."""
    logging.info("Building content-based features...")
    
    movies_content = movies.copy()
    movies_content = movies_content.merge(merged_movies[['tmdbId', 'movieId']], left_on='id', right_on='tmdbId', how='left')
    
    if watched_tmdb_ids:
        watched_movies = movies[movies['id'].isin(watched_tmdb_ids)]
        watched_movies = watched_movies.merge(merged_movies[['tmdbId', 'movieId']], left_on='id', right_on='tmdbId', how='left')
        movies_content = pd.concat([movies_content, watched_movies]).drop_duplicates(subset=['id']).reset_index(drop=True)
    
    logging.info(f"Filtered movies for content features: {len(movies_content)} rows")
    
    def combine_metadata(row):
        fields = []
        genres = str(row['genres']) if pd.notnull(row['genres']) else 'Unknown'
        cast = str(row['cast']) if pd.notnull(row['cast']) else 'Unknown'
        director = str(row['director']) if pd.notnull(row['director']) else 'Unknown'
        keywords = str(row['keywords']) if pd.notnull(row['keywords']) else ''
        tagline = str(row['tagline']) if pd.notnull(row['tagline']) else ''
        fields.extend([genres.replace('|', ' ').replace(',', ' ')] * 7)
        fields.extend([cast.replace('|', ' ').replace(',', ' ')] * 8)
        fields.extend([director.replace('|', ' ').replace(',', ' ')] * 4)
        fields.extend([keywords.replace('|', ' ').replace(',', ' ')] * 1)
        fields.extend([tagline.replace('|', ' ').replace(',', ' ')] * 1)
        return ' '.join(fields)
    
    movies_content['metadata'] = movies_content.apply(combine_metadata, axis=1)
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
    tfidf_matrix = tfidf.fit_transform(movies_content['metadata'])
    tfidf_matrix = csr_matrix(tfidf_matrix)
    
    n_movies = len(movies_content)
    sim_dict = {}
    batch_size = 1000
    for start in range(0, n_movies, batch_size):
        end = min(start + batch_size, n_movies)
        batch_matrix = tfidf_matrix[start:end]
        sim_batch = cosine_similarity(batch_matrix, tfidf_matrix)
        
        for i in range(end - start):
            global_i = start + i
            sim_scores = sim_batch[i]
            valid_indices = [idx for idx in range(len(sim_scores)) if not np.isnan(sim_scores[idx]) and idx != global_i]
            top_indices = heapq.nlargest(top_k, valid_indices, key=lambda x: sim_scores[x])
            top_scores = [sim_scores[idx] for idx in top_indices]
            sim_dict[global_i] = [(int(idx), float(score)) for idx, score in zip(top_indices, top_scores)]
    
    logging.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}, Top-{top_k} similarities computed")
    return sim_dict, movies_content

# Normalize popularity
def normalize_popularity(movies):
    """Normalize vote_count and vote_average for popularity score."""
    logging.info("Normalizing popularity metrics...")
    movies['vote_count'] = movies['vote_count'].fillna(0)
    movies['vote_average'] = movies['vote_average'].fillna(0)
    max_vote_count = movies['vote_count'].max() or 1
    max_vote_average = movies['vote_average'].max() or 1
    movies['popularity_score'] = (movies['vote_count'] / max_vote_count * 0.5 +
                                 movies['vote_average'] / max_vote_average * 0.5)
    return movies

# Get content-based recommendations
def get_content_recommendations(movie_id, movies, sim_dict, top_n=30):
    """Get content-based recommendations using sparse similarities."""
    if movie_id not in movies['id'].values:
        logging.warning(f"Movie ID {movie_id} not found in filtered movies")
        return []
    idx = int(movies.index[movies['id'] == movie_id].tolist()[0])
    if idx not in sim_dict:
        logging.warning(f"Similarity index {idx} not in sim_dict")
        return []
    sim_scores = sim_dict[idx]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_n]
    movie_indices = [i for i, _ in sim_scores if i < len(movies)]
    if not movie_indices:
        logging.warning(f"No valid similar movies for movie_id {movie_id}")
        return []
    return movies.iloc[movie_indices][['id', 'title', 'genres', 'director', 'cast']].to_dict('records')

# Get hybrid recommendations
def get_hybrid_recommendations(user_id, svd, trainset, movies, sim_dict, watched_movie_ids, watchlist_movie_ids, watched_ratings, input_years, top_n=15):
    """Generate hybrid recommendations with year weighting and rating boosts."""
    logging.info(f"Generating recommendations for user {user_id}...")
    
    movies = normalize_popularity(movies)
    
    id_mapping = movies[movies['movieId'].notnull()][['id', 'movieId']].set_index('id')['movieId'].to_dict()
    logging.info(f"TMDb id to MovieLens movieId mapping: {len(id_mapping)} entries")
    
    svd_scores = {}
    if user_id in trainset._raw2inner_id_users:
        chunk_size = 1000
        for start in range(0, len(movies), chunk_size):
            chunk = movies.iloc[start:start+chunk_size]
            for _, row in chunk.iterrows():
                if 'movieId' in row and pd.notnull(row['movieId']):
                    movie_id = row['movieId']
                    if movie_id in trainset._raw2inner_id_items:
                        pred = svd.predict(user_id, movie_id)
                        svd_scores[movie_id] = pred.est
    
    watched_tmdb_ids = [movies[movies['movieId'] == mid]['id'].iloc[0] for mid in watched_movie_ids if not movies[movies['movieId'] == mid].empty]
    ratings_dict = dict(watched_ratings)
    for movie_id in watched_movie_ids:
        tmdb_id = id_mapping.get(movies[movies['movieId'] == movie_id]['id'].iloc[0]) if not movies[movies['movieId'] == movie_id].empty else None
        if tmdb_id in ratings_dict:
            svd_scores[movie_id] = ratings_dict[tmdb_id]
        else:
            svd_scores[movie_id] = 4.9
    if watchlist_movie_ids:
        for movie_id in [movies[movies['id'] == mid]['movieId'].iloc[0] for mid in watchlist_movie_ids if not movies[movies['id'] == mid].empty]:
            svd_scores[movie_id] = 4.9
    logging.info(f"Simulated SVD scores for watched: {watched_movie_ids}, watchlist: {watchlist_movie_ids}, ratings: {ratings_dict}")
    
    content_scores = {}
    director_scores = {}
    cast_scores = {}
    rating_similarities = {}
    watched_similarities = {}
    
    preferred_directors = set(movies[movies['id'].isin(watched_tmdb_ids + watchlist_movie_ids)]['director'].dropna())
    preferred_actors = set()
    for cast in movies[movies['id'].isin(watched_tmdb_ids + watchlist_movie_ids)]['cast'].dropna():
        preferred_actors.update(str(cast).split(', '))
    
    # Genre ratings
    genre_ratings = Counter()
    for movie_id, rating in ratings_dict.items():
        movie_row = movies[movies['id'] == movie_id]
        if not movie_row.empty:
            genres = str(movie_row['genres'].iloc[0]).split('|')
            rating_weight = (rating / 5.0) ** 2
            for genre in genres:
                genre_ratings[genre] += rating_weight
    
    if watchlist_movie_ids or watched_tmdb_ids:
        for movie_id in set(watchlist_movie_ids + watched_tmdb_ids):
            if movie_id in movies['id'].values:
                rating = ratings_dict.get(movie_id, 4.9)
                rating_weight = (rating / 5.0) ** 2
                content_recs = get_content_recommendations(movie_id, movies, sim_dict, top_n=30)
                for rec in content_recs:
                    rec_id = rec['id']
                    sim_score = next((score for idx, score in sim_dict[movies.index[movies['id'] == movie_id].tolist()[0]] if movies.iloc[idx]['id'] == rec_id), 0.0)
                    content_scores[rec_id] = content_scores.get(rec_id, 0) + 2.0 * rating_weight * sim_score
                    rating_similarities[rec_id] = rating_similarities.get(rec_id, []) + [rating]
                    watched_similarities[rec_id] = watched_similarities.get(rec_id, []) + [sim_score]
                    if rec['director'] and rec['director'] == movies[movies['id'] == movie_id]['director'].iloc[0]:
                        director_scores[rec_id] = director_scores.get(rec_id, 0) + 1.5 * rating_weight
                    rec_cast = set(str(rec['cast']).split(', ')) if pd.notnull(rec['cast']) else set()
                    movie_cast = set(str(movies[movies['id'] == movie_id]['cast'].iloc[0]).split(', ')) if pd.notnull(movies[movies['id'] == movie_id]['cast'].iloc[0]) else set()
                    if rec_cast.intersection(movie_cast):
                        cast_scores[rec_id] = cast_scores.get(rec_id, 0) + 1.2 * len(rec_cast.intersection(movie_cast)) * rating_weight
    
    watched_genres = set()
    for tmdb_id in watched_tmdb_ids + watchlist_movie_ids:
        movie_row = movies[movies['id'] == tmdb_id]
        if not movie_row.empty:
            genres = str(movie_row['genres'].iloc[0]).split('|')
            watched_genres.update(genres)
    
    # Median year for weighting
    median_year = int(np.median(input_years)) if input_years else 2005
    logging.info(f"Median release year: {median_year}")
    
    core_genres = {'Drama', 'Science Fiction', 'Thriller', 'Crime', 'Romance', 'Music'}
    conditional_exclude = {'Horror', 'Documentary', 'Animation'}
    excluded_genres = conditional_exclude - set(genre_ratings.keys())
    allowed_genres = watched_genres | core_genres | set(genre_ratings.keys())
    allowed_genres.add('Comedy') if {'Music', 'Drama'} <= watched_genres and 'Comedy' in genre_ratings else None
    logging.info(f"Core genres: {core_genres}, Allowed genres: {allowed_genres}, Excluded genres: {excluded_genres}")
    
    final_scores = {}
    director_counter = Counter()
    genre_counter = Counter()
    filter_counts = {
        'total': len(movies),
        'watched': 0,
        'excluded_genres': 0,
        'no_allowed_genres': 0,
        'light_comedy': 0,
        'light_action': 0,
        'light_tone': 0,
        'director_limit': 0,
        'low_year_weight': 0
    }
    
    for movie_id in movies['id']:
        if movie_id in watched_tmdb_ids + watchlist_movie_ids:
            filter_counts['watched'] += 1
            continue
        movie_lens_id = id_mapping.get(movie_id)
        svd_score = svd_scores.get(movie_lens_id, 3.0) if movie_lens_id else 3.0
        content_score = content_scores.get(movie_id, 0)
        director_score = director_scores.get(movie_id, 0)
        cast_score = cast_scores.get(movie_id, 0)
        popularity_score = movies[movies['id'] == movie_id]['popularity_score'].iloc[0]
        
        genres = str(movies[movies['id'] == movie_id]['genres'].iloc[0]).split('|')
        if any(g in excluded_genres for g in genres):
            filter_counts['excluded_genres'] += 1
            continue
        if not any(g in allowed_genres for g in genres):
            filter_counts['no_allowed_genres'] += 1
            continue
        is_light_comedy = 'Comedy' in genres and not any(g in list(genre_ratings.keys()) + ['Music', 'Drama', 'Crime', 'Thriller'] for g in genres)
        if is_light_comedy:
            filter_counts['light_comedy'] += 1
            continue
        is_light_action = ('Action' in genres or 'Adventure' in genres) and not any(g in allowed_genres for g in genres)
        if is_light_action:
            filter_counts['light_action'] += 1
            continue
        keywords = str(movies[movies['id'] == movie_id]['keywords'].iloc[0]).lower() if pd.notnull(movies[movies['id'] == movie_id]['keywords'].iloc[0]) else ''
        is_light_tone = any(kw in keywords for kw in ['light-hearted', 'family-friendly'])
        if is_light_tone:
            filter_counts['light_tone'] += 1
            continue
        
        # Year weighting
        movie_year = movies[movies['id'] == movie_id]['release_year'].iloc[0]
        year_weight = np.exp(-((movie_year - median_year) / 20) ** 2)
        if year_weight < 0.1:
            filter_counts['low_year_weight'] += 1
            continue
        
        matching_core_genres = sum(1 for g in genres if g in core_genres)
        matching_allowed_genres = sum(1 for g in genres if g in allowed_genres)
        genre_rating_boost = sum(genre_ratings.get(g, 0) for g in genres) / len(genres) if genres else 1.0
        genre_boost = (6.0 + matching_core_genres * 2.5 + matching_allowed_genres * 0.6 + genre_rating_boost) if matching_core_genres >= 2 else (4.0 + matching_core_genres * 2.0 + matching_allowed_genres * 0.5 + genre_rating_boost)
        
        director = movies[movies['id'] == movie_id]['director'].iloc[0]
        cast = set(str(movies[movies['id'] == movie_id]['cast'].iloc[0]).split(', ')) if pd.notnull(movies[movies['id'] == movie_id]['cast'].iloc[0]) else set()
        director_boost = 4.0 if director in preferred_directors else 1.0
        actor_boost = 2.0 + 0.8 * len(cast.intersection(preferred_actors)) if cast.intersection(preferred_actors) else 1.0
        
        director_limit = 2 if director in preferred_directors else 1
        if director_counter[director] >= director_limit:
            filter_counts['director_limit'] += 1
            continue
        
        # Diversity penalty
        diversity_penalty = 1.0 / (1 + genre_counter[genres[0]] + director_counter[director])
        
        avg_rating_similarity = np.mean(rating_similarities.get(movie_id, [4.9])) / 5.0 if movie_id in rating_similarities else 1.0
        rating_boost = 1.0 + 0.5 * avg_rating_similarity
        avg_watched_similarity = np.mean(watched_similarities.get(movie_id, [0.0])) if movie_id in watched_similarities else 0.0
        watched_similarity_boost = 1.0 + 1.0 * avg_watched_similarity
        
        final_score = (W_SVD * svd_score / 5.0 +
                       W_CONTENT * (content_score + director_score + cast_score) +
                       W_POPULARITY * popularity_score) * genre_boost * director_boost * actor_boost * rating_boost * watched_similarity_boost * year_weight * diversity_penalty
        final_scores[movie_id] = final_score
        director_counter[director] += 1
        for g in genres:
            genre_counter[g] += 1
    
    logging.info(f"Filter counts: {filter_counts}")
    logging.info(f"Number of candidates after filtering: {len(final_scores)}")
    
    top_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    logging.info(f"Top 20 movies before selection: {[movies[movies['id'] == k]['title'].iloc[0] + f' ({k}, score: {v:.3f}, genres: {movies[movies['id'] == k]['genres'].iloc[0]})' for k, v in top_scores]}")
    
    recommendations = []
    director_counter = Counter()
    sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    for movie_id, score in sorted_scores:
        director = movies[movies['id'] == movie_id]['director'].iloc[0]
        director_limit = 2 if director in preferred_directors else 1
        if len(recommendations) < top_n and director_counter[director] < director_limit:
            recommendations.append((movie_id, score))
            director_counter[director] += 1
        if len(recommendations) >= top_n:
            break
    
    rec_movies = movies[movies['id'].isin([rec[0] for rec in recommendations])][['id', 'title', 'genres', 'director', 'cast', 'release_date']].drop_duplicates()
    
    for _, row in rec_movies.iterrows():
        if row['director'] in preferred_directors:
            logging.info(f"Recommendation '{row['title']}' boosted by director: {row['director']}")
        if set(str(row['cast']).split(', ')).intersection(preferred_actors):
            logging.info(f"Recommendation '{row['title']}' boosted by cast: {row['cast']}")
        if row['id'] in rating_similarities:
            logging.info(f"Recommendation '{row['title']}' rating similarity: {np.mean(rating_similarities[row['id']]):.2f}")
        if row['id'] in watched_similarities:
            logging.info(f"Recommendation '{row['title']}' watched similarity: {np.mean(watched_similarities[row['id']]):.3f}")
    
    genre_counts = rec_movies['genres'].str.split('|', expand=True).stack().value_counts()
    logging.info(f"Genre distribution: {genre_counts.to_dict()}")
    
    logging.info(f"Generated {len(rec_movies)} recommendations for user {user_id}")
    return rec_movies

# Fallback: Popular movies
def get_popular_movies(movies, watched_tmdb_ids, watchlist_movie_ids, input_years, top_n=15):
    """Get popular movies with dynamic genre filtering and year weighting."""
    logging.info("Generating popular movies fallback...")
    
    movies = normalize_popularity(movies)
    
    watched_genres = set()
    genre_ratings = Counter()
    for tmdb_id in watched_tmdb_ids + watchlist_movie_ids:
        movie_row = movies[movies['id'] == tmdb_id]
        if not movie_row.empty:
            genres = str(movie_row['genres'].iloc[0]).split('|')
            watched_genres.update(genres)
            # Assume default rating for watchlist
            rating_weight = 0.81  # (4.5 / 5.0) ** 2
            for genre in genres:
                genre_ratings[genre] += rating_weight
    
    median_year = int(np.median(input_years)) if input_years else 2005
    logging.info(f"Median release year: {median_year}")
    
    core_genres = {'Drama', 'Science Fiction', 'Thriller', 'Crime', 'Romance', 'Music'}
    conditional_exclude = {'Horror', 'Documentary', 'Animation'}
    excluded_genres = conditional_exclude - set(genre_ratings.keys())
    allowed_genres = watched_genres | core_genres | set(genre_ratings.keys())
    allowed_genres.add('Comedy') if {'Music', 'Drama'} <= watched_genres and 'Comedy' in genre_ratings else None
    logging.info(f"Core genres: {core_genres}, Allowed genres: {allowed_genres}, Excluded genres: {excluded_genres}")
    
    preferred_directors = set(movies[movies['id'].isin(watched_tmdb_ids + watchlist_movie_ids)]['director'].dropna())
    preferred_actors = set()
    for cast in movies[movies['id'].isin(watched_tmdb_ids + watchlist_movie_ids)]['cast'].dropna():
        preferred_actors.update(str(cast).split(', '))
    
    recommendations = []
    director_counter = Counter()
    popular = movies.sort_values(by='popularity_score', ascending=False)
    filter_counts = {
        'total': len(popular),
        'watched': 0,
        'excluded_genres': 0,
        'no_allowed_genres': 0,
        'light_comedy': 0,
        'light_action': 0,
        'light_tone': 0,
        'director_limit': 0,
        'low_year_weight': 0
    }
    
    for _, row in popular.iterrows():
        if row['id'] in watched_tmdb_ids + watchlist_movie_ids:
            filter_counts['watched'] += 1
            continue
        genres = str(row['genres']).split('|')
        if any(g in excluded_genres for g in genres):
            filter_counts['excluded_genres'] += 1
            continue
        if not any(g in allowed_genres for g in genres):
            filter_counts['no_allowed_genres'] += 1
            continue
        is_light_comedy = 'Comedy' in genres and not any(g in list(genre_ratings.keys()) + ['Music', 'Drama', 'Crime', 'Thriller'] for g in genres)
        if is_light_comedy:
            filter_counts['light_comedy'] += 1
            continue
        is_light_action = ('Action' in genres or 'Adventure' in genres) and not any(g in allowed_genres for g in genres)
        if is_light_action:
            filter_counts['light_action'] += 1
            continue
        keywords = str(row['keywords']).lower() if pd.notnull(row['keywords']) else ''
        is_light_tone = any(kw in keywords for kw in ['light-hearted', 'family-friendly'])
        if is_light_tone:
            filter_counts['light_tone'] += 1
            continue
        # Year weighting
        year_weight = np.exp(-((row['release_year'] - median_year) / 20) ** 2)
        if year_weight < 0.1:
            filter_counts['low_year_weight'] += 1
            continue
        director_limit = 2 if row['director'] in preferred_directors else 1
        if director_counter[row['director']] >= director_limit:
            filter_counts['director_limit'] += 1
            continue
        if len(recommendations) < top_n:
            recommendations.append(row['id'])
            director_counter[row['director']] += 1
    
    logging.info(f"Fallback filter counts: {filter_counts}")
    logging.info(f"Number of fallback candidates after filtering: {len(recommendations)}")
    rec_movies = movies[movies['id'].isin(recommendations)][['id', 'title', 'genres', 'director', 'cast', 'release_date']].drop_duplicates()
    genre_counts = rec_movies['genres'].str.split('|', expand=True).stack().value_counts()
    logging.info(f"Genre distribution in fallback: {genre_counts.to_dict()}")
    return rec_movies

# Fuzzy match movie titles
def find_movie_matches(titles_years, df, column='title', n_matches=1, cutoff=0.95):
    """Find closest matching movie titles with user-specified years without API."""
    matched_titles = []
    all_titles = df[[column, 'release_year', 'vote_count']].dropna(subset=[column]).copy()
    all_titles['vote_count'] = pd.to_numeric(all_titles['vote_count'], errors='coerce').fillna(0)
    
    title_corrections = {
        'The Batman': 'The Batman',
        'Batman': 'The Batman',
        'La la Land': 'La La Land',
        'La La land': 'La La Land',
        'Blue Valentine': 'Blue Valentine',
        'Blade Runner': 'Blade Runner 2049',
        'Toy Story': 'Toy Story 4',
        'Gladiator': 'Gladiator',
        'The Avengers': 'The Avengers',
        'Hangover': 'The Hangover',
        'Blue valentine': 'Blue Valentine',
        'Notebook': 'The Notebook',
        'Trainspotting': 'Trainspotting'
    }
    
    for title, year in titles_years:
        corrected_title = title_corrections.get(title, title)
        matches = get_close_matches(corrected_title, all_titles[column].unique(), n=n_matches * 2, cutoff=cutoff)
        if matches:
            matched_rows = all_titles[all_titles[column].isin(matches)]
            if year:
                matched_rows = matched_rows[matched_rows['release_year'] == year]
                if matched_rows.empty:
                    matched_rows = all_titles[all_titles[column].isin(matches)]  # Fallback
            matched_rows = matched_rows.sort_values(by=['vote_count', 'release_year'], ascending=[False, False])
            matched_titles.append(matched_rows[column].iloc[0] if not matched_rows.empty else None)
            if matched_titles[-1]:
                logging.info(f"Matched title: {title} (year: {year}) -> {matched_titles[-1]} (year: {matched_rows['release_year'].iloc[0]}, vote_count: {matched_rows['vote_count'].iloc[0]})")
        else:
            # Try partial match
            partial_matches = all_titles[all_titles[column].str.contains(corrected_title, case=False, na=False, regex=False)]
            if year:
                partial_matches = partial_matches[partial_matches['release_year'] == year]
                if partial_matches.empty:
                    partial_matches = all_titles[all_titles[column].str.contains(corrected_title, case=False, na=False, regex=False)]
            if not partial_matches.empty:
                partial_matches = partial_matches.sort_values(by=['vote_count', 'release_year'], ascending=[False, False])
                matched_titles.append(partial_matches[column].iloc[0])
                logging.info(f"Matched partial: {title} (year: {year}) -> {matched_titles[-1]} (year: {partial_matches['release_year'].iloc[0]}, vote_count: {partial_matches['vote_count'].iloc[0]})")
            else:
                matched_titles.append(None)
                logging.warning(f"No match for: {title} (year: {year})")
                print(f"Warning: Could not find '{title}' (year: {year}).")
    
    return matched_titles

# Get user input
def get_user_input():
    """Prompt user for input with ratings and optional years."""
    print("Welcome to the Movie Recommender!")
    print("Note: Use exact TMDb titles (e.g., 'Blue Valentine', 'Blade Runner 2049').")
    print("For watched movies, use format 'Title:Year:Rating' (e.g., 'The Batman:2022:5.0'), 'Title:Rating', or 'Title'.")
    print("For watchlist, use 'Title:Year', or 'Title'.")
    user_id = input("Enter user ID (e.g., 1, or Enter to skip): ").strip()
    
    def clean_input(input_str, is_watched=False):
        """Clean input string, parse titles, ratings, and years, ensuring consistent tuples."""
        cleaned = re.sub(r'[^\w\s,\':.]', '', input_str)
        titles_ratings_years = []
        seen = set()
        for item in cleaned.split(','):
            item = item.strip()
            if not item:
                continue
            parts = item.split(':')
            title = parts[0].strip()
            rating = None
            year = None
            if is_watched:
                if len(parts) == 3:  # Title:Year:Rating
                    try:
                        year = int(parts[1])
                        rating = float(parts[2])
                        if not 0 <= rating <= 5:
                            print(f"Warning: Rating {rating} for '{title}' must be 0-5; ignoring rating.")
                            rating = None
                    except ValueError:
                        print(f"Warning: Invalid year or rating in '{item}'; treating as Title.")
                        year = None
                        rating = None
                elif len(parts) == 2:  # Title:Rating or Title:Year
                    second_part = parts[1].strip()
                    try:
                        # Try as Title:Rating
                        rating = float(second_part)
                        if not 0 <= rating <= 5:
                            print(f"Warning: Rating {rating} for '{title}' must be 0-5; ignoring rating.")
                            rating = None
                    except ValueError:
                        # Try as Title:Year
                        try:
                            year = int(second_part)
                        except ValueError:
                            print(f"Warning: Invalid year or rating in '{item}'; treating as Title.")
                            year = None
                elif len(parts) == 1:  # Title
                    pass
                else:
                    print(f"Warning: Invalid format in '{item}'; treating as Title.")
            else:  # Watchlist
                if len(parts) == 2:  # Title:Year
                    try:
                        year = int(parts[1])
                    except ValueError:
                        print(f"Warning: Invalid year '{parts[1]}' for '{title}'; ignoring year.")
                        year = None
                elif len(parts) == 1:  # Title
                    pass
                else:
                    print(f"Warning: Invalid format in '{item}'; treating as Title.")
            if title.lower() not in seen:
                seen.add(title.lower())
                titles_ratings_years.append((title, rating, year))
                logging.info(f"Parsed input: {title}, rating={rating}, year={year}")
            else:
                print(f"Warning: Duplicate title '{title}' removed.")
        return titles_ratings_years
    
    watchlist_input = input("Enter watchlist movies (comma-separated, e.g., Prisoners:2013, La La Land): ").strip()
    watched_input = input("Enter watched movies (comma-separated, e.g., The Batman:2022:5.0, Interstellar:4.5): ").strip()
    
    watchlist_movies = clean_input(watchlist_input, is_watched=False) if watchlist_input else []
    watched_movies = clean_input(watched_input, is_watched=True) if watched_input else []
    
    # Remove duplicates across watched and watchlist
    watched_titles = {t.lower() for t, _, _ in watched_movies}
    watchlist_movies = [(t, None, y) for t, _, y in watchlist_movies if t.lower() not in watched_titles]
    
    logging.info(f"Watched movies: {watched_movies}")
    logging.info(f"Watchlist movies: {watchlist_movies}")
    return user_id or None, [(t, y) for t, _, y in watchlist_movies], watched_movies

# Collect user feedback
def get_feedback(user_id, rec_movies):
    """Prompt user to rate recommended movies."""
    feedback = []
    print("\nRate recommended movies (0-5, or Enter to skip):")
    for title in rec_movies['title']:
        rating = input(f"Rate '{title}' (0-5, Enter to skip): ").strip()
        if rating:
            try:
                rating = float(rating)
                if 0 <= rating <= 5:
                    feedback.append((title, rating, None))  # Consistent with watched_movies format
                else:
                    print(f"Warning: Rating {rating} for '{title}' must be 0-5.")
            except ValueError:
                print(f"Warning: Invalid rating for '{title}'.")
    
    # Save feedback
    if feedback and user_id:
        try:
            with open(FEEDBACK_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                for title, rating, _ in feedback:
                    writer.writerow([user_id, title, rating])
            logging.info(f"Saved feedback for user {user_id}: {[(t, r) for t, r, _ in feedback]}")
        except Exception as e:
            logging.error(f"Error saving feedback: {e}")
    
    return feedback

# Main recommendation function (corrected)
def recommend_movies(user_id=None, watchlist_movies=None, watched_movies=None, top_n=15, sample_ratings=False):
    """Generate recommendations with user ratings and feedback."""
    logging.info("Starting recommendation process...")
    
    ratings, merged_movies, tmdb_all = load_datasets(sample_ratings=sample_ratings)
    
    # Extract years correctly from watched_movies (3-tuples) and watchlist_movies (2-tuples)
    watched_years = [year for title, rating, year in watched_movies if year is not None] if watched_movies else []
    watchlist_years = [year for title, year in watchlist_movies if year is not None] if watchlist_movies else []
    input_years = watched_years + watchlist_years
    
    # Extract watched titles and years for matching
    watched_titles_years = [(title, year) for title, rating, year in watched_movies] if watched_movies else []
    watched_ratings = [(tmdb_all[tmdb_all['title'] == title]['id'].iloc[0], rating) 
                       for title, rating, _ in watched_movies 
                       if title in tmdb_all['title'].values and rating is not None]
    
    if watched_titles_years:
        matched_watched = find_movie_matches(watched_titles_years, tmdb_all)
        watched_titles = [m for m in matched_watched if m is not None]
        logging.info(f"Matched watched: {watched_titles}")
    else:
        watched_titles = []
    
    if watchlist_movies:
        matched_watchlist = find_movie_matches(watchlist_movies, tmdb_all)
        watchlist_titles = [m for m in matched_watchlist if m is not None]
        logging.info(f"Matched watchlist: {watchlist_titles}")
    else:
        watchlist_titles = []
    
    svd, trainset = train_svd(ratings)
    
    watched_movie_ids = []
    watched_tmdb_ids = []
    if watched_titles:
        watched_tmdb_ids = tmdb_all[tmdb_all['title'].isin(watched_titles)]['id'].tolist()
        watched_movie_ids = merged_movies[merged_movies['id'].isin(watched_tmdb_ids)]['movieId'].tolist()
        if not watched_movie_ids:
            print(f"Warning: Watched movies {watched_titles} not in MovieLens. Using TMDb IDs.")
    
    watchlist_movie_ids = tmdb_all[tmdb_all['title'].isin(watchlist_titles)]['id'].tolist() if watchlist_titles else []
    
    sim_dict, movies_content = build_content_features(tmdb_all, merged_movies, watched_tmdb_ids)
    
    if user_id and user_id in ratings['userId'].values:
        recommendations = get_hybrid_recommendations(user_id, svd, trainset, movies_content, sim_dict, watched_movie_ids, watchlist_movie_ids, watched_ratings, input_years, top_n)
    elif watchlist_movie_ids or watched_tmdb_ids:
        content_scores = {}
        director_scores = {}
        cast_scores = {}
        rating_similarities = {}
        watched_similarities = {}
        watched_genres = set()
        genre_ratings = Counter()
        
        for tmdb_id in watched_tmdb_ids + watchlist_movie_ids:
            movie_row = movies_content[movies_content['id'] == tmdb_id]
            if not movie_row.empty:
                genres = str(movie_row['genres'].iloc[0]).split('|')
                watched_genres.update(genres)
                rating_weight = (ratings_dict.get(tmdb_id, 4.5) / 5.0) ** 2
                for genre in genres:
                    genre_ratings[genre] += rating_weight
        
        preferred_directors = set(movies_content[movies_content['id'].isin(watched_tmdb_ids + watchlist_movie_ids)]['director'].dropna())
        preferred_actors = set()
        for cast in movies_content[movies_content['id'].isin(watched_tmdb_ids + watchlist_movie_ids)]['cast'].dropna():
            preferred_actors.update(str(cast).split(', '))
        
        ratings_dict = dict(watched_ratings)
        for movie_id in set(watchlist_movie_ids + watched_tmdb_ids):
            rating = ratings_dict.get(movie_id, 4.5)
            rating_weight = (rating / 5.0) ** 2
            content_recs = get_content_recommendations(movie_id, movies_content, sim_dict, top_n=30)
            for rec in content_recs:
                rec_id = rec['id']
                sim_score = next((score for idx, score in sim_dict[movies_content.index[movies_content['id'] == movie_id].tolist()[0]] if movies_content.iloc[idx]['id'] == rec_id), 0.0)
                content_scores[rec_id] = content_scores.get(rec_id, 0) + 2.0 * rating_weight * sim_score
                rating_similarities[rec_id] = rating_similarities.get(rec_id, []) + [rating]
                watched_similarities[rec_id] = watched_similarities.get(rec_id, []) + [sim_score]
                if rec['director'] and rec['director'] == movies_content[movies_content['id'] == movie_id]['director'].iloc[0]:
                    director_scores[rec_id] = director_scores.get(rec_id, 0) + 1.5 * rating_weight
                rec_cast = set(str(rec['cast']).split(', ')) if pd.notnull(rec['cast']) else set()
                movie_cast = set(str(movies_content[movies_content['id'] == movie_id]['cast'].iloc[0]).split(', ')) if pd.notnull(movies_content[movies_content['id'] == movie_id]['cast'].iloc[0]) else set()
                if rec_cast.intersection(movie_cast):
                    cast_scores[rec_id] = cast_scores.get(rec_id, 0) + 1.2 * len(rec_cast.intersection(movie_cast)) * rating_weight
        
        movies_content = normalize_popularity(movies_content)
        final_scores = {}
        
        median_year = int(np.median(input_years)) if input_years else 2005
        logging.info(f"Median release year: {median_year}")
        
        core_genres = {'Drama', 'Science Fiction', 'Thriller', 'Crime', 'Romance', 'Music'}
        conditional_exclude = {'Horror', 'Documentary', 'Animation'}
        excluded_genres = conditional_exclude - set(genre_ratings.keys())
        allowed_genres = watched_genres | core_genres | set(genre_ratings.keys())
        allowed_genres.add('Comedy') if {'Music', 'Drama'} <= watched_genres and 'Comedy' in genre_ratings else None
        logging.info(f"Core genres: {core_genres}, Allowed genres: {allowed_genres}, Excluded genres: {excluded_genres}")
        
        filter_counts = {
            'total': len(movies_content),
            'watched': 0,
            'excluded_genres': 0,
            'no_allowed_genres': 0,
            'light_comedy': 0,
            'light_action': 0,
            'light_tone': 0,
            'director_limit': 0,
            'low_year_weight': 0
        }
        
        director_counter = Counter()
        genre_counter = Counter()
        for movie_id in movies_content['id']:
            if movie_id in watched_tmdb_ids + watchlist_movie_ids:
                filter_counts['watched'] += 1
                continue
            genres = str(movies_content[movies_content['id'] == movie_id]['genres'].iloc[0]).split('|')
            if any(g in excluded_genres for g in genres):
                filter_counts['excluded_genres'] += 1
                continue
            if not any(g in allowed_genres for g in genres):
                filter_counts['no_allowed_genres'] += 1
                continue
            is_light_comedy = 'Comedy' in genres and not any(g in list(genre_ratings.keys()) + ['Music', 'Drama', 'Crime', 'Thriller'] for g in genres)
            if is_light_comedy:
                filter_counts['light_comedy'] += 1
                continue
            is_light_action = ('Action' in genres or 'Adventure' in genres) and not any(g in allowed_genres for g in genres)
            if is_light_action:
                filter_counts['light_action'] += 1
                continue
            keywords = str(movies_content[movies_content['id'] == movie_id]['keywords'].iloc[0]).lower() if pd.notnull(movies_content[movies_content['id'] == movie_id]['keywords'].iloc[0]) else ''
            is_light_tone = any(kw in keywords for kw in ['light-hearted', 'family-friendly'])
            if is_light_tone:
                filter_counts['light_tone'] += 1
                continue
            # Year weighting
            movie_year = movies_content[movies_content['id'] == movie_id]['release_year'].iloc[0]
            year_weight = np.exp(-((movie_year - median_year) / 20) ** 2)
            if year_weight < 0.1:
                filter_counts['low_year_weight'] += 1
                continue
            content_score = content_scores.get(movie_id, 0)
            director_score = director_scores.get(movie_id, 0)
            cast_score = cast_scores.get(movie_id, 0)
            popularity_score = movies_content[movies_content['id'] == movie_id]['popularity_score'].iloc[0]
            matching_core_genres = sum(1 for g in genres if g in core_genres)
            matching_allowed_genres = sum(1 for g in genres if g in allowed_genres)
            genre_rating_boost = sum(genre_ratings.get(g, 0) for g in genres) / len(genres) if genres else 1.0
            genre_boost = (6.0 + matching_core_genres * 2.5 + matching_allowed_genres * 0.6 + genre_rating_boost) if matching_core_genres >= 2 else (4.0 + matching_core_genres * 2.0 + matching_allowed_genres * 0.5 + genre_rating_boost)
            
            director = movies_content[movies_content['id'] == movie_id]['director'].iloc[0]
            cast = set(str(movies_content[movies_content['id'] == movie_id]['cast'].iloc[0]).split(', ')) if pd.notnull(movies_content[movies_content['id'] == movie_id]['cast'].iloc[0]) else set()
            director_boost = 4.0 if director in preferred_directors else 1.0
            actor_boost = 2.0 + 0.8 * len(cast.intersection(preferred_actors)) if cast.intersection(preferred_actors) else 1.0
            
            director_limit = 2 if director in preferred_directors else 1
            if director_counter[director] >= director_limit:
                filter_counts['director_limit'] += 1
                continue
            
            # Diversity penalty
            diversity_penalty = 1.0 / (1 + genre_counter[genres[0]] + director_counter[director])
            
            avg_rating_similarity = np.mean(rating_similarities.get(movie_id, [4.9])) / 5.0 if movie_id in rating_similarities else 1.0
            rating_boost = 1.0 + 0.5 * avg_rating_similarity
            avg_watched_similarity = np.mean(watched_similarities.get(movie_id, [0.0])) if movie_id in watched_similarities else 0.0
            watched_similarity_boost = 1.0 + 1.0 * avg_watched_similarity
            
            final_score = (W_CONTENT * (content_score + director_score + cast_score) +
                           W_POPULARITY * popularity_score) * genre_boost * director_boost * actor_boost * rating_boost * watched_similarity_boost * year_weight * diversity_penalty
            final_scores[movie_id] = final_score
            director_counter[director] += 1
            for g in genres:
                genre_counter[g] += 1
        
        logging.info(f"Content-based filter counts: {filter_counts}")
        logging.info(f"Number of content-based candidates after filtering: {len(final_scores)}")
        
        top_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        logging.info(f"Top 20 before selection: {[movies_content[movies_content['id'] == k]['title'].iloc[0] + f' ({k}, score: {v:.3f}, genres: {movies_content[movies_content['id'] == k]['genres'].iloc[0]})' for k, v in top_scores]}")
        
        recommendations = []
        director_counter = Counter()
        sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        for movie_id, score in sorted_scores:
            director = movies_content[movies_content['id'] == movie_id]['director'].iloc[0]
            director_limit = 2 if director in preferred_directors else 1
            if len(recommendations) < top_n and director_counter[director] < director_limit:
                recommendations.append((movie_id, score))
                director_counter[director] += 1
            if len(recommendations) >= top_n:
                break
        
        rec_movies = movies_content[movies_content['id'].isin([rec[0] for rec in recommendations])][['id', 'title', 'genres', 'director', 'cast', 'release_date']].drop_duplicates()
        
        for _, row in rec_movies.iterrows():
            if row['director'] in preferred_directors:
                logging.info(f"Recommendation '{row['title']}' boosted by director: {row['director']}")
            if set(str(row['cast']).split(', ')).intersection(preferred_actors):
                logging.info(f"Recommendation '{row['title']}' boosted by cast: {row['cast']}")
            if row['id'] in rating_similarities:
                logging.info(f"Recommendation '{row['title']}' rating similarity: {np.mean(rating_similarities[row['id']]):.2f}")
            if row['id'] in watched_similarities:
                logging.info(f"Recommendation '{row['title']}' watched similarity: {np.mean(watched_similarities[row['id']]):.3f}")
        
        genre_counts = rec_movies['genres'].str.split('|', expand=True).stack().value_counts()
        logging.info(f"Genre distribution: {genre_counts.to_dict()}")
        
        print("\nRecommended Movies:")
        print(rec_movies[['title', 'genres', 'director', 'cast', 'release_date']].to_string(index=False))
        return rec_movies
    else:
        recommendations = get_popular_movies(tmdb_all, watched_tmdb_ids, watchlist_movie_ids, input_years, top_n)
    
    print("\nRecommended Movies:")
    print(recommendations[['title', 'genres', 'director', 'cast', 'release_date']].to_string(index=False))
    
    # Collect feedback
    if recommendations.empty:
        logging.warning("No recommendations generated; skipping feedback.")
    else:
        feedback = get_feedback(user_id, recommendations)
        if feedback:
            # Update watched_movies with feedback
            watched_movies.extend(feedback)
            # Re-run recommendations with updated input
            logging.info("Re-running recommendations with feedback...")
            return recommend_movies(
                user_id=user_id,
                watchlist_movies=watchlist_movies,
                watched_movies=watched_movies,
                top_n=top_n,
                sample_ratings=sample_ratings
            )
    
    logging.info("Recommendation process completed")
    return recommendations

if __name__ == "__main__":
    try:
        user_id, watchlist_movies, watched_movies = get_user_input()
        recommend_movies(
            user_id=user_id,
            watchlist_movies=watchlist_movies,
            watched_movies=watched_movies,
            top_n=15,
            sample_ratings=True
        )
    except Exception as e:
        logging.error(f"Error: {e}")
        print(f"Error: {e}")