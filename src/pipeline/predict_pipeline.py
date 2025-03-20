import sys
import pandas as pd
import numpy as np
from src.exception.exception import CustomException
from src.utils.main_utils.utils import load_object

# Assuming movies_with_ratings, movies, mlb, and movie_index are defined elsewhere
# You'll need to load these from your saved files or data sources.

class PredictPipeline:
    def __init__(self, model_path="final_model/model.pkl", movies_with_ratings_path="Artifacts\03_19_2025_16_20_30\data_validation\validatedMoviesRatings.csv", movies_path="Artifacts\03_19_2025_16_20_30\data_validation\validatedMovies.csv", mlb_path="Artifacts\03_19_2025_16_20_30\data_transformation\transformed_object\preprocessing.pkl", movie_index_path="Artifacts\03_19_2025_16_50_58\model_trainer\trained_model\movie_index.pkl"):
        self.model_path = model_path
        self.movies_with_ratings_path = movies_with_ratings_path
        self.movies_path = movies_path
        self.mlb_path = mlb_path
        self.movie_index_path = movie_index_path

    def load_data(self):
        try:
            self.movies_with_ratings = pd.read_csv(self.movies_with_ratings_path)
            self.movies = pd.read_csv(self.movies_path)
            self.mlb = load_object(file_path=self.mlb_path)
            self.movie_index = load_object(file_path=self.movie_index_path)
        except Exception as e:
            raise CustomException(e, sys)

    def recommend_items(self, user_id, n=5):
        try:
            self.load_data()  # Load data before making predictions
            model = load_object(file_path=self.model_path)
            item_ids = np.arange(self.movies_with_ratings['movieId'].nunique())
            user_array = np.full(len(item_ids), user_id)
            genres_array = np.zeros((len(item_ids), len(self.mlb.classes_)))
            predictions = model.predict([user_array, item_ids, genres_array]).flatten()
            top_n_items = predictions.argsort()[-n:][::-1]
            recommended_movie_ids = [movie_id + 1 for movie_id in self.movie_index[top_n_items]]
            recommended_movies = self.movies[self.movies["movieId"].isin(recommended_movie_ids)]
            recommended_titles = recommended_movies["title"].tolist()
            return recommended_titles
        except Exception as e:
            raise CustomException(e, sys)