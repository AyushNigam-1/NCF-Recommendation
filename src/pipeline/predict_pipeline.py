import sys
import pandas as pd
import numpy as np
from src.exception.exception import CustomException
from src.utils.main_utils.utils import load_object
import tensorflow as tf

class PredictPipeline:
    def __init__(self, model_path="/home/ayush/NCF-Recommendation/Artifacts/03_23_2025_20_47_23/model_trainer/trained_model/model.keras", movies_with_ratings_path="/home/ayush/NCF-Recommendation/Artifacts/03_23_2025_17_39_40/movies_data.csv", movies_path="/home/ayush/NCF-Recommendation/Artifacts/03_23_2025_20_47_23/data_validation/validatedMovies.csv", mlb_path="/home/ayush/NCF-Recommendation/Artifacts/03_23_2025_20_47_23/data_transformation/transformed_object/preprocessing.pkl", movie_index_path="/home/ayush/NCF-Recommendation/Artifacts/03_23_2025_20_47_23/model_trainer/trained_model/movie_index.pkl"):
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
            self.model = tf.keras.models.load_model(self.model_path)
        except Exception as e:
            raise CustomException(e, sys)

    def recommend_items(self, user_id, n=5):
        try:
            self.load_data()
            item_ids = np.arange(self.movies_with_ratings['movieId'].nunique())
            print(item_ids)
            user_array = np.full(len(item_ids), user_id)
            print(len(user_array))
            genres_array = np.zeros((len(item_ids), len(self.mlb.classes_)))
            print(len(genres_array))
            predictions = self.model.predict([user_array, item_ids,genres_array]).flatten()
            top_n_items = predictions.argsort()[-n:][::-1]
            recommended_movie_ids = [movie_id + 1 for movie_id in self.movie_index[top_n_items]]
            recommended_movies = self.movies[self.movies["movieId"].isin(recommended_movie_ids)]
            recommended_titles = recommended_movies["title"].tolist()
            return recommended_titles
        except Exception as e:
            print(e)
            raise CustomException(e, sys)


