import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
import sys

movies = pd.read_csv("data/movies.csv")

movies["genres"] = movies["genres"].apply(
    lambda x: x.split("|")
)
mlb = MultiLabelBinarizer()

genre_matrix = mlb.fit_transform(
    movies["genres"]
)
model = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)
model.fit(genre_matrix)
movie_indices = pd.Series(
    movies.index,
    index=movies["title"]
).drop_duplicates()
def recommend(movie_title):
    movie_index = movie_indices[
        movie_title
    ]

    distances, indices = model.kneighbors(
        genre_matrix[movie_index].reshape(1, -1),
        n_neighbors=6
    )

    recommendations = movies.iloc[
        indices[0][1:]
    ]["title"]

    return recommendations
if __name__ == "__main__":

    movie_name = sys.argv[1]

    recommendations = recommend(movie_name)

    print("\nRecommended movies:\n")

    for movie in recommendations:
        print(movie)
