#  Movie Recommendation System

## 1. Introduction
In this assignment we nned to create a movie recommendation system. It will recommend five movies based on the input movie. The system uses content filtering by comparing movie genres to find similar movies with same/similar genras.

## 2. Dataset
MovieLens dataset was used for this assignment.
The file "movies.csv" was selected because it contains movie titles and genras.

These columns bellow were used:
- movie ID
- Title
- genras

The genome files were ignored because the assignment requerments

## 3. Method
movies were sorted by their types (genges), like action or comedy. Needed to change the gemres into numbers so the computer can understand them. 
So the system looks for movies that are similar by checking how close their types are using a method called KNN and a way to measure similarity called cosine similarity.

## 4. Feature Selection
Genres were selected as feature for recommendations.
The genras column originally have values separated by the "|" symbol. These values were split into lists and changed into 
one hot encoded vectors.
For example "Adventure|Comedy" became "Adventure, Comedy" and transformed into numerical values.
## 5. Recommendation Process
Recomendation works like:
1. input a movie title in the command, the system finds the movie in the dataset and KNN searches for the nearest movies using cosine similarity. The input movie is removed and five movie recommendations are returned.

## 6. Limitations
One limitation of this system is that recommendations are based only on genres. 
## 7. Conclusion
content filtering was used to make a movie reccomendation system. Movie genres were converted into numerical values by one hot encoding. KNN and cosine similarity were used to find recomandations