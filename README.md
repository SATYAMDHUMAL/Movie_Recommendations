# Movie_Recommendationsimport pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the MovieLens dataset
url = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.data'
names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(url, sep='\t', names=names)

# Load the data into the Surprise library's Dataset class
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Use a basic collaborative filtering algorithm (KNN)
sim_options = {
    'name': 'cosine',
    'user_based': True,
}

model = KNNBasic(sim_options=sim_options)

# Train the model
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the model
accuracy.rmse(predictions)

# Function to get movie recommendations for a given user
def get_recommendations(user_id, num_recommendations=5):
    user_movies = df[df['user_id'] == user_id]['item_id'].tolist()
    unwatched_movies = df[~df['item_id'].isin(user_movies)]['item_id'].unique()

    # Get predictions for unwatched movies
    pred_ratings = [model.predict(user_id, movie) for movie in unwatched_movies]

    # Sort predictions by predicted rating
    sorted_predictions = sorted(pred_ratings, key=lambda x: x.est, reverse=True)

    # Get top recommendations
    top_recommendations = sorted_predictions[:num_recommendations]

    # Get movie titles for recommended movie IDs
    movie_titles = [df[df['item_id'] == pred.iid]['item_id'].values[0] for pred in top_recommendations]

    return movie_titles

# Example: Get recommendations for user 1
user_id = 1
recommendations = get_recommendations(user_id)
print(f"Top 5 movie recommendations for user {user_id}: {recommendations}")
 
