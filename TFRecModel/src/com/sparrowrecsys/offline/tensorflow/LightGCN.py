import tensorflow as tf
import numpy as np

# Load datasets as tf dataset
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name='label',
        na_value="0",
        num_epochs=1,
        ignore_errors=True)
    return dataset

# Define paths for training and testing samples
train_samples_file_path = 'D:\SparrowRecSys\src\main\\resources\webroot\sampledata\\trainingSamples.csv'  # Adjust to your path
test_samples_file_path = 'D:\SparrowRecSys\src\main\\resources\webroot\sampledata\\testSamples.csv'  # Adjust to your path

train_dataset = get_dataset(train_samples_file_path)
test_dataset = get_dataset(test_samples_file_path)

# Genre vocabulary
genre_vocab = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western', 'Documentary',
               'Sci-Fi', 'Drama', 'Thriller', 'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery', 'Children', 'Musical']

# Embedding feature columns
movie_col = tf.feature_column.categorical_column_with_identity(key='movieId', num_buckets=1001)
user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=30001)

user_genre_col = tf.feature_column.categorical_column_with_vocabulary_list(key="userGenre1", vocabulary_list=genre_vocab)
item_genre_col = tf.feature_column.categorical_column_with_vocabulary_list(key="movieGenre1", vocabulary_list=genre_vocab)

# Create embedding columns
movie_emb_col = tf.feature_column.embedding_column(movie_col, 10)
user_emb_col = tf.feature_column.embedding_column(user_col, 10)
user_genre_emb_col = tf.feature_column.embedding_column(user_genre_col, 10)
item_genre_emb_col = tf.feature_column.embedding_column(item_genre_col, 10)

# Define input layers
inputs = {
    'movieId': tf.keras.layers.Input(name='movieId', shape=(), dtype='int32'),
    'userId': tf.keras.layers.Input(name='userId', shape=(), dtype='int32'),
    'userGenre1': tf.keras.layers.Input(name='userGenre1', shape=(), dtype='string'),
    'movieGenre1': tf.keras.layers.Input(name='movieGenre1', shape=(), dtype='string'),
    'releaseYear': tf.keras.layers.Input(name='releaseYear', shape=(), dtype='int32'),
    'movieRatingCount': tf.keras.layers.Input(name='movieRatingCount', shape=(), dtype='int32'),
    'userRatingCount': tf.keras.layers.Input(name='userRatingCount', shape=(), dtype='int32'),
    'movieAvgRating': tf.keras.layers.Input(name='movieAvgRating', shape=(), dtype='float32'),
    'userAvgRating': tf.keras.layers.Input(name='userAvgRating', shape=(), dtype='float32'),
}

# LightGCN requires a neighbor matrix, simulate a simple adjacency matrix for user-item interactions
def create_adjacency_matrix(num_users, num_items, interactions):
    """
    Create an adjacency matrix for LightGCN from user-item interactions.
    This is a sparse matrix, where non-zero entries represent interactions (e.g., user rated movie).
    """
    adj_matrix = np.zeros((num_users + num_items, num_users + num_items))

    for user_id, movie_id in interactions:
        adj_matrix[user_id, num_users + movie_id] = 1
        adj_matrix[num_users + movie_id, user_id] = 1

    return adj_matrix

# Assuming you have some user-item interactions data (user_id, movie_id) in `interactions`
interactions = [(1, 2), (1, 3), (2, 1)]  # Replace with your actual user-item interaction data
num_users = 30000  # Total number of users
num_items = 1000   # Total number of items
adj_matrix = create_adjacency_matrix(num_users, num_items, interactions)

# LightGCN Layer (Graph Convolution Layer)
class LightGCNLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim):
        super(LightGCNLayer, self).__init__()
        self.embedding_dim = embedding_dim

    def call(self, user_embeddings, item_embeddings, adj_matrix):
        # Perform simple propagation over the adjacency matrix
        all_embeddings = tf.concat([user_embeddings, item_embeddings], axis=0)
        updated_embeddings = tf.matmul(adj_matrix, all_embeddings)
        return updated_embeddings

# Create embeddings for users and items
user_embeddings = tf.keras.layers.Embedding(input_dim=num_users, output_dim=10)(inputs['userId'])
item_embeddings = tf.keras.layers.Embedding(input_dim=num_items, output_dim=10)(inputs['movieId'])

# Apply LightGCN Layer
lightgcn_layer = LightGCNLayer(embedding_dim=10)
updated_embeddings = lightgcn_layer(user_embeddings, item_embeddings, adj_matrix)

# Use updated embeddings for final prediction
user_updated_embeddings = updated_embeddings[:num_users]
item_updated_embeddings = updated_embeddings[num_users:]

# Calculate user-item interaction score (dot product between user and item embeddings)
interaction_score = tf.reduce_sum(user_updated_embeddings * item_updated_embeddings, axis=1)

# Output layer
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(interaction_score)

# Build and compile the model
model = tf.keras.Model(inputs, output_layer)

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')]
)

# Train the model
model.fit(train_dataset, epochs=5)

# Evaluate the model
test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test ROC AUC: {test_roc_auc}, Test PR AUC: {test_pr_auc}')

# Save the model
model.save("path_to_save_model")
