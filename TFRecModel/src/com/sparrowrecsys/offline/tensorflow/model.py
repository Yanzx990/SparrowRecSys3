import tensorflow as tf

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

# FM first-order term columns
fm_first_order_columns = [
    tf.feature_column.indicator_column(movie_col),
    tf.feature_column.indicator_column(user_col),
    tf.feature_column.indicator_column(user_genre_col),
    tf.feature_column.indicator_column(item_genre_col)
]

# DeepFM Model Architecture
# FM part: Cross different categorical features
item_emb_layer = tf.keras.layers.DenseFeatures([movie_emb_col])(inputs)
user_emb_layer = tf.keras.layers.DenseFeatures([user_emb_col])(inputs)
item_genre_emb_layer = tf.keras.layers.DenseFeatures([item_genre_emb_col])(inputs)
user_genre_emb_layer = tf.keras.layers.DenseFeatures([user_genre_emb_col])(inputs)

fm_first_order_layer = tf.keras.layers.DenseFeatures(fm_first_order_columns)(inputs)

# FM interactions (dot products)
product_layer_item_user = tf.keras.layers.Dot(axes=1)([item_emb_layer, user_emb_layer])
product_layer_item_genre_user_genre = tf.keras.layers.Dot(axes=1)([item_genre_emb_layer, user_genre_emb_layer])

# Deep part: MLP layers for high-order feature interaction
deep_feature_columns = [movie_emb_col, user_emb_col, user_genre_emb_col, item_genre_emb_col,
                        tf.feature_column.numeric_column('releaseYear'),
                        tf.feature_column.numeric_column('movieRatingCount'),
                        tf.feature_column.numeric_column('userRatingCount'),
                        tf.feature_column.numeric_column('movieAvgRating'),
                        tf.feature_column.numeric_column('userAvgRating')]

deep = tf.keras.layers.DenseFeatures(deep_feature_columns)(inputs)
deep = tf.keras.layers.Dense(64, activation='relu')(deep)
deep = tf.keras.layers.Dense(64, activation='relu')(deep)

# Concatenate FM part and Deep part
concat_layer = tf.keras.layers.concatenate([fm_first_order_layer, product_layer_item_user, product_layer_item_genre_user_genre, deep], axis=1)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(concat_layer)

# Build and compile model
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
