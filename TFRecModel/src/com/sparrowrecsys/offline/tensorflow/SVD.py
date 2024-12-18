import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

# 获取数据集的函数
def get_dataset(file_path):
    # 使用 TensorFlow 读取 CSV 文件并创建数据集
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name='rating',  # 假设目标列是 'rating'
        na_value="0",         # 缺失值处理
        num_epochs=1,
        ignore_errors=True
    )
    return dataset

# 假设训练集和测试集文件路径
training_samples_file_path = tf.keras.utils.get_file(
    "trainingSamples.csv",
    "file:///D:/SparrowRecSys/src/main/resources/webroot/sampledata/trainingSamples.csv"
)

test_samples_file_path = tf.keras.utils.get_file(
    "testSamples.csv",
    "file:///D:/SparrowRecSys/src/main/resources/webroot/sampledata/testSamples.csv"
)

# 加载数据集
train_dataset = get_dataset(training_samples_file_path)
test_dataset = get_dataset(test_samples_file_path)

# 获取评分矩阵的函数
def get_rating_matrix(dataset):
    user_ids = []
    movie_ids = []
    ratings = []

    for feature, label in dataset:
        user_ids.extend(feature['userId'].numpy())  # 获取用户 ID
        movie_ids.extend(feature['movieId'].numpy())  # 获取电影 ID
        ratings.extend(label.numpy())  # 获取评分

    return np.array(user_ids), np.array(movie_ids), np.array(ratings)

# 获取训练集的评分数据
train_user_ids, train_movie_ids, train_ratings = get_rating_matrix(train_dataset)

# 获取用户和电影的数量
num_users = np.max(train_user_ids) + 1  # 用户数量（假设从 0 开始）
num_movies = np.max(train_movie_ids) + 1  # 电影数量（假设从 0 开始）

# 构建评分矩阵（稠密矩阵）
R = np.zeros((num_users, num_movies))
for user, movie, rating in zip(train_user_ids, train_movie_ids, train_ratings):
    R[user, movie] = rating

# 使用 SVD 进行矩阵分解
svd = TruncatedSVD(n_components=10)  # 选择 10 个潜在因子
U = svd.fit_transform(R)  # 用户矩阵
Sigma = np.diag(svd.singular_values_)  # 奇异值矩阵
Vt = svd.components_  # 电影矩阵的转置

# 计算预测的评分矩阵
R_pred = np.dot(np.dot(U, Sigma), Vt)

# 计算训练集上的均方误差
train_loss = mean_squared_error(R, R_pred)
print(f"训练集的均方误差 (MSE): {train_loss}")

# 将 SVD 结果作为嵌入矩阵
user_matrix = tf.Variable(U, dtype=tf.float32)
movie_matrix = tf.Variable(Vt.T, dtype=tf.float32)

# 定义矩阵分解模型
def matrix_factorization(user_ids, movie_ids):
    user_embeddings = tf.gather(user_matrix, user_ids)  # 获取用户嵌入
    movie_embeddings = tf.gather(movie_matrix, movie_ids)  # 获取电影嵌入
    return tf.reduce_sum(user_embeddings * movie_embeddings, axis=1)  # 计算预测评分

# 定义损失函数：均方误差（MSE）
def loss_fn(ratings, predicted_ratings):
    return tf.reduce_mean(tf.square(ratings - predicted_ratings))

# 优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# 训练过程
epochs = 10
batch_size = 12

# 计算训练集的总批次数
total_batches = sum(1 for _ in train_dataset)

# 在训练过程中打印损失时，使用手动计算的 total_batches
for epoch in range(epochs):
    total_loss = 0
    for feature, label in train_dataset:
        user_ids = feature['userId'].numpy()
        movie_ids = feature['movieId'].numpy()
        ratings = label.numpy()

        with tf.GradientTape() as tape:
            predicted_ratings = matrix_factorization(user_ids, movie_ids)
            loss = loss_fn(ratings, predicted_ratings)

        gradients = tape.gradient(loss, [user_matrix, movie_matrix])
        optimizer.apply_gradients(zip(gradients, [user_matrix, movie_matrix]))  # 更新矩阵

        total_loss += loss

    # 使用手动计算的 total_batches
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss.numpy() / total_batches}')

# 评估模型
test_user_ids, test_movie_ids, test_ratings = get_rating_matrix(test_dataset)
predicted_ratings = matrix_factorization(test_user_ids, test_movie_ids)

test_loss = loss_fn(test_ratings, predicted_ratings)
print(f'\n测试集的损失 (MSE): {test_loss.numpy()}')

# 打印一些预测结果
for i in range(12):
    print(f"预测评分: {predicted_ratings[i]:.2f} | 实际评分: {test_ratings[i]}")
