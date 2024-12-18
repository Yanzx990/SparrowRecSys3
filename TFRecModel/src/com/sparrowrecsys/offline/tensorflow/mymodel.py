import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate

# Load dataset
train_data = pd.read_csv('D:/SparrowRecSys/src/main/resources/webroot/sampledata/trainingSamples.csv')
test_data = pd.read_csv('D:/SparrowRecSys/src/main/resources/webroot/sampledata/testSamples.csv')

scaler = StandardScaler()
encoder = OneHotEncoder(sparse_output=False)

# Feature preprocessing
def preprocess_data(train_data):
    # Movie features
    movie_features = ['releaseYear', 'movieRatingCount', 'movieAvgRating', 'movieRatingStddev']

    # User features
    user_features = ['userAvgRating', 'userRatingCount', 'userAvgReleaseYear', 'userReleaseYearStddev']

    # Normalize continuous features
    scaler = StandardScaler()
    train_data[movie_features] = scaler.fit_transform(train_data[movie_features])
    train_data[user_features] = scaler.fit_transform(train_data[user_features])

    # One-hot encode categorical features (genres)
    encoder = OneHotEncoder(sparse_output=False)
    movie_genres = encoder.fit_transform(train_data[['movieGenre1', 'movieGenre2', 'movieGenre3']])
    user_genres = encoder.fit_transform(train_data[['userGenre1', 'userGenre2', 'userGenre3', 'userGenre4', 'userGenre5']])

    # Combine the features into feature matrix
    X_user = train_data[user_features].values
    X_movie = train_data[movie_features].values
    X_movie_genre = movie_genres
    X_user_genre = user_genres

    # Combine all features
    X = np.concatenate([X_user, X_movie, X_movie_genre, X_user_genre], axis=1)
    y = train_data['rating'].values

    return X, y

X, y = preprocess_data(train_data)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. XGBoost Model (to capture feature interactions)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, max_depth=7, learning_rate=0.05)
xgb_model.fit(X_train, y_train)

# 2. Deep Neural Network Model (DNN) with complex architecture

input_layer = Input(shape=(X_train.shape[1],))  # Ensure input dimension matches the number of features


x = Dense(512, activation='relu')(input_layer)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)

x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

output_layer = Dense(1)(x)

dnn_model = Model(inputs=input_layer, outputs=output_layer)
dnn_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the DNN model
dnn_model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_val, y_val))

# 3. Hybrid Model: Using predictions from XGBoost as an input to DNN model
xgb_train_preds = xgb_model.predict(X_train)
xgb_val_preds = xgb_model.predict(X_val)

# Combine the predictions from XGBoost with the features for final DNN model
X_train_final = np.concatenate([X_train, xgb_train_preds.reshape(-1, 1)], axis=1)
X_val_final = np.concatenate([X_val, xgb_val_preds.reshape(-1, 1)], axis=1)

print(X_train_final.shape)
print(X_val_final.shape)

# Train final DNN model with combined features and XGBoost predictions
input_layer = Input(shape=(X_train_final.shape[1],))  # 使用最终特征矩阵的维度
x = Dense(512, activation='relu')(input_layer)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)

x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

output_layer = Dense(1)(x)

# 重新编译模型
dnn_model = Model(inputs=input_layer, outputs=output_layer)
dnn_model.compile(optimizer='adam', loss='mean_squared_error')
dnn_model.fit(X_train_final, y_train, epochs=10, batch_size=64, validation_data=(X_val_final, y_val))

# Predict ratings for test data
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 预处理测试集数据
def preprocess_test_data(test_data, scaler, encoder):
    # 使用训练集的标准化器和编码器进行相同的预处理
    movie_features = ['releaseYear', 'movieRatingCount', 'movieAvgRating', 'movieRatingStddev']
    user_features = ['userAvgRating', 'userRatingCount', 'userAvgReleaseYear', 'userReleaseYearStddev']

    # 标准化连续特征
    test_data[movie_features] = scaler.transform(test_data[movie_features])
    test_data[user_features] = scaler.transform(test_data[user_features])

    # 对电影和用户类别特征进行 One-Hot 编码
    movie_genres = encoder.transform(test_data[['movieGenre1', 'movieGenre2', 'movieGenre3']])
    user_genres = encoder.transform(test_data[['userGenre1', 'userGenre2', 'userGenre3', 'userGenre4', 'userGenre5']])

    # 提取特征矩阵
    X_user = test_data[user_features].values
    X_movie = test_data[movie_features].values
    X_movie_genre = movie_genres
    X_user_genre = user_genres

    # 合并所有特征
    X = np.concatenate([X_user, X_movie, X_movie_genre, X_user_genre], axis=1)
    y = test_data['rating'].values

    return X, y

# 加载和预处理测试集
X_test, y_test = preprocess_test_data(test_data, scaler, encoder)

# XGBoost 预测
xgb_test_preds = xgb_model.predict(X_test)

# 合并 XGBoost 的预测结果作为 DNN 的输入
X_test_final = np.concatenate([X_test, xgb_test_preds.reshape(-1, 1)], axis=1)

# DNN 预测
dnn_test_preds = dnn_model.predict(X_test_final).flatten()

# 评估模型性能
mse = mean_squared_error(y_test, dnn_test_preds)
mae = mean_absolute_error(y_test, dnn_test_preds)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# 如果需要保存每个样本的预测结果
test_data['predicted_rating'] = dnn_test_preds
test_data[['userId', 'movieId', 'rating', 'predicted_rating']].to_csv('test_predictions.csv', index=False)

print("预测结果已保存到 test_predictions.csv")

