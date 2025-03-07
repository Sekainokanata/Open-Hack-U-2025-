import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# データの設定
num_samples = 1000  # データサンプル数
num_landmarks = 52  # ランドマークの数
num_coordinates = 7  # 各ランドマークの座標数 (x, y, z)
num_classes = 10  # クラス数（10クラス分類）

# ランダムに座標データを生成（入力データ: ランドマーク座標）
X = np.random.rand(num_samples, num_landmarks * num_coordinates)  # ランダムに [0, 1] の値を生成

# ラベルデータを生成（ランダムに10クラスに振り分け）
y = np.random.randint(0, num_classes, num_samples)

# ラベルをone-hotエンコード（クロスエントロピー用）
y_one_hot = to_categorical(y, num_classes)

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# 作成したデータを表示
print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')


# 1次元データ（例: 手のランドマーク座標の数が21個、x, y, z座標がある場合）
input_dim = 52 * 7  # 21個のランドマーク×3次元（x, y, z）

# MLPモデルを定義
def create_mlp_model(input_dim):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))  # クラス分類用の出力層（10クラスの分類）

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# モデルを作成
mlp_model = create_mlp_model(input_dim)
mlp_model.summary()


# モデルの学習
mlp_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
mlp_model.evaluate(X_test,  y_test, verbose=2)
