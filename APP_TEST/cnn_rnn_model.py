import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import os

# パラメータ設定
num_landmarks = 21  # 手のランドマークの数
num_coordinates = 6  # 各ランドマークのx, y, zの3次元座標と回転rx, ry, rz
time_steps = 60  # 30フレーム分の手の動き
init_position = 2 # キーボードの初期位置

# CSVファイルからデータを読み込む関数
def load_csv_data(csv_file_path):
    # CSVファイルを読み込み、各フレームに対するランドマーク座標情報を取得
    df = pd.read_csv(csv_file_path, header=None)
    
    # データをnumpy配列に変換
    # 各行がフレームごとのランドマーク座標を表していると仮定（ランドマーク数×3次元）
    data = df.values  # shape: (time_steps, num_landmarks * num_coordinates + init_position)
    
    return data

# すべてのCSVファイルからデータを読み込み
def load_all_csv_files(directory_path):
    X = []
    file_list = sorted(os.listdir(directory_path))  # ディレクトリ内のファイルをソートして取得
    for file_name in file_list:
        if file_name.endswith('.csv'):
            csv_file_path = os.path.join(directory_path, file_name)
            samples = load_csv_data(csv_file_path)

            X.append(samples)
    return np.array(X)  # shape: (num_samples, time_steps, num_landmarks * num_coordinates + init_position)

# ランダムにラベルを生成（今回は10クラス分類を想定）
def generate_random_labels(num_samples, num_classes=10):
    return np.random.randint(0, num_classes, size=num_samples)

# 1D CNN + RNN（LSTM）モデルの定義
def create_cnn_rnn_model():
    model = models.Sequential()

    # 1D CNN部分
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(time_steps, num_landmarks * num_coordinates + init_position)))
    model.add(layers.Conv1D(128, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.5))

    # RNN部分（LSTM）
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.LSTM(64))

    # 出力層
    model.add(layers.Dense(10, activation='softmax'))

    # コンパイル
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# データの前処理とモデルのトレーニング
def main(csv_directory):
    # CSVデータの読み込み
    X = load_all_csv_files(csv_directory)
    
    # 読み込んだデータの形状確認
    print(f"データの形状: {X.shape}")  # shape: (num_samples, time_steps, num_landmarks * num_coordinates + init_position)

    # ラベル生成（ランダム）
    y = generate_random_labels(X.shape[0])

    # モデルの作成
    cnn_rnn_model = create_cnn_rnn_model()
    
    # 学習（ここではエポック数を少なく設定）
    cnn_rnn_model.fit(X, y, epochs=10, batch_size=32)

# メインプログラムの実行
if __name__ == '__main__':
    # CSVファイルが保存されているディレクトリを指定
    csv_directory = '.\INPUT_csv'  # ここにCSVファイルのパスを指定
    main(csv_directory)
