import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

# パラメータ設定
num_landmarks = 52  # 手のランドマークの数
num_coordinates = 7  # 各ランドマークのx, y, zの3次元座標と回転を4個
time_steps = 60  # 60フレーム分の手の動き
init_position = 2 * 3 # キーボードの初期位置
camera_position = 2 * 3 #左右のカメラの位置

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
            X.append(samples[:,1:])
    return np.array(X)  # shape: (num_samples, time_steps, num_landmarks * num_coordinates + init_position)

# ラベルデータの読み込み
def load_answer_labels(directory_path): 
    Y = []  # 初期化をリストにする
    file_list = sorted(os.listdir(directory_path))
    for file_name in file_list:
        if file_name.endswith('.csv'):   
            csv_file_path = os.path.join(directory_path, file_name)
            labels = load_csv_data(csv_file_path)
            labels = labels.flatten()  # フラット化
            print(labels)  # 各ファイルのラベルを表示
            Y.append(labels)  # リストに追加
    
    Y = np.concatenate(Y, axis=0)  # 1次元で結合
    print(Y.shape)  # 最終的な配列の形状を表示
    print(Y)  # 結果の配列を表示
    return Y  # 結果の配列を返す

# 1D CNN + RNN（LSTM）モデルの定義
def create_cnn_rnn_model():
    model = models.Sequential()

    # 1D CNN部分
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(time_steps, 367)))
    model.add(layers.Conv1D(128, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.5))

    # RNN部分（LSTM）
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.LSTM(64))

    # 出力層
    model.add(layers.Dense(29, activation='softmax'))

    # コンパイル
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# データの前処理とモデルのトレーニング
def main(csv_directory, answer_directory):
    # CSVデータの読み込み
    X = load_all_csv_files(csv_directory)
    
    # 読み込んだデータの形状確認
    print(f"データの形状: {X.shape}")  # shape: (num_samples, time_steps, num_landmarks * num_coordinates + init_position)

    # ラベル生成（ランダム）
    y = load_answer_labels(answer_directory)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # モデルの作成
    cnn_rnn_model = create_cnn_rnn_model()
    
    # 学習（ここではエポック数を少なく設定）
    cnn_rnn_model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = cnn_rnn_model.evaluate(X_test, y_test, batch_size=32)
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    #print("Generate predictions for 3 samples")
    #predictions = model.predict(x_test[:3])
    #print("predictions shape:", predictions.shape)

# メインプログラムの実行
if __name__ == '__main__':
    # CSVファイルが保存されているディレクトリを指定
    csv_directory = '.\INPUT_csv'  # ここにCSVファイルのパスを指定
    answer_directory = '.\AnswerLABEL'  # ここに正解ラベルのファイルのパスを指定
    main(csv_directory, answer_directory)
