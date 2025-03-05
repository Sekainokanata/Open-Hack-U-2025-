import numpy as np
import pandas as pd
import os


# CSVファイルからデータを読み込む関数
def load_csv_data(csv_file_path):
    # CSVファイルを読み込み、各フレームに対するランドマーク座標情報を取得
    df = pd.read_csv(csv_file_path, header=None)
    
    # データをnumpy配列に変換
    # 各行がフレームごとのランドマーク座標を表していると仮定（ランドマーク数×3次元）
    data = df.values  # shape: (time_steps, num_landmarks * num_coordinates + init_position)
    
    return data

directory_path = '.\AnswerLABEL'
file_name = 'tanaka_AnswerLABEL.csv'
csv_file_path = os.path.join(directory_path, file_name)
y = load_csv_data(csv_file_path)
y = y.flatten()
print(y.shape)
print(y)
y_ord = [ord(char) - 97 for char in y]
print(y_ord)
np.array(y_ord)  

print(y_ord)