import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os
import re
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_all_csv_files(directory_path):
    dfs = []
    file_list = sorted(os.listdir(directory_path), key=lambda f: int(re.search(r'\d+', f).group()))
    #print(file_list)
    for file_name in file_list:
        if file_name.endswith('.csv'):
            csv_file_path = os.path.join(directory_path, file_name)
            #print(f"Loading {csv_file_path}")
            df = pd.read_csv(csv_file_path, header=None)
            dfs.append(df)
        
    if len(dfs) == 0:
        return pd.DataFrame()  # 空のDataFrameを返す
    # CSVファイルを縦方向に連結し、1つのDataFrameにする
    return pd.concat(dfs, ignore_index=True)


def main():
    train_csv_directory = r'.\train_csv'
    train_answer_directory = r'.\train_answerLABEL'
    test_csv_directory = r'.\test_csv'
    test_answer_directory = r'.\test_answerLABEL'
    
    # 各ディレクトリ内の CSV ファイルを連結して1つの DataFrame にする
    train_data_df = load_all_csv_files(train_csv_directory)
    train_label_df = load_all_csv_files(train_answer_directory)
    test_data_df = load_all_csv_files(test_csv_directory)
    test_label_df = load_all_csv_files(test_answer_directory)
    
    # DataFrame が空でないか確認
    if train_data_df.empty:
        raise ValueError("train_dataのdfが空です。CSVファイルの内容を確認してください。")
    if train_label_df.empty:
        raise ValueError("train_labelのdfが空です。CSVファイルの内容を確認してください。")
    if test_data_df.empty:
        raise ValueError("test_dataのdfが空です。CSVファイルの内容を確認してください。")
    if test_label_df.empty:
        raise ValueError("test_labelのdfが空です。CSVファイルの内容を確認してください。")
    
    # 学習とテストのデータに変換（各サンプルを1行として扱う）  
    # ※ サンプルが複数行になっている場合、適宜前処理が必要です。
    X_train, y_train = train_data_df.values, train_label_df.values.ravel()
    X_test, y_test = test_data_df.values, test_label_df.values.ravel()
    
    # ★ CatBoostClassifier の初期化 ★
    model = CatBoostClassifier(
        iterations=10000,
        learning_rate=0.01,#0.01わりとよき
        depth=8,
        early_stopping_rounds=100,
        verbose=10,
        #l2_leaf_reg=3,
    )
    
    model.fit(X_train, y_train, eval_set=(X_test, y_test), plot=True)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_per = accuracy * 100
    print(f"Accuracy: {accuracy_per:.2f}%")
    
    # 予測と評価の後に追加
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    
if __name__ == '__main__':
    main()