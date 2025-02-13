import numpy as np
import pandas as pd
import sys
import os
import math
import sound_prechange

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm




# CSVファイルから特徴ベクトルを読み込む
def read_data( filename ):
    if not os.path.exists( filename ):
        print(f'Error: No such file: {filename}')
        sys.exit()
    #else:
    #    print(f'Reading: {filename}')

    # pandasライブラリを用いてcsvファイルをデータフレームdfに読み込み
    df = pd.read_csv( filename )

    y = df['class'] # 1列目を抽出（クララベル）
    x = df.drop(['class'],axis=1)  # 2列目以降を抽出（特徴ベクトル）

    # numpy配列に変換する
    X = np.array(x)
    Y = np.array(y).astype(int) # クラス番号は整数に変換

    return X,Y


#----------------------------------------------------------------
#  評価
#----------------------------------------------------------------

# 評価用データをCSVファイルから読み込み
filename_test = 'gesture_0_to_6_test_furuya_new.csv'
X_test, Y_test = read_data( filename_test )

# 特徴ベクトルを計算したものを受け取る関数
X_test  = sound_prechange( X_test )

# N: データ数，M: 特徴ベクトルの次元
N_test,M_test = np.shape(X_test)

print(f'評価用データ: {filename_test}: N={N_test}, M={M_test}')



# 正解率やクラス毎の正解率を表示する機能を追加する
collect = 0
class_num = 7
collect_list = [[0 for i in range(2)] for j in range(7)]

for i in range(N_test):
    # 評価用データ X_test[i] の認識結果が Y_test[i] と一致すれば正解
    x = X_test[i]
    y = Y_test[i]
    results = classifier.predict( [x] )#ここどうしよう
    cls = results[0]
    collect_list[y][0] += 1
    if cls == y:
        collect += 1
        collect_list[y][1] += 1
    print(f'{i} true {y} predicted {cls}')

print(f'true%all : {collect * 100 / N_test}')
for i in range(class_num):
    print(f'true%{i} : {collect_list[i][1] * 100 / collect_list[i][0]}')
