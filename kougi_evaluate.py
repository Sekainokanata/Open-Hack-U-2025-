import numpy as np
import pandas as pd
import sys
import os
import math

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
#from sklearn.model_selection import cross_validate

#----------------------------------------------------------------
#  read_data() - CSVファイルを読み込む
#----------------------------------------------------------------

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
#  特徴ベクトルを作成する関数
#----------------------------------------------------------------
def make_feature_vector_origin(x, M):
    xout = list(x)
    
    for i in range(0, M, 2):
        xout[i] -= x[0]
        xout[i] -= x[1]
    return xout

def make_feature_vector_relative_distance(x):
    x_list = [8, 16, 24, 30]
    y_list = [x + 1 for x in x_list]
    xout = []
    
    for i in range(len(x_list)):
        for j in range(i + 1, len(x_list)):
            dx1 = x[x_list[i]] - x[x_list[j]]
            dx2 = x[y_list[i]] - x[y_list[j]]
            
            dist = math.sqrt(dx1 ** 2 + dx2 ** 2)
            xout.append(dist)
    
    return xout

#----------------------------------------------------------------
#  特徴ベクトルを計算する関数
#----------------------------------------------------------------
def compute_feature_vector( X ):
    # 引数は特徴ベクトルのリスト X[i] (0<=i<N) 

    # N: データ数，M: 特徴ベクトルの次元
    N,M = np.shape(X)

    # 計算される特徴ベクトルのリスト Xout[i] (0<=i<N)
    Xout = []  

    # （特徴抽出の例 ここから）
    # 21個の関節点の座標を，そのまま42次元の特徴ベクトルにする
    for i in range(N):
        x = list(X[i])  # xは関節点のベクトル (x[j], 0<=j<42)
        
        ### ここに特徴ベクトルを作成するプログラムを書く
        #xout =  make_feature_vector_origin(x, M)   # xを手首原点とする座標変換
        xout = make_feature_vector_relative_distance(x) # それぞれの指先の相対距離（小指を除く）のみを特徴ベクトルとする
        #xout = x # xを変換しない
        Xout.append(xout)   
    #  （特徴抽出の例 ここまで）

    return np.array(Xout)

#----------------------------------------------------------------
#  学習
#----------------------------------------------------------------

# 学習用データをCSVファイルから読み込み
filename_train = 'gesture_0_to_6_train_iwado_new.csv'
X_train, Y_train = read_data( filename_train )

# 評価用データをCSVファイルから読み込み
filename_test = 'gesture_0_to_6_test_furuya_new.csv'
X_test, Y_test = read_data( filename_test )

# 特徴ベクトルを計算し，X_train, X_test に上書き代入する
# 学習用，評価用の両方に対して行う
X_train = compute_feature_vector( X_train )
X_test  = compute_feature_vector( X_test )

# N: データ数，M: 特徴ベクトルの次元
N,M = np.shape(X_train)
N_test,M_test = np.shape(X_test)

# データ数 N，特徴ベクトルの次元 M を表示
# 特徴ベクトルの次元が同じかどうかをチェック
print(f'学習用データ: {filename_train}: N={N}, M={M}')
print(f'評価用データ: {filename_test}: N={N_test}, M={M_test}')
if M!=M_test:
    print('学習用/評価用データの特徴ベクトルの次元が異なります．')
    sys.exit()
print('')

# 識別器の初期化
#classifier = LinearDiscriminantAnalysis() # 線形判別分析 (LDA)
#classifier = svm.SVC( kernel='rbf' ) # SVM（rbfカーネル）
classifier = svm.SVC( kernel='linear' ) # SVM（線形カーネル）

# 識別器を学習データを用いて学習
classifier.fit( X_train, Y_train )

#----------------------------------------------------------------
#  評価
#----------------------------------------------------------------

# 正解率やクラス毎の正解率を表示する機能を追加する
collect = 0
class_num = 7
collect_list = [[0 for i in range(2)] for j in range(7)]

for i in range(N_test):
    # 評価用データ X_test[i] の認識結果が Y_test[i] と一致すれば正解
    x = X_test[i]
    y = Y_test[i]
    results = classifier.predict( [x] )
    cls = results[0]
    collect_list[y][0] += 1
    if cls == y:
        collect += 1
        collect_list[y][1] += 1
    print(f'{i} true {y} predicted {cls}')

print(f'true%all : {collect * 100 / N_test}')
for i in range(class_num):
    print(f'true%{i} : {collect_list[i][1] * 100 / collect_list[i][0]}')

