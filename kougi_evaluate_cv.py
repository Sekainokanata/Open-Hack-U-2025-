import numpy as np
import pandas as pd
import sys
import os

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.model_selection import cross_validate

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

    y = df['class'] # 1列目を抽出（クラスラベル）
    x = df.drop(['class'],axis=1)  # 2列目以降を抽出（特徴ベクトル）

    # numpy配列に変換する
    X = np.array(x)
    Y = np.array(y).astype(int) # クラス番号は整数に変換

    return X,Y

#----------------------------------------------------------------
#  特徴ベクトルを計算する関数
#----------------------------------------------------------------
def compute_feature_vector( X ):
    # 引数は特徴ベクトルのリスト X[i] (0<=i<N) 

    # N: データ数，M: 特徴ベクトルの次元
    N,M = np.shape(X)

    # 計算される特徴ベクトルのリスト Xout[i] (0<=i<N)
    Xout = []  

    # （例0）21個の関節点の座標を，そのまま42次元の特徴ベクトルにする
    for i in range(N):
        x = list(X[i])  # xは関節点のベクトル (x[j], 0<=j<42)

        ### ここに特徴ベクトルを作成するプログラムを書く
        xout = x        # x を変換せずにそのままコピーする

        Xout.append(xout)   

    return np.array(Xout)

#----------------------------------------------------------------
#  学習
#----------------------------------------------------------------

# データをCSVファイルから読み込み
# 交差検証法により，学習用にも評価用にも用いる
filename = 'gesture_0_to_6_train_iwado_new.csv'
X, Y = read_data( filename )

# 特徴ベクトルを計算し，X に上書き代入する
X = compute_feature_vector( X )

# N: データ数，M: 特徴ベクトルの次元
N,M = np.shape(X)

# 特徴ベクトルの次元が同じかどうかをチェック
print(f'学習/評価用: {filename}: N={N}, M={M}')

# 識別器の初期化
classifier = LinearDiscriminantAnalysis() # 線形判別分析 (LDA)
#classifier = svm.SVC( kernel='rbf' ) # SVM（rbfカーネル）
#classifier = svm.SVC( kernel='linear' ) # SVM（線形カーネル）

#----------------------------------------------------------------
#  交差検証法による評価
#  cross_validate() 関数を用いる
#----------------------------------------------------------------
scoring = ['precision_macro']
#scoring = ['precision_macro', 'recall_macro', 'f1_macro']
scores = cross_validate( classifier, X, Y, scoring=scoring )

print('')
precision_scores = scores['test_precision_macro']
print('accuracy for all folds:')
print(precision_scores)
score = np.average(precision_scores)
print(f'average score: {score}')
