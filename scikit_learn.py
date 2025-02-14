import numpy as np
import pandas as pd
import sys

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm

def svm_learn(X_train, Y_train, X_test, Y_test):
    
    # N: データ数，M: 特徴ベクトルの次元
    N,M = np.shape(X_train)
    N_test,M_test = np.shape(X_test)

    # データ数 N，特徴ベクトルの次元 M を表示
    # 特徴ベクトルの次元が同じかどうかをチェック
    #print(f'学習用データ: {filename_train}: N={N}, M={M}')
    #print(f'評価用データ: {filename_test}: N={N_test}, M={M_test}')
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
    
    # 正解率やクラス毎の正解率を表示する機能を追加する
    collect = 0
    for i in range(N_test):
        # 評価用データ X_test[i] の認識結果が Y_test[i] と一致すれば正解
        x = X_test[i]
        y = Y_test[i]
        results = classifier.predict( [x] )
        cls = results[0]
        if cls == y:
            collect += 1
        print(f'{i} true {y} predicted {cls}')

    print(f'true%all : {collect * 100 / N_test}')
    
    return classifier
    

