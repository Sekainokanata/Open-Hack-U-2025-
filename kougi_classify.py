import numpy as np
import pandas as pd
import math
import cv2
import copy
import hand_detection
import os
import sys

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
#from sklearn import preprocessing
#import statistics

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

    df = pd.read_csv( filename )

    y = df['class'] # 1列目を抽出（クラスラベル）
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
# 学習
#----------------------------------------------------------------

# ここにデータファイルを読み込み、認識モデルを学習するプログラムを追加

# 学習用データをCSVファイルから読み込み
filename_train = 'gesture_0_to_6_train_iwado_new.csv'
X_train, Y_train = read_data( filename_train )

X_train = compute_feature_vector(X_train)

# 識別器の初期化
#classifier = LinearDiscriminantAnalysis() # 線形判別分析 (LDA)
classifier = svm.SVC( kernel='rbf' ) # SVM（rbfカーネル）
#classifier = svm.SVC( kernel='linear' ) # SVM（線形カーネル）

# 識別器を学習データを用いて学習
classifier.fit( X_train, Y_train )

#----------------------------------------------------------------
#  カメラの準備
#----------------------------------------------------------------

# カメラ
camera_id = 0
#cap = cv2.VideoCapture(camera_id)
cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

# mediapipeライブラリを用いたジェスチャ認識クラス
detector = hand_detection.detection()

# スクリーンショット回数
screenshot_count = 0

#----------------------------------------------------------------
#  画像を取得して，学習した識別器でジェスチャ認識
#----------------------------------------------------------------

while cap.isOpened():
    #-----------------------------    
    # カメラから画像を１枚取得 (変数imageに格納)
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # 関節点を検出
    results = detector.process( image )

    # 検出された関節点を画像 image に描画
    detector.draw( image, results )

    # 関節点の特徴ベクトルを求める
    x = detector.data_frame( image, results )

    #-----------------------------
    # ジェスチャ認識
    cls = -1
    if len(x) > 0:
        # 関節点の位置からジェスチャ認識
        # cls = 0,1,2,.. は推定されたジェスチャのクラス番号

        # 特徴ベクトル xx の計算
        [xx] = compute_feature_vector( [x] )

        # クラスラベル cls の推定        
        # cls = 0,1,2,.. は推定されたジェスチャのクラス番号        
        [cls] = classifier.predict( [xx] )
        
        print(f'推定クラスラベル: {cls}')

    #-----------------------------
    # 画面に表示する画像を作成

    # 左右反転
    image = cv2.flip( image, 1 )

    # ここに認識結果を画像に描画
    cv2.putText( image,
    text= f'{int(cls)}', # 描画する文字（英数字のみ）
    org=(0,70), # 座標
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,# フォント
    fontScale=3, # 文字の大きさ
    color=(255,255,255), # 文字の色（青, 緑, 赤）, 各0～255
    thickness=5, # 文字の太さ
    lineType=cv2.LINE_4 ) # 文字の描画方法

    # 画像を画面に描画
    cv2.imshow('MediaPipe Hands', image)


    #-----------------------------
    # ESC キーで終了
    k = cv2.waitKey(5) & 0xff
    if k == 27 or k== ord('q'): # ESC key / q key
        break
    elif k == ord('s'):  # 's' key for screenshot
        screenshot_filename = f'screenshot_{screenshot_count}.png'
        cv2.imwrite(screenshot_filename, image)
        print(f'Screenshot saved as {screenshot_filename}')
        screenshot_count += 1

# カメラの撮影を終了
cap.release()
