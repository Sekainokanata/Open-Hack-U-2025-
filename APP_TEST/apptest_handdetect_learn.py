from cvzone.HandTrackingModule import HandDetector
import cv2
import keyboard
import csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
import numpy as np
import pandas as pd
import sys
import os
import joblib
import time

# グローバル変数として定義（キーイベントでもアクセス可能にするため）
lmList1 = []
lmList2 = []

# PCカメラの初期化
cap = cv2.VideoCapture(0)

#初期位置を取得したかどうかの関数
FLAG = 0

# HandDetectorのパラメータ設定
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

data = []

# CSVファイルから特徴ベクトルを読み込む
def read_data(filename):
    if not os.path.exists(filename):
        print(f'Error: No such file: {filename}')
        sys.exit()

    # header=None でヘッダーなしとして読み込む
    df = pd.read_csv(filename, header=None)

    y = df.iloc[:, 0]     # 1列目をラベルとして取得
    x = df.iloc[:, 1:]    # 2列目以降を特徴ベクトルとして取得

    X = np.array(x)
    Y = np.array(y)

    return X, Y

def flatten_landmarks(lmList):
    # 各ランドマークは [x, y, z] のようなリスト
    return [coordinate for point in lmList for coordinate in point]

# キーイベントコールバックの定義
def handle_key_event(e):
    global lmList1, lmList2
    if e.event_type == keyboard.KEY_DOWN:
        features = flatten_landmarks(lmList1)
        if lmList2:
            features += flatten_landmarks(lmList2)
        data.append([e.name] + features)

# 非同期的にキーイベントを監視
keyboard.hook(handle_key_event)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, draw=True, flipType=True)
    
    # もし両手検出され、先頭の手が右手でなければ入れ替える
    if len(hands) == 2:
        if hands[0]["type"] != "Right" and hands[1]["type"] == "Right":
            hands[0], hands[1] = hands[1], hands[0]
    
    if FLAG == 0:
        print("初期位置の取得をします。カウントダウン開始。")
        for i in range(3, 0, -1):
            print(i)
            time.sleep(1)

    if hands:
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        bbox1 = hand1["bbox"]
        center1 = hand1["center"]
        handType1 = hand1["type"]
        fingers1 = detector.fingersUp(hand1)
        #print(f'H1 = {fingers1.count(1)}', end=" ")
        length, info, img = detector.findDistance(lmList1[8][0:2], lmList1[12][0:2], img, color=(255, 0, 255), scale=10)

        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            bbox2 = hand2["bbox"]
            center2 = hand2["center"]
            handType2 = hand2["type"]
            fingers2 = detector.fingersUp(hand2)
            #print(f'H2 = {fingers2.count(1)}', end=" ")
            length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img, color=(255, 0, 0), scale=10)
        else:
            lmList2 = []  # 2つ目の手がない場合は空リスト

        #print(" ")
        
    cv2.imshow("Image", img)
    
    
    if FLAG == 0:
        with open("first_position.csv", "w", newline="") as hand_position_file:
            writer = csv.writer(hand_position_file)
            # 2つ目の手がない場合はbbox2は空リストになるので注意
            writer.writerow(bbox1)
            writer.writerow(bbox2)
        FLAG = 1  # 一度保存したらFLAGを更新して再度保存しないようにする
        print("初期位置を取得が完了しました")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

with open("hand_data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for row in data:
        writer.writerow(row)

print("記録終了！ラーニングに入るよ！")

# 学習用データをCSVファイルから読み込み
filename_train = 'hand_data.csv'
X_train, Y_train = read_data( filename_train )

classifier = svm.SVC( kernel='linear' ) # SVM（線形カーネル）

classifier.fit( X_train, Y_train )

# モデルの保存
joblib.dump(classifier, 'svm_model.pkl')
print("学習終了！モデルをsvm_model.pklに保存しました")