from cvzone.HandTrackingModule import HandDetector
import cv2
import csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
import joblib
import time
import numpy as np
# PCカメラの初期化
cap = cv2.VideoCapture(0)


# --- パラメータ設定 ---
THRESHOLD_UP = -50    # 指が上がるときの速度の閾値（調整必須）
THRESHOLD_DOWN = 50   # 指が下がるときの速度の閾値（調整必須）


####ゆびをあげて、おろした場合を判別するFLAG

move_FLAG = 0

#初期位置を取得したかどうかの関数
FLAG = 0
###手の位置を保存する変数
R_hand_now = []
L_hand_now = []

###learn時の手の位置を保存する変数
R_hand_learn = []
L_hand_learn = []

#Offsetを保存する変数
offset_R_x = 0
offset_R_y = 0
offset_L_x = 0
offset_L_y = 0


# HandDetectorのパラメータ設定
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

def flatten_landmarks(lmList):
    # 各ランドマークは [x, y, z] のようなリスト
    return [coordinate for point in lmList for coordinate in point]

data = []#全てのランドマーク座標を格納するリスト


# --- 学習用データの記録リスト ---
# 今回はシンプルに「速度」を特徴量とする
speed = []  # 各フレームの速度を記録（後で拡張可能）
labels = []    # 0: stationary, 1: finger raise, 2: finger drop

# --- 前フレームのデータ ---
prev_time = time.time()
prev_y = None  # インデックス指先（例: ランドマークID8）のy座標


# モデルの読み込み
classifier = joblib.load('svm_model.pkl')

# カメラのフレーム取得を繰り返す
while True:
    success, img = cap.read()
    # 現在のフレーム画像から手を検出
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

    # 手を検出したら情報を算出
    if hands:
        hand1 = hands[0]
        # 手のランドマーク座標
        lmList1 = hand1["lmList"]
        # 手の境界となる座標(矩形)
        bbox1 = hand1["bbox"]
        # 手の中心座標
        center1 = hand1['center']
        # 右手か左手か
        handType1 = hand1["type"]
        # 立っている指の数
        fingers1 = detector.fingersUp(hand1)
        #print(f'H1 = {fingers1.count(1)}', end=" ") 
        length, info, img = detector.findDistance(lmList1[8][0:2], lmList1[12][0:2], img, color=(255, 0, 255),
                                                  scale=10)

        # 手が二つ検出されている場合は、もう一つの手の情報を算出
        if len(hands) == 2:
            # Information for the second hand
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            bbox2 = hand2["bbox"]
            center2 = hand2['center']
            handType2 = hand2["type"]
            fingers2 = detector.fingersUp(hand2)
            #print(f'H2 = {fingers2.count(1)}', end=" ")
            length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img, color=(255, 0, 0),
                                                      scale=10)
        else:
            lmList2 = []  # 2つ目の手がない場合は空リスト

        #print(" ")  # New line for better readability of the printed output
    
    if FLAG == 0:
        with open("first_position.csv", "r", newline="") as file:
            reader = csv.reader(file)
            rows = list(reader)
            
            # rows[0]が右手の座標、rows[1]が左手の座標と仮定
            R_hand_learn = [int(rows[0][0]), int(rows[0][1])]
            L_hand_learn = [int(rows[1][0]), int(rows[1][1])]
            
            R_hand_now = [int(bbox1[0]), int(bbox1[1])]
            L_hand_now = [int(bbox2[0]), int(bbox2[1])]
            
            offset_R_x = R_hand_now[0] - R_hand_learn[0]
            offset_R_y = R_hand_now[1] - R_hand_learn[1]
            
            offset_L_x = L_hand_now[0] - L_hand_learn[0]
            offset_L_y = L_hand_now[1] - L_hand_learn[1]
            
            print(f"Offset R: ({offset_R_x}, {offset_R_y})")
            print(f"Offset L: ({offset_L_x}, {offset_L_y})")
            
            FLAG = 1  # 一度保存したらFLAGを更新して再度保存しないようにする
        print("初期位置の取得が完了しました")
    

    
    features = flatten_landmarks(lmList1)
    n_landmarks_R = len(lmList1)
    for i in range(0, n_landmarks_R * 3, 3):
        features[i]   = features[i]   - offset_R_x  # x座標の補正
        features[i+1] = features[i+1] - offset_R_y  # y座標の補正
 
    if lmList2:
        features += flatten_landmarks(lmList2)
        n_landmarks_L = len(lmList2)
        # featuresの右手分の後ろから左手のデータが連結されていると仮定
        start_index = n_landmarks_R * 3
        for i in range(start_index, start_index + n_landmarks_L * 3, 3):
            features[i]   = features[i]   - offset_L_x
            features[i+1] = features[i+1] - offset_L_y
            
    data.append(features)
    
    cls = -1
    x, y, _ = lmList1[8]  # インデックス指先（例: ランドマークID8）の座標
    
    current_time = time.time()
    dt = current_time - prev_time
    
    if prev_y is not None and dt > 0:
        # 速度 = Δy / Δt（画面の座標系ではyが下方向に増えるので、上げる動作は負の値になる）
        velocity = (y - prev_y) / dt

        # 特徴量として速度を記録
        speed.append([velocity])

        # 閾値に応じてラベルを割り振る
        if velocity < THRESHOLD_UP:
            label = 1  # 指が上がってる＝"raise"
            cv2.putText(img, "Raise", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            move_FLAG = 1
        elif velocity > THRESHOLD_DOWN:
            label = 2  # 指が下がってる＝"drop"
            cv2.putText(img, "Drop", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            if move_FLAG == 1:
                move_FLAG += 1
        else:
            label = 0  # stationary（ホールド状態）
            cv2.putText(img, "Stationary", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            if move_FLAG == 2:
                move_FLAG += 1


        labels.append(label)
        
     # 前回値の更新
    prev_y = y
    prev_time = current_time
    
    
    
    
    
    
    
    
    
    
    if len(data) > 0:
        # 関節点の位置からジェスチャ認識
        # cls = 0,1,2,.. は推定されたジェスチャのクラス番号

        # cls = 0,1,2,.. は推定されたジェスチャのクラス番号        
        [cls] = classifier.predict(data)
        
        #print(f'推定クラスラベル: {cls}')
        cv2.putText(img, str(cls), (300,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        
    



    data.clear()  
    
    
    if (move_FLAG == 3):
        print(str(cls) + "が入力されました！")
        move_FLAG = 0
    
    # 手のトラッキングを画像上に表示
    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()