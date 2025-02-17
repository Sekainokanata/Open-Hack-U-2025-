import cv2
import time
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from sklearn.svm import SVC
import joblib

# --- パラメータ設定 ---
THRESHOLD_UP = -50    # 指が上がるときの速度の閾値（調整必須）
THRESHOLD_DOWN = 50   # 指が下がるときの速度の閾値（調整必須）

# --- カメラとハンドディテクタの初期化 ---
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# --- 学習用データの記録リスト ---
# 今回はシンプルに「速度」を特徴量とする
features = []  # 各フレームの速度を記録（後で拡張可能）
labels = []    # 0: stationary, 1: finger raise, 2: finger drop

# --- 前フレームのデータ ---
prev_time = time.time()
prev_y = None  # インデックス指先（例: ランドマークID8）のy座標

while True:
    success, img = cap.read()
    if not success:
        break

    # --- 手の検出 ---
    hands, img = detector.findHands(img)  # 検出結果をimgに描画してくれる
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]  # 各ランドマークの座標リスト
        
        # 例として、インデックスの指先（ランドマークID8）の位置を取得
        x, y, _ = lmList[8]
        current_time = time.time()
        dt = current_time - prev_time
        
        if prev_y is not None and dt > 0:
            # 速度 = Δy / Δt（画面の座標系ではyが下方向に増えるので、上げる動作は負の値になる）
            velocity = (y - prev_y) / dt

            # 特徴量として速度を記録
            features.append([velocity])

            # 閾値に応じてラベルを割り振る
            if velocity < THRESHOLD_UP:
                label = 1  # 指が上がってる＝"raise"
                cv2.putText(img, "Raise", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            elif velocity > THRESHOLD_DOWN:
                label = 2  # 指が下がってる＝"drop"
                cv2.putText(img, "Drop", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                label = 0  # stationary（ホールド状態）
                cv2.putText(img, "Stationary", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            labels.append(label)

        # 前回値の更新
        prev_y = y
        prev_time = current_time

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()