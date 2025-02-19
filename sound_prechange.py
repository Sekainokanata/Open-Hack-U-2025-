import librosa
import librosa.display
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


def Sound_prechange():
    # === 1. MP3ファイルを読み込む ===
    filename = "sample_daipan.mp3"  # MP3ファイルのパス
    y, sr = librosa.load(filename, sr=None)  # MP3ファイルを読み込む

    # === 2. 1.5kHz～3kHzのバンドパスフィルタを適用 ===
    y_filtered = bandpass_filter(y, sr)

    # === 3. 特徴量（MFCC, スペクトル特徴）を抽出 ===
    n_mfcc = 13  # MFCCの次元数
    target_frames = 100  # フレーム数を固定
    mfccs = librosa.feature.mfcc(y=y_filtered, sr=sr, n_mfcc=n_mfcc)

    # フレーム数を固定
    mfccs = librosa.util.fix_length(mfccs, size=target_frames, axis=1)

    # # === 4. 結果を可視化（軸を固定） ===
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(mfccs, sr=sr, x_axis='time', vmin=-100, vmax=100)  # 軸固定
    # plt.colorbar()
    # plt.title("MFCC (1.5kHz-3kHz Enhanced)")
    # plt.xlabel("Time (Fixed Frame)")
    # plt.ylabel("MFCC Coefficients")
    # plt.show()


def bandpass_filter(data, sr, lowcut=1500, highcut=3000, order=5):
    nyquist = sr / 2  # ナイキスト周波数
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)


# 関数を実行
Sound_prechange()

