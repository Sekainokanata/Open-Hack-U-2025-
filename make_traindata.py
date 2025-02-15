import pyaudio
import matplotlib.pyplot as plt
import numpy as np
import keyboard

# 録音の設定pip
FORMAT = pyaudio.paInt16  # 音声のフォーマット
CHANNELS = 1  # モノラル
RATE = 44100  # サンプリングレート（44.1kHz）
CHUNK = 1024  # データを一度に読み取る量
RECORD_SECONDS = 0.5  # 録音時間（秒）
OUTPUT_FILENAME = "output.wav"  # 保存するファイル名



def record_sound():
    
    # PyAudioオブジェクトの生成
    audio = pyaudio.PyAudio()

    # ストリームの開始
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("録音を開始します...")

    frames = []

    # 録音のメインループ
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("録音終了。ファイルに保存します...")

    # ストリームの停止と終了
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 録音データを音声配列に変換 (WAVファイルには保存しない)
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

    # 時間軸の生成
    time = np.linspace(0, RECORD_SECONDS, num=len(audio_data))

    return audio_data, time




data = []

while True:
    k = keyboard.read_event()
    if k.event_type == keyboard.KEY_DOWN:  # キーが押されたか確認
        if k.name == "0" or k.name == "1":  # "0" か "1" のキー入力を確認
            
            audio_data, time = record_sound()  # データを生成
            key = int(k.name)
            print(key)
            print(audio_data)
            a = [key] + audio_data.tolist()
            #print(a)
            data.append(a)  # データをリストに追加
        
        elif k.name == "esc":  # "esc" キーが押されたら終了
            break

#print(data)
# 最後にリストをNumPy配列に変換
data = np.array(data, dtype=object)

print(data)

print(data[0][0])
print(data[1][0])

