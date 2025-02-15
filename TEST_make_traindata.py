#これはmake_traindata.pyの実行テストをするプログラムです．
import matplotlib.pyplot as plt
import make_traindata

audio_data, time = make_traindata.make_traindata()

 # グラフのプロット
plt.figure(figsize=(10, 4))
plt.plot(time, audio_data)
plt.title("Recorded Audio Waveform")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()