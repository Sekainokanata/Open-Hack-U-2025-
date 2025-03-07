from tensorflow.python.client import device_lib # GPUデバイス情報取得ライブラリのインポート
device_lib.list_local_devices() # GPUデバイス情報取得
