import os

# リネームしたいフォルダのパス
folder_path = "./@@@@@@@@@"

# フォルダ内のすべてのファイルを取得
for filename in os.listdir(folder_path):
    old_file = os.path.join(folder_path, filename)  # 古いファイルのフルパス
    if os.path.isfile(old_file):  # ファイルかどうか確認
        new_file = os.path.join(folder_path, f"1{filename}")  # 新しいファイル名を定義
        os.rename(old_file, new_file)  # リネーム
        print(f"Renamed: {old_file} -> {new_file}")


