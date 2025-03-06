import csv

# CSVファイルを読み込む
input_file = './AnswerLABEL/AnswerLABEL2.csv'  # 読み込むCSVファイル名
output_file = 'output.csv'  # 書き込むCSVファイル名

with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    rows = [row for row in reader]  # CSVのすべての行をリストに保存

# 数値を-1する処理
for i in range(len(rows)):
    for j in range(len(rows[i])):
        try:
            # 数値に変換できる場合のみ -1 する
            rows[i][j] = str(float(rows[i][j]) - 1)
        except ValueError:
            pass  # 数値でない場合はそのまま

# 新しいCSVファイルに書き込む
with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(rows)

print("CSVファイルを処理して保存しました。")
