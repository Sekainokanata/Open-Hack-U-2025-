import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml  # DirectMLを利用するためのライブラリ
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# DirectMLデバイスの設定
device = torch_directml.device()

# パラメータ設定
num_landmarks = 52    # 手のランドマークの数
num_coordinates = 7   # 各ランドマークのx, y, zの3次元座標と回転
time_steps = 60       # フレーム数
input_features = num_landmarks * num_coordinates + 3 + 6 # csvのカラム数(手動調整)
num_classes = 26 - 2

# CSVファイルからデータを読み込む関数（PyTorch用の前処理）
def load_csv_data(csv_file_path):
    df = pd.read_csv(csv_file_path, header=None)
    data = df.values  # shape: (time_steps, input_features)
    return data

def load_all_csv_files(directory_path):
    X = []
    file_list = sorted(os.listdir(directory_path))
    for file_name in file_list:
        if file_name.endswith('.csv'):
            csv_file_path = os.path.join(directory_path, file_name)
            samples = load_csv_data(csv_file_path)
            # 必要なら1列目を除くなどの処理（元コードではsamples[:, 1:]）
            X.append(samples)  
    return np.array(X)

def load_answer_labels(directory_path):
    Y = []
    file_list = sorted(os.listdir(directory_path))
    for file_name in file_list:
        if file_name.endswith('.csv'):
            csv_file_path = os.path.join(directory_path, file_name)
            labels = load_csv_data(csv_file_path)
            labels = labels.flatten()
            Y.append(labels)
    Y = np.concatenate(Y, axis=0)
    return Y

# PyTorch Datasetの定義
class HandDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, hidden):
        h, c = hidden
        gates = self.x2h(x) + self.h2h(h)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        c_next = (forgetgate * c) + (ingate * cellgate)
        h_next = outgate * torch.tanh(c_next)
        return h_next, c_next

class UnfusedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(UnfusedLSTM, self).__init__()
        self.cell = MyLSTMCell(input_size, hidden_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.cell.hidden_size, device=x.device)
        c_t = torch.zeros(batch_size, self.cell.hidden_size, device=x.device)
        outputs = []
        for t in range(seq_len):
            h_t, c_t = self.cell(x[:, t, :], (h_t, c_t))
            outputs.append(h_t.unsqueeze(1))
        return torch.cat(outputs, dim=1)

# 既存のCNNRNNモデル内でのLSTM部分の置き換え例
class CNNRNN(nn.Module):
    def __init__(self):
        super(CNNRNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        # nn.LSTMの代わりにUnfusedLSTMを利用
        self.lstm1 = UnfusedLSTM(input_size=128, hidden_size=128)
        self.lstm2 = UnfusedLSTM(input_size=128, hidden_size=64)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

def main(csv_directory, answer_directory):
    # データ読み込み
    X = load_all_csv_files(csv_directory)  # shape: (num_samples, time_steps, input_features - 1)
    y = load_answer_labels(answer_directory)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = HandDataset(X_train, y_train)
    test_dataset = HandDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = CNNRNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # トレーニングループ（エポック数は10）
    model.train()
    for epoch in range(20):
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
    
    # 評価
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            pred = outputs.argmax(dim=1)
            total += batch_y.size(0)
            correct += (pred == batch_y).sum().item()
    print("Test Accuracy: {:.2f}%".format(100 * correct / total))
    
if __name__ == '__main__':
    csv_directory = r'.\INPUT_csv'
    answer_directory = r'.\AnswerLABEL'
    main(csv_directory, answer_directory)
