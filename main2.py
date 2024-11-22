import torch
import torch.nn as nn
import torch.nn.functional as F  
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

# 学習時と同じニューラルネットワーク構造を定義
class ImprovedNet(nn.Module):
    def __init__(self):
        super(ImprovedNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 保存した改良モデルを読み込む
net = ImprovedNet()
net.load_state_dict(torch.load('improved_mnist_model.pth'))
net.eval()  # 推論モードに切り替え

# 画像を読み込んで推論
def predict_image(image_path):
    img = Image.open(image_path).convert('L')  # グレースケールに変換
    img = img.resize((28, 28))  # サイズを28x28にリサイズ
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    img_tensor = transform(img).unsqueeze(0)  # バッチサイズを1に追加

    # 推論
    with torch.no_grad():
        output = net(img_tensor)
        _, predicted = torch.max(output, 1)
    messagebox.showinfo("推論結果", f"予測された数字: {predicted.item()}")

# ファイル選択ダイアログを表示して画像を選択
def open_file_dialog():
    file_path = filedialog.askopenfilename(title="画像ファイルを選択してください", filetypes=[("PNGファイル", "*.png"), ("すべてのファイル", "*.*")])
    if file_path:
        display_image(file_path)  # 画像を表示
    return file_path

# リサイズされた画像をウィンドウで表示
def display_image(image_path):
    img = Image.open(image_path).convert('L')  # グレースケールに変換
    img = img.resize((280, 280))  # ウィンドウに表示しやすいサイズにリサイズ
    img_tk = ImageTk.PhotoImage(img)  # Tkinterで表示できる形式に変換

    label_image.config(image=img_tk)
    label_image.image = img_tk  # 参照を保持するために変数に保存
    label_image.pack()
    
    # 実行ボタンの表示
    btn_run.config(state=tk.NORMAL)
    btn_run.pack()

# 実行ボタンが押されたら推論を開始
def on_run_button():
    global selected_image_path
    if selected_image_path:
        predict_image(selected_image_path)

# メイン処理
if __name__ == '__main__':
    # ウィンドウのセットアップ
    root = tk.Tk()
    root.title("PyTorch MNIST GuiCtrlPanel")

    # 画像を表示するラベル
    label_image = tk.Label(root)
    
    # ファイル選択ボタン
    btn_select = tk.Button(root, text="画像を選択", command=lambda: open_file_dialog())
    btn_select.pack()

    # 実行ボタン（初期状態では無効）
    btn_run = tk.Button(root, text="推論を実行", state=tk.DISABLED, command=on_run_button)
    
    # 選択された画像のパスを保持する変数
    selected_image_path = None

    def display_image(image_path):
        global selected_image_path
        selected_image_path = image_path
        img = Image.open(image_path).convert('L')  # グレースケールに変換
        img = img.resize((280, 280))  # ウィンドウに表示しやすいサイズにリサイズ
        img_tk = ImageTk.PhotoImage(img)  # Tkinterで表示できる形式に変換

        label_image.config(image=img_tk)
        label_image.image = img_tk  # 参照を保持するために変数に保存
        label_image.pack()

        # 実行ボタンの表示
        btn_run.config(state=tk.NORMAL)
        btn_run.pack()

    root.mainloop()
