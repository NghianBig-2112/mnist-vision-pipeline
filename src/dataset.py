import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import pandas as pd
import numpy as np


class KaggleCSVDataset(Dataset):
    """
    Dataset dùng cho định dạng CSV của Kaggle (ví dụ: cuộc thi Digit Recognizer)
    File train.csv có cấu trúc: label, pixel0, pixel1, ..., pixel783
    """
    def __init__(self, csv_file: str, transform=None):
        df = pd.read_csv(csv_file)
        self.labels = df.iloc[:, 0].values
        # Reshape từ 784 về ảnh 28x28
        self.images = df.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0

        return image, label


def get_mnist_loaders(data_dir: str = "./data", batch_size: int = 64, from_csv: str = None):
    """
    Hàm lấy DataLoader:
    - Nếu truyền `from_csv`: Đọc từ file CSV của Kaggle (/kaggle/input/...)
    - Nếu không: Tải tự động qua torchvision trực tiếp trên Kaggle (chỉ mất ~2s)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Mean và Std chuẩn của MNIST
    ])

    if from_csv and os.path.exists(from_csv):
        print(f"--> Đang nạp dữ liệu từ CSV: {from_csv}")
        dataset = KaggleCSVDataset(csv_file=from_csv, transform=transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    else:
        print(f"--> Đang tải/nạp MNIST qua torchvision (Lưu tại: {data_dir})...")
        train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        val_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
