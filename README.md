# MNIST Vision Pipeline

Hệ thống huấn luyện mô hình CNN phân loại chữ số viết tay MNIST với PyTorch, tối ưu hoá cho việc code trên VS Code và train trên Kaggle GPU.

---

## 1. Cấu trúc thư mục
```
mnist-vision-pipeline/
├── src/
│   ├── dataset.py        # Module nạp dữ liệu (tự động tải hoặc đọc từ Kaggle CSV)
│   └── preview.py        # In thông tin batch & xuất ảnh minh hoạ ra màn hình
├── notebooks/            # Jupyter Notebooks thí nghiệm
├── requirements.txt      # Thư viện yêu cầu
└── README.md
```

---

## 2. Hướng dẫn chạy trên Kaggle

### Bước 1: Đẩy code từ VS Code lên GitHub
```bash
git add .
git commit -m "Add dataset loader and preview script"
git push origin main
```

### Bước 2: Chạy trên Kaggle Notebook
1. Tạo một Notebook mới trên Kaggle: `File -> New Notebook`.
2. Bật GPU: **Settings -> Accelerator -> GPU T4** (hoặc P100).
3. Bật Internet: **Settings -> Internet -> On**.
4. Trong một cell của Kaggle Notebook, chạy các lệnh:

```bash
# 1. Clone repository về Kaggle
!git clone https://github.com/<tai-khoan-cua-ban>/mnist-vision-pipeline.git
%cd mnist-vision-pipeline

# 2. Cài thư viện (nếu cần)
!pip install -r requirements.txt -q

# 3. Chạy script in thông tin và xuất ảnh mẫu
# Cách A: Dùng torchvision tải trực tiếp (khuyên dùng, mất 2s)
!python src/preview.py

# Cách B: Nếu bạn dùng dataset Kaggle Digit Recognizer (train.csv)
# !python src/preview.py --csv /kaggle/input/digit-recognizer/train.csv

# 4. Hiển thị ảnh ngay trong Notebook
from IPython.display import Image
Image("mnist_preview.png")
```
