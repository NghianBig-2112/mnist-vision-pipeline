import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# Visualization tools
import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

# --- Check GPU is availabel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device using GPU: {torch.cuda.is_available()}")

# Download training dataset
train_set = torchvision.datasets.MNIST(
    root = "./data",
    train = True,
    download = True
)

# Downlaod validation dataset
valid_set = torchvision.datasets.MNIST(
    root = "./data",
    train = False,
    download = True
)

