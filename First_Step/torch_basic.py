import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# install pytorch verification on local machine
print(f"version of pytorch: {torch.__version__}")

if torch.cuda.is_available():
    print(f"PyTorch can access GPU: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch cannot access GPU, check your CUDA installation.")

# tensors
x = np.random.randn(10, 1)
x = torch.tensor(x, requires_grad=True)
print(x)

w = np.random.randn(10, 1)
w = torch.tensor(w.T, requires_grad=True)
print(w)

b = torch.tensor(1., requires_grad=True)
print(b)

# build a computational graph
y = w @ x + b
print(y)