import torch
if torch.cuda.is_available():
    print(f"PyTorch can access GPU: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch cannot access GPU, check your CUDA installation.")