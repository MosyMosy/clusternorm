import torch

if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
global_device = torch.device(dev)