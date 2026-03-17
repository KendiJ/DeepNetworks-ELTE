import numpy as np
import torch
import copy


a = torch.arange(1, 6, dtype=torch.float32)
b = a * a[:, None]

print(b)
print(b.shape)
print(b.stride())
print(b.dtype)
print(b.itemsize)
print(b.numel())
print(b.device)
print(b.ndim)