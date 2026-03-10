import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import torch
import torch.nn as nn

xs_train, ys_train = sklearn.datasets.make_moons(n_samples=100, noise=0.1)
print("Training samples, input shape: ", xs_train.shape)                    # 2D samples: [[x1, x2]]
print("Training samples, labels shape:", ys_train.shape)                    # labels of samples: 0 or 1
plt.scatter(xs_train[:,0], xs_train[:,1], c=['r' if y == 1 else 'b' for y in ys_train])  # red for label 1, blue for 0
plt.show()