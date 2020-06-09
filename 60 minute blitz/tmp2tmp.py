# 临时用用的，求证某些事情的 test

from __future__ import print_function
import torch

x = torch.tensor([5.5,3])

x = x.new_ones(5,3,dtype=torch.double)  # new_* 这类方法需要输入一个sizes
print(x)