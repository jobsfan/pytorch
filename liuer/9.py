# 多分类问题 不用sigmod，而是用softmax NLLLOSSS(nagative log likelihood loss)
# CrossEntropyLoss <===> LogSoftmax + NLLLoss

import numpy as np
import torch

y = np.array([1,0,0])
z = np.array([0.2,0.1,-0.1])
y_pred = np.exp(z) / np.exp(z).sum()

loss = (-y * np.log(y_pred)).sum()
print(loss)

# ==================================================================

y = torch.LongTensor([0])
z = torch.Tensor([[0.2,0.1,-0.1]])
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(z,y)
print(loss)

# ==================================================================

criterion = torch.nn.CrossEntropyLoss()
Y = torch.LongTensor([2,0,1])

Y_pred1 = torch.Tensor([[0.1,0.2,0.9],
                        [1.1,0.1,0.2],
                        [0.2,2.1,0.1]])

Y_pred2 = torch.Tensor([[0.8,0.2,0.3],
                        [0.2,0.3,0.5],
                        [0.2,0.2,0.5]])

l1 = criterion(Y_pred1,Y)
l2 = criterion(Y_pred2,Y)
print("Batch Loss1 =",l1.data, "\nBatch Loss2 =",l2.data)

# CrossEntropyLoss <===> LogSoftmax + NLLLoss