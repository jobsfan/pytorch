# 多维特征输入，目前还没搞懂啥意思，感觉y值不像是个分类，像是个回归
import numpy as np
import torch

x_ = np.loadtxt('diabetes_data.csv.gz',delimiter=' ',dtype=np.float32)
y_ = np.loadtxt('diabetes_target.csv.gz',delimiter=' ',dtype=np.float32)
y_ = np.expand_dims(y_,axis=1)

x_data = torch.from_numpy(x_)
y_data = torch.from_numpy(y_)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(10,8)
        self.linear2 = torch.nn.Linear(8,6)
        self.linear3 = torch.nn.Linear(6,4)
        self.linear4 = torch.nn.Linear(4,1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x

model = Model()

criterion = torch.nn.BCELoss(reduction='mean')

optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()