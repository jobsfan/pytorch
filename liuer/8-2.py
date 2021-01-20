import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

train_dataset = datasets.MNIST(root='../dataset/mnist',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = datasets.MNIST(root='../dataset/mnist',train=False,transform=transforms.ToTensor(),download=True)

train_loader = DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=32,shuffle=False)

for batch_idx,(inputs,target) in enumerate(train_loader):
    pass