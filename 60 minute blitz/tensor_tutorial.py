'''
https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html

Pytorch 是什么？
它是一个基于python的科学计算库，致力于下面方面：
1、在更强悍的GPU上面替代numpy
2、一个深度学习的研究平台，提供最大的灵活性和速度

Tensors张量
所谓tensor和numpy的ndarray非常相似，能用于在GPU上面加速运算。
'''

from __future__ import print_function
import torch

'''
注意
一个违背初始化的矩阵是不被鼓励提倡的，但不包含用在使用它之前用已知的值定义的这种情况。
当一个未初始化的矩阵被创建后，不论设置值被分配内存的时候，就是它被初始化值的时候。
'''

# 创建一个未初始化的5x3的矩阵
x = torch.empty(5,3)
print(x)  # 手册上写的out全部是0，我运行的并不是！不知道是不是我使用python3.6的缘故


# 创建一个随机初始化的矩阵
x = torch.rand(5,3)
print(x)


# 创建一个dtype为long的5x3的全零矩阵
x = torch.zeros(5,3,dtype=torch.long)
print(x)


# 从数据data直接创建tensor张量
x = torch.tensor([5.5,3])
print(x)


# 基于已经存在的tensor来创建一个tensor。 除非用户提供了新值，否则这些方法将重用输入tensor的属性，例如dtype
x = x.new_ones(5,3,dtype=torch.double)  # new_* 这类方法需要输入一个sizes
print(x)