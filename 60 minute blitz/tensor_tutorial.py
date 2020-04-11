'''
https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html

Pytorch 是什么？
它是一个基于python的科学计算库，致力于下面方面：
1、在更强悍的GPU上面替代numpy
2、一个深度学习的研究平台，提供最大的灵活性和速度

Tensors张量
所谓tensor和numpy的ndarray非常相似，能用于在GPU上面加速运算。
'''

from __future__ import print_function  # 在开头加上from __future__ import print_function这句之后，即使在python2.X，使用print就得像python3.X那样加括号使用。python2.X中print不需要括号，而在python3.X中则需要。
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
x = x.new_ones(5,3,dtype=torch.double)  # new_* 这类方法需要输入一个sizes，x是要已经有的！！结果size可以和原来不同
print(x)

x = torch.randn_like(x, dtype=torch.float)  # 覆盖dtype
print(x)  # 结果是和原来的size相同


# 获取tensor的size，用x.size()
print(x.size())

'''
注意
torch.Size事实上是一个tuple，因此它支持所有的tuple操作
'''

'''
操作
'''

# 加法的语法1
y = torch.rand(5,3)
print(x + y)

# 加法的语法2
print(torch.add(x,y))

# 加法：提供一个输出的参数tensor来接收结果
result = torch.empty(5,3)
torch.add(x,y,out=result)
print(result)

# 加法：结果原地替换的方法 in-place，把x加到y上面，y的值也因此改变
y.add_(x)
print(y)


'''
注意
任何的操作，只要它后面跟了个_，例如x.copy_(y)，x.t_()，都会改变x自身！
'''

# 对于所有的pytorch的tensor，你都可以使用标准的numpy-like（类似于numpy）的索引！ 分片
print(x[:,1])

# 改变tensor的形状，resizing，如果你想要resize或者说reshape一个tensor张量，你可以使用torch.view
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1,8)  # 哪一维的size如果写了-1，那么它会自动计算，是参照着其他维的变化而变化的，不想计算的懒人写法
print(x.size(),y.size(),z.size())

# 如果你有一个只有一个元素的tensor，那么使用 .item() 方法得到它的值（它的值是一个python的数值）
x = torch.randn(1)
print(x)
print(x.item())

'''
扩充阅读
有超过100的tensor的操作，包括变形，索引，截取，数学计算，线性代数，随机数值，等等
https://pytorch.org/docs/torch
'''


'''
numpy的桥梁作用
可以将一个tensor转换成numpy的array，反之亦然。
tensor和numpy的array将共享他们的下面的内存地址（在CPU上面运行的话），改变一个，另一个也随之改变
'''

# 将tensor转换成numpy的array
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

# 我们来看看tensor改变，numpy的array的值是不是也变了
a.add_(1)
print(a)
print(b)


# 将numpy array转换成torch tensor
# 见证一下改变np array的值，对应的tensor也自动改变
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# 除了字符类的tensor之外，在CPU上面所有的tensor都支持和numpy的相互转换

'''
CUDA tensor
使用.to方法，可以将tensor移动到任何设备
'''

# 只有到cuda有的时候，才能运行，我这穷屌的电脑，应该运行不了。
# 我们使用"torch.devie"对象将tensor移进GPU，移出GPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # 一个CUDA设备的对象
    y = torch.ones_like(x, device=device)  # 直接在GPU上面创建一个tensor
    x = x.to(device)  # 或者使用to方法，移动到GPU
    z = x + y
    print(z)
    print(z.to("cpu",torch.double))  # to方法移动到CPU，to方法同时可以改变dtype！！！

# 果然是没有GPU，没有CUDA！！