'''
自动求导：自动微分
PyTorch使用了一种叫做自动微分的技术。也就是说，它会有一个记录我们所有执行操作的记录器，之后再回放记录来计算我们的梯度。这一技术在构建神经网络时尤其有效，因为我们可以通过计算前路参数的微分来节省时间。

在pytorch中所有神经网络的中心，就是autograd包。让我们先快速的访问一下它，然后训练我们自己的第一个神经网络。

autograd包为所有在tensor上的的操作提供自动微分。它是一个运行时刻定义的框架，这意味着你的反向传播(backprop)是在你的代码运行时候定义的，取决于你代码怎么运行，并且每次迭代都是不同的。
'''

'''
tensor
torch.tensor是自动求导的中心类。如果你把它的属性.requires_grad设置成true，它将开始追踪所有在它上面的操作。当你完成你所有的计算之后，你可以调用.backward()方法来把所有的梯度自动计算一遍。此tensor的梯度将会累积到它的.grad属性里面。

如果你要停止一个tensor的历史追踪，你可以调用.detach()方法来把计算历史进行分离开来，并且阻止后续的计算被追踪。

如果要停止追踪历史停止使用多余的内存的话，你可以把代码块藏进 with torch.no_grad():里面。这招在评估模型时特别有用，因为模型可能具有可训练的参数，设置了requires_grad=True，但是评估模型的时候并不需要梯度。训练时候需要，评估时候不需要。

这里有另一个对于自动求导实现的非常重要的类，Function类。

tensor和function这两个类是内部关联的，一起组成了一个有向无环图（acyclic graph），这个图描述了一个完整的计算历史。每个tensor有一个.grad_fn的属性，它引用了创建张量的函数（除了那些用户自己创建的tensor，它们的grad_fn是None）。

如果你想计算导数，你可以在tensor上面调用.backward()方法。如果tensor是一个标量scalar（就是它只包含了一个元素），那么你backward()的时候不需要指定任何参数。但是如果它不止一个元素，你需要指定一个于tensor的形状匹配的gradient参数。
'''

import torch

# 创建一个tensor并设置requires_grad=True来记录它经历的计算。
x = torch.ones(2,2,requires_grad=True)
print(x)