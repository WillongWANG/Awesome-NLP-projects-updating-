# 1.DAPT

几个结论： 
1. 在目标领域的数据集上继续预训练（DAPT）可以提升效果；目标领域与语言模型的原始训练语料越不相关，DAPT的效果提升越明显。 
2. 在具体任务的数据集上继续预训练（TAPT）可以十分“廉价”地提升效果。 
3. 结合二者（先进行DAPT，再进行TAPT）可以进一步提升效果。 
4. 如果能够获取更多的，任务相关的无标注数据继续训练（Curated-TAPT）,效果最佳。 
5. 如果无法获取更多的、任务相关的无标注数据，采取一种十分轻量化的简单数据选择策略，效果也会提升。作者采用的方法很简单，对于无标注的句子先做sentence-embedding，然后使用kNN最近邻方法，选取k个最相似的句子作为任务相关的无标注数据，然后进行TAPT。

# 2.分布式训练  

> 实现

1. 在使用 distributed 包的任何其他函数之前，需要使用 init_process_group 初始化进程组，同时初始化 distributed 包.  
2. 创建分布式并行模型 DDP(model, device_ids=device_ids)  
3. 构建分布式 Sampler和 Dataloader    
4. 执行训练  


# 3.混合精度


> 混合精读目的

- 使用float16来运算的目的：
    1. 减少显存占用
    2. 加快训练和推断的计算

- float16带来的问题：
    1. 溢出错误
    2. 舍入误差

- 解决办法
    1. 混合精度训练
    2. 损失放大

> 实现

三行代码实现：

```
from apex import amp
# 初始化模型和优化器
model, optimizer = amp.initialize(model, optimizer, opt_level="O1") # 这里是“欧一”，不是“零一”

# 计算梯度
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
```