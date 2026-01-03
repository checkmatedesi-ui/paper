### FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

### 一、简述

##### 原因：

##### 牺牲模型质量来降低计算复杂度，但往往无法实现实际运行速度的提升



##### 做出的改进：

##### 使attention算法具备**IO感知**能力——即考虑 GPU不同层级内存之间的读写操作。

通过分块技术减少HBM与SRAM之间的内存读写次数



##### 结果：

##### 证明其比标准注意力机制需要更少的 HBM访问次数，并且在特定 SRAM容量范围内达到最优。将 FlashAttention扩展至块稀疏注意力，得到一种比现有所有近似注意力方法更快的近似注意力算法。



### 二、细节

##### 技术细节：

1.**分块计算**



- 在计算Attention输出时，通过外层循环遍历KV的块，内层循环遍历Q的块，直接在SRAM中完成局部Attention的计算，而不是像标准做法那样一次性对整个大矩阵进行运算。

  传统

![image-20260102100552155](C:\Users\z1318\AppData\Roaming\Typora\typora-user-images\image-20260102100552155.png)

改进分块

![image-20260102095138543](C:\Users\z1318\AppData\Roaming\Typora\typora-user-images\image-20260102095138543.png)

2.**在线Softmax重组**

- 这意味着可以在不读取完整行的情况下，逐步计算并累积最终的Attention结果，从而支持上述的分块策略。
- safe softmax，由于FP16容易溢出

![image-20260102100356222](C:\Users\z1318\AppData\Roaming\Typora\typora-user-images\image-20260102100356222.png)

3.**反向传播中的重计算**

- 标准Attention在**反向传播**时需要**复用**前向传播计算出的巨大的注意力矩阵，这通常需要将其**存储在HBM**中，消耗大量显存
- **策略**：FlashAttention选择**不存储**这个巨大的注意力矩阵。相反，在反向传播时，利用存储在HBM中的**输入块**（Q, K, V）和前向传播时计算出的轻量级统计量（归一化因子），在**SRAM**中**重新计算**所需的Attention块。
- **收益**：虽然这增加了计算量（FLOPs），但由于大幅减少了HBM的读写操作（IO），且SRAM计算速度极快，最终的总运行时间反而更短，且显存占用从二次方降低到了线性 



### 三、结果

- **实验场景与任务**：
  1. **BERT-large 训练**：在MLPerf 1.1基准上，任务是掩码语言建模。
  2. **GPT-2 训练**：在OpenWebText数据集上训练不同规模的GPT-2（从Small到Medium）。
  3. **长序列基准（Long-range Arena, LRA）**：测试处理长序列（1k-4k）的能力，包括图像分类（Image）、文本匹配等任务。

- **主要结果**：
  - 在**BERT-large**上，训练速度比MLPerf记录保持者（NVIDIA）快15%。
  - 在**GPT-2**上，训练速度比HuggingFace标准实现快3倍，比Megatron-LM快1.7倍。
  - 在**LRA基准测试**中，FlashAttention作为精确注意力方法，速度不仅超过了标准Attention（2.4倍），甚至超过了许多近似注意力方法（如Linformer），同时保持了最高的准确率。