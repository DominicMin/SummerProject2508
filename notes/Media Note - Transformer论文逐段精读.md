---
media: https://www.youtube.com/watch?v=nzqlFIcCSWQ
mindmap-plugin: basic
---
问题：
1. 残差连接本身的原理不太理解，以及它如何“连接”各sublayer
2. MLP本身的原理，以及它放在attention层之后的作用
# Transformer精读

## 引言
- 主流是RNN
    - 无法并行化paralization
    - 序列长的话内存开销大
- attention mechanism的应用
    - 之前用在RNN
    - 现在纯attention，并行化高

## 背景
- 之前的工作
    - 卷积神经网络——多通道——多头注意力机制

## 模型架构
- 嵌入层
    - 目标：将每个词转化为向量，方便计算
    - transformer模型采用基于神经网络的训练方法生成查找表，最后的结果是语义相近的词在高维空间中距离近
- encoder
    - 嵌入层——转化为向量
    - 基本单位（layer） Nx
        - sub layer 1：注意力层（==第1个注意力层==）
            - 属于自注意力机制
        - sub layer 2：前馈神经网络
            - 一个Multilayer Perceptron（MLP）多层感知机，最基本的神经网络
        - sub layer之间：
            - 残差链接
                - 长度统一为512
            - layer normalization
                - 先介绍batchnorm的概念
                    - 一批batch数据：一个矩阵，每一行是一个样本，每一列是一个特征
                    - 操作：将每一列均值变成0，方差变成1
                        - 就是zscore标准化$x'=\frac{x-\mu}{\sigma}$
                - 而layernorm是把每一个样本标准化
                - 为什么改用layernorm？
                    - 序列模型中序列长度有变化
                        - 为了长度一致需要padding填充（一般是0）
                        - 一个feature batch，有些地方是真实值，有些地方是填充值，均值和方差不准确而且没有意义
            - 两个加一起，sublayer的输出就是$\mathrm{LayerNorm(x+Sublayer(x))}$
                - 这一步用到了残差连接，让残差函数Sublayer(x)学习“与理想输出之间的差值”，构建了一个“双车道的信息通道”
                - 残差连接能使多个layer进行stacking堆叠，使得最终模型的能力变得很强大
    - 编码器最终的输出是一个捕捉到语义连接的矩阵（向量序列）
- decoder
    - decoder的输入：当前已经生成的预测序列，经过嵌入，再作为输入
    - 基本单位 Nx
        - Masked带掩码的多头注意力层（==第2个注意力层==）
            - 属于自注意力机制
            - 目的：保证训练和预测时行为一致
            - 训练
                - 完成的输入和输出序列
                - 但是decoder自回归生成序列，我们不能让它看到后面的词
                - 因此采用masking matrix将后面的词权重变为负无穷——经过softmax变成0
            - 预测
                - 和训练保持一致
            - decoder第一个注意力层的输出结果：当前已生成的“半句话”经过注意力机制捕捉后的信息矩阵，可以理解为“现有输出的语义信息”
        - 一般多头注意力层（==第3个注意力层==）
            - 这里==不是自注意力==
            - K和V来自编码器的输出（输入中捕捉到的语义信息），Q来自解码器前一个层的输出（现有输出中捕捉到的语义信息）
            - 通过注意力机制把来自不同地方（输入和现有输出）的信息有机结合，解释“现有输出与输入中每个词的语义有什么关联”，从而更好预测下一个输出
            - 这就是cross-attention交叉注意力
        - 前馈神经网络
    - 最终使用softmax多分类器，得到输出词概率
- 注意力机制
    - attention注意力函数：理解上下文的关键
        - Q，K，V是什么
            - Q：当前词的向量经过特定线性变换
                - n个query并在一起，成为$n \times d_k$的矩阵
            - K，V都是序列中的其它词向量经过不同线性变换之后的向量，代表词“能提供什么信息”和“实际提供什么信息”
                - $m \times d_k$的矩阵
        - $\mathrm{Attention}(Q,K,V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
            - scaled dot product：点积注意力+scale
                - 点积注意力很好并行化
                - scale除以$\sqrt{d_k}$：点积结果太大，softmax容易把最大值映射到接近1，另外都是0，太过“尖锐”了，这样模型使用梯度下降训练很难收敛，所以要scale缩放
            - $\mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})$得到权重表
            - 通过与V相乘，得到“原始词向量序列的上下文感知版本”
        - 自注意力：一个词作为Q，句子中的所有词作为KV，这样可以让这个词和整个句子有机连接，得到信息丰富的向量结果
            - self指“同一个序列内”，和之前的注意力机制做区分
    - 多头注意力
        - 将Q，K，V投影到低维——做h次注意力函数——concat简单连接输出再映射，达到最终结果
        - 目标：希望模型可以学习不同的投影方法，从而捕捉更多信息
            - 单个注意力函数其实没什么可以学习的pattern
- position-wise feed-forward network
    - feed-forward network是MLP
    - position-wise
        - 每个词进入同一个MLP处理
        - 每个词经过完全相同的处理，但是不产生信息交换
            - 意义：保持并行化，提高效率。上下文关联已经由attention完成了，不是MLP的任务
        - MLP的作用：进行非线性的转换（ReLU等非线性激活函数），相当于深加工的特征工程
            - DL的本质就是特征工程由机器自己完成
- embedding and softmax
    - learned embeddings：通过神经网络把含义相近的次放在相近的位置
- positional encoding
    - attention机制有无序性
    - 因此encoding的时候要刻画词之间的位置关系
        - 将索引**数值**通过周期变动的三角函数转化为512位的向量
        - 嵌入后的向量乘上根号d，也差不多位于±1之间
        - 两个向量简单相加，就把位置信息囊括进去了
            - 向量和数量不同，向量相加应该理解为平移和叠加，而不是覆盖和抹除
    - DL模型能通过解耦（decouple）在某几个注意力头解出位置和语义信息