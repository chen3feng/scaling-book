---
layout: distill
title: "你需要了解的所有 Transformer 数学知识"
# permalink: /main/
description: "这里我们将对 Transformer 架构做一个快速回顾，特别是如何计算 FLOPs、字节以及其他感兴趣的量。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 4

previous_section_url: "../sharding"
previous_section_name: "Part 3: Sharding"

next_section_url: ../training
next_section_name: "Part 5: Training"

giscus_comments: true

authors:
  - name: Jacob Austin
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: Google DeepMind
  - name: Sholto Douglas
    url: "https://x.com/_sholtodouglas"
  - name: Roy Frostig
    url: "https://cs.stanford.edu/~rfrostig/"
  - name: Anselm Levskaya
    url: "https://anselmlevskaya.com/"
  - name: Charlie Chen
    url: "https://x.com/charliexychen"
  - name: Sharad Vikram
    url: "https://sharadvikram.com/"
  - name: Federico Lebron
    url: "https://fedelebron.com/"
  - name: Peter Choy
    url: "https://x.com/pchoy95"
  - name: Vinay Ramasesh
    url: "https://x.com/vinayramasesh"
  - name: Albert Webson
    url: "https://representation.ai/"
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope

bibliography: main.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "Counting Dots"
  - subsections:
    - name: "Forward and reverse FLOPs"
  - name: "Transformer Accounting"
  - name: "Global FLOPs and Params Calculation"
  - name: "Miscellaneous Math"
  - subsections:
    - name: "Sparsity and Mixture-of-Experts"
    - name: "Gradient checkpointing"
    - name: "Key-Value (KV) caching"
  - name: "What Should You Take Away from this Section?"
  - name: "A Few Problems to Work"
  - name: "Appendix"
  - subsections:
    - name: "Appendix A: How does Flash Attention work?"

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## 数点计数

让我们从以下形状的向量 $$x$$、$$y$$ 和矩阵 $$A$$、$$B$$ 开始：

$$
\def \red#1{\textcolor{red}{#1}}
\def \green#1{\textcolor{green}{#1}}
\def \blue#1{\textcolor{blue}{#1}}
\def \purple#1{\textcolor{purple}{#1}}
\def \orange#1{\textcolor{orange}{#1}}
\def \gray#1{\textcolor{gray}{#1}}

\begin{array}{cc}
\textrm{array}  & \textrm{shape} \\ \hline
x               & \textrm{[P]}   \\
y               & \textrm{[P]}   \\
A               & \textrm{[N P]} \\
B               & \textrm{[P M]} \\
\hline
\end{array}
$$

- 点积 $$x \cdot y$$ 需要 $$P$$ 次 _加法_ 和 _乘法_，即总共 $$2P$$ 次浮点运算。
- 矩阵-向量积 $$Ax$$ 沿着 $$A$$ 的行执行 $$N$$ 次点积，总共需要 $$2NP$$ 次 FLOPs。
- 矩阵-矩阵积 $$AB$$ 对 $$B$$ 的每个 $$M$$ 列执行一次矩阵-向量积，总共需要 $$2NPM$$ 次 FLOPs。
- 一般来说，如果我们有两个更高维的数组 $$C$$ 和 $$D$$，其中一些维度是 <span style="color:red">CONTRACTING</span>，另一些是 <span style="color:blue">BATCHING</span>（例如 $$C[\blue{GH}IJ\red{KL}], D[\blue{GH}MN\red{KL}]$$），那么这个收缩操作的 FLOPs 成本是所有 $$C$$ 和 $$D$$ 维度乘积的两倍，其中批处理和收缩维度只计算一次（例如 $$2\blue{GH}IJMN\red{KL}$$）。请注意，只有当维度出现在两个乘数中时，它才是批处理维度。（还请注意，如果没有收缩维度，这只是一个逐元素乘积，那么 2 的系数将不适用。）<d-footnote><b>Contracting</b> 维度是在操作过程中被求和的轴（它们出现在两个输入中但不出现在输出中），就像矩阵乘法中的内维度一样。<b>Batching</b> 维度是出现在两个输入中并被原样传递到输出中的共享轴；它们索引独立的子问题，在 FLOP 计数中不会被相乘。用 einsum 的术语来说：同时出现在两个输入和输出中的标签是批处理维度；同时出现在两个输入中但不出现在输出中的标签是收缩维度。</d-footnote>

$$
\begin{array}{ccc}
\textrm{Operation} & \textrm{FLOPs} & \textrm{Data} \\
\hline
x \cdot y  & 2P   & 2P      \\
A x        & 2NP  & NP + P  \\
AB         & 2NPM & NP + PM \\
[c_0,...,c_N] \cdot [d_0,...,d_N] &
2 \prod c_i \times \prod_{\substack{d_j \notin \blue{BATCH} \\ d_j \notin \red{CONTRACT}}} d_j
&
  \prod c_i + \prod d_j \\
\hline
\end{array}
$$

请注意这样一个事实：对于矩阵-矩阵乘法，*算力* 是按立方级缩放 $$O(N^3)$$，而数据传输仅按平方级缩放 $$O(N^2)$$ —— 这意味着随着我们增大 matmul 的规模，达到算力饱和的限制会变得 *更容易*。这非常不寻常，也从很大程度上解释了为什么我们使用以矩阵乘法为主的架构 —— 它们适合进行缩放！

{% include figure.liquid path="assets/img/matmul-flops.gif" class="img-fluid" %}

### 前向和反向 FLOPs

在训练过程中，我们并不特别关心某个矩阵乘法的结果；我们真正关心的是它的导数。事实证明，计算这个导数的成本大约是直接进行矩阵乘法本身的 3 倍。

如果我们想象 **B** 只是更大网络中的一个矩阵，而 **A** 是我们的输入激活值，且 **C = A B**，那么损失 **L** 对 **B** 的导数由链式法则给出：

$$\frac{\partial L}{\partial B} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial B} = A^T \left(\frac{\partial L}{\partial C}\right)$$

这需要 $2NPM$ FLOPs 来计算（因为它在 $N$ 维度上进行收缩）。同样，损失对 **A** 的导数是

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial A} = \left(\frac{\partial L}{\partial C}\right) B^T$$

这仍然是 $2NPM$ FLOPs，因为 **dL/dC** 是一个大小为 $$[N, M]$$ 的矩阵。虽然这个量不是参数的导数，但它被用来计算网络前面层的导数（例如，就像上面用 dL/dC 计算 dL/dB 一样）。

将这些加起来，我们看到**在训练期间，我们总共有 6NPM FLOPs**，而推理期间为 2NPM：前向传递为 2NPM，反向传递为 4NPM。由于 PM 是矩阵中的参数数量，这就是著名的 $$6 * \text{num parameters} * \text{num tokens}$$ 对训练期间 Transformer FLOPs 的近似值的最简单形式：每个词元需要 $$6 * \text{num parameters}$$ FLOPs。我们将在下面展示一个更准确的推导。

## Transformer 计算

Transformer 是未来。或者说，它们已经是现在了。也许几年前，它们只是众多架构之一。但今天，了解架构的几乎所有细节都是值得的。我们不会重新介绍架构，但[这篇博客](https://jalammar.github.io/illustrated-transformer/)和[原始 Transformer 论文](https://arxiv.org/abs/1706.03762)可能作为有用的参考。

这是 Transformer 解码器架构的基本示意图：

{% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid" caption="<b>图：</b> 该图展示了标准 Transformer 的一层以及从上到下的数据流动。我们采用单字母惯例来描述 Transformer 中数组的形状和布局，再次用红色表示收缩维度，用蓝色表示批量维度。在给定操作中，输入形状位于左上方，参数形状位于右上方，结果形状位于下方，例如 BTD 是门控 einsum 的输入形状，DF 是权重形状。" %}

**注意 [gating einsum]**：上面的图表使用了“[gating einsum](https://arxiv.org/abs/2002.05202)”<d-cite key="glu"></d-cite>，其中我们将上投影矩阵拆分为两个矩阵（$W_\text{In1}$ 和 $W_\text{In2}$ 上方所示），它们的输出通过逐元素相乘作为一种“门控函数”。并非所有大语言模型都使用这种方式，因此有时你会看到一个单独的 $W_\text{In}$ 矩阵，以及总 MLP 参数数量为 2DF 而不是 3DF。通常在这种情况下，D 和 F 会被放大以保持参数数量与三矩阵情况相同。话虽如此，LLaMA、DeepSeek 和许多其他模型都使用了某种形式的 gating einsum。

**注意 2 [MHA 注意力]**: 使用自注意力时，T 和 S 是相同的，但在交叉注意力中它们可能不同。使用标准的多头注意力（MHA）时，N 和 K 是相同的，而在[多查询注意力](https://arxiv.org/abs/1911.02150)（MQA）<d-cite key="mqa"></d-cite> 中 K=1，而在[分组 MQA](https://arxiv.org/abs/2305.13245)（GMQA）<d-cite key="gmqa"></d-cite> 中，K 只需要能整除 N。

**注意 3 [预归一化与后归一化]:** 上述图表展示的是所谓的“预归一化”架构，其中归一化操作发生在残差连接之前，通常如 `x + attn(norm(x))` 所示。目前像 LLaMA-3 这样的模型使用的就是这种架构。原始的 Transformer 论文使用的是“后归一化”架构，其中层归一化操作发生在残差连接之后，即 `norm(x + attn(x))`。

## 全局 FLOPs 和参数计算

让我们计算 Transformer 的每层 FLOPs（这样我们就不必在每个地方都写上 **L** 的因子）。请注意，下面的训练 FLOPs 几乎总是推理 FLOPs 的 3 倍，因此你可以将任何总数除以 3，以得到仅前向传递的成本。

### MLPs

Transformer 的 MLP 通常由 2 个输入矩阵乘法组成，它们按元素组合，并接一个输出矩阵乘法：

$$
\begin{array}{ccc}
\textrm{operation} & \textrm{train FLOPs} & \textrm{params} \\
\hline \\
A[B,T,\red{D}] \cdot W_{in1}[\red{D}, F] & 6BTDF & DF \\[10pt]
A[B,T,\red{D}] \cdot W_{in2}[\red{D}, F] & 6BTDF & DF \\[10pt]
\sigma\left(A_{in1}\right)[B,T, F] * A_{in2}[B,T, F] & \gray{O(BTF)} \\[10pt]
A[B,T,\red{F}] \cdot W_{out}[\red{F}, D] & 6BTDF & DF \\[10pt]
\hline \\
& \approx 18BTDF & 3DF
\end{array}
$$

### 注意

对于具有不同 **Q** 和 **KV** 头数的通用分组查询注意力情况，假设 **Q**、**K**、**V** 投影的头维度 H 相等，并估计 **QKVO** 矩阵乘法的成本：

$$
\begin{array}{ccc}
\textrm{operation} & \textrm{train FLOPs} & \textrm{params} \\
\hline \\
A[B,T,\red{D}] \cdot W_{Q}[\red{D}, N, H] & 6BTDNH & DNH \\[10pt]
A[B,T,\red{D}] \cdot W_{K}[\red{D}, K, H] & 6BTDKH & DKH \\[10pt]
A[B,T,\red{D}] \cdot W_{V}[\red{D}, K, H] & 6BTDKH & DKH \\[10pt]
A[B,T,\red{N}, \red{H}] \cdot W_{O}[\red{N}, \red{H}, D] & 6BTDNH & DNH \\[10pt]
\hline \\ & 12BTD(N+K)H & 2D(N+K)H
\end{array}
$$

点积注意力操作更为微妙，实际上是将 $$TH \cdot HS$$ 矩阵乘法在 $$B$$、$$K$$ 维度上进行批处理，接着是一个 softmax 操作，然后再将 $$TS \cdot SH$$ 矩阵乘法在 $$B$$、$$K$$ 维度上进行批处理。我们用蓝色突出显示批处理的维度：

$$
\begin{array}{cc}
\textrm{operation} & \textrm{train FLOPs} \\
\hline \\[3pt]
Q[\blue{B}, T, \blue{K}, G, \red{H}] \cdot K[\blue{B}, S, \blue{K}, \red{H}]
& 6BTSKGH = 6BTSNH  \\[3pt]
\textrm{softmax}_S \;\; L[B, T, S, K, G] & \gray{O(BTSKG) = O(BTSN)} \\[3pt]
S[\blue{B}, T, \red{S}, \blue{K}, G] \cdot V[\blue{B}, \red{S}, \blue{K}, H]
& 6BTSKGH = 6BTSNH \\[3pt]
\hline \\
& \approx 12BTSNH = 12BT^2NH \\
\end{array}
$$

**注意 [因果掩码]**：大多数最新的 Transformer 使用因果掩码，而不是全双向注意力。在这种情况下，点积操作的有用 FLOPs 会减少一半。为了在实践中实现这一减少，我们需要使用一个注意力内核，而不是简单的 einsum。

### 其他操作

Transformer 中还有其他几种操作。层归一化（Layernorms）相对便宜，可以在第一阶成本估计中忽略。请注意，每一层通常有两个层归一化（一个在注意力机制之前，一个在 MLP 之前）。还有一个最终的巨大（尽管不是每层都有的）反嵌入矩阵乘法。

$$
\begin{array}{ccc}
\textsf{operation} & \textsf{train FLOPs} & \textsf{params} \\
\hline \\
2 \times \textrm{layernorm}_D \;\; A[B,T,\red{D}] & \gray{O\left(BTD\right)} & \gray{2D} \\[10pt]
A[B,T,\red{D}] \cdot W_{unembed}[\red{D}, V] & 6BTDV & DV \\
\end{array}
$$

### Transformer FLOPs 的一般经验法则

如果我们忽略点积注意力的计算成本（这对于短上下文训练是合理的），那么所有层的总 FLOPs 为

$$
\begin{align*}
(18BTDF + 12BTD(N+K)H)L = 6 *BT * (3DF + 2D(N+K)H)L \\ = 6 * \textrm{num tokens} * \textrm{parameter count}
\end{align*}
$$

这导致了一个著名的经验法则，用于估算密集 Transformer 的 FLOP 数量，忽略注意力的 FLOPs。（反嵌入是另一个简单的矩阵乘法，具有 $6BTDV$ FLOPs 和 $DV$ 参数，并遵循相同的规则。）

### 上下文长度的注意力分数成本

如果我们考虑上述的点积注意力并假设 $$F=4D$$、$$D=NH$$（这是常见的）和 $$N=K$$，点积注意力 FLOPs 与所有矩阵乘法 FLOPs（包括注意力投影）的比值是：

$$\small{\frac{\textrm{attention FLOPs}}{\textrm{matmul FLOPs}} = \frac{12BT^2NH}{18BTDF + 24BTDNH} = \frac{12BT^2D}{4*18 BTD^2 + 24 BTD^2} = \frac{12BT^2D}{96 BTD^2} = \frac{T}{8D}}$$

结果是，**点积注意力的 FLOPs 只有在 T>8D 时才会在训练过程中占据主导地位**。对于 D ~ 8k 的情况，这大约是 64K 个词元。这在一定程度上是有道理的，因为这意味着随着 MLP 规模的增加，注意力的 FLOPs 变得不那么关键。对于大模型而言，注意力的二次成本实际上并不是长上下文训练的严重障碍。然而，对于较小的模型，例如 D=4608 的 Gemma-27B，注意力在约 37k 的序列长度时变得占主导地位。<d-footnote>请注意，一些现代开源模型引入了局部注意力或其他优化方法，这些方法降低了注意力的成本并改变了这一上限。</d-footnote> Flash Attention 也有助于缓解长上下文的成本，我们将在[附录 A](#appendix-a-how-does-flash-attention-work)中简要讨论。

## 其他数学

### 稀疏性与混合专家

我们如果不简要讨论专家混合（MoE）模型<d-cite key="moe"></d-cite>，那就显得疏忽了。这类模型用一组独立的MLP取代标准Transformer中每个层的单一密集MLP块，这些MLP可以动态地进行路由。粗略地说，**一个MoE模型只是每个层包含E个MLP块的普通密集模型**，而不是仅仅一个。每个词元通常激活$k \ll E$个专家中的$k$个。该比例$E / k$称为稀疏度，通常在8到64之间（例如，[DeepSeek v3](https://arxiv.org/pdf/2412.19437)实际上具有$k=8$和$E=256$）。这使参数量增加了$O(E)$，同时将每个词元激活的参数总数乘以$k$，与密集版本相比。

{% include figure.liquid path="assets/img/moe.png" class="img-fluid img-small" caption="<b>图：</b> 一个包含 $n$ 个专家的示例 MoE 层。门控专家将每个词元路由到 $k$ 个专家中，这些 $k$ 个 MLP 的输出被相加。我们的参数量是每个专家大小的 $n$ 倍，但每个词元仅使用 $k$ 个专家。<a href=\"https://deepgram.com/learn/mixture-of-experts-ml-model-guide\">来源</a>。" %}

与密集模型相比，MoE 引入了新的通信操作，主要是两个 AllToAll（一个在 MoE 模块之前，一个在之后），用于将词元路由到正确的专家，并将其带回其所属设备。<d-footnote> 从技术上讲，这只有在我们沿与专家相同轴进行数据或序列分片时才会发生。</d-footnote> 然而，正如我们在上一节中看到的，每个 AllToAll 的成本仅为沿单个轴进行的等效 AllGather 的 1/4（对于双向环）。

### 梯度检查点机制

反向传播作为一种算法，以算力换取内存。与其说反向传播需要 $$O(n_\text{layers}^2)$$ FLOPs 的反向传递，**不如说它需要 $$O(n_\text{layers})$$ 的内存**，保存前向传递过程中生成的所有中间激活值。虽然这比二次方的算力要好，但从内存角度来看却极其昂贵：一个具有 $$B * T=4M$$（每批 4M 总词元）、L=64 和 D=8192 的模型，如果避免所有不必要的反向传递计算，就需要保存大约 $$2 * 20 * B * T * D * L = 84TB$$ 的激活值（使用 bfloat16）。这个 20 是大致统计了上面 Transformer 图中每个中间节点的数量，例如

$$f(x) = \exp(g(x))$$

$$\frac{df}{dx} = \exp(g(x)) \cdot \frac{dg}{dx}$$

因此，为了避免重复计算，我们需要从正向传播中保存 $$g(x)$$ 和 $$\exp(g(x))$$。为了避免占用过多内存，我们可以选择只保存一部分中间激活值。以下是一些我们使用的方法。

* **Block remat**：仅保存每层的输入。这是我们使用的一种最激进的方法，每层仅保存一个检查点，这意味着在上面的例子中我们只保存 4.2TB。这迫使我们在反向传播中重复几乎所有前向传播的 FLOPs，这意味着我们的 FLOPs 从 $$6 \cdot \text{num params} \cdot \text{num tokens}$$ 增加到大约 $$8 \cdot \text{num params} \cdot \text{num tokens}$$。  
* **Big matmuls only**：另一种简单的策略是仅保存大型矩阵乘法的输出。这使我们能够在反向传播中避免重新计算任何大型矩阵乘法，但仍需要重新计算其他激活函数和注意力的部分。这将每层上面的 20 降低到每层约 7。

这绝非全面。使用 JAX 时，这些通常由 `jax.remat`/`jax.checkpoint` 控制（你可以在 [此处](https://jax.readthedocs.io/en/latest/_autosummary/jax.checkpoint.html) 了解更多）。

### 键值（KV）缓存

如我们在[第7章](../inference)中将看到，大语言模型推理包含两个关键部分：预填充和生成。

* **预填充**处理一个长提示，并将其注意力激活值保存在键值缓存（KV 缓存）中，供生成阶段使用，具体为注意力块中的键值投影。
* **生成**将多个这些 KV 缓存进行批处理，并从每个缓存中采样词元。

每个 KV 缓存实际上是一个大小为 $[2, S, L, K, H]$ 的数组，其中 2 代表键和值。这个规模相当大！使用 int8 表示的 Key-Value 缓存总大小为 $2SLKH$。对于一个中等规模的模型，其上下文长度为 8k，有 64 层，且 $KH = NH = D = 8192$，这将是 $2 \cdot 8192 \cdot 64 \cdot 8192 = 8\text{GiB}$。你可以理解为什么我们希望使用 GMQA 且 $K \ll N$。

## 你应从本节中记住什么？

* Transformer 的总体参数和 FLOPs 相对容易计算，这里进行了总结，假设使用 MHA（批量大小为 B，词汇量为 V，序列长度为 T，D=d<sub>model</sub>，F=d<sub>ff</sub>）：


<!-- $$
\begin{array}{ccc}
\textrm{组件} & \textrm{每层参数量} & \textrm{每层训练 FLOPs} \\
\hline \\
\textbf{MLP} & 3DF & 18BTDF \\[10pt]
\textbf{Attention} & 4DNH & 24BTDNH + 12BT^2NH \\[10pt]
\textbf{Other} & D & BTD \\[10pt]
\textbf{Vocab} & DB \text{（总参数量，非每层）} & 12BTDV \\[10pt]
\end{array}
$$ -->


| Component     | Params per layer          | Training FLOPs per layer      |
| :------------ | :------------------------ | :---------------------------- |
| **MLP**       | 3DF                       | 18BTDF                        |
| **Attention** | 4DNH                      | 24BTDNH \+ 12BT<sup>2</sup>NH |
| **Other**     | 2D                        | BTD                           |
| **Vocab**     | DV (total, not per-layer) | 12BTDV                        |

* MLP 模块的参数量占总参数量的主导地位，只要序列长度 $T < 8D$，MLP 模块也主导了 FLOPs 预算。
* 在合理的上下文长度下，训练期间的总算力预算可由 $$6 \cdot \text{num_params} \cdot \text{num_tokens}$$ 很好地近似。
* 在推理期间，我们的 KV 缓存大约为 $$2 \cdot S \cdot L \cdot K \cdot H$$ 每个缓存（其中 K 是 KV 头的数量），尽管架构修改通常可以降低这一数值。

## 一些需要解决的问题

**问题 1:** 具备 $D=4096$、$F=4 \cdot D$、$V=32,000$ 和 $L=64$ 的模型有多少个参数？这些参数中有多少比例是注意力参数？每个词元的 KV 缓存有多大？*你可以假设 $N\cdot H=D$ 和使用 int8 KVs 的多头注意力。*

{% details 点击此处查看答案。 %}

1. 总参数量约为 $$L \cdot (3DF + 4DNH + 2D) + 2DV$$（每层计算两个层归一化）。对于给定的数值，这相当于 $$64 \cdot (3 \cdot 4e3 \cdot 16e3 + 4 \cdot 4e3 \cdot 4e3 + 2 \cdot 4e3) + 2 \cdot 4e3 \cdot 32e3 = 16e9$$，或 16B 参数。  
2. 一般来说，注意力参数与总参数的比例为 $$4DNH / (4DNH + 3DF) = 4D^2 / (4D^2 + 12D^2) = 1/4$$。这意味着大约 1/4 的参数用于注意力。  
3. 每个词元，我们的 KV 缓存以 int8 格式为 $$2 \cdot L \cdot N \cdot H = 2 \cdot 64 \cdot 4096$$，即 `512 KiB / token`。

{% enddetails %}

**Question 2:** How many total FLOPs are required to perform A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F] on `{'X': 4, 'Y': 8, 'Z': 4}`? How many FLOPs are performed by each TPU?

{% details 点击此处查看答案。 %}

该操作的总“理论”FLOPs为$$2 \cdot B \cdot D \cdot F$$。然而，由于计算没有在Z维度上进行分片，我们实际上执行了Z额外的FLOPs，这意味着总FLOPs为$$2 \cdot B \cdot D \cdot F \cdot Z$$。由于计算在其他维度上进行了分片，每台设备的总FLOPs大致为$$2 \cdot B \cdot D \cdot F / (X \cdot  Y)$$。

{% enddetails %}

**问题 3:** 执行 $A[I,J,K,L] * B[I,J,M,N,O] \rightarrow C[K,L,M,N,O]$ 涉及多少 FLOPs？

{% details 点击此处查看答案。 %}

按照上述规则，I 和 J 是收缩维度，而 K、L、M、N 和 O 是非收缩维度。我们没有“批处理维度”，因此这仅仅是 $$2 \cdot I \cdot J \cdot K \cdot L \cdot M \cdot N \cdot O$$，即所有轴的乘积。如果我们有一个共享轴，它只会被计算一次。

{% enddetails %}

**问题 4:** 分组多查询注意力的算术强度是多少（忽略 Q/K/V/O 投影）？*将答案表示为 Q 和 KV 序列长度 T 和 S 以及多查询因子 G 的函数。* 在什么上下文长度下，注意力的 FLOPs 会成为瓶颈？给定我们 TPU 的 HBM 带宽，绘制随着上下文长度增长，注意力相对于 FFW 模块的有效相对成本。*提示：假设我们使用的是一个高效的注意力实现，不会进行任何不必要的读写操作。考虑 T = S 和 T << S 两种极限情况。*

{% details 点击此处查看答案。 %}

自注意力需要加载 $$Q$$、$$K$$ 和 $$V$$ 激活值，然后计算 $$\text{softmax}(Q \cdot K) \cdot V$$，再将结果写回 HBM。这将使用 Flash Attention 来完成，因此这个数学计算有一些需要注意的地方，但基本上在 bfloat16 中自注意力的性能是

$$\text{Q[B,T,N,H]} \rightarrow_\text{reshape} \text{Q[B, T, K, G, H]} \cdot \text{K[B, S, K, H]} \rightarrow \text{O[B, T, S, K, G]}$$

$$U=\text{softmax}_S(\text{O[B, T, S, K, G]})$$

$$\text{U[B, T, S, K, G]} \cdot \text{V[B, S, K, H]} \rightarrow \text{X[B, T, K, G, H]}$$

因此，我们的总字节数为 $$2 * \text{sizeof}(Q) + 2 * \text{sizeof(K or V)} = 4BTNH + 4BSKH = 4BHK * (TG + S)$$，总 FLOPs 为 $$4BTSNH + O(BTSN)$$，算术强度为 $$4BTSKGH / (4BHK * (TG + S))$$。

因此，基本上在预填充阶段，我们有 $$S=T$$，因此算术强度为 $$4BT^2KGH / 4BHKT \cdot (G+1) = TG/(G + 1) = O(T)$$。在生成阶段，$$T=1$$，因此我们有 $$4BSKGH / (4BHK \cdot (G + S)) = SG / (G + S) \rightarrow G$$，假设 $$S$$ 非常大。根据你对问题的理解，在预填充或训练期间，假设没有序列分片，自注意力在 S=240 时是算力受限的。在生成阶段，我们永远不会算力受限，因为 $$G$$ 很小。尽管如此，你可以看到增加 $$G$$ 会使我们更接近算力受限。

{% enddetails %}

**问题 5:** 在什么序列长度下，自注意力 FLOPs 等于 QKVO 投影 FLOPs？

{% details 点击此处查看答案。 %}

这纯粹是关于何时 $$24BTDNH = 12BT^2NH$$ 的问题。简化后我们得到 $$2D = T$$，因此例如对于 $$D=4096$$，这为 $$8192$$。这告诉我们，对于大多数合理的上下文长度，矩阵乘法 FLOPs 更大。

{% enddetails %}

**问题 6:** 假设在我们的前向传递过程中，我们只保存 Transformer 层中每个的 7 个主要矩阵乘法（Q, K, V, O + 三个 FFW 矩阵）的输出。在反向传递过程中，我们需要额外多少 FLOPs 来“重新计算”这些值？

{% details 点击此处查看答案。 %}

仅保存七个矩阵乘法输出（Q, K, V, O, W₁, W₂, W₃）意味着反向传播必须重新计算两个注意力矩阵乘法

$$QK^{\top} \quad\text{and}\quad \operatorname{softmax}(QK^{\top})V$$

为了获得 $\frac{\partial L}{\partial W_\text{O}}$。

每个都是对 $B$ 个序列和 $N$ 个头进行批处理的 $T \times T$ 矩阵乘法，因此额外的 FLOPs 是

$$4 \; B \, T^{2} \, N \, H.$$

其他需要重新计算的操作包括：
1. $O(BTD)$ 用于 $\frac{\partial L}{\partial W_\text{In1}}$ 和 $\frac{\partial L}{\partial W_\text{In2}}$。
2. 以及 $O(BTF)$ 用于 $\frac{\partial L}{\partial W_\text{Out}}$。

{% enddetails %}

**问题 7:** DeepSeek v3 表示它在 14.8T 词元上训练了 2.79M H800 小时 ([来源](https://arxiv.org/pdf/2412.19437v1))。考虑到它有 37B 激活参数，他们大致实现了多少硬件利用率？*提示：请注意他们使用了 FP8 FLOPs 而没有结构化稀疏性。*

{% details 点击此处查看答案。 %}

从规格表[这里](https://lenovopress.lenovo.com/lp1814.pdf)可知，启用稀疏性的 FP8 性能为 3,026 TFLOPs/s，而未启用稀疏性时通常为一半（`1.513e15` FLOPs/s）。2.79M H800 小时意味着 `2.79e6 * 1.513e15 * 60 * 60 = 1.52e25` 总 FLOPs。考虑到激活的参数量为 37B，这次训练运行应使用了约 `6 * 37e9 * 14.8e12 = 3.3e24` FLOPs。这意味着 FLOPs 利用率约为 `3.3e24 / 1.52e25 = 21.7%`。

{% enddetails %}

**问题 8:** 混合专家（MoE）模型有 $E$ 个标准密集 MLP 模块的副本，每个词元会激活 $k$ 个专家。在 TPU v5e 上使用 int8 权重的 MoE 模型，需要多大的词元批量大小才能达到算力限制？对于 DeepSeek，它有 256（路由）专家和 $k=8$，这个数值是多少？

{% details 点击此处查看答案。 %}

由于我们有 $E$ 个每个专家的 int8 拷贝，对于每个权重矩阵，我们需要加载 $E \cdot D \cdot F$ 字节。由于每个词元激活 $k$ 个专家，对于每个权重矩阵，我们有 $2\cdot k \cdot B \cdot D \cdot F$ FLOPs。为了在使用 int8 权重和 bfloat16 FLOPs 时达到算力限制，算术强度（每加载字节的 FLOPs）需要超过 TPU 的 ~240 FLOPs/byte，这发生在 $(2\cdot k \cdot BDF) / EDF > 240$ 或 $k \cdot B / E > 120$ 时。

因此，我们需要 $B > 120 \cdot E / k$ 是算力受限的。对于 DeepSeek 来说，这给了我们 $B > 120 \cdot 256 / 8 = 3840$。这在生成阶段是一个非常大的批量大小。

{% enddetails %}

<h3 markdown=1 class="next-section">第4部分就到这里！第5部分（关于扩展Transformer训练），[点击此处](../training)！</h3>

## 附录

### 附录 A：Flash Attention 是如何工作的？

传统上，反对将 Transformer 缩放至非常长上下文的论点是，注意力 FLOPs 和内存使用量会随着上下文长度呈二次方增长。虽然确实注意力 QK 乘积的形状为 $[B, T, S, N]$，其中 B 是批量大小，T 和 S 是 Q 和 K 序列的维度，N 是头的数量，但这一说法伴随着一些严重的限制：

1. 正如我们之前所指出的，尽管这是二次方的，但注意力 FLOPs 仅在 $$T > 8 \cdot D$$ 时才占主导地位，而且在训练期间，单个注意力矩阵的内存占用与内存中所有权重和激活检查点相比非常小，尤其是在分片的情况下。
2. 为了计算注意力，我们不需要显式生成完整的注意力矩阵！我们可以计算局部的和与最大值，并且避免显式生成数组中超过一小块的部分。虽然总的 FLOPs 仍然是二次方的，但我们显著降低了内存压力。

这一第二个观察结果最早由 [Rabe 等人 2021](https://arxiv.org/abs/2112.05682) 提出，后来在 [Flash Attention 论文](https://arxiv.org/abs/2205.14135)（Dao 等人 2022）中再次提到。基本思路是按 K/V 的块来计算注意力，在其中计算局部的 softmax 和一些辅助统计信息，然后将它们传递给下一个块，由该块将其与自身的局部块进行合并。具体来说，我们计算

1. **M:** 在序列维度上 $$q \cdot k$$ 的运行最大值  
2. **O:** 在序列维度上完整的注意力 softmax 的运行结果  
3. **L:** 运行的分母 $$\sum_i \exp(q \cdot k_i - \text{running max})$$

有了这些，我们仅用常量大小的内存就可以计算新的最大值、新的运行总和以及新的输出。为了大致描述这一过程是如何工作的，注意力大致对应于以下操作：

$$\text{Attn}(Q, K, V) = \sum_i \frac{\exp(Q \cdot K_i - \max_j Q \cdot K_j) V_i}{\sum_l \exp(Q \cdot K_l - \max_j Q \cdot K_j)}$$

为保证数值稳定性，最大值被减去，且由于 $$\sum_i \exp(a_i + b) = \exp(b) \sum \exp(a)$$ 的原因，减去最大值不会影响结果。仅看上面的分母，如果我们想象有两个连续的关键向量块 $$K^1$$ 和 $$K^2$$，并为每个块计算局部 softmax 和 $$L^1$$ 和 $$L^2$$。

$$L^1 = \sum_i \exp(Q \cdot K_i^1 - \max_j Q \cdot K_j^1)$$

$$L^2 = \sum_i \exp(Q \cdot K_i^2 - \max_j Q \cdot K_j^2)$$

然后我们可以通过使用这些内容，将这两个块的完整 Softmax 求和组合在一起。

$$L^\text{combined} = \exp(M^1 - \max(M^1, M^2)) \cdot L^1 + \exp(M^2 - \max(M^1, M^2)) \cdot L^2$$

其中

$$M^1 = \max_j Q \cdot K_j^1 \text{ and } M^2 = \max_j Q \cdot K_j^2$$

这也可以应用于完整的 softmax，为我们提供了一种累积任意大 softmax 和的方法。以下是 Flash Attention 论文中的完整算法。

{% include figure.liquid path="assets/img/flash-algo.png" class="img-fluid" %}

从硬件角度来看，这使我们能够将 Q 的一块数据放入 VMEM（上述算法中称为片上 SRAM），这样我们只需在每次迭代时加载 KV 块，从而提高算术强度。我们还可以将运行时统计信息保留在 VMEM 中。

还有一个值得强调的细微之处是注意力 softmax 的一个性质，该性质用于使 Flash VJP（反向模式导数）计算在训练中变得可行。我们定义了一个中间的 softmax 数组：

$$S_{ij} = \frac{e^{\tau q_i \cdot k_j}}{\sum_l e^{\tau q_i \cdot k_l}}$$

在注意力机制中，我们从反向模式的 *dO* 和 *V* 数组中获得 *dS*：

$$dS_{ij} = dO_{id} \cdot_d V_{jd} = \sum_d dO_{id} V_{jd}$$

在将此梯度反向传播到 Q 和 K 时

$$d(q_i \cdot k_j) = (dS_{ij} - S_{ij} \cdot_j dS_{ij}) S_{ij}$$

我们利用了一个恒等式，使我们能够将沿大键 **长度** 维度的收缩转换为沿特征 **深度** 维度的局部收缩。

$$\begin{align*}
S_{ij} \cdot_j dS_{ij} &= \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} \sum_d dO_{id} V_{jd} \\
&= \sum_d dO_{id} \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} V_{jd} \\
&= \sum_d dO_{id} O_{id} \\
&= dO_{id} \cdot_d O_{id}
\end{align*}$$

这种替换对于实现 VJP 的序列块 *局部* 计算至关重要，并启用了更聪明的分片方案，如环形注意力。
