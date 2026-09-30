---
layout: distill
title: "如何对 Transformer 进行训练并行化"
# permalink: /main/
description: "这里我们讨论在大语言模型训练过程中使用的四种主要并行方案：数据并行、完全分片数据并行（FSDP）、张量并行和流水线并行。对于每种方案，我们计算在什么情况下会因通信而成为瓶颈。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 5

previous_section_url: "../transformers"
previous_section_name: "Part 4: Transformers"

next_section_url: ../applied-training
next_section_name: "Part 6: Training LLaMA"

bibliography: main.bib

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

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "What Do We Mean By Scaling?"
  - subsections:
    - name: "Data Parallelism"
    - name: "Fully-Sharded Data Parallelism (FSDP)"
    - name: "Tensor Parallelism"
    - name: "Combining FSDP and Tensor Parallelism"
    - name: "Pipelining"
    - name: "Scaling Across Pods"
  - name: "Takeaways from LLM Training on TPUs"
  - name: "Some Problems to Work"
  - name: "Appendix"
  - subsections:
    - name: "Appendix A: Deriving the backward pass comms"

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

## 我们所说的缩放是指什么？

“模型缩放”的目标是能够在增加用于训练或推理的芯片数量的同时，实现吞吐量的成比例、线性增长（我们称之为*强缩放*）。虽然单个芯片上的性能取决于显存带宽与FLOPs之间的权衡，但集群层面的性能则取决于通过与有用的FLOPs重叠来隐藏芯片间的通信。这并非易事，因为增加芯片数量会增加通信负载，同时减少可用于隐藏通信的每设备计算量。正如我们在[第3节](../sharding)中看到的，分片矩阵乘法通常需要代价高昂的AllGathers或ReduceScatters，这些操作可能会阻止TPUs执行有用的工作。本节的目标是找出这些操作何时会变得*过于昂贵*。

在本节中，我们将讨论四种常见的并行方案：（纯）**数据并行**、**全分片数据并行**（FSDP / ZeRO 分片）、**张量并行**（也称为模型并行）以及（简要地）**流水线并行**。对于每种方案，我们将展示我们产生的通信成本以及该成本何时开始成为算力成本的瓶颈。<d-footnote>我们将关注通信限制——虽然内存容量限制很重要，但当我们使用重计算（激活检查点）和预训练时使用大量芯片时，这些限制通常不会限制我们。我们在这里也不讨论 MoE 的专家并行——这会显著扩展设计空间，我们只讨论密集 Transformer 的基本情况。</d-footnote>对于本节，你可以专注于芯片间的通信成本，因为只要我们有足够大的单芯片批量大小，从 HBM 到 MXU 的数据传输已经与计算重叠。

我们将使用以下符号来简化本节中的所有计算。

|符号表示|含义（模型参数）|
| :------- | :--------------------------------------------------------------------- ||D|**d**<sub>model</sub>（隐藏维度/残差流维度）|
|F|**d**<sub>ff</sub>（前馈维度）|
|B|批量维度（批量中的词元数量；总数量，而非每设备数量）|
|T|序列长度|
|L|模型中的层数|

|符号表示|含义（硬件特性）|
| :------- | :------------------------------------------------------------------------------------------------ ||C|每芯片 FLOPS/s|
|W|网络带宽（双向，通常以 $W_{\text{ici}}$ 或 $W_{\text{dcn}}$ 等形式作为下标）|
|X|沿网格 X 轴的芯片数量|
|Y|沿另一条网格轴的芯片数量，标记为 Y|
|Z|沿第三个网格轴的芯片数量，标记为 Z|

为了简化起见，**我们将 Transformer 近似为一组 MLP 模块的堆叠** —— 正如我们在[第 4 章](../transformers)中看到的那样，对于更大的模型，注意力操作所占的 FLOPs 比例相对较小。我们还将忽略门控矩阵乘法，这样每层的结构就简化为如下形式：

{% include figure.liquid path="assets/img/transformer-layer.png" class="img-fluid" caption="<b>图：</b> 一个简化的 Transformer 层。我们将每个 FFW 模块视为两个矩阵的堆叠 <b>W<sub>in</sub></b>：<code>bf16[D, F]</code>（上投影）和 <b>W<sub>out</sub></b>：<code>bf16[F, D]</code>（下投影），输入为 <b>In</b>：<code>bf16[B, D]</code>。" %}

{% details 这是我们没有并行性的小小 Transformer 的完整算法。 %}

<div markdown=1 class="algorithm">

**前向传播：** 需要计算 Loss[B]

1. Tmp[B, F] = In[B, D] *<sub>D</sub> W<sub>in</sub>[D, F]  
2. Out[B, D] = Tmp[B, F] *<sub>F</sub> W<sub>out</sub>[F, D]  
3. Loss[B] = ...

**反向传播：** 需要计算 dW<sub>out</sub>[F, D], dW<sub>in</sub>[D, F]

1.  dOut[B, D] = ...  
2.  dW<sub>out</sub>[F, D] = Tmp[B, F] *<sub>B</sub> dOut[B, D]  
3.  dTmp[B, F] = dOut[B, D] *<sub>D</sub> W<sub>out</sub>[F, D]  
4.  dW<sub>in</sub>[D, F] = In[B, D] *<sub>B</sub> dTmp[B, F]  
5.  dIn[B, D] = dTmp[B, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*供前一层使用*)

</div>

我们提供此内容，以便与添加了通信的算法进行比较。

{% enddetails %}

这里是我们将讨论的 4 种并行方案。每种方案都可以看作是由上述图表中 **In**、**W<sub>in</sub>**、**W<sub>out</sub>** 和 **Out** 的分片唯一定义的。

**1. 数据并行：** *沿批量划分激活值，参数和优化器状态在每个设备上复制。通信仅在反向传播过程中发生。*

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

**2. 全分片数据并行（FSDP 或 ZeRO-3）：** 激活值沿批量维度分片（类似于纯数据并行），参数沿同一网格轴分片，并在前向传递中使用前即时通过 AllGather 汇聚。优化器状态也沿批量维度分片。减少了内存重复。

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$$

**3. 张量并行（也称为 Megatron 分片或模型并行）：** 激活值沿 D 维度分片（$d_\text{model}$），参数沿 F 维度分片（$d_{ff}$）。每个模块前后对激活值进行 AllGather 和 ReduceScatter 操作。与 FSDP 兼容。

$$\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$$

**4. 流水线并行：** *沿层维度对权重进行分片，激活值微批量处理并在层维度上滚动。流水线阶段之间的通信极少（仅需将激活值传递一次）。为了便于表述：*

$$\text{In}[L_Z, B, D][i] \cdot_D W_\text{in}[L_Z, D, F][i] \cdot_F W_\text{out}[L_Z, F, D][i] \rightarrow \text{Out}[L_Z, B, D][i]$$

### 数据并行

**语法:** $$\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

当你的模型可以单芯片运行，即使使用非常小的批量大小（>240 词元，从而达到算力限制）时，**你应该始终使用简单的数据并行。** 纯数据并行可以将我们的激活值分配到任意数量的 TPU 上，只要 TPU 的数量小于批量大小。前向传播过程中不需要通信，但每一步结束时，**每个 TPU 会对本地梯度执行 AllReduce 操作，以在更新参数前同步这些梯度。**

{% include figure.liquid path="assets/img/data-parallelism.png" class="img-fluid" caption="<b>图：</b> 纯数据并行（前向传播）的示意图。我们的激活值（左）沿批量维度被完全分片，而权重被完全复制，因此每个TPU都有权重的完整副本。这意味着我们的权重总内存增加了N倍，但前向传播不需要通信。" %}

{% details 这里是前向和反向传播的完整算法。我们为了简洁性，滥用符号将 dL/dOut 写作 dOut。 %}

<div markdown=1 class="algorithm">

**纯数据并行算法：**

**前向传播：** 需要计算 Loss[B<sub>X</sub>]

1. Tmp[B<sub>X</sub>, F] = In[B<sub>X</sub>, D] *<sub>D</sub> W<sub>in</sub>[D, F]  
2. Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F] *<sub>F</sub> W<sub>out</sub>[F, D]  
3. Loss[B<sub>X</sub>] = ...

**反向传播：** 需要计算 dW<sub>out</sub>[F, D], dW<sub>in</sub>[D, F]

1.  dOut[B<sub>X</sub>, D] = ...  
2.  dW<sub>out</sub>[F, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F] \*<sub>B</sub> dOut[B<sub>X</sub>, D]  
3.  dW<sub>out</sub>[F, D] = **AllReduce**(dW<sub>out</sub>[F, D] {U<sub>X</sub>}) (*不在关键路径上，可异步完成*)  
4.  dTmp[B<sub>X</sub>, F] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F, D]  
5.  dW<sub>in</sub>[D, F] {U<sub>X</sub>} = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F]  
6.  dW<sub>in</sub>[D, F] = **AllReduce**(dW<sub>in</sub>[D, F] {U<sub>X</sub>}) (*不在关键路径上，可异步完成*)  
7.  dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*供前一层使用*)

</div>

我们忽略损失函数的细节，并将 $\text{Tmp} = W_\text{in} \cdot \text{In}$ 简写。请注意，尽管我们的最终损失是平均 **AllReduce**(Loss[B<sub>X</sub>])，但我们在平均权重梯度时，只需在反向传播过程中计算 AllReduce。

{% enddetails %}

请注意，前向传播过程中没有通信 —— **所有通信都在反向传播中进行**！反向传播还有一个很好的特性，即 AllReduces 不在“关键路径”上，这意味着每个 AllReduce 可以在任何方便的时候执行，不会阻碍后续操作的进行。如果总的通信成本超过了我们的总算力成本，它仍然可能成为瓶颈，但从实现角度来看，它要宽容得多。我们将看到，模型/张量并行不具备这一特性。

**为什么要这样做？** 纯数据并行通过在批量维度上分割激活值，减少了激活内存压力，只要我们有更多的芯片来分割批量维度，就可以几乎任意地增加批量大小。特别是在训练期间，当我们的激活值通常主导内存使用时，这非常有帮助。

**为什么不做这件事呢？** 纯数据并行无法减少模型参数或优化器状态带来的内存压力，这意味着在大规模模型中，当我们的参数 + 优化器状态无法放入单个 TPU 时，纯数据并行几乎毫无用处。为了说明规模，如果我们用 bf16 训练参数，用 fp32 存储 Adam<d-footnote>Adam 的优化器状态（包括参数、一阶和二阶累加器），由于参数是 bfloat16，优化器状态是 float32，这会给我们每个参数带来 `2 + 8 = 10` 字节的内存占用。</d-footnote>，我们能容纳的最大模型有 $$\text{TPU memory} / 10$$ 个参数，例如在一块拥有 96GB HBM 的 TPUv5p 芯片上，使用纯数据并行时，大约可以容纳 90 亿个参数。

<p markdown=1 class="takeaway">**要点**：使用 Adam 和纯数据并行训练的模型规模最大为 $$\text{num_params} = \text{HBM per device} / 10$$。对于 TPU v5p 来说，这大约是 9B 参数。<d-footnote>请注意，这不包括梯度检查点，因此实际上并没有太大用处。这是在批量大小为 1 个词元时的绝对下限。</d-footnote></p>

*为了在训练期间使这在实际模型中可用，我们需要至少部分地对模型参数或优化器进行分片。*

**我们何时会因通信成为瓶颈？** 如上所述，我们在每一层都有两次 AllReduce，每次的规模为 $$2DF$$（针对 bf16 权重）。数据并行在何时会使我们因通信而受限？

如上表所示，令 $C$ = 每芯片 FLOPs，$W_{\text{ici}}$ = **双向** 网络带宽，$X$ = 批量被划分到的分片数量<d-footnote>。我们假设这种划分是在 ICI 网格上进行的，因此相关的网络带宽是 $W_\text{ici}$</d-footnote>。让我们计算执行相关矩阵乘法 $$T_\text{math}$$ 所需的时间，以及所需的通信时间 $$T_\text{comms}$$。由于这种并行方案在前向传递中不需要通信，我们只需为反向传递计算这些量。

*通信时间:* 从前一节我们知道，在 1D 网格中执行 AllReduce 所需的时间仅取决于被 AllReduce 的数组的总字节数和 ICI 带宽 $W_\text{ici}$；具体来说，AllReduce 时间是 $2 \cdot \text{total bytes} / W_\text{ici}$。由于我们需要对 $W_\text{in}$ 和 $W_\text{out}$ 都进行 AllReduce，因此每层有 2 次 AllReduce。每次 AllReduce 都是对一个权重矩阵，即一个包含 $DF$ 个参数的数组，或 $2DF$ 字节。将所有这些因素综合起来，单层中 AllReduce 的总时间是

$$\begin{align}
T_\text{comms} &= \frac{2 \cdot 2 \cdot 2 \cdot D \cdot F}{W_\text{ici}}. \\
\end{align}$$

*Matmul time:* 每一层在前向传递中包含两个矩阵乘法，或在反向传递中包含四个矩阵乘法，每个矩阵乘法需要 $2(B/X)DF$ FLOPs。因此，对于反向传递中的单一层，我们有

$$\begin{align}
T_\text{math} &= \frac{2 \cdot 2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
\end{align}$$

由于我们进行了重叠，每层的总时间是这两个量的最大值：

$$\begin{aligned}
T &\approx \max(\frac{8 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{8 \cdot D \cdot F}{W_\text{ici}}) \\
T &\approx 8 \cdot D \cdot F \cdot \max(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}})
\end{aligned}$$

当 $$T_\text{math}/T_\text{comms} > 1$$ 时，或当

$$\begin{align}
\frac{B}{X} > \frac{C}{W_\text{ici}}.
\end{align}$$

其结果是，为了在使用数据并行时保持算力受限，我们需要每个设备的批量大小 $$B / X$$ 超过 ICI 运算强度 $C / W_\text{ici}$。这最终是由于计算时间随着每个设备的批量大小而变化，而通信时间与此量无关（因为我们正在传输模型权重）。请注意 $B/X > C/W_\text{ici}$ 条件与单设备算力受限规则 $B > 240$ 的相似之处；在那种情况下，规则也来自于计算时间随批量大小变化，而数据传输大小（在 $B \ll F, D$ 区间）与批量大小无关这一事实。

让我们用一些实际的数字来感受一下规模。对于 TPUv5p，`C=4.6e14` 和 `W=2 * 9e10` 是在 ICI 上进行 1D 数据并行的，因此 **每块芯片的批量大小必须至少为 2,550，以避免成为通信瓶颈**。由于我们可以在多个轴上进行数据并行，如果我们把 TPUv5p 机架的三个轴全部用于纯粹的数据并行，我们可以将带宽提升 3 倍 $W_\text{ici}$，并将每块 TPU 的批量大小降低到仅 850，或每批次每机架（含 8960 块芯片）处理 7.6M 词元！**这告诉我们，纯粹的数据并行很难成为瓶颈！**

<p markdown=1 class="takeaway">**注意 [上下文并行]：** 在本节中，$B$ 始终指的是以 **词元** 为单位的总批量大小。显然，我们的批量由许多不同的序列组成，那么这该如何处理呢？就 MLP 而言，**词元就是词元**！它们是否属于同一序列或两个不同序列并不重要。因此，我们在批量和序列维度上几乎可以自由地进行数据并行：我们称之为上下文并行或序列并行，但你可以将其视为另一种形式的数据并行。与 MLP 相比，注意力机制更为复杂，因为我们需要进行跨序列计算，但可以通过在注意力过程中收集 KVs 或 Qs，并仔细重叠 FLOPs 和通信（通常使用一种称为“环形注意力”的方法）来处理。在本节中，我们将完全忽略序列维度，并假设存在一定程度的批量或序列并行。</p>

**关于多个网格轴的说明：** 我们应快速说明多个轴如何影响可用带宽。当我们对给定的并行策略使用多个网格轴时，可以获得更多的带宽。

* **定义:** $M_X$（$M_Y$、$M_Z$ 等）是指给定并行策略所跨越的硬件网格轴数。  
* **影响（带宽受限）:** 使用 $M$ 轴可提供（$\approx M$ 倍）的聚合链路带宽，因此集合通信时间按 $\propto 1/M_X$ 缩放。

### 全分片数据并行（FSDP）

**语法:** $$\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$$

全分片数据并行（通常称为 FSDP 或 ZeRO 分片 <d-cite key="zero"></d-cite>）将模型优化器状态和权重分布在数据并行分片中，并在需要时高效地进行收集和分发。**与纯数据并行相比，FSDP 显著减少了每台设备的内存使用，并节省了反向传播的 FLOPs，且开销非常小。**

{% include figure.liquid path="assets/img/fsdp.png" class="img-fluid" caption="<b>图：</b> FSDP 沿数据维度对 Win 的收缩维度和 Wout 的输出维度进行分片。这减少了内存使用，但（见第 3 节）要求我们在执行矩阵乘法之前先收集 W 的权重。请注意，激活值（左）<i>并未沿收缩维度进行分片</i>，这正是迫使我们进行收集的原因。<b>请注意，我们的权重优化器状态也沿收缩维度进行了分片。</b>" %}

你将记得（参见[第3节](../sharding)），AllReduce可以分解为AllGather和ReduceScatter。这意味着，我们不必为标准数据并行执行完整的梯度AllReduce，而可以在芯片之间分片权重和优化器状态，在前向传递过程中在每一层AllGather它们，并在反向传递过程中在权重之间ReduceScatter，而不会产生额外成本。

{% details 这是 FSDP 的完整算法。 %}

<div markdown=1 class="algorithm">

**全分片数据并行（FSDP）：**

**前向传播：** 需要计算 Loss[B<sub>X</sub>]

1.  W<sub>in</sub>[D, F] = **AllGather**(W<sub>in</sub>[D<sub>X</sub>, F]) (*不在关键路径上，可以在前一层进行*)
2.  Tmp[B<sub>X</sub>, F] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F] (*现在可以丢弃 W<sub>in</sub>[D, F]*)
3.  W<sub>out</sub>[F, D] = **AllGather**(W<sub>out</sub>[F, D<sub>X</sub>]) (*不在关键路径上，可以在前一层进行*)
4.  Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>out</sub>[F, D]
5.  Loss[B<sub>X</sub>] = ...

**反向传播：** 需要计算 dW<sub>out</sub>[F, D<sub>X</sub>], dW<sub>in</sub>[D<sub>X</sub>, F]

1.  dOut[B<sub>X</sub>, D] = ...  
2.  dW<sub>out</sub>[F, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F] \*<sub>B</sub> dOut[B<sub>X</sub>, D]  
3.  dW<sub>out</sub>[F, D<sub>X</sub>] = **ReduceScatter**(dW<sub>out</sub>[F, D] {U<sub>X</sub>}) (*不在关键路径上，可异步完成*)  
4.  W<sub>out</sub>[F, D] = **AllGather**(W<sub>out</sub>[F, D<sub>X</sub>]) (*可提前完成*)  
5.  dTmp[B<sub>X</sub>, F] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F, D] *(可在此处丢弃 W<sub>out</sub>[F, D])*  
6.  dW<sub>in</sub>[D,F] {U<sub>X</sub>} = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F]  
7.  dW<sub>in</sub>[D<sub>X</sub>, F] = **ReduceScatter**(dW<sub>in</sub>[D, F] {U<sub>X</sub>}) *(不在关键路径上，可异步完成)*  
8.  W<sub>in</sub>[D, F] = **AllGather**(W<sub>in</sub>[D<sub>X</sub>, F]) (*可提前完成*)  
9.  dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*供前一层使用) (可在此处丢弃 W<sub>in</sub>[D, F]*)

</div>

{% enddetails %}

这也被称为“ZeRO 分片”，源自“Zero Redundancy Optimizer”，因为我们不执行任何不必要的计算，也不存储任何不必要的状态。ZeRO-{1,2,3} 分别用于指代以这种方式对优化器状态、梯度和权重进行分片。由于所有操作的通信成本相同<d-footnote>从技术上讲，FSDP 在前向传递中会增加纯 DP 没有的通信，但这种增加与反向传递中的比例相同，因此对通信上限应没有影响。关键在于，ZeRO-3 将反向传递中的 AllReduce 转换为 AllGather 和 ReduceScatter，它们的总通信量相同。</d-footnote>，我们基本上可以始终使用 ZeRO-3 分片，它将参数、梯度和优化器状态在一组设备之间进行分片。

**为什么要这样做？** 标准的数据并行涉及大量重复的工作。每个 TPU 都会对完整的梯度执行 AllReduce，然后更新完整的优化器状态（所有 TPU 上的工作完全相同），然后更新参数（再次完全重复）。对于 ZeRO 分片（分片梯度/优化器状态），而不是使用 AllReduce，你可以使用 ReduceScatter 对梯度进行分片，仅更新你分片中的优化器状态，更新参数的一个分片，然后根据前向传递的需要 AllGather 参数。

**在什么情况下会受到通信的瓶颈限制？** 我们的相对 FLOPs 和通信成本与纯粹的数据并行完全相同，因为反向传播中的每个 AllReduce 都变成了 AllGather + ReduceScatter。请记住，AllReduce 是通过 AllGather 和 ReduceScatter 实现的，每种操作的成本都是一半。在这里我们建模前向传播，因为它与反向传播具有相同的 FLOPs-to-comms 比例：

$$\begin{aligned}
T_\text{math} &= \frac{2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
T_\text{comms} &= \frac{2 \cdot 2 \cdot D \cdot F}{W_\text{ici}} \\
T &\approx \max\left(\frac{4 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{4 \cdot D \cdot F}{W_\text{ici}}\right) \\
T &\approx 4 \cdot D \cdot F \cdot \max\left(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}}\right)
\end{aligned}$$

因此，与纯粹的数据并行一样，当 $$B / X > C / W_\text{ici}$$ 时，我们会受到算力的限制，即当每个设备的批量大小 $B/X$ 超过“ICI 操作强度” $C/W_\text{ici}$（v5p 为 `4.59e14 / 1.8e11 = 2550`）时。这对我们将是非常有利的，因为它意味着如果我们每个设备的批量大小足够大，以至于在纯粹的数据并行中受到算力限制，我们就可以——无需担心离开算力限制区间——直接升级到 FSDP，从而节省大量参数和优化器状态内存！虽然我们确实需要在前向传递中增加通信，但这一成本可以忽略不计，因为它仅与前向传递的 FLOPs 重叠。

<p markdown=1 class="takeaway">**要点：** 当每设备批量大小小于 $2550 / M_X$ 时，FSDP 和纯数据并行在 TPUv5 上会受到带宽限制，其中 $M_X$ 是网格轴的数量。</p>

例如，DeepSeek-V2（最近为数不多的公开其训练批量大小的强模型之一）使用了约 4000 万词元的批量大小。**这使我们能够在达到带宽限制之前，扩展到约 47,000 张芯片，或大约 5 个 TPUv5 机架。**

对于训练了大约 `6.3e24 (15e12 * 70e9 * 6)` FLOPs 的 LLaMA-3 70B，我们可以将一个包含 16M 词元的批次分配到大约 `16e6 / (2550 / 3) = 18,823` 张芯片（大约 2 个包含 8960 张芯片的机架）上，每张芯片具有 `4.59e14` FLOPs 的算力，并以 50% 的峰值 FLOPs 利用率（通常称为 MFU）运行，**并在大约 17 天内完成训练**。还不错！但让我们探讨一下如何做得更好。

<p markdown=1 class="takeaway">**关于临界批量大小的说明**：令人有些意外的是，随着总批量大小的减少（芯片数量固定），我们更容易受到通信瓶颈的限制。数据并行和FSDP使我们能够扩展到任意数量的芯片，只要我们能够不断增加批量大小！然而，在实践中，随着批量大小的增加，由于梯度几乎不再含有噪声，我们通常会看到训练收益的递减。我们有时还会看到训练的不稳定性。因此，在“无限制算力区间”中寻找最优分片方案的游戏，通常从一个由缩放定律确定的固定批量大小和一个已知（较大）的芯片数量开始，然后旨在找到一种分片方式，使我们能够在这么多芯片上容纳这个小批量大小。</p>

### 张量并行

**语法:** $$\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$$（我们使用 $$Y$$ 最终与 FSDP 结合）

在完全分片的数据并行 AllReduce 中，我们会在芯片之间移动权重。我们也可以对模型的前馈维度进行分片，并在层内移动激活值——这被称为“1D 模型并行”或 Megatron 分片<d-cite key="megatron"></d-cite>。这可以实现每个 pod 更小的高效批量大小。下图展示了一个以这种方式分片的单个矩阵的示例：

{% include figure.liquid path="assets/img/model-parallelism.png" class="img-fluid" caption="<b>Figure:</b> 展示了一个基本张量并行的示例。由于我们仅在 Y 维度上对激活进行分片（与 FSDP 在 X 维度上分片不同），我们在 X 维度上复制激活。使用我们的标准语法，这表示为 <b>A</b>[B, D<sub>Y</sub>] * <b>B</b>[D, F<sub>Y</sub>] -> <b>C</b>[B, F<sub>Y</sub>]。因为我们仅在一个收缩维度上进行分片，通常在矩阵乘法之前我们会对激活进行 AllGather <b>A</b>。" %}

如前所述，**In\[B, D<sub>Y</sub>\] \*<sub>D</sub> W<sub>in</sub>\[D, F<sub>Y</sub>\] \*<sub>F</sub> W<sub>out</sub>\[F<sub>Y</sub>, D\] \-\> Out\[B, D<sub>Y</sub>\] 意味着我们必须在第一次矩阵乘法之前收集激活值。当激活值小于权重时，这比 ZeRO 分片更便宜。** 这通常只有在添加了一定程度的 ZeRO 分片（这会减小收集的规模）时才成立。这也是我们倾向于混合使用 ZeRO 分片和张量并行的原因之一。

{% details 这是张量并行的算法！ %}

<div markdown=1 class="algorithm">

**张量并行：**

**前向传播：** 需要计算 Loss[B]

1.  In[B, D] = **AllGather**(In[B, D<sub>Y</sub>]) *(在关键路径上)*  
2.  Tmp[B, F<sub>Y</sub>] = In[B, D] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *(沿收缩维分片，因此无通信)*  
3.  Out[B, D] {U<sub>Y</sub>} = Tmp[B, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D]  
4.  Out[B, D<sub>Y</sub>] = **ReduceScatter**(Out[B, D] {U<sub>Y</sub>}) *(在关键路径上)*  
5.  Loss[B] = ...

**反向传播：** 需要计算 dW<sub>out</sub>[F<sub>Y</sub>, D], dW<sub>in</sub>[D, F<sub>Y</sub>]

1.  dOut[B, D<sub>Y</sub>] = ...  
2.  dOut[B, D] = **AllGather**(dOut[B, D<sub>Y</sub>]) *(在关键路径上)*  
3.  dW<sub>out</sub>[F<sub>Y</sub>, D] = Tmp[B, F<sub>Y</sub>] \*<sub>B</sub> dOut[B, D]  
4.  dTmp[B, F<sub>Y</sub>] = dOut[B, D] \*<sub>D</sub> W<sub>out</sub>[F<sub>Y</sub>, D] *(可在此处丢弃 dOut[B, D])*  
5.  In[B, D] = **AllGather**(In[B, D<sub>Y</sub>]) *(可与前向传播的 (1) 共享，跳过此步)*  
6.  dW<sub>in</sub>[D, F<sub>Y</sub>] = In[B, D] \*<sub>B</sub> dTmp[B, F<sub>Y</sub>]  
7.  dIn[B, D] {U<sub>Y</sub>} = dTmp[B, F<sub>Y</sub>] \*<sub>F</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *(供前一层使用)*  
8.  dIn[B, D<sub>Y</sub>] = **ReduceScatter**(dIn[B, D] {U<sub>Y</sub>}) *(在关键路径上)*

</div>

{% enddetails %}

张量并行的一个优点是它与我们在 Transformer 前向传播中的两个矩阵很好地交互。直觉上，我们会在每个矩阵之后执行一次 AllReduce。但在这里，我们首先执行 **In[B, D<sub>Y</sub>] \* W<sub>in</sub>[D, F<sub>Y</sub>] -> Tmp[B, F<sub>Y</sub>]**，然后执行 **Tmp[B, F<sub>Y</sub>] \* W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B, D<sub>Y</sub>]**。这意味着我们在开始时对 **In** 执行 AllGather，最后对 **Out** 执行 ReduceScatter，而不是执行 AllReduce。

**这会带来多大的成本？** 让我们只建模前向传播——反向传播只是此处每个操作的转置。在 1D 张量并行中，我们在第一次矩阵乘法之前对激活值进行 AllGather，第二次矩阵乘法之后进行 ReduceScatter，每次发送两个字节（bf16）。让我们弄清楚何时会因通信成为瓶颈。

$$\begin{align}
T_\text{math} & = \frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} \\
T_\text{comms} & =
\frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\\
\textnormal{T} & \approx \max \left(\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C}, \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\right)
\end{align}$$

考虑到我们希望算力成本大于通信成本，我们得到：

$$\begin{align}
\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} > \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}
\end{align}$$

$$\begin{align}
\frac{F}{Y \cdot C} > \frac{1}{W_\text{ici}}
\end{align}$$

$$\begin{align}
F > Y \cdot \frac{C}{W_\text{ici}}
\end{align}$$

因此，例如对于 TPUv5p，在 bf16 中 $C / W_{ici} = 2550$，因此我们只能将张量并行度做到 $Y < F / 2550$。当我们有多个 ICI 轴时，我们的 $T_\text{comms}$ 会减少 $M_Y$ 倍，因此我们得到 $Y < M_Y \cdot F / 2550$。

<p markdown=1 class="takeaway">**要点**: 当 $Y > M_Y \cdot F / 2550$ 时，张量并行会受到通信的限制。对于大多数模型来说，这通常在 8 到 16 路张量并行之间。</p>

**请注意，这与计算的精度无关**，例如对于 int8，在 TPUv5p 上，$$C_\text{int8} / W_{ici}$$ 是 $$5100$$ 而不是 $$2550$$，但通信量也减少了一半，因此两个两倍的因素相互抵消。

**让我们思考一些例子：**

* 在 TPUv5p 上使用 LLaMA 3-70B 与 $$D = 8192,$$ $$F \approx 30,000$$ 时，我们可以轻松实现 8 路张量并行，但在 16 路张量并行时将受到通信限制。8 路模型分片所需的 F 为 20k。

* 对于 Gemma 7B，$$F \approx 50k$$，因此我们使用 19 路张量并行时变得受通信限制。这意味着我们可能使用 16 路仍能获得良好的性能。

### 结合 FSDP 和张量并行

**语法:** $$\text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y]$$

FSDP 和张量并行的优点在于它们可以结合使用。通过沿着两个轴对 **W<sub>in</sub>** 和 **W<sub>out</sub>** 进行分片，我们既可以节省内存，也可以节省算力。由于我们沿着 X 轴对 B 进行分片，我们减少了模型并行 AllGathers 的规模，而由于我们沿着 Y 轴对 F 进行分片，我们减少了 FSDP 的通信开销。这意味着两者的结合可以使我们达到比上面看到的更低的有效批量大小。

{% include figure.liquid path="assets/img/mixed-fsdp-model-parallelism.png" class="img-fluid" caption="<b>图：</b> 一张结合 FSDP 和张量并行的示意图。与其它情况不同，这里没有模型参数的重复。" %}

{% details 这是混合 FSDP + 张量并行的完整算法。虽然我们有很多通信，但我们的 AllGathers 和 ReduceScatters 都更小，因为我们对激活值进行了批量分片，对权重进行了更大量的张量分片！ %}

<div markdown=1 class="algorithm">

**前向传播：** 需要计算 Loss[B]

1. In[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(In[B<sub>X</sub>, D<sub>Y</sub>]) *(在关键路径上)*  
2. W<sub>in</sub>[D, F<sub>Y</sub>] = **AllGather**<sub>X</sub>(W<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]) *(可提前完成)*  
3. Tmp[B<sub>X</sub>, F<sub>Y</sub>] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>]  
4. W<sub>out</sub>[F<sub>Y</sub>, D] = **AllGather**<sub>X</sub>(W<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>]) *(可提前完成)*  
5. Out[B<sub>X</sub>, D] {U<sub>Y</sub>} = Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D]  
6. Out[B<sub>X</sub>, D<sub>Y</sub>] = **ReduceScatter**<sub>Y</sub>(Out[B<sub>X</sub>, D] {U<sub>Y</sub>}) *(在关键路径上)*  
7. Loss[B<sub>X</sub>] = ...

**反向传播：** 需要计算 dW<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>], dW<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]

1.  dOut[B<sub>X</sub>, D<sub>Y</sub>] = ...  
2.  dOut[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(dOut[B<sub>X</sub>, D<sub>Y</sub>]) *(在关键路径上)*  
3.  dW<sub>out</sub>[F<sub>Y</sub>, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>B</sub> dOut[B<sub>X</sub>, D]  
4.  dW<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>] = **ReduceScatter**<sub>X</sub>(dW<sub>out</sub>[F<sub>Y</sub>, D] {U<sub>X</sub>})  
5.  W<sub>out</sub>[F<sub>Y</sub>, D] = **AllGather**<sub>X</sub>(W<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>]) *(可提前完成)*  
6.  dTmp[B<sub>X</sub>, F<sub>Y</sub>] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F<sub>Y</sub>, D] *(可在此处丢弃 dOut[B, D])*  
7. In[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(In[B<sub>X</sub>, D<sub>Y</sub>]) *(不在关键路径上，且可与前一层的 (2) 共享)*  
8.  dW<sub>in</sub>[D, F<sub>Y</sub>] {U<sub>X</sub>} = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F<sub>Y</sub>]  
9.  dW<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>] = **ReduceScatter**<sub>X</sub>(dW<sub>in</sub>[D, F<sub>Y</sub>] {U<sub>X</sub>})  
10. W<sub>in</sub>[D, F<sub>Y</sub>] = **AllGather**<sub>X</sub>(W<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]) *(可提前完成)*  
11. dIn[B<sub>X</sub>, D] {U<sub>Y</sub>} = dTmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *(供前一层使用)*  
12. dIn[B<sub>X</sub>, D<sub>Y</sub>] = **ReduceScatter**<sub>Y</sub>(dIn[B<sub>X</sub>, D] {U<sub>Y</sub>}) *(在关键路径上)*

</div>

{% enddetails %}

**FSDP 和 TP 的最佳组合是什么？** 一个简单但关键的原则是，FSDP 移动权重，而张量并行移动激活值。这意味着当我们的批量大小减小时（尤其是当我们进行更多数据并行时），张量并行变得更为经济，因为每个分片的激活值更小。

* 张量并行执行 $$\mathbf{AllGather}_Y([B_X, D_Y])$$，该操作随着 $$X$$ 的增长而减少。  
* FSDP 执行 $$\mathbf{AllGather}_X([D_X, F_Y])$$，该操作随着 $$Y$$ 的增长而减少。

因此，通过结合两者，我们可以进一步降低每个副本的最小批量大小。我们可以以与上述相同的方式计算 FSDP 和 TP 的最佳数量：

令 $$X$$ 表示专用于 FSDP 的芯片数量，$$Y$$ 表示专用于张量并行的芯片数量。令 $$N$$ 表示我们切片中的总芯片数，$$N=XY$$ 表示该切片的总芯片数。令 $$M_X$$ 和 $$M_Y$$ 分别表示我们进行 FSDP 和 TP 的网格轴数量（这些数量应大致加起来等于 3）。我们将仅对前向传递进行建模，因为它每 FLOP 的通信量最大。然后将上述算法中的通信量相加，我们得到

$$T_\text{FSDP comms}(B, X, Y) = \frac{2\cdot 2\cdot D \cdot F}{Y \cdot W_\text{ici} \cdot M_X}$$

$$T_\text{TP comms}(B, X, Y) = \frac{2 \cdot 2 \cdot B \cdot D}{X \cdot W_\text{ici} \cdot M_Y}$$

同样，我们的总 FLOPs 时间是

$$T_\text{math} = \frac{2\cdot 2 \cdot B \cdot D \cdot F}{N \cdot C}.$$

为简化分析，我们做出两个假设：首先，我们允许 $X$ 和 $Y$ 采用非整数值（只要它们为正且满足 $XY=N$）；其次，我们假设可以在 $X$ 和 $Y$ 轴上完全重叠通信。在第二个假设下，总的通信时间是

$$T_\text{comms} = \max\left(T_\text{FSDP comms}, T_\text{TP comms}\right)$$

在我们探讨在什么条件下会受到算力限制之前，让我们先找到 $X$ 和 $Y$ 的最优值，以最小化我们的总通信量。由于我们的 FLOPs 与 $X$ 和 $Y$ 无关，因此最优设置就是那些能简单地最小化通信的设置。为此，让我们将上述的 $T_\text{comms}$ 用 $X$ 和 $N$（这是一个固定值，因为它是我们系统中的芯片数量）而不是 $X$ 和 $Y$ 来表示：

$$T_\text{comms} (X) = \frac{4D}{W_\text{ici}} \max\left(\frac{F \cdot X}{N \cdot M_X}, \frac{B}{X \cdot M_Y}\right)$$

由于 $T_\text{FSDP comms}$ 随着 $X$ 单调递增，而 $T_\text{TP comms}$ 随着 $X$ 单调递减，最大值必须在 $T_\text{FSDP comms} = T_\text{TP comms}$ 时最小化，这发生在

$$\begin{align*}
\frac{FX_{opt}}{M_X} = \frac{BN}{X_{opt} M_Y} \rightarrow \\
X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}
\end{align*}$$

这非常有用！这告诉我们，对于给定的 $B$、$F$ 和 $N$，最优的 FSDP 数量是多少。让我们对规模有个大致了解。代入实际值，即 $N = 64$（对应 4x4x4 芯片阵列）、$B=48,000$、$F=32768$，得到的结果约为 $X\approx 13.9$。因此，我们会选择 $X$ 为 16，$Y$ 为 4，接近我们计算出的最优值。

<p markdown=1 class="takeaway">**要点：** 一般来说，在训练过程中，FSDP 的最优数量为 $$X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}$$。 </p>

现在让我们回到一直在向所有并行策略提出的问题：**在什么条件下我们会受到算力的限制？** 由于我们可以重叠 FLOPs 和通信，当

$$\max\left(T_\text{FSDP comms}, T_\text{TP comms}\right) < T_\text{math}$$

令 $\alpha \equiv C / W_\text{ici}$，即 ICI 算力强度，我们可以简化：

$$\max\left(\frac{F}{Y \cdot M_X}, \frac{B}{X \cdot M_Y}\right) < \frac{B \cdot F}{N \cdot \alpha}$$

由于我们计算 $X_{opt}$ 使左侧达到最大值相等，我们可以直接将其代入任一侧（注意 $Y_{opt} = N/X_{opt}$），即

$$\frac{F}{N \cdot W_\text{ici} \cdot M_X} \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N} < \frac{B \cdot F}{N \cdot C}$$

进一步简化，我们发现

$$ \sqrt{\frac{B\cdot F}{M_X \cdot M_Y \cdot N}} < \frac{B \cdot F}{N \cdot \alpha},$$

左侧与通信时间成正比，右侧与计算时间成正比。请注意，虽然计算时间随批量大小线性增长（无论是否并行均如此），但通信时间随批量大小的平方根增长。因此，计算时间与通信时间的比值也随批量大小的平方根增长：

$$ \frac{T_\text{math}}{T_\text{comms}} = \frac{\sqrt{BF}\sqrt{M_X M_Y}}{\alpha \sqrt{N}}. $$

为了确保这一比例大于一从而使我们处于算力受限状态，我们需要

$$ \frac{B}{N} > \frac{\alpha^2}{M_X M_Y F}$$

为了得到近似数值，再次代入 $F=32,768$、$\alpha=2550$ 和 $M_X M_Y=2$（这必须是对一个三维网格而言）。这给出了大约 $B/N > 99$。与纯粹的数据并行（或 FSDP）情况相比，这使我们获得了一个大约八倍的优势，其中假设是三维网格，我们计算得出 $B/N$ 必须超过约 $850$ 才会成为算力瓶颈。

<p markdown=1 class="takeaway">**要点：** 将张量并行与 FSDP 结合使用，使我们能够将 $B/N$ 降低到 $$2550^2 / 2F$$。这使我们能够每芯片处理少至 100 的批量，这大约比仅使用 FSDP 能达到的规模小八倍。</p>

下图中，我们绘制了混合 FSDP + TP 的 FLOPs 与通信时间的比值，并将其与仅使用张量并行（TP）和仅使用数据并行（FSDP）进行比较，所用的是一个具有代表性的 4x4x4 芯片阵列。虽然纯 FSDP 并行在非常大的批量大小下占主导地位，但在批量大小与芯片数量之比大致在 100 到 850 之间的区间内，为了达到算力限制，需要采用混合 FSDP + TP 策略。

{% include figure.liquid path="assets/img/mixed-fsdp-comms-2.png" class="img-fluid" caption="<b>图：</b> 在 TPUv5p 4x4x4 切片上，F=30k 时最优混合 FSDP/TP 的 FLOPs 与通信时间的比率。如预期所示，张量并行与批量大小具有固定比率；理想的混合 FSDP + TP 随着 $\sqrt{B}$ 缩放，FSDP 随着 $B$ 缩放。然而，在中间批量大小区间，只有 FSDP + TP 实现了大于 1 的比率。"%}

这是另一个 TPU v5p 16x16x16 的示例，展示了不同分片方案下，FLOPs 和通信时间作为批量大小的函数。

{% include figure.liquid path="assets/img/math-comms-time.png" class="img-fluid" caption="<b>图：</b> 不同并行方案的通信耗时。黑色虚线表示矩阵乘法 FLOPs 所需的时间，因此任何位于该线以上的曲线均受通信限制。我们注意到，当批量大小低于 6e5 时，所有策略均受通信限制，这与我们的预期一致：4096 * 2550^2 / (2 * 8192 * 4) = 4e5。" %}

黑线表示用于模型 FLOPs 的时间量，这意味着任何批次大小，只要这个值低于所有通信成本，就严格受通信限制。你会注意到黑线在大约 `4e5` 处与绿线相交，正如预测的那样。

这里有一个交互式动画可供体验，展示了不同批量大小的总算力时间和通信时间：

<figure class="plotly-embed">
  <iframe src="{{ 'assets/plotly/training-roofline.html' | relative_url }}" title="MLP roofline for mixed FSDP/TP" loading="lazy" scrolling="no"></iframe>
  <figcaption><b>图：</b> 在 TPU v5p 16x16x16（$D=8192$, $F=32768$）上，前向传递 MLP 的计算和通信时间作为批量大小的函数，使用上述 $T_\text{FSDP comms}$、$T_\text{TP comms}$ 和 $T_\text{math}$。拖动滑块以更改 FSDP/TP 分割。阴影区域中的任何批量大小都是通信受限的。</figcaption>
</figure>

你会发现这与上述内容基本一致：最佳划分方式，即 FSDP=512 和 TP=8，在批量大小超过约 4e5 时会受到算力限制，而所有使用更多张量并行的划分方式在所有批量大小下都受到通信限制。使用较少张量并行的划分方式需要更大的批量。

### 流水线

你可能会注意到，我们在前面的章节中完全避开了对流水线的讨论。流水线是GPU并行性的一种主导策略，在TPU上则重要性稍低。简而言之，流水线训练涉及将模型的层分布在多个设备上，并在前向和反向传递过程中在流水线阶段之间传递激活值。该算法大致如下：

1. 在 TPU 0 上用跨层维度分片的权重初始化数据（$W_\text{in}[L_Z, D_X, F_Y]$ 用于与 FSDP 和张量并行结合的流水线）。
2. 在 TPU 0 上执行第一层，然后将生成的激活值复制到 TPU 1，重复此过程直到到达最后一个 TPU。
3. 计算损失函数及其导数 $\partial L / \partial x_L$。
4. 对于最后一个流水线阶段，计算导数 $\partial L / \partial W_L$ 和 $\partial L / \partial x_{L-1}$，然后将 $\partial L / \partial x_{L-1}$ 复制到前一个流水线阶段，并重复此过程直到到达 TPU 0。

{% details 这里是一些（可用的）Python 伪代码 %}

此伪代码应在 Cloud TPU VM 上运行。虽然它效率不高且不现实，但它能让你了解数据是如何在设备之间传播的。

```python
batch_size = 32
d_model = 128
d_ff = 4 * d_model

num_layers = len(jax.devices())

key = jax.random.PRNGKey(0)

# Pretend each layer is just a single matmul.
x = jax.random.normal(key, (batch_size, d_model))
weights = jax.random.normal(key, (num_layers, d_model, d_model))

def layer_fn(x, weight):
  return x @ weight

# Assume we have num_layers == num_pipeline_stages
intermediates = [x]
for i in range(num_layers):
  x = layer_fn(x, weights[i])
  intermediates.append(x)

  if i != num_layers - 1:
    x = jax.device_put(x, jax.devices()[i+1])

def loss_fn(batch):
  return jnp.mean(batch ** 2)  # make up some fake loss function

loss, dx = jax.value_and_grad(loss_fn)(x)

for i in range(num_layers - 1, -1, -1):
  _, f_vjp = jax.vjp(layer_fn, intermediates[i], weights[i])
  dx, dw = f_vjp(dx)  # compute the jvp dx @ J(L)(x[i], W[i])
  weights[i] = weights[i] - 0.01 * dw  # update our weights

  if i != 0:
    dx = jax.device_put(dx, jax.devices()[i-1])
```

{% enddetails %}

**为什么这是一个好主意？** 流水线并行有许多优点：流水线阶段之间的通信成本较低，这意味着即使使用带宽较低的互连设备，也可以训练非常大的模型。这在 GPU 上尤其有用，因为 GPU 并不像 TPU 那样通过 ICI 密集连接。

**为什么这会困难/令人烦恼？** 你可能在上面的伪代码中注意到，TPU 0 几乎总是处于空闲状态！它只在流水线的第一步和最后一步执行工作。这种空闲期被称为流水线气泡，处理起来非常令人烦恼。通常我们首先尝试通过微批量处理来缓解这一问题，即把多个小批量发送到流水线中，从而至少在总步时间的更大比例上保持 TPU 0 的利用率。

第二种方法是仔细重叠前向矩阵乘法 $W_i @ x_i$、反向 $dx$ 矩阵乘法 $W_i @ \partial L / \partial x_{i+1}$ 和 $dW$ 矩阵乘法 $\partial L / \partial x_{i+1} @ x_i$。由于这些操作都需要一些 FLOPs，我们可以重叠它们以完全隐藏气泡。这是来自最近 DeepSeek v3 论文 <d-cite key="DeepSeek3"></d-cite> 的一张图表，展示了他们的“无气泡”流水线调度：

{% include figure.liquid path="assets/img/deepseek-pipeline.png" class="img-fluid" caption="<b>图：</b> DeepSeek v3 管道调度（来自他们的 <a href=\"https://arxiv.org/pdf/2412.19437\">最近论文</a>）。橙色表示前向矩阵乘法，绿色表示 dL/dx 矩阵乘法，蓝色表示 dL/dW 矩阵乘法。通过优先处理反向 dL/dx 乘法，我们可以避免“闲置”FLOPs。" %}

由于TPU（具有更大的互联机架）对此不太关键，我们不会深入探讨这一点，但了解关键的流水线瓶颈是一个很好的练习。

### 跨 Pod 缩放

最大的 TPU 切片是一个包含 8960 个芯片（和 2240 个主机）的 TPU v5p SuperPod。当我们想要扩展到超过这个规模时，就需要跨越数据中心网络（DCN）的边界。每个 TPU 主机都配备一个或多个 NIC（网络接口卡），通过以太网将主机连接到其他 TPU v5p 模块。如 [TPU 章节](../tpus) 中所述，每个主机拥有约 200Gbps（25GB/s）的全双工 DCN 带宽，每个 TPU 的全双工（出站）带宽约为 6.25GB/s。

通常，在扩展到单个 pod 以外时，我们会在 ICI 域内进行某种形式的模型并行或 FSDP，然后在多个 pod 之间进行纯数据并行。令 $N$ 表示我们希望扩展到的 TPU 数量，$M$ 表示每个 ICI 连接的 slice 中的 TPU 数量。要在 DCN 上执行 AllReduce，我们可以对 pod 集合执行环形归约，在反向传播中得到：

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{N \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF}{M \cdot W_\text{dcn}}$$

通信带宽随着 $M$ 缩放，因为与 ICI 不同，总带宽随着我们扩展 ICI 领域并获取更多 NIC 的过程而增长。简化后，我们发现当 $T_\text{math} > T_\text{comms}$ 时

$$\frac{B}{\text{slice}} > \frac{C}{W_\text{dcn}}$$

对于 TPU v5p，$\frac{C}{W_\text{dcn}}$ 约为 `4.59e14 / 6.25e9 = 73,440`。这告诉我们，为了在 DCN 上高效缩放，每个 ICI 域需要一个最小的批量大小，以使每个节点能够出站。

**这个问题有多严重？** 以一个具体例子来说明，假设我们想在 TPU v5p 上使用 2M 词元的批量大小（BS）来训练 LLaMA-3 70B。LLaMA-3 70B 具有 $F\approx 30,000$。从上面的章节中，我们知道以下内容：

* 我们可以将张量并行扩展到 $Y = M_Y \cdot F / 2550 \approx 11 \cdot M_Y$。
* 只要 $B / N > 2550 / M_X$，我们就可以使用 FSDP。这意味着如果我们想用 BS=2M 和三个数据并行轴进行训练，最多只能使用 $\approx 2400$ 芯片，大约相当于一个 TPU v5p 机柜的四分之一。
* 当我们将 FSDP 与张量并行结合时，在 $B / N < 2550^2 / (2 \cdot 30000) = 108$ 的情况下会受到通信的限制，因此这使我们能够扩展到大约 18,000 个芯片！然而，TPU v5p 机柜的最大规模是 8,000 个芯片，超过这个规模后我们必须使用 DCN。

TLDR 是我们有一个不错的训练配方，使用 BS=1M，大约 X（FSDP）= 1024 和 Y（TP）= 8，但使用 BS=2M 时需要使用 DCN。如上所述，我们的 DCN 算术强度为 $\text{73,440}$，因此我们只需确保每个 ICI 域的批量大小大于这个值。对我们来说这很简单，因为使用 2 个 pod 时，每个 pod 的 BS 为 1M，每个 TPU 的批量大小为 111，这非常好（可能稍微接近临界值，但理论上是成立的）。

<p markdown=1 class="takeaway">**要点：** 只要每个 TPU 节点的批量大小至少为 73k 词元，使用纯数据并行在多个 TPU 节点上进行缩放就相当直接。</p>

## TPU 上大语言模型训练的要点

* 增加并行度或减少批量大小都会使我们更容易受到通信的限制，因为它们降低了每块芯片执行的算力。

* 在合理的上下文长度（~32k）以内，我们可以将 Transformer 建模为 MLP 模块的堆叠，并通过它们如何对每层的两个/三个主要矩阵乘法进行分片，来定义几种并行方案。

* 在训练过程中，我们考虑了 4 种主要的并行方案，每种方案都有其各自的带宽和算力需求（数据并行、FSDP、张量并行和混合 FSDP + 张量并行）。

|**策略**|**描述**|
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ ||**数据并行**|激活值按批量分片，其余部分全部复制，在反向传播过程中我们对梯度进行全归约。|
|**FSDP**|激活值、权重和优化器均按批次分片，权重在使用前刚刚被聚合，梯度被reduce-scattered。|
|**张量并行（又称 Megatron，模型）**|激活值沿 $$d_\text{model}$$ 分片，权重沿 $$d_{ff}$$ 分片，在 W<sub>in</sub> 之前收集激活值，结果在 W<sub>out</sub> 之后进行 reduce-scatter。|
|**混合 FSDP + 张量并行**|上述两种情况，其中 FSDP 会收集模型分片的权重。|

以下是每种方法的“公式”：

$$\small
\begin{array}{cc}
\text{Strategy} & \text{Formula}\\
\hline
\text{DP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D] \\
\text{FSDP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D] \\
\text{TP} & \text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y] \\
\text{TP + FSDP}  & \text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y] \\
\hline
\end{array}$$

* 这些策略都存在一个极限，当达到该极限时，它们会受到网络/通信的限制，这取决于每台设备的算力和通信能力。假设 $$X$$ 是 FSDP，$$Y$$ 是张量并行，以下是每层的算力和通信能力。

$$
\small
\begin{array}{ccc}
\text{Strategy} & \text{Compute per layer} & \text{Comms per layer} \\
& \text{(ignoring gating einsum)} & \text{(bytes, forward + backward pass)}\\
\hline
\text{DP} & 4BDF/X + 8BDF/X & 0 + 8DF \\
\text{FSDP} & 4BDF/X + 8BDF/X & 4DF + 8DF \\
\text{TP} & 4BDF/Y + 8BDF/Y & 4BD + 4BD \\
\text{FSDP + TP} & 4BDF/(XY) + 8BDF/(XY) & (4BD/X + 4DF/Y) + (8BD/X + 8DF/Y) \\
\hline
\end{array}$$

* 纯数据并行很少有用，因为模型及其优化器状态使用的字节数 = 参数量的 10 倍。这意味着我们很少能将超过几十亿参数放入内存中。

* 当 $$\text{batch size per shard} < C / W$$ 的算术强度达到一定水平时，数据并行和 FSDP 会受到通信的限制。对于 ICI 来说，这一数值是 2,550，对于 DCN 来说约为 71,000。通过增加更多的并行轴，可以提高这一数值。

* 当 $$\lvert Y\rvert > F / 2550$$ 时，张量并行会受到通信的限制。**这通常在大多数模型中为 8-16 路。** 这与批量大小无关。

* 混合 FSDP + 张量并行允许我们将批量大小降低至低至 $$2550^2 / 2F \approx 100$$。这非常低。

* 跨 pod 的数据并行需要每个 pod 的最小批量大小约为 71,000，否则会受到 DCN 的限制。

* 基本上，如果你的批量大小较大或模型较小，情况会比较简单。你可以选择在 DCN 上进行数据并行，或者使用 FSDP + 数据并行。中间部分才是事情变得有趣的地方。

## 一些需要解决的问题

让我们以 LLaMA-2 13B 作为本节的基本模型。以下是模型详细信息：

|超参数|价值|
| ---------- | ------ ||L|40|
|D|5,120|
|F|13824|
|N|40|
|K|40|
|H|128|
|V|32,000|

LLaMA-2 有单独的嵌入矩阵和输出矩阵，并包含一个门控 MLP 模块。

**问题 1：** LLaMA-2 13B 有多少个参数（我知道这听起来很傻，但请做一下计算）？*请注意，如[Transformer 数学](../transformers)中所述，LLaMA-3 有 3 个大的 FFW 矩阵，两个上投影和一个下投影。我们在本节中忽略了两个“门控”einsum 矩阵，但它们在此部分的行为与 W<sub>in</sub> 相同。*

{% details 点击此处查看答案。 %}

* FFW 参数：$$3LDF$$ = `8.5e9`  
* 注意力参数：$$4DNHL$$ = `4.2e9`  
* 词汇参数：$$2VD$$ = `0.33e9`  
* 总计：`8.5e9 + 4.2e9 + 0.33e9 = 13.0e9`，正如预期！

{% enddetails %}

**问题 2:** 假设我们使用 BS=16M 词元进行训练，并使用 Adam 优化器。暂时忽略并行性，模型的参数、优化器状态和激活值总共会使用多少内存？*假设参数以 bf16 格式存储，优化器状态以 fp32 格式存储，并且每个层的激活值检查点存储三次（在三个大的 FFW 矩阵乘法之后）。*

{% details 点击此处查看答案。 %}

参数（bf16）和两个优化器状态（fp32，第一个和第二个矩形累积器）所使用的总内存为 `(2 + 4 + 4) * 13e9 ~ 130GB`。前两次矩阵乘法后的激活值形状为 $BF$，最后一次矩阵乘法后的激活值形状为 $BD$（如上图 Transformer 所示），因此 bf16 的总内存为 $2 \cdot L \cdot (BD + 2 * BF) = 2LB \cdot (D + 2F)$ 或 `2 * 40 * 16e6 * 5,120 * (1 + 2 * 2.7) ~ 4.2e13 = 42TB`，因为 `B=16e6`。其他所有激活值都可以忽略不计。

{% enddetails %}

**问题 3:** 假设我们希望使用 32k 序列长度和总批量大小为 3M 词元，在 TPUv5p 16x16x16 切片上进行训练。假设我们希望像上面一样使用 bfloat16 权重和 float32 优化器。

1. 我们能否仅使用纯数据并行？为什么能或不能？  
2. 我们能否在不受到通信限制的情况下仅使用纯 FSDP？为什么能或不能？如果使用纯 FSDP，每个设备将使用多少内存（假设仅在三个大的 FFW 矩阵之后进行梯度检查点）？  
3. 我们能否混合使用 FSDP 和张量并行？为什么能或不能？如果是，$X$ 和 $Y$ 应该是什么？每个设备将存储多少内存？仅使用屋顶线 FLOPs 估计并忽略注意力机制，在 40% MFU 的情况下，每个训练步需要多长时间？

{% details 点击此处查看答案。 %}

首先，让我们写下一些数字。在序列长度为 32k 且批量大小为 3M 的情况下，我们得到一个序列批量大小为 96。在 TPU v5p 16x16x16 切片上，我们有 `393TB` 的 HBM。

1. 我们不能单纯使用数据并行，因为它会在每块芯片上复制参数和优化器状态，这些参数和优化器状态已经达到了约130GB（来自Q2），超过了每块芯片的HBM容量（96GB）。

2. 让我们先纯粹从显存的角度来看。在 Q2 中将 BS=16M 替换为 3M，我们得到 `~7.86e12` 总的检查点激活值，再加上 1.3e11 的优化器状态，这使我们达到几乎正好 8e12 = 8TB。TPUv5p 切片总共有 `393TB` 的 HBM，因此我们在 HBM 限制内是安全的。接下来让我们看看我们是否会受到通信或算力的限制。拥有 4096 个芯片和 3 个并行轴，我们可以实现最小的批量大小为 `850 * 4096 = 3.48M` 个词元。这略高于我们的 3M 批量大小。因此，实际上我们是受通信限制的，这令人遗憾。因此，总体的回答是 **不，我们不能在不受到通信限制的情况下单独使用 FSDP**。

3. 现在我们知道我们主要的关注点是受通信限制，因此让我们代入一些数字。首先，我们从上面知道，使用混合 FSDP + 张量并行时，此处每块芯片的批量大小需要高于 $2550^2 / 2F = 235$。这意味着理论上我们可以做到这一点！让我们计算各自的部分。

我们有规则 $X_{opt} = \sqrt{(B / F) \cdot (M_X / M_Y) \cdot N}$，所以这里我们有 `sqrt(3e6 * 2 * 4096 / 13824) = 1333`，这意味着我们将进行大约 1024 路 DP 和 4 路 TP。每块 TPU 的内存将如 (2) 所示，步时将只是 `6 * 3e6 * 13e9 / (4096 * 4.6e14 * 0.4) = 300ms`。

{% enddetails %}

<h3 markdown=1 class="next-section">第五部分就到这里！第六部分将把本内容应用到实际的LLaMA模型中，[点击此处](../applied-training)！</h3>

## 附录

### 附录 A：推导反向传播通信

上述内容中，我们将 Transformer 层的前向传播简化为 Out[B, D] = In[B, D] *<sub>D</sub> W<sub>in</sub>[D, F] *<sub>F</sub> W<sub>out</sub>[F, D]。如何推导反向传播所需的通信？

这与上一节中针对单个矩阵乘法 **Y = X * A** 的规则非常自然地一致。

$$\frac{dL}{dA} = \frac{dL}{dY}\frac{dY}{dA} = X^T \left(\frac{dL}{dY}\right)$$

$$\frac{dL}{dX} = \frac{dL}{dY}\frac{dY}{dX} = \left(\frac{dL}{dY}\right) A^T$$

使用这个，我们得到以下公式（令 Tmp[B, F] 表示 In[B, D] * W<sub>in</sub>[D, F]）：

<div markdown=1 class="algorithm">

1. dW<sub>out</sub>[F, D] = Tmp[B, F] *<sub>B</sub> dOut[B, D]  
2. dTmp[B, F] = dOut[B, D] *<sub>D</sub> W<sub>out</sub>[F, D]  
3. dW<sub>in</sub>[D, F] = In[B, D] *<sub>B</sub> dTmp[B, F]  
4. dIn[B, D] = dTmp[B, F] *<sub>F</sub> W<sub>in</sub>[D, F]

</div>

请注意，这些公式是数学表达式，未提及分片。反向传播的任务是计算这四个量。因此，为了确定所需的通信操作，我们只需取上述四个方程中所有需要进行矩阵乘法的量（Tmp、dOut、W<sub>out</sub>、W<sub>in</sub>）的分片方式，这些分片方式由我们的并行方案指定，然后根据分片矩阵乘法的规则确定需要进行哪些通信操作。请注意，dOut 的分片方式与 Out 相同。
