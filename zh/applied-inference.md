---
layout: distill
title: "在 TPUs 上部署 LLaMA 3-70B"
# permalink: /main/
description: "让我们仔细看看如何在 TPU v5e 上部署 LLaMA 3-70B 模型。在屋顶线模型中，不同模型的部署成本有多高？它们的 KV 缓存有多大？我们应该使用多大的批量大小？在推理过程中，参数和激活值是如何分片的？让我们对生产环境中的延迟和吞吐量做一些粗略估计。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 8

previous_section_url: "../inference"
previous_section_name: "Part 7: Inference"

next_section_url: ../profiling
next_section_name: "Part 9: Profiling"

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
  - name: "What's the LLaMA Serving Story?"
  - subsections:
    - name: "Thinking about throughput"
    - name: "What about prefill?"
  - name: "Visualizing the Latency Throughput Tradeoff"
  - name: "Worked Problems"

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

*本节将探讨如何部署 LLaMA-3 以及其部署效率。与前一个“应用”部分一样，尝试在查阅答案之前，用纸笔自行推导出答案！*

## LLaMA 服务的故事是什么？

让我们回想一下LLaMA 3-70B的结构（参见[第6章](../applied-training)作为参考）：

|**超参数**|**价值函数**|
| --------------------------- | :-------: ||$$n_\text{layers}$$ (L)|80|
|$$d_\text{model}$$ (D)|8,192|
|$$d_{ff}$$ (F)|28,672|
|$$n_\text{heads}$$ (N)|64|
|$$n_\text{kv heads}$$ (K)|8|
|$$d_\text{qkv}$$ (H)|128|
|$$n_\text{embeddings}$$ (V)|128,256|

让我们从一个简单的问题开始：**我们应该在什么硬件上进行服务？** 答案基本上是，选择每 FLOP / 美元成本最低的硬件。<d-footnote> 这并不总是正确的，有时更多的 HBM 或 ICI 带宽比 FLOPs 更关键，但这是一个不错的启发式方法。</d-footnote> 正因如此，我们通常希望在 TPU v5e 上进行服务，这是我们当前专用的推理芯片（截至 2025 年 2 月，成本来自 [Google Cloud 定价](https://cloud.google.com/tpu/pricing)）：

|**TPU 类型**|**bfloat16 FLOPs/s**|**Google Cloud USD / 小时**|**FLOPs / $**|
| ------------ | :------------------: | :-------------------------: | :-----------: ||H100|9.9e14|$10.8|3.3e17|
|v5p|4.59e14|$4.2|3.9e17|
|v5e|1.97e14|$1.2|**5.8e17**|

每块 TPU v5e 配备 16GB 的 HBM，这将迫使我们相当激进地对模型进行分片。让我们先考虑一些可能对我们重要的基本量：

**问题：** 每个词元的 LLaMA 3-70B 的 KV 缓存有多大？*你可以假设我们以 int8 格式存储它们。这决定了在给定拓扑结构下我们的批量大小可以有多大。*

{% details 想好了就点击这里！ %}

LLaMA 3-70B 有 8 个 KV 头，因此每个词元的大小是 `2 * K * H * L = 2 * 8 * 128 * 80 = 160kB`。

**请注意这个数值有多大！** 如果我们有一个序列长度为 32k 词元（这是常见的），这将使用 `160e3 * 32,768 = 5.3GB / sequence`。对于 BS=240，这需要 1.3TB 的内存！由于 TPU v5e 每块只有 16GB，我们需要大约 `(70e9 + 1.3e12) / 16e9 = 86` 块 TPU v5e 芯片才能容纳这么多内存。同时请注意，与 70GB 的模型参数相比，这个数值有多大。

{% enddetails %}

**问题：** 假设我们希望以批量大小 32 和 8192 的序列长度来部署 L3 70B 模型，并且将所有参数和 KV 缓存都使用 int8 格式。这将总共使用多少内存？我们能在最小的硬件配置上部署它吗？

{% details 答案 %}

由于我们的 KVs 是 `160e3` 字节的 int8，我们的总 KV 内存是 `160e3 * 8192 * 32 = 41.9e9` 字节。我们的参数是 `70e9` 字节，因为我们每个参数有 1 字节。因此，我们的总内存使用量是 `41.9e9 + 70e9 = 112GB`。

我们能使用的最小切片将包含 `112e9 / 16e9 = 7` TPUs，或者（四舍五入到偶数大小），TPU v5e `4x2`。这将是一个非常紧凑的配置，考虑到其他开销，我们可能无法完全容纳，因此我们可能至少需要 `4x4`（或者减少批量大小）。

{% enddetails %}

**问题：** 在这个批量大小和量化设置下，使用 TPU v5e `4x2`，每个解码步骤大约会有什么样的延迟？吞吐量（词元 / 秒 / 芯片）是多少？如果是 `4x4` 呢？*假设我们使用 bfloat16 进行 FLOPs 计算，并且所有内容都完全分片。*

{% details 答案 %}

我们可以调用前一节中的公式

$$\begin{align*}
\tiny \text{Theoretical Step Time (General)} = \underbrace{\frac{\text{Batch Size} \times \text{KV Cache Size}}{\tiny \text{Total Memory Bandwidth}}}_{\text{Attention (always bandwidth-bound)}} + \underbrace{\max\left(\frac{2 \times \text{Batch Size} \times \text{Parameter Count}}{\text{Total FLOPs/s}}, \frac{\text{Parameter Size}}{\text{Total Memory Bandwidth}}\right)}_{\tiny \text{MLP (can be compute-bound)}}
\end{align*}$$

由于我们的参数是 int8 格式，而我们的 FLOPs 是 bfloat16 格式，因此我们的关键批量大小约为 120。我们也可以手动计算 RHS 最大值，但这基本上是我们已经多次进行过的计算。**因此，我们的矩阵乘法和 FLOPs 都已经进入了内存受限区间。**

严格从显存带宽来看，我们的步时间基本上是 `(KV size + param size) / (8 * HBM bandwidth) = 112e9 / (8 * 8.2e11) = 17ms`。**因此理论上我们的步时间约为 17 毫秒。** 我们的吞吐量将是 `32 / .017 = 1882 tokens / sec`，或 `1882 / 8 = 235 tokens / sec / chip`。

这里有一个需要注意的地方，即要检查我们是否可能在矩阵乘法上受到 ICI 的限制。我们可以在这里将两个轴分配给它，因此理论上当 $Y > 2 * F / 2200 = 2 * 28672 / 2200 = 26$ 时，我们是 ICI 限制的，所以我们的做法是完美的！

如果我们使用 `4x4` 运行，从 ICI 的角度来看我们仍然没有问题，因此我们的延迟将降至 `17 / 2 = 8.5ms`，但每芯片的吞吐量将保持不变。

{% enddetails %}

### 关于吞吐量的思考

让我们花一点时间纯粹地思考一下吞吐量。当我们优化吞吐量时，我们希望达到算力限制，这意味着我们能够充分利用所有 TPU MXU 的容量。通常这意味着我们希望批量大小尽可能大，这样我们就能尽可能多地完成工作。

**问题：** 在 TPU v5e 上，使用 bfloat16 权重和激活值，我们的批量大小需要多大，才能在矩阵乘法中达到算力限制？如果我们使用 int8 权重，但用 bfloat16 执行 FLOPs 呢？如果是 int8 权重和 int8 FLOPs 呢？

{% details 答案 %}

如第 7 节所述，对于任何满足 $B \ll D, F$ 的 bfloat16 矩阵乘法，我们有

$$\begin{equation*}
T_\text{math} > T_\text{comms} \leftrightarrow \frac{2BDF}{2DF} \geq \frac{\text{TPU bfloat16 FLOPs/s}}{\text{HBM bandwidth}} = 240
\end{equation*}$$

当我们的权重为 int8 时，分母会损失一个因子 2，因此我们得到 $2BDF / DF = 2B > 240$，或者说 $B > 120$，即之前临界批量大小的一半。这对我们非常有帮助！当我们使用 int8 权重和 int8 FLOPs 时，我们必须使用 int8 值来表示 TPU FLOPs/s，其数值从 bfloat16 的 1.97e14 增加到 3.94e14，几乎翻了一倍。这意味着我们又回到了最初的位置，大约在 $B > 240$。

使用 int8 权重和 bfloat16 FLOPs 的情况相当常见，因为无损量化参数通常比进行低精度算术运算更容易。

{% enddetails %}

**问题：** 使用 bfloat16、int8 和 int4（包括 KVs 和参数）在 8k 上下文长度下，我们能用最小的 TPU v5e 拓扑部署 LLaMA 3-70B 吗？*你可以认为对于这个问题，KV 缓存的大小可以忽略不计。*

{% details 答案 %}

这很简单！如果我们能接受一个非常小的批量大小，那么唯一的限制就是将参数内存适配到 HBM 中，即它仅仅是 `ceil(num_params * sizeof(dtype) / HBM per TPU)`，或者将 `ceil(70e9 * sizeof(dtype) / 16e9)` 四舍五入到最合理的拓扑结构（2 的某个倍数）：

|数据类型|参数量|KV 缓存大小 / 词元（字节）|最小 TPU v5es|实际最小切片|剩余的 HBM 用于 KV 缓存|8k 个 KV 缓存|
| :---: | :--------: | :---------------------: | :----------: | :--------------: | :-------------------------: | :----------------: ||bfloat16|140GB|324kB|8.75|4x4 = 16 芯片|116|43|
|int8|70GB|162kB|4.38|4x2 = 8 芯片|58|43|
|int4|35GB|81kB|2.81|2x2 = 4 芯片|29|43|

这真的很酷！它告诉我们，如果我们想的话，可以将 LLaMA 70B 拟合到 TPU v5e 2x2 上。不过你会注意到 KV 缓存的数量非常小。这就是我们的批量大小！这意味着我们将得到非常糟糕的 FLOPs 利用率。我们非常希望使用更大的拓扑结构，以将我们的批量大小提升到 240。

{% enddetails %}

**问题：** 假设我们使用能适应这些拓扑结构的最大批量大小，每个生成步骤的延迟可能有多大？

{% details 答案 %}

这也很容易，因为我们选择批量大小是为了填满所有的 HBM！这只是一个问题，即需要多长时间才能将一个完整的 TPU v5e 的字节加载到 MXU 中。这仅仅是 `v5e HBM / v5e HBM memory bandwidth = 16GB / 8.2e11 = 19ms`，因此这是 **19ms / 步**。假设我们的生成的中位长度为 512 个词元，那么每个解码大约需要 9 秒。请注意，如果我们使用较小的批量大小，例如仅将模型参数视为 int4，那么我们的最低延迟约为 10ms / 步，因为 HBM 不再是满的。

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：我们可以通过询问从 HBM 将模型的所有参数加载到 MXU 所需的时间，来始终对解码延迟进行下界估计。当我们的 KV 缓存较小时，可以认为每一层只是逐块加载权重，然后丢弃它们。除非我们使用较大的批量大小或大量的设备间通信，否则这通常是一个合理的界限（在 1.5 倍以内）。当我们的批量大小较大时，还需要对 KV 缓存的加载进行建模，因为这会主导参数的加载时间。</p>

同样地，在 FLOPs 限制区间（例如训练或大批次推理）中，我们可以使用 $$\text{Total FLOPs} / (N \cdot C) = 2 \cdot \text{param count} \cdot B / (N \cdot C)$$ 下限，该下限假设不存在通信。

**问题：** 对于这些情况，这能给我们每块芯片带来多大的吞吐量（以查询数/芯片为单位）？*你可以假设我们的平均解码长度是 512 个词元。*

{% details 答案 %}

这是一个重要问题，因为它与成本 / 词元完全相关。

基于我们对中位解码长度的假设，我们的吞吐量仅为 $$B / (\text{per-step latency} \cdot \text{median steps} \cdot N) \approx 43 / (0.019 * 512 * N)$$。这使我们大约达到 $$(4.42 / N)$$ QPS，因此代入 $$N$$ 我们得到：

|数据类型|QPS / 芯片|
| :------: | :--------: ||bfloat16|0.27|
|int8|0.55|
|int4|1.11|

请注意，这种估计相当乐观，因为它完全忽略了前向传递的 working memory（用于激活值和注意力的内存）。使用 Flash Attention 时，这并不荒谬，但也不现实。实际数值可能约为这个值的一半。为了实现绝对最大的吞吐量，我们可能需要将芯片数量增加一倍以上，并显著增加批量大小。

{% enddetails %}

**问题：** 如果我们将上述每个示例的拓扑结构翻倍，我们的峰值吞吐量会如何变化？

{% details 答案 %}

如果我们使用 4x8 的 bfloat16 切片，我们将有 372GB 的剩余空间用于 KV 缓存，这将使我们能够将批量大小增加到 140。然后，由于我们的步时间将保持不变，我们的吞吐量将达到 `14.39 / num_chips`，或

|数据类型|QPS / 芯片|
| :---------------: | :--------: ||bfloat16（4x8）|0.44|
|int8（4x4）|0.90|
|int4（在 2x4 上）|1.80|

进一步增加将带来更大的收益！主要结论是，如果我们受到 KV 缓存大小的限制，**最小的拓扑结构并非在所有情况下都是性能最佳的拓扑结构**。

{% enddetails %}

**问题：** 现在让我们深入探讨分片的问题。假设我们希望在 TPU v5e 4x8 上以 bfloat16 格式进行服务。在生成过程中，我们在 TPU v5e 4x8 上应使用何种分片方式？我们能否避免通信瓶颈？

{% details 答案 %}

如前一节所述，在生成过程中，我们实际上只有一种分片选项：模型并行。在变得受通信限制之前，我们能做多少呢？如前一节所述，我们的模型在大约以下情况下会变得受通信限制：

$$Y > \frac{F \cdot M_Y}{2200}$$

对于 LLaMA 3-70B，我们有 `F = 28,672`，因此如果我们进行两个轴的模型分片，这将给我们大约 $$Y = 28672 \cdot 2 / 2200 = 26$$，因此一般来说，我们可以在不受到通信限制的情况下扩展到大约 16 个芯片，这使我们能够使用 `4x4`，但不能使用 `4x8`。通常，由于我们不能完全重叠计算，即使这个估计也过于乐观。

**要点：我们实际上无法仅通过纯模型并行来在 4x8 上进行服务。** 在这里我们能做到的最好的是 4x2 或 _或许_ 是 4x4。

然而，正如我们之前讨论的，当我们的批量大小较小时，我们通常可以在不显著影响吞吐量的情况下进行更多的模型并行，因为我们的模型是显存带宽受限而非 FLOPs 受限。我们之前说过这个值大约是 $Y=F / (8\cdot B)$，因此如果我们使用批量大小 64，理论上在成为 ICI 受限之前，我们可以达到高达 `Y = 28,672 / (8 * 64) = 56` 的模型并行程度。为了进行合理性检查，我们可以查看 $T_\text{ici comms}$、$T_\text{hbm comms}$ 和 $T_\text{math}$ 中单个矩阵乘法的情况。我们显然有：

$$\begin{align*}T_\text{ici comms} = \frac{2BD}{W_\text{ici}} && T_\text{hbm comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{math} = \frac{2BDF}{Y \cdot C}\end{align*}$$

对于 `4x8`，这将给我们 $T_\text{ici comms}$ = `(2 * 64 * 8192) / 9e10 = 11us`，$T_\text{hbm comms}$ = `(2 * 8192 * 28,672) / (32 * 8.2e11) = 18us`，以及 $T_\text{math}$ = `(2 * 64 * 8192 * 28,672) / (32 * 1.97e14) = 4us`，因此从理论上讲，我们仍然受 HBM 带宽限制，这很好！*请注意，从 `4x4` 扩展到 `4x8` 从吞吐量角度来看可能没有帮助，但它会降低我们的延迟！*

如果我们看一下 int8 和 int4 的配置，我们 _可以_ 通过纯模型并行来实现这些。因此，我们已经到达了一个点，即量化实际上在更快的 FLOPs 之外给我们带来了有意义的优势：它让我们在成为通信瓶颈之前能够使用更大的批量大小。**所以这个故事的结局是，我们无法在 4x8 上实现峰值吞吐量，但对于 int8 和 int4 配置，我们可以实现纯模型并行。**

{% enddetails %}

<p markdown=1 class="takeaway">**提示**：有用的模型并行最大数量取决于 $$d_{ff}$$ 以及你对模型进行分片的轴数。最大值通常在 8 到 32 之间，具体取决于模型规模。你可以超过这个限制以提高延迟，但会付出吞吐量方面的代价。</p>

### 预填充如何？

我们在这里 mostly 忽略了预填充，因为它要简单得多。让我们将几个概念结合起来，思考整体的端到端情况。

**问题：** 假设预填充期间实现了 40% 的 FLOPs 利用率。在 16 个 TPU v5e 芯片上，长度为 8192 的预填充需要多长时间？

{% details 答案 %}

在 8k 词元时，我们已经明确受到算力的限制，因此只需考虑 FLOPs。我们知道我们的模型有 `70e9` 个参数，因此每次前向传递使用 `2 * 70e9 * B` FLOPs。假设模型 FLOPs 利用率为 40%，这将给我们带来大约 `2 * 70e9 * 8192 / (16 * 1.97e14 * 0.4) = 0.91s` 的运行时间。与我们之前看到的数字相比，这实际上相当多！

{% enddetails %}

**问题：** 假设我们的预填充长度中位数为 8192 个词元，解码长度中位数为 4096 个词元。假设我们的生成批量大小为 32。平均而言，每一步有多少序列完成解码？平均而言，每一步我们的 KV 缓存中会驱逐多少个词元？

{% details 答案 %}

这在某种程度上是显而易见的。由于我们的中位解码长度为 4096 个词元，一个序列大约每 1 / 4096 个词元就会完成。给定批量大小为 32，这意味着每一步我们将有 `32 / 4096` 个序列被移除。由于我们的 KV 缓存长度大约为 `8192 + 4096`，这意味着每一步将移除 `32 * (8192 + 4096) / 4096 = 96` 个词元。一般公式为 $B * (P + G) / G$，其中 $P$ 和 $G$ 是预填充和生成长度。

{% enddetails %}

**问题：** 假设我们采用解耦服务，预填充长度中位数为 8192，生成长度中位数为 512。假设上述计算的预填充和生成延迟是在 bfloat16 下进行的。需要预填充：生成服务器的什么比例才能使两者都保持完全饱和。

{% details 答案 %}

这算是一个有趣的问题。设 $P$ 为预填充服务器的数量，$G$ 为生成服务器的数量。总的来说，这是一个流水线问题，我们以 `P / prefill_latency` 的速率将序列输入，以 `B * G / (generate_latency * median_decode_length)` 的速率消耗它们。我们计算出在批量大小为 43（我们称之为 32）的情况下，每个预填充步骤需要 `910ms`，每个解码步骤需要 `19ms`。因此我们需要 `P / 0.91 = 32 * G / (0.019 * 512)` 或 `P = 3G`，也就是说，我们需要大约 3 倍于生成服务器数量的预填充服务器！

{% enddetails %}

## 可视化延迟与吞吐量的权衡

继续以 LLaMA 70B 为例，让我们实际看一下生成过程中不同批量大小的延迟和吞吐量。正如我们在上一节对 PaLM 模型的展示一样，这为我们提供了吞吐量/延迟的帕累托前沿。我们假设使用 16 路张量并行，因为这是在保持 MLP 模块算力受限的情况下可以合理使用的上限。这里我们将使用 TPU v5e 4x4 拓扑结构。**滑块控制序列长度，以便您可以看到更大的 KV 缓存带来的影响。**

<figure class="plotly-embed">
  <iframe src="{{ 'assets/plotly/pareto.html' | relative_url }}" title="Latency/throughput Pareto frontier for LLaMA 3-70B" loading="lazy" scrolling="no"></iframe>
  <figcaption><b>图：</b> 每芯片吞吐量与每词元延迟对比，针对 LLaMA 3-70B（int8 权重，16 路张量并行）在 TPU v5e 4x4 上的表现，沿每条曲线调整批量大小。拖动滑块以更改上下文长度；每条曲线在 KV 缓存适合 HBM 的最大批量处停止。</figcaption>
</figure>

* **请看看成本与延迟之间的权衡有多显著。** 通过将每词元延迟翻倍，我们可以实现每词元成本大约减少 100 倍。此外，我们的延迟范围可以从小批量时的 5.5 毫秒到大批量时的 20 毫秒。
* 请注意，当在 2k 上下文长度时，吞吐量在达到 BS 120 的上限（此处为 120，因为我们使用 int8 权重但 bf16 FLOPs）时，每芯片的吞吐量有效稳定在约 1 词元 / 毫秒。然而，随着序列长度的增加，我们无法再将此批量大小放入内存中，因此我们从未达到完全饱和的点。
* 请注意，在相同吞吐量下，大批量时的延迟要高得多，因为 KV 加载变得占主导地位（而不是参数加载）。

我们可以通过将成本和延迟的来源分解为参数加载时间、KV加载时间和FLOPs时间，来更好地理解这一点。阴影区域表示我们预计在MLP块中会受到算力限制的区域。

<figure class="plotly-embed">
  <iframe src="{{ 'assets/plotly/latency_breakdown_log.html' | relative_url }}" title="Decode step time breakdown for LLaMA 3-70B" loading="lazy" scrolling="no"></iframe>
  <figcaption><b>Figure:</b> 每个解码步骤的时间在 TPU v5e 4x4 上运行 LLaMA 3-70B 时随批量大小变化的情况，分为参数加载、KV 缓存加载和 FLOPs。总时间是 KV 缓存加载时间加上参数加载或 FLOPs 中较大的那个。拖动滑块以更改上下文长度。</figcaption>
</figure>

这说明了一个相当有趣的现象。你可以看到，最初，参数加载占据了大部分的延迟，直到批量大小变得足够大，FLOPs 和 KV 加载变得更为显著。值得注意的是，在所有序列长度超过 2048 的情况下，我们在 KV 缓存加载上花费的时间超过了 FLOPs 的时间！**因此，虽然我们可以通过增加批量大小来提高硬件利用率，但在长上下文长度的情况下，KV 加载始终主导着总步时间。**

<p markdown=1 class="takeaway">**要点：** 对于 LLaMA 3-70B，几乎在所有这些配置中，我们均受到 KV 缓存显存带宽（以及 HBM 带宽）的限制，这突显了减少 KV 缓存大小对生成吞吐量的重要性。同时请注意，这里的延迟/吞吐量权衡依然非常明显。</p>

{% details 这段代码非常简单。 %}

以下是计算这些屋顶线的代码：

```py
import numpy as np

num_chips = 16  # we fix 16 as the amount of total model parallelism we do
bytes_per_param = 1  # int8 means 1 byte per param
param_count = 70e9
param_size = bytes_per_param * param_count
sequence_length = 8192  # can vary this

hbm_bandwidth = 8.20E+11  # v5e
flops = 1.97E+14  # v5e

def kv_cache_size(bs):
    return 2 * bs * 128 * 8 * 80

def min_topology(bytes):
    return 2 ** np.ceil(np.log2(bytes / 16e9))

def get_max_batch_size(
    num_chips: int,
    sequence_length: int,
    param_size: float,
) -> int:
  batch_sizes = np.arange(1, 1024, 4)
  kv_sizes = kv_cache_size(sequence_length * batch_sizes)
  required_chips = min_topology(kv_sizes + param_size)
  max_idx = np.where(required_chips <= num_chips)[0][-1]
  return max_idx

max_idx = get_max_batch_size(
    num_chips=num_chips,
    sequence_length=sequence_length,
    param_size=param_size,
)  # get the largest batch size that can fit
batch_sizes = np.arange(1, 512, 1)[:max_idx]
kv_sizes = kv_cache_size(sequence_length * batch_sizes)

kv_comms_time = kv_sizes / (num_chips * hbm_bandwidth)

param_comms_time = param_size / (num_chips * hbm_bandwidth)
param_comms_time = np.asarray([param_comms_time] * batch_sizes.shape[0])

flops_time = 2 * param_size * batch_sizes / (num_chips * flops)  # roughly true in a 2ND sense

mlp_time = np.maximum(flops_time, param_comms_time)
attn_time = kv_comms_time  # always bandwidth-bound for generate

latency = 1000 * (mlp_time + attn_time)
throughput = batch_sizes / (latency * num_chips)
```

注意我们如何明确地将延迟分解为两个来源：KV 加载和参数加载，并注意延迟要么受 FLOPs 限制，要么受通信限制，取决于哪一方更大。

{% enddetails %}

## 习题解答

这里有几个已解决的例题。其中一些重复了上面已经讲解过的内容，但可能在教学上有用。

**问题 1:** 每个词元的 LLaMA 3-405B 前向传递使用多少 FLOPs？假设我们受 FLOPs 限制，在 N 个 TPU v5e 芯片上单次前向传递的下限是多少？如果我们受通信限制呢？*忽略该模型无法在单个芯片上运行这一事实。*

**问题 2:** 假设我们想使用 int8 权重和 int8 KV 缓存以 BS240 的方式部署 LLaMA 3-8B。模型参数、KV 缓存和峰值工作激活值（大致）各占用多少字节？我们可以在哪种最小的拓扑结构上运行这个模型？

**问题 3:** 如何在 TPU v5e 上部署 LLaMA 3-405B？假设权重为 int8，FLOPs 为 bfloat16。假设我们对每个词元的延迟有严格的 15ms 限制，我们能实现的最高吞吐量配置是什么？理论上的最小步时间是多少？

**问题 4:** 了解大语言模型的最佳方式是从零开始实现一个。构建完整的训练流程既麻烦又昂贵，但将你可以从 [HuggingFace 下载](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/) 的训练权重转换为一个可用的推理实现是非常有启发性的。简而言之，你应该尝试完成以下步骤：

1. 下载 LLaMA 3 8B 权重并在 Colab 中加载它们。可视化权重（我推荐使用 [TreeScope](https://github.com/google-deepmind/treescope)），并尝试识别每个张量。统计参数的数量。它是否与你的预期一致？MLP 和注意力部分各有多少参数？

2. 实现模型的完整前向传播。你不需要担心预填充/解码或任何复杂的内容。只需实现到可以输入一个序列并输出合理的下一个词元概率的阶段。你应该能够输入一个提示，输出概率，选择概率最高的一个，并重复这个过程。这是一种速度较慢但功能正常的采样方式。尽量只以LLaMA-3论文作为参考，但[`jax-ml/jax-llm-examples/llama3`](https://github.com/jax-ml/jax-llm-examples/tree/main/llama3)仓库可以作为细节的良好参考（注意位置嵌入和因果掩码）。你的目标是从模型中输出连贯的词元。*如果你能获得超过16GiB的HBM，就可以在单个TPU上运行所有内容。*

3. 现在是时候让这个过程更快了。实现 KV 缓存，你可以将前向传递中的键/值激活值保存下来，并在后续的传递中对其进行注意力计算。这将大大加快你的采样循环。

4. 实现独立的预填充和解码服务器。可以在不同的芯片组上分别执行这些任务。预填充一次仅处理单个提示，然后将其发送到一个处理词元批次的批量解码服务器。

5. 在 Pallas 中实现 Flash Attention。这将使注意力机制更加高效。*我认为在没有参考的情况下尝试实现这一点是很好的。*

<h3 markdown=1 class="next-section">第 8 部分就到这里！第 9 部分将深入探讨 XLA 和 TPU 性能分析，点击[这里](../profiling)。</h3>
