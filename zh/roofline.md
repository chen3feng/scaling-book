---
layout: distill
title: "关于屋顶线的一切"
# permalink: /main/
description: "当我们在硬件上运行算法时，我们受到三方面限制：计算机执行数学运算的速度（OPs/秒）、用于数据传输的带宽（字节/秒）以及可用于存储数据的总内存（字节）。这些“屋顶线”约束让我们能够对给定计算的时间进行上下限界定。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 1

previous_section_url: ".."
previous_section_name: "Part 0: Introduction"

next_section_url: ../tpus
next_section_name: "Part 2: TPUs"

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

  - name: Where Does the Time Go?
  - subsections:
    - name: "Visualizing rooflines"
    - name: "Matrix multiplication"
    - name: "Network communication rooflines"
  - name: A Few Problems to Work

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

## 时间都去哪儿了？

让我们从一个极其简单的问题开始：*为什么一个算法需要 50 毫秒而不是 50 秒或 5 毫秒*？模型内部到底发生了什么，导致了显著的时间消耗，以及我们预期它需要多长时间？

**算力：** 深度学习模型本质上是一系列矩阵乘法，每一步都由浮点乘法和加法“操作”（FLOPs）组成。我们的加速器速度决定了这些操作的计算时间：

$$\begin{equation}
T_\text{math} = \frac{\text{Computation FLOPs}}{\text{Accelerator FLOPs/s}}
\end{equation}$$

例如，NVIDIA H100 可以实现约 9.89e14 bfloat16<d-footnote>bf16 是 <a href="https://en.wikipedia.org/wiki/Bfloat16_floating-point_format">bfloat16</a> 的缩写，一种常用于机器学习的 16 位浮点格式。</d-footnote> FLOPs/s，而 TPU v6e 可以实现 9.1e14 FLOPs/s。<d-footnote> H100 和 B200 通常只能达到标称峰值 FLOPs 的约 80-85%，而 TPU 在正常使用情况下可以接近 95%。</d-footnote> 这意味着在 H100 上执行 1e12 FLOPs 大约需要 `1e12 / 9.89e14 = 1.01ms` 和 `1e12 / 9.1e14 = 1.1ms` 在 TPU v6e 上完成。<d-footnote> 请注意，这些芯片的价格不同，此比较未按成本进行归一化处理。</d-footnote>

**芯片内部的通信：** *在加速器内部*，张量需要在加速器内存（HBM）和计算核心之间传输。你将会看到这条链路的带宽被称为“HBM 带宽”。<d-footnote>NVIDIA 也将其称为“内存带宽”。</d-footnote> 在 H100 上，[这大约是 3.35TB/s](https://www.nvidia.com/en-us/data-center/h100/)，在 TPU v6e 上，[这大约是 1.6TB/s](https://cloud.google.com/tpu/docs/v6e)。

**芯片间的通信：** 当我们将模型 *分布到* 多个加速器上时，张量经常需要在它们之间传输。我们的硬件上通常有几种选项（ICI、DCN 和 PCIe），每种选项的带宽都不同。

无论通信是在芯片内部还是芯片之间，我们均以字节/秒为单位进行测量，并使用以下公式估算总的通信时间：

$$\begin{equation}
T_\text{comms} = \frac{\text{Communication Bytes}}{\text{Network/Memory Bandwidth Bytes/s}}
\end{equation}$$

通常（但并非总是），单个芯片内的计算可以与芯片内及芯片间的通信重叠。这意味着**我们可以通过使用计算和通信时间的最大值来下界训练和推理时间**。我们也可以**用它们的总和作为上界**。在实践中，我们以最大值为目标进行优化，因为代数更简单，并且通过重叠通信和计算，通常可以接近这个界限。如果我们以最大值为目标进行优化，那么下界和上界之间的差异最多为2倍，因为$T_\text{math} + T_\text{comms} \leq 2 * \max(T_\text{math}, T_\text{comms})$。然后，我们通过建模“重叠区域”和开销来提高精度，这些信息可以通过对特定模型和目标系统进行性能分析来获得。

$$\begin{equation}
T_\text{lower}=\max(T_\text{math}, T_\text{comms})
\end{equation}$$

$$\begin{equation}
T_\text{upper} = T_\text{math} + T_\text{comms}
\end{equation}$$

如果我们假设可以完美地重叠通信和计算，当 $T_\text{math} > T_\text{comms}$ 时，我们会看到硬件的充分利用。我们将这种情况称为“算力受限”。当 $T_\text{comms} > T_\text{math}$ 时，我们往往处于“通信受限”<d-footnote>。在本书中，我们将“通信受限”、“comms-bound”、“内存受限”和“带宽受限”互换使用。</d-footnote> 并且至少有一部分我们的加速器 FLOPs/s 被浪费在等待数据传递上。判断一个操作是算力受限还是通信受限的一种方法是查看其“**算术强度**”或“**操作强度**”。

**定义：** 算法的算术强度由其执行的总 FLOPs 与需要通信的字节数之比给出——无论是芯片内部还是芯片之间。

$$\begin{equation}
\text{Arithmetic Intensity} = \frac{\text{Computation FLOPs}}{\text{Communication Bytes}}
\end{equation}$$

算力强度衡量的是给定操作的“每字节 FLOPs”。从第一阶来看，当我们的算力强度较高时，$T_\text{math}$ 相比 $T_\text{comms}$ 较大，我们通常会使用大部分可用的 FLOPs。当情况相反时，我们会花费更多时间在通信上，浪费了 FLOPs。这种转换发生的临界点是我们的硬件的“峰值算力强度”，即峰值加速器 FLOPs/s 与加速器带宽的比值。

$$\begin{align*}
T_\text{math} > T_\text{comms} \Leftrightarrow \frac{\text{Computation FLOPs}} {\text{Accelerator FLOPs/s}} > \frac{\text{Communication Bytes}}{\text{Bandwidth Bytes/s}} & \\[0.5em]
\Leftrightarrow \frac{\text{Computation FLOPs}}{\text{Communication Bytes}} > \frac{\text{Accelerator FLOPs/s}}{\text{Bandwidth Bytes/s}} & \\[0.5em]
\Leftrightarrow \text{Intensity}(\text{Computation}) > \text{Intensity}(\text{Accelerator}) & \\
\end{align*}$$

$\text{Intensity}(\text{Accelerator})$ 是我们的加速器达到峰值 FLOPs/s 时的算术强度。**对于 TPU v5e MXU 来说，这一数值约为 240 FLOPs/字节**，因为 TPU 每秒可以执行 `1.97e14` FLOPs，并且可以从 HBM 每秒加载 `8.2e11` 字节。<d-footnote> MXU 是 TPU 上的矩阵乘法单元。我们在这里说明这一点，是因为 TPU 还有其他加速器，如 VPU，它们负责执行逐元素操作，其峰值 FLOPs/s 不同。</d-footnote> 这意味着，如果一个算法的算术强度低于 240 FLOPs/字节，它将受到字节加载的限制，因此我们无法充分利用硬件。<d-footnote> 这仅在算法从 HBM 加载权重并在 MXU 中运行时才成立。正如我们将在下一节中讨论的那样，有时可以将参数存储在 VMEM 中，VMEM 的带宽要高得多。许多算法也运行在 VPU 中，VPU 具有不同的性能特征。</d-footnote> 让我们来看一个这样的例子：

**<span style="color:#7ab5ff">示例（点积）</span>:** 要在 bfloat16 精度下计算两个向量的点积 `x • y: bf16[N], bf16[N] → bf16[1]`，需要从内存中加载 $x$ 和 $y$，每个占用 $2 * N = 2N$ 字节，执行 $N$ 次乘法和 $N-1$ 次加法，然后将 $2$ 字节写回 HBM。
$$\begin{equation}
\text{Intensity}(\text{dot product}) = \frac{\text{Total FLOPs}}{\text{Total Bytes}} = \frac{N + N - 1}{2N + 2N + 2} = \frac{2N - 1}{4N + 2} \rightarrow \frac{1}{2}
\end{equation}$$

如 $N\rightarrow\infty$。因此，点积的算术强度为 $\frac{1}{2}$，或者说，点积每加载一个字节执行 0.5 次浮点运算。这意味着我们的算术强度低于硬件的强度，我们将受到通信的限制。<d-footnote>上面的 240 这个数字在这里并不是正确的比较对象，因为正如你将在下一节看到的，点积是在 VPU 上执行，而不是在 MXU 上。TPU v5p 的 VPU 每个核心每秒可以执行大约 7e12 FLOPs，因此其关键强度约为 3，这意味着我们在这里仍然在一定程度上受到通信的限制。不管怎样，我们的强度低且恒定这一事实意味着在大多数硬件上很难达到算力限制。</d-footnote>

### 可视化屋顶线

我们可以使用 **屋顶线图** 来可视化内存和算力之间的权衡，该图将算法在我们硬件上可达到的峰值 FLOPs/s（吞吐量）（y 轴）与其算术强度（x 轴）进行对比。以下是一个对数-对数图的示例：

{% include figure.liquid path="assets/img/roofline-improved.png" class="img-fluid" caption="<b>图：</b> 一个示例屋顶线图，展示了两种具有不同算术强度的算法（算法 1 和算法 2）以及它们在不同带宽（BW1 和 BW2）下的理论峰值吞吐量。在红色区域，算法在两种带宽下均受带宽限制，并浪费了硬件峰值 FLOPs/s 的一部分。黄色区域仅在较低带宽（BW1）下受带宽限制。绿色区域在所有带宽下均受算力限制。此处，我们使用加速器的峰值 FLOPs/s，增加带宽或提高强度不会带来任何好处。" %}

如上所述，随着强度的增加（从左到右），我们最初会看到算法性能（以 FLOPs/s 为单位）呈线性增长，直到达到硬件的关键算术强度，TPU v5e 的这一强度为 240。任何强度较低的算法都将受带宽（BW）限制，并受限于峰值内存带宽（以红色显示）。任何位于右侧的算法都将充分利用我们的 FLOPs（以绿色显示）。在这里，Algo 1 受通信限制，仅使用了总硬件 FLOPs/s 的一小部分。Algo 2 受算力限制。我们通常可以通过提高算法的算术强度或增加可用内存带宽（从 BW1 移动到 BW2）来提高算法的性能。

### 矩阵乘法

让我们来看我们即将喜爱的算法：矩阵乘法（又称 matmul）。给定两个矩阵 $X$ 和 $Y$，我们可以写成 $X * Y \rightarrow Z$，其中 $X$ 的形状为 $\text{bf16}[B, D]$，$Y$ 的形状为 $\text{bf16}[D, F]$，$Z$ 的形状为 $\text{bf16}[B, F]$。为了执行矩阵乘法，我们需要加载 $2DF + 2BD$ 字节，执行 $2BDF$ FLOPs，并将 $2BF$ 字节写回。<d-footnote>从技术上讲，我们执行了 $BF \times (2D - 1)$ FLOPs，但这个数值已经足够接近。这来自于 $BDF$ 次乘法和 $BF * (D-1)$ 次加法。第 4 节有更多细节。</d-footnote> <d-footnote>虽然矩阵乘法的输出在技术上是 float32，但我们通常在将其复制回 HBM 之前将其转换为 bfloat16。</d-footnote> 因此：

$$\begin{equation}
\text{Intensity}(\text{matmul}) = \frac{2BDF}{2BD + 2DF + 2BF} = \frac{BDF}{BD + DF + BF}
\end{equation}$$

如果我们假设我们的“批量大小”$B$相对于$D$和$F$来说较小，那么我们可以得到一个很好的简化。然后我们得到

$$\begin{equation}
\frac{BDF}{BD + DF + BF} \approx \frac{BDF}{DF} = B
\end{equation}$$

$$\begin{equation}
\text{Intensity}(\text{matmul}) > \text{Intensity}(\text{TPU}) \implies B > \frac{1.97e14}{8.20e11} = 240
\end{equation}$$

这是对 Transformer 矩阵乘法的一个合理假设，因为我们通常有一个本地（每个副本）的词元批量大小 $B < 1024$（*注意，不是序列*），但 $D$ 和 $F > 8000$。因此，当我们的每个副本 <d-footnote> 我们说每个副本，因为我们如果对模型进行分片以增加矩阵乘法中使用的芯片数量，我们会按相同的比例扩展可用算力和显存带宽。因此，关键批量大小是每个模型权重独立副本的真实值。</d-footnote> 当批量大小大于 240 个词元时，这是一个非常简单的规则！

<p markdown=1 class="takeaway">**要点：** 要使 bfloat16 矩阵乘法在大多数 TPU 上成为算力受限的，我们需要将批量大小设置为大于 240（词元）。<d-footnote>请注意，这并不是通常意义上的批量大小，通常意义上的批量大小指的是序列的批量大小。事实证明，大多数性能上限完全取决于词元的数量，无论这些词元属于同一序列还是不同序列。例如，如果你在 2048 个 GPU 上有 512 个序列，每个序列包含 4096 个词元，那么你总的批量大小为 `512 * 4096 = 2M` 个词元，而本地批量大小为 1k 个词元。</d-footnote></p>

这伴随着一些值得注意的限制条件，我们将在下面的问题中探讨，特别是关于量化（例如，如果我们量化了激活值但仍使用全精度 FLOPs），但这是一个值得记住的规则。对于 GPU 来说，这个数值略高（接近 300），但总体结论仍然成立。当我们[将一个大的矩阵乘法分解为较小的矩阵乘法](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html#your-first-matrix-multiplication-kernel)时，瓦片大小也很重要。<d-footnote>当我们进行大矩阵乘法时，我们需要将其分解为能够放入 VMEM/SMEM/TMEM（高带宽片上内存）的更小瓦片。这会导致我们多次加载数据块，因此不再完全正确地说我们只加载 $O(N^2)$ 字节。考虑一个 $(m, k) \cdot (k, n)$ 矩阵乘法，其瓦片大小为 $bm$、$bk$、$bn$。设 $tm = m / bm$ 等。那么总的 FLOPs 是 $2 \cdot tm \cdot tn \cdot tk \cdot bm \cdot bn \cdot bk$，总的字节数是 $2 \cdot tm \cdot tn \cdot (tk \cdot (bm \cdot bk + bk \cdot bn) + bm \cdot bn)$。忽略最后一项，我们得到一个强度为 $bm \cdot bn / (bm + bn)$，这与上面的数值类似。</d-footnote> 我们将在[下一节](../tpus)中讨论更底层的 GPU 和 TPU 细节。

### 网络通信上限

到目前为止，我们讨论的所有性能极限线都是显存带宽性能极限线，_且都局限于单个芯片内部_。这不应被当作一条规则。事实上，本书中我们将关注的大多数性能极限线涉及芯片之间的通信：通常是涉及跨多个TPU分片的矩阵相乘操作。

为了举一个稍微有些人为的例子，假设我们想要相乘两个大矩阵 $X\sim \text{bf16}[B, D]$ 和 $Y \sim \text{bf16}[D, F]$，它们在 $D$ 维度上被均匀地分配到 2 个 TPU/GPU 上。为了完成这个乘法（我们将在 [第 3 章](../sharding) 中看到），我们可以在每个 TPU 上分别计算每个矩阵的一半（TPU 0 上计算 `Z0 = X[:, :D // 2] @ Y[:D // 2, :]`，TPU 1 上计算 `Z1 = X[:, D // 2:] @ Y[D // 2:, :]`），然后将得到的“部分和”复制到另一个 TPU 上并相加。假设我们每个方向的复制速度为 `4.5e10` 字节/秒，每个芯片的计算能力为 `1.97e14` FLOPs/秒。那么 $T_\text{math}$ 和 $T_\text{comms}$ 分别是什么？

$T_\text{math}$ 显然是之前的二分之一，因为每个 TPU 只完成了原来一半的工作，即 <d-footnote>。我们忽略了将两个部分和相加所需的 FLOPs（另一个 BF 加法），但这一点基本上可以忽略不计。</d-footnote>

$$T_\text{math} = \frac{2BDF}{2 \cdot \text{Accelerator FLOPs/s}} = \frac{BDF}{1.97e14}$$

现在 $T_\text{comms}$ 指的是芯片之间的通信时间！这仅仅是总发送字节数除以网络带宽，即

$$T_\text{comms} = \frac{2BF}{\text{Network Bandwidth}} = \frac{2BF}{4.5e10}$$

因此，当 $$\text{Intensity}(\text{matmul (2-chips)}) > \text{Intensity}(\text{TPU w.r.t. inter-chip network})$$ 或等效地当 $\frac{BDF}{2BF} = \frac{D}{2} > \frac{1.97e14}{4.5e10} = 4377$ 或 $D > 8755$ 时，我们变得受算力限制（现在是相对于芯片间的网络）。请注意，与之前不同，临界阈值现在取决于 $D$ 而不是 $B$！试着思考为什么会出现这种情况。这只是其中一个例子，但我们强调，这种屋顶线对于了解何时可以跨多个 TPU 并行执行操作至关重要。

## 一些需要解决的问题

**问题 1 [int8 matmul]:** 假设我们想用 int8 精度（每个参数 1 字节）而不是 bfloat16（每个参数 2 字节）来进行 matmul $X[B, D] \cdot_D Y[D, F] \rightarrow Z[B, F]$<d-footnote>。这里及之后我们将使用 $A \cdot_D B$ 表示法，表示乘法在 D 维度上进行收缩。这是一种对 einsum 符号的误用。</d-footnote> 因为 TPUs/GPUs 在更低精度下可以更快地执行 matmul。

1. 需要从内存中加载多少字节？需要写回内存多少字节？  
2. 总共执行了多少个操作（OP）？  
3. 计算强度是多少？  
4. $T_\text{math}$ 和 $T_\text{comms}$ 的屋顶线估计是什么？整个操作的运行时间的合理上限和下限是什么？

假设我们的 HBM 带宽为 `8.2e11` 字节/秒，我们的 int8 峰值 OPs/s 为 `3.94e14`（约为 bfloat16 的 2 倍）。

{% details 点击此处查看答案。 %}

1. 因为我们使用 int8 存储参数，每个参数占用 1 字节，因此从 HBM 加载了 $$BD + DF$$ 字节，并写回了 $$BF$$ 字节。  
2. 这与 bfloat16 的情况相同，但理论上 int8 的 OPs/s 应该更快。因此，这仍然是 $2BDF$ OPs。  
3. 算术强度为 $$2BDF / (BD + DF + BF)$$。如果我们对 $$B \ll D$$ 和 $$B \ll F$$ 做出与上述相同的假设，得到的算术强度为 $$2B$$，这意味着我们的规则变为 $B > \text{HBM int8 arithmetic intensity} / 2$。使用给出的数值，这个 int8 强度为 `3.94e14 / 8.2e11 = 480`，因此规则为 $B > 480 / 2 = 240$。请注意，这基本上没有变化！  
4. $$T_\text{math} = 2BDF / 3.94e14$$ 和 $$T_\text{comms} = (BD + DF + BF) / 8.2e11$$，因此一个合理的下界是 $$\max(T_\text{math}, T_\text{comms})$$，上界是 $$T_\text{math} + T_\text{comms}$$。

{% enddetails %}

**问题 2 [int8 + bf16 matmul]:** 实际应用中，我们经常对权重和激活值进行不同的量化，因此可能会将权重存储在非常低的精度中，但保持激活值（以及计算）在更高的精度中。假设我们希望将权重量化为 int8，但将激活值（以及计算）保持在 bfloat16。在什么批量大小时，我们会受到算力的限制？假设 `1.97e14` bfloat16 FLOPs/s。

*提示：这特指 `bf16[B, D] * int8[D, F] -> bf16[B, F]`，其中 $B$ 是“批量大小”。*

{% details 点击此处查看答案。 %}

再次假设 B 很小，我们有 2BDF bfloat16 FLOPs，但只有 DF 权重（而不是 bfloat16 中的 2DF）。这意味着当 $$2B > 240$$ 或 $$B > 120$$ 时，我们会变得受算力限制。这要低得多，这意味着如果我们能进行 int8 权重量化（这相对容易实现），但仍然使用 bfloat16 FLOPs，我们能在效率上获得显著提升（尽管 int8 操作会更好）。

{% enddetails %}

**问题 3:** 在问题 2 的设置基础上，为 $F = D = 4096$ 和 $F = D = 1024$ 绘制一个峰值 FLOPs/s 与 $B$ 的屋顶线图。*使用精确的字节数加载量，而非近似值。*

{% details 点击此处查看答案。 %}

这里就是所讨论的图表：

{% include figure.liquid path="assets/img/roofline-plot-q3.png" class="img-fluid img-small" %}

请注意，两种模型最终都达到了峰值硬件 FLOPs/s，但更大的 D/F 更早达到这一水平。D=F=1024 几乎使临界批量大小翻倍。生成此图的代码在这里：

```py
import matplotlib.pyplot as plt
import numpy as np

bs = np.arange(1, 512)

def roofline(B, D, F):
  total_flops = 2*B*D*F
  flops_time = total_flops / 1.97e14
  comms_time = (2*B*D + D*F + 2*B*F) / 8.2e11
  total_time = np.maximum(flops_time, comms_time)
  return total_flops / total_time

roofline_big = roofline(bs, 4096, 4096)
roofline_small = roofline(bs, 1024, 1024)

plt.figure(figsize=(8, 4))
plt.plot(bs, roofline_big, label='F=D=4096')
plt.plot(bs, roofline_small, label='F=D=1024')
plt.legend()
plt.xlabel('batch size')
plt.ylabel('peak bfloat16 FLOPs/s on TPU v5e')
plt.grid()
```

{% enddetails %}

**问题 4:** 如果我们想要执行 $\text{int8}[B, D] \cdot_D \text{int8}[B, D, F] \rightarrow \text{int8}[B, F]$，其中我们想象每个批量元素都有一个不同的矩阵。该操作的算术强度是多少？

{% details 点击此处查看答案。 %}

让我们先看一下总 FLOPs 和通信。

1. 总 FLOPs：FLOPs 基本相同，因为我们正在做 $$B$$ 个独立的 $$[D] \times [D, F]$$ 产品，总工作量与单个 $$[B, D] \times [D, F]$$ 矩阵乘法相同（这在第 4 节中讨论得更详细）。因此，这仅仅是 $$2BDF$$。

2. 总通信量：我们这里的通信量要多得多：$$BD + BDF + BF$$。

3. 因此，我们的算术强度现在实际上是 $$2BDF / (BD + BDF + BF)$$。由于 $$BDF$$ 主导了分母，这大致是 $$2$$。因此，它不再依赖于批量大小，而是基本上保持恒定。这很糟糕，因为它意味着无论怎样，我们基本上总是会受到通信的限制。

{% enddetails %}

**问题 5 [GPU 的内存屋顶线]：** 使用 NVIDIA 提供的 [H100 SXM 的规格表](https://www.nvidia.com/en-us/data-center/h100/)，计算 bfloat16 矩阵乘法变为算力受限的批量大小。*请注意，Tensor Core 的 FLOPs 数值是真实值的两倍，因为它们仅在结构化稀疏性下可实现。*

{% details 点击此处查看答案。 %}

从规格表中可以看到，报告的 bfloat16 FLOPs 值为 `1.979e15` FLOPs/s，并带有注释“带有稀疏性”的星号。在没有稀疏性的情况下，真实值是这个值的一半，即 `9.89e14` FLOPs/s。内存带宽为 3.35TB/s，即 `3.35e12` 字节/秒。因此 $B_\text{crit}$ 是 `9.89e14 / 3.35e12 = 295`，与 TPU 相当相似。

{% enddetails %}

<h3 markdown=1 class="next-section">第一部分就到这里！第二部分将探讨真实TPU如何处理FLOPs和通信，[点击此处](../tpus)。</h3>
