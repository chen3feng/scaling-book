---
layout: distill
title: "分片矩阵及其乘法方法"
# permalink: /main/
description: "当我们训练大型机器学习模型时，必须将它们的参数或输入分布在许多加速器上。由于大语言模型主要由矩阵乘法组成，理解这一点归结为理解如何在设备之间分割矩阵时进行矩阵乘法。我们基于 TPU 通信原语的成本，开发了一种关于分片矩阵乘法的简单理论。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 3

previous_section_url: "../tpus"
previous_section_name: "Part 2: TPUs"

next_section_url: ../transformers
next_section_name: "Part 4: Transformer Math"

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
  - name: "Partitioning Notation and Collective Operations"
  - subsections:
    - name: "A unified notation for sharding"
    - name: "How do we describe this in code?"
  - name: "Computation With Sharded Arrays"
  - subsections:
    - name: "Case 1: neither multiplicand has a sharded contracting dimension"
    - name: "Case 2: one multiplicand has a sharded contracting dimension"
    - name: "Case 3: both multiplicands have sharded contracting dimensions"
    - name: "Case 4: both multiplicands have a non-contracting dimension sharded along the same axis"
  - name: "A Deeper Dive into TPU Communication Primitives"
  - subsections:
    - name: "Our final communication primitive: the AllToAll"
    - name: "More about the ReduceScatter"
    - name: "How to overlap matmul communication with compute"
  - name: "What Have We Learned?"
  - name: "Some Problems to Work"

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

## 分区符号与集合操作

当我们在一万块 TPU 或 GPU 上训练一个大语言模型时，本质上与在一块芯片上训练所进行的计算是一样的。不同之处在于 **我们的数组无法放入单个 TPU/GPU 的 HBM 中**，因此我们必须将它们拆分。<d-footnote> 值得注意的是，我们可能也会为了提高速度而选择并行化。即使我们能够将模型放入更少的芯片上，扩展到更多芯片上可以简单地为我们提供更多的 FLOPs/s。例如，在推理过程中，我们有时可以将模型放入更小的拓扑结构中，但为了减少延迟，我们可能选择扩展到更大的拓扑结构。同样地，在训练过程中，我们经常扩展到更多的芯片上以减少每一步的时间。</d-footnote> 我们将这种操作称为“*分片*”或“*分区*”我们的数组。缩放的艺术在于弄清楚如何对模型进行分片，以保持计算的高效性。

这是跨 4 个 TPU 分片的二维数组 **A** 的一个示例：

{% include figure.liquid path="assets/img/sharding-example.png" class="img-fluid" caption="<b>图：</b> 一个形状为 <b>A</b>[I, J] 的数组被分片到 4 个设备上。两个维度均均匀地分片到 2 个设备上，分片方式为 <b>A</b>[I<sub>X</sub>, J<sub>Y</sub>]。每个 TPU 保存总显存的 1/4。" %}

注意分片数组仍然具有与未分片数组（如 `(4, 128)`）相同的 *全局* 或 *逻辑形状*，但它还具有一个 *设备本地形状*（如 `(2, 64)`），这给出了每个 TPU 实际持有的字节数（在上图中，每个 TPU 持有整个数组的 ¼）。现在我们将这一概念推广到任意数组。

### 一种统一的分片表示法

我们使用一种 *命名轴符号法* 的变体，来描述张量如何在设备之间以块的形式进行分片：我们假设存在一个称为 **设备网格** 的 2D 或 3D 设备网格，其中每个轴都被赋予了 **网格轴名称**，**例如 X**、**Y 和 Z**。然后，我们可以通过描述数组中每个命名维度如何在物理网格轴上进行划分，来指定矩阵数据在设备网格中的布局。我们将这种分配方式称为 **分片**。

**示例（上方的图表）**：对于上方的图表，我们有：
* **网格（Mesh）**：上方的 `Mesh(devices=((0, 1), (2, 3)), axis_names=('X', 'Y'))`，它告诉我们我们有一个 2x2 的网格，其中有 4 个 TPUs，轴名为 $X$ 和 $Y$。
* **分片（Sharding）**：$A[I_X, J_Y]$，它告诉我们沿着网格轴 $X$ 对第一个轴 $I$ 进行分片，沿着网格轴 $Y$ 对第二个轴 $J$ 进行分片。这种分片方式告诉我们每个分片保存了数组的 $1 / (\lvert X\rvert \cdot \lvert Y\rvert)$。

综合来看，我们了解到数组的局部形状（单个设备所持有的分片的大小）为 $(\lvert I\rvert / 2, \lvert J\rvert / 2)$，其中 $$\lvert I\rvert$$ 是 A 的第一个维度的大小，$$\lvert J\rvert$$ 是 A 的第二个维度的大小。

<b markdown=1 style="color: #048affff;">快速测试 [2D 分片跨越 1 个轴]:</b> 考虑一个数组 `fp32[1024, 4096]`，其分片方式为 $A[I_{XY}, J]$，网格为 `{'X': 8, 'Y': 2}`。每个设备持有多少数据？从 HBM 加载此数组到 H100 上需要多长时间（假设每块芯片的内存带宽为 `3.4e12`）？

{% details 点击此处查看答案。 %}

$A[I_{XY}, J]$ 沿 X 和 Y 硬件轴对第一个维度（I）进行分片。在此示例中，本地形状为 $(\lvert I\rvert /(\lvert X\rvert \cdot \lvert Y\rvert), \lvert J\rvert)$。对于给定的示例，全局形状为 `fp32[1024, 4096]`，因此本地形状为 `fp32[64, 4096]`。

由于每块 GPU 有 `4 * 64 * 4096 = 1MiB` 字节，这将需要大约 `1e6 / 3.4e12 = 294ns`，但由于各种开销，实际所需可能显著更多，因为这个数值非常小。

{% enddetails %}

**可视化这些分片：** 让我们通过观察一个跨 4 个设备分割的二维数据数组来尝试可视化这些分片：

{% include figure.liquid path="assets/img/sharding-colored1.png" class="img-fluid img-small" %}

我们将矩阵的 *fully-replicated* 形式简单地写为 $A[I, J]$，不进行任何分片分配。这意味着 *每个* 设备都包含整个矩阵的完整副本。

{% include figure.liquid path="assets/img/sharding-colored2.png" class="img-fluid img-small" %}

我们可以用带下标 mesh 轴的符号来表示这些维度中的某一个已经在 mesh 轴上被划分。例如 $A[I_X, J]$ 表示 **I** 逻辑轴已经在 **X** mesh 维度上被划分，但 **J** 维度并未被划分，而块在 **Y** mesh 轴上保持 *部分复制*。

{% include figure.liquid path="assets/img/sharding-colored3.png" class="img-fluid img-small" %}

$A[I_X, J_Y]$ 表示 **I** 逻辑轴已在 **X** 网格轴上进行了分区，且 **J** 维度已在 **Y** 网格轴上进行了分区。

{% include figure.liquid path="assets/img/sharding-colored4.png" class="img-fluid img-small" %}

我们将在下图中说明其他可能性：

{% include figure.liquid path="assets/img/sharding-colored5.png" class="img-fluid" %}

此处 $A[I_{XY}, J]$ 表示我们将 **X** 和 **Y** 网格轴视为一个更大的展平维度，并将 **I** 命名轴在所有设备之间进行划分。多个网格轴下标的顺序很重要，因为它指定了在网格上划分的遍历顺序。

{% include figure.liquid path="assets/img/sharding-colored6.png" class="img-fluid img-small" %}

最后，请注意我们 *不能* 沿着 *同一个* mesh 维度对多个命名轴进行分片。例如，$A[I_X, J_X]$ 是一个毫无意义且被禁止的分片方式。一旦某个 mesh 维度被用来对数组的一个维度进行分片，从某种意义上来说，这个维度就“被用完了”。

<b markdown=1 style="color: #57cf57;">小测验：</b> 设 **A** 是一个形状为 `int8[128, 2048]`、分片为 $A[I_{XY}, J]$、网格为 `Mesh({'X': 2, 'Y': 8, 'Z': 2})`（共 32 个设备）的数组。**A** 每个设备使用多少内存？**A** 在所有设备上总共使用多少内存？

{% details 点击此处查看答案。 %}

**答案：** 我们的数组 **A** 在 X 和 Y 上进行分片，在 Z 上进行复制，因此每个设备上的形状为 `int8[128 / (2 * 8), 2048] = int8[8, 2048]`，大小为 `8 * 2048 = 16,384` 字节。由于它在 Z 上进行了复制，而在每个 Z 平面内它在 X 和 Y 上完全分片，因此原始数组有两个完整的副本（每个 Z 平面一个）。因此，所有设备上的总大小为：原始数组大小 × Z 副本数 = 128 * 2048 * 2 = 512 KiB 总计。或者，我们也可以这样验证：32 个设备 × 每个设备 16,384 字节 = 512 KiB 总计。

{% enddetails %}

### 我们如何在代码中描述这一点？

到目前为止，我们一直避免谈论代码，但现在是窥探一下的好时机。JAX 使用一种命名分片语法，与我们上面描述的抽象语法非常接近。我们将在[第 10 章](../jax-stuff)中更详细地讨论这一点，但这里先做个简短预览。你可以在 [这里](https://colab.research.google.com/drive/15cxw66eABwZPG-V4QFmbLfiykPFf_gaP?usp=sharing) 的 Google Colab 中尝试这段代码，并对结果进行性能分析，以查看 JAX 如何处理不同的分片方式。这段代码做了三件事：

1. 创建一个 **jax.Mesh**，将我们的 8 个 TPU 映射为一个 4x2 网格，并将两个轴分别命名为 'X' 和 'Y'。
2. 创建矩阵 A 和 B，其中 A 在两个维度上进行分片，而 B 在输出维度上进行分片。
3. 编译并执行一个简单的矩阵乘法，返回一个分片数组。

```py
import jax
import jax.numpy as jnp

# Create our mesh! We're running on a TPU v2-8 4x2 slice with names 'X' and 'Y'.
# The Auto axis type tells JAX to let the XLA compiler infer intermediate shardings.
assert len(jax.devices()) == 8
Auto = jax.sharding.AxisType.Auto
mesh = jax.make_mesh(axis_sizes=(4, 2), axis_names=('X', 'Y'), axis_types=(Auto, Auto))

# A little utility function to help define our sharding. A PartitionSpec is our
# sharding (a mapping from axes to names).
def P(*args):
  return jax.NamedSharding(mesh, jax.sharding.PartitionSpec(*args))

# We shard both A and B over the non-contracting dimension and A over the contracting dim.
A = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=P('X', 'Y'))
B = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=P(None, 'Y'))

# We can perform a matmul on these sharded arrays! out_shardings tells us how we want
# the output to be sharded. JAX/XLA handles the rest of the sharding for us.
y = jax.jit(lambda A, B: jnp.einsum('BD,DF->BF', A, B), out_shardings=P('X', 'Y'))(A, B)
```

JAX 的妙处在于，这些数组表现得就好像它们是未分片的一样！`B.shape` 会告诉我们全局或逻辑形状（2048, 8192）。我们必须实际查看 `B.addressable_shards` 才能了解它是如何在本地进行分片的。我们可以对这些数组执行操作，JAX 会尝试推断出如何对它们进行广播或重塑以执行这些操作。例如，在上面的例子中，**A** 的本地形状是 `[2, 1024]`，而 **B** 的本地形状是 `[2048, 4096]`。JAX/XLA 会根据需要在这些数组之间自动添加通信，以完成最终的乘法运算。

## 使用分片数组进行计算

如果你拥有一组分布在许多设备上的数据，并希望对其执行数学运算，那么对数据和计算进行分片会带来哪些开销？

显然，这取决于所涉及的计算。

* 对于*逐元素*操作，在分布式数组上进行操作**没有开销**。  
* 当我们希望对驻留在许多设备上的元素执行操作时，事情会变得复杂。幸运的是，对于大多数机器学习任务而言，几乎所有计算都以矩阵乘法的形式进行，而它们相对简单，易于分析。

本节其余部分将讨论如何相乘分片矩阵。粗略地说，这涉及移动矩阵的块，以便能够完全相乘或求和每个块。**每种分片方式将涉及不同的通信。** 例如，$A[I_X, J] \cdot B[J, K_Y] \to C[I_X, K_Y]$ 可以在不进行任何通信的情况下相乘，因为 *收缩维度*（J，我们实际求和的维度）是未分片的。然而，如果我们希望输出未分片（即 $A[I_X, J] \cdot B[J, K_Y] \to C[I, K]$），则需要将 $A$ 和 $B$ 或 $C$ 复制到每个设备上（使用 *AllGather*）。这两种选择具有不同的通信成本，因此我们需要计算这种成本并选择最低的一种。

{% details 你可以从“块矩阵乘法”的角度来理解这一点。 %}

要理解这一点，回忆“块矩阵”或矩阵的嵌套矩阵的概念可能会有帮助：

$$\begin{equation}
\begin{pmatrix}
a_{00} & a_{01} & a_{02} & a_{03} \\
a_{10} & a_{11} & a_{12} & a_{13} \\
a_{20} & a_{21} & a_{22} & a_{23} \\
a_{30} & a_{31} & a_{32} & a_{33}
\end{pmatrix}
=
\left(
\begin{matrix}
\begin{bmatrix}
a_{00} & a_{01} \\
a_{10} & a_{11}
\end{bmatrix} \\
\begin{bmatrix}
a_{20} & a_{21} \\
a_{30} & a_{31}
\end{bmatrix}
\end{matrix}
\begin{matrix}
\begin{bmatrix}
a_{02} & a_{03} \\
a_{12} & a_{13}
\end{bmatrix} \\
\begin{bmatrix}
a_{22} & a_{23} \\
a_{32} & a_{33}
\end{bmatrix}
\end{matrix}
\right)
=
\begin{pmatrix}
\mathbf{A_{00}} & \mathbf{A_{01}} \\
\mathbf{A_{10}} & \mathbf{A_{11}}
\end{pmatrix}
\end{equation}$$

矩阵乘法具有一个很好的性质，即当矩阵乘数以块的形式表示时，其乘积可以按照标准规则以块矩阵乘法的形式表示：

$$\begin{equation}
\begin{pmatrix}
A_{00} & A_{01} \\
A_{10} & A_{11}
\end{pmatrix}
\cdot
\begin{pmatrix}
B_{00} & B_{01} \\
B_{10} & B_{11}
\end{pmatrix}
=
\begin{pmatrix}
A_{00}B_{00} + A_{01}B_{10} & A_{00}B_{01} + A_{01}B_{11} \\
A_{10}B_{00} + A_{11}B_{10} & A_{10}B_{01} + A_{11}B_{11}
\end{pmatrix}
\end{equation}$$

这意味着实现分布式矩阵乘法归结为将这些分片块通过网络传输，在块上执行*本地*矩阵乘法，并对结果求和。**问题在于需要添加哪些通信操作，以及这些操作的代价有多大。**

{% enddetails %}

方便的是，我们可以将所有可能的分片方式简化为大约 4 种需要考虑的情况，每种情况都有一个规则，说明需要添加哪些通信操作。  
1. **[情况 1](#case-1-neither-multiplicand-has-a-sharded-contracting-dimension):** 输入在收缩维度上均未被分片。_我们可以在不进行任何通信的情况下直接相乘本地分片。_  
2. **[情况 2](#case-2-one-multiplicand-has-a-sharded-contracting-dimension):** 一个输入的收缩维度被分片。_我们通常沿收缩维度对分片输入执行“AllGather”操作。_  
3. **[情况 3](#case-3-both-multiplicands-have-sharded-contracting-dimensions):** 两个输入的收缩维度均被分片。_我们可以相乘本地分片，然后对结果执行“AllReduce”操作。_  
4. **[情况 4](#case-4-both-multiplicands-have-a-non-contracting-dimension-sharded-along-the-same-axis):** 两个输入的非收缩维度沿同一轴被分片。_在不先对其中一个输入执行“AllGather”操作的情况下，我们无法继续进行。_

你可以将这些视为只需遵循的规则，但了解这些规则为何成立以及它们的成本有多高也同样有价值。我们现在将逐一详细讲解这些规则。

### 案例 1：两个乘数均没有分片的收缩维度

**引理：** 当相乘分片矩阵时，计算是有效的，输出遵循输入的分片方式 *除非* 合同维度被分片或两个矩阵沿同一轴被分片。例如，这可以正常工作

$$\begin{equation*}
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y] \rightarrow \mathbf{C}[I_X, K_Y]
\end{equation*}$$

完全不进行任何通信，结果是一个在 X 和 Y 硬件维度上都进行了张量分片的张量。试着思考一下为什么会出现这种情况。基本上，计算与分片是*独立*的，因为每个批量条目都可以乘以并减少它所对应的被收缩轴的局部块。这些情况中的任何一种都可以正常工作，并遵循这一规则：

$$\begin{align*}
\mathbf{A}[I, J] \cdot \mathbf{B}[J, K] \rightarrow &\ \mathbf{C}[I, K] \\
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K] \rightarrow &\ \mathbf{C}[I_X, K]\\
\mathbf{A}[I, J] \cdot \mathbf{B}[J, K_Y] \rightarrow &\ \mathbf{C}[I, K_Y]\\
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y] \rightarrow &\ \mathbf{C}[I_X, K_Y]
\end{align*}$$

由于 **A** 和 **B** 都没有沿分片维度 **J** 的收缩维度，因此我们只需对输入执行本地块矩阵乘法，结果 *已经* 按照期望的输出分片方式进行了分片。当两个乘数的非收缩维度沿同一轴进行分片时，这一情况不再成立（详情请参见[无效分片](#case-4-both-multiplicands-have-a-non-contracting-dimension-sharded-along-the-same-axis)部分）。

### 案例 2：一个乘数具有分片的收缩维度

让我们考虑当一个输入 **A** 沿着收缩的 **J** 维度进行分片，而 **B** 完全复制时应怎么做：

$$\mathbf{A}[I, J_X] \cdot \mathbf{B}[J, K] \rightarrow \mathbf{C}[I, K]$$

我们不能简单地将 **A** 和 **B** 的本地块相乘，因为我们需要对 **A** 的完整收缩维度求和，而该维度在 X 轴上被分割。通常，我们首先对 **A** 的分片执行 "**AllGather**"，使每个设备都拥有完整的副本，然后才与 **B** 相乘：

$$\textbf{AllGather}_X[I, J_X] \rightarrow \mathbf{A}[I, J]$$

$$\mathbf{A}[I, J] \cdot \mathbf{B}[J, K] \rightarrow \mathbf{C}[I, K]$$

这样，实际的乘法就可以在每个设备上完全执行。

<p markdown=1 class="takeaway">**要点：** 当其中一个矩阵沿收缩维度进行分片时，我们通常先对其进行 AllGather 操作，使收缩操作不再分片，然后再执行本地矩阵乘法。</p>

请注意，当 **B** 沿 X 方向未被分片时，我们也可以先执行本地部分矩阵乘法，然后对分片的部分和进行求和（或 *AllReduce*），这使我们能够对计算进行分片，但通常通信成本更高。在某些情况下这可能更快，尽管在实践中 **B** 通常会被分片。问题 4 [下方](#some-problems-to-work) 会探讨在什么情况下这种方法更优。

**什么是 AllGather？** AllGather 是我们首先讨论的核心 [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) 通信原语。AllGather 沿着一个轴 *移除分片*，并将分布在各个设备上的分片重新组装到该轴上的 *每个* 设备上。使用上述符号表示法，AllGather 会从一组轴中移除一个下标，例如：

$$\textbf{AllGather}_{XY}(A[I_{XY}, J]) \rightarrow A[I, J]$$

我们不需要为给定维度移除所有下标，例如 $$A[I_{XY}, J] \rightarrow A[I_Y, J]$$ 也是一个 AllGather，只是仅在一个轴上进行。还请注意，我们可能也希望使用 AllGather 来移除非收缩维度的分片，例如在矩阵乘法中：

$$A[I_X, J] \cdot B[J, K] \rightarrow C[I, K]$$

我们可以在最初对 **A** 进行 AllGather 以消除输入分片，或者我们可以进行分片矩阵乘法，然后对结果 **C** 进行 AllGather。

**AllGather 是如何实际执行的？** 为了在单个 TPU 轴（一个环）上执行一维 AllGather，我们基本上让每个 TPU 将其分片在环中传递，直到每个设备都拥有一份副本。<d-footnote> 一个 GPU AllGather 也可以这样工作，即你从节点中的 GPU 创建一个环，并按照该（任意）顺序传递数据块。</d-footnote> 这里有一个动画：

{% include figure.liquid path="assets/img/all-gather.gif" caption="<b>图：</b> 一个动画演示如何在一组8个TPU或GPU设备上执行AllGather操作。每个设备最初拥有数组的1/8，最终会获得完整的副本。" %}

我们可以在一个方向上执行 AllGather，也可以在两个方向上执行（上面显示了两个方向）。如果我们执行一个方向，每个 TPU 会通过环形网络中的 $N - 1$ 跳步，发送大小为 $\text{bytes} / N$ 的数据块。如果我们执行两个方向，我们将有大小为 $2 \cdot \text{bytes} / N$ 的 $\lfloor \frac{N}{2} \rfloor$ 跳步。

**这需要多长时间？** 让我们以双向 AllGather 为例，计算它需要多长时间。设 $$V$$ 为数组中的字节数，$X$ 为收缩维度上的分片数量。那么根据上述图示，每次跳转在每个方向上传输 $V / \lvert X\rvert$ 字节，因此每次跳转需要

$$T_{hop} = \frac{2 \cdot V}{\lvert X \rvert \cdot W_\text{ici}}$$

其中 $W_\text{ici}$ 是 **双向** 的 ICI 带宽。<d-footnote>分子中的因子 2 来源于我们使用的是双向带宽。我们在每个方向发送 $V / X$，总共发送 $2V / X$。</d-footnote> 要到达每个 TPU，总共需要发送 $\lvert X\rvert / 2$ 次跳转<d-footnote>技术上讲，是 $\lfloor X / 2 \rfloor$</d-footnote>，因此总的缩减操作需要

$$T_{total} = \frac{2 \cdot V \cdot X}{2 \cdot X \cdot W_\text{ici}}$$

$$T_{total} = \frac{V}{W_\text{ici}}$$

请注意，这**与 $X$ 无关！** 这有点令人惊讶，因为它意味着即使我们的 TPUs 仅在本地连接，连接的局部性也不重要。我们只是受限于每个链路的速度。

<p markdown=1 class="takeaway">**要点：** 在吞吐量受限的场景下执行 AllGather（或 ReduceScatter 或 AllReduce）时，实际的通信时间仅取决于数组的大小和可用带宽，而与数组被分片到的设备数量无关！</p>

**关于 ICI 延迟的说明：** 每次通过 ICI 链路的跳转都会有一些固有的开销，与数据量无关。这通常约为 1 微秒。这意味着当我们的数组 $$A$$ 非常小且每次跳转耗时少于 1 微秒时，我们可以进入一个“延迟受限”的区间，在这个区间中，计算 _确实_ 依赖于 $X$。

{% details 如需详细了解，请点击此处。 %}

令 $$T_\text{min}$$ 为单次跳跃的最短时间。然后

$$T_{hop} = \max \left[ T_{min}, \frac{2 \cdot V}{X \cdot W_\text{ici}} \right]$$

$$T_{total} = \max \left[ \frac{T_{min} \cdot X}{2}, \frac{V}{W_\text{ici}} \right]$$

由于我们执行了 $X / 2$ 次跳转。对于大规模的缩减或收集操作，我们受到带宽的严格限制。我们传输的数据量如此之大，以至于每次跳转的开销几乎可以忽略不计。但对于小数组（例如从模型中采样时），这种开销就不可忽视了，而且 ICI 带宽也不相关。此时我们仅受延迟限制。另一种说法是，对于特定的 TPU，例如具有 `4.5e10` 单向 ICI 带宽的 TPU v5e，传输任何小于 `4.5e10 * 1e-6 = 45kB` 的缓冲区都将受到延迟限制。

{% enddetails %}

这是在 TPU v5e 8x16 切片上对 AllGather 带宽的实测结果。数组在 16 轴上进行分片，因此它具有完整的双向环结构。

{% include figure.liquid path="assets/img/all-gather-bandwidth.png" class="img-small" caption="<b>Figure:</b> 实际带宽和估计链路带宽在 TPU v5e 的 AllGather 过程中的表现。橙色表示实际的每秒 AllGather 字节数，而蓝色曲线则根据已知的集合通信成本计算出的经验单向链路带宽。" %}

请注意，我们不仅实现了约 95% 的峰值声称带宽（`4.5e10`），而且在约 10MB 的情况下也达到了这一峰值，当进行 16 路分片时，每个设备可获得约 625kB（*备注*：这比 GPU 的表现好得多）。

**当我们对多个轴进行 AllGather 操作时会发生什么？** 当我们对多个轴进行 gather 操作时，我们有多个 ICI 维度可用于执行 gather。例如，AllGather<sub>XY</sub>([B, D<sub>XY</sub>]) 在两个硬件网格轴上进行操作。这使可用带宽增加了 $N_\text{axes}$ 倍。

在考虑延迟时，我们最终得到一般规则：

$$T_{total} = \max \left[ \frac{T_{min} \cdot \sum_{i} |X_i|}{2}, \frac{V}{W_\text{ici} \cdot N_\text{axes}} \right]$$

其中 $$\sum_i \lvert X_i \rvert / 2$$ 是 TPU 网格中最长路径的长度。

<b markdown=1 style="color:rgb(144, 92, 255);">小测验 2 [AllGather 时间]:</b> 使用 [Part 2](../tpus) 中的数字，TPU v5e 上的 2D 网格 `{'X': 8, 'Y': 4}`, $$E = 2048$$, $$F = 8192$$ 在 bfloat16 中执行 AllGather<sub>Y</sub>([E<sub>Y</sub>, F]) → [E, F] 需要多长时间？如果是 $$E=256, F=256$$ 呢？

{% details 点击此处查看答案。 %}

**答案：** 让我们从计算一些基本量开始：

1) 每个 TPU v5e 的两个轴方向的单向 ICI 带宽为 4.5e10 字节/秒。  
2) 在 (a) 的 bfloat16 情况下，我们有 $A[E_Y, F]$，因此每个设备保存一个形状为 bf16[512, 8192] 的数组，其大小为 512 * 8192 * 2 = 8.4MB。整个数组的大小为 2048 * 8192 * 2 = 34MB。

对于第(1)部分，我们可以使用上面的公式。由于我们是在一个轴上执行 AllGather，我们有 $T_{\text{comms}} = \text{34e6} / \text{9e10} = \text{377us}$。为了确认我们不是受延迟限制的，我们知道在一个大小为 4 的轴上，最多会有 3 次跳转，因此我们的延迟上限大约是 3 微秒，所以我们离这个限制还很远。然而，TPU v5e 只有在某个轴的大小为 16 时才具有绕环连接，因此在这里 *我们实际上无法执行完全双向的 AllGather*。数据从边缘到达另一侧需要 3 次跳转，因此理论上我们会有大约 $T_{\text{comms}} = 3 * \text{8.4e6} / \text{4.5e10} = 560\mu s$。[**这是**](https://imgur.com/a/RkvpRGQ) **一个实际的性能分析** 来自 [这个 Colab](https://colab.research.google.com/drive/15tDZMfNqm2vJjvSzw5VC9qtSwc5td-oV?usp=sharing)，显示了 $680 \mu s$，这是合理的，因为我们很可能无法达到理论带宽的 100%！对于第(2)部分，每个分片的大小为 `64 * 256 * 2 = 32kB. 32e3 / 4.5e10 = 0.7us`，因此我们是受延迟限制的。由于有 3 次跳转，这将大约需要 3 * 1us = 3us。[实际上，它接近 8us。](https://imgur.com/a/HZLQmYs)

{% enddetails %}

<p markdown=1 class="takeaway">**注意：** 当我们有一个二维网格如 `{'X': 16, 'Y': 4}` 时，并不需要每个轴都对应特定的 _硬件_ 轴。这意味着例如上面的描述可以表示一个 4x4x4 TPU v5p 立方体，其中 $X$ 轴上有两个轴。这将在我们后续描述多个轴上的数据并行时变得重要。</p>

### 案例 3：两个乘数的收缩维度均已分片

第三个基本情形是当两个乘数都在其收缩维度上沿着同一网格轴进行分片：

$$\textbf{A}[I, J_X] \cdot \textbf{B}[J_X, K] \rightarrow C[I, K]$$

在这种情况下，*局部* 的分片块矩阵乘法至少*可以*执行，因为它们将共享相同的收缩索引集。但每个乘积仅表示完整目标乘积的*部分和*，并且沿 **X** 维度的每个设备将保留这个最终目标乘积的不同*部分和*。这种情况非常常见，因此我们扩展了我们的符号表示，以明确标记这一条件：

$$\textbf{A}[I, J_X] \cdot_\text{LOCAL} \textbf{B}[J_X, K] \rightarrow C[I, K] \{\ U_X \}$$

符号 **{ U<sub>X</sub> }** 表示“沿 X 网格轴未归约”，指的是该操作在某种意义上是“不完整的”，因为它只有在进行最终求和后才会完成。$\cdot_\text{LOCAL}$ 语法表示我们执行局部求和，但保留结果未归约。

这可以看作是关于矩阵乘法和外积的以下结果：

$$A \cdot B = \sum_{i=1}^{P} \underbrace{A_{:,i} \otimes B_{i,:}}_{\in \mathbb{R}^{n \times m}}$$

其中 ⊗ 表示外积。因此，如果在轴 **X** 上的 TPU **i** 拥有矩阵 **A** 的第 **i** 列和矩阵 **B** 的第 **i** 行，我们可以进行局部矩阵乘法以得到 $$A_{:,i} \otimes B_{i,:} \in \mathbb{R}_{n\times m}$$。该矩阵的每个元素对应于 **A • B** 在该位置的和中的第 **i** 项。我们仍然需要在轴 **X** 上对 **P** 进行求和，以得到完整的 **A • B**。如果我们将 **A** 和 **B** 按块（即分片）写入，然后对每个结果分片进行求和，其工作方式是相同的。

我们可以通过在 **X** 轴上执行完整的 **AllReduce** 来解决这个问题：

$$\begin{align*}
A[I, J_X] \cdot_\text{LOCAL} B[J_X, K] \rightarrow &\ C[I, K] \{ U_X \} \\
\textbf{AllReduce}_X C[I, K] \{ U_X \} \rightarrow &\ C[I, K]
\end{align*}$$

AllReduce 会移除部分和，从而使轴上的 *每个* 设备都拥有相同的完全求和后的值。AllReduce 是我们在本节中将要讨论的几个关键通信中的第二个，第一个是 AllGather，其余的是 ReduceScatter 和 AllToAll。AllReduce 接收一个具有未求和（部分求和）轴的数组，并通过在未求和轴上传递这些分片并累加结果来执行求和操作。其函数签名是

$$\textbf{AllReduce}_Y A[I_X, J] \{U_Y\} \rightarrow A[I_X, J]$$

这意味着它只是去掉了 $\\{U_Y\\}$ 后缀，但其他部分保持不变。

**AllReduce 的代价有多高？** 一种理解 AllReduce 执行方式的思维模型是，每个设备将其分片发送给其邻居，并对其接收到的所有分片进行求和。显然，这比 AllGather 更昂贵，因为每个“分片”与完整数组的形状相同。一般来说，**AllReduce 的代价是 AllGather 的两倍**。一种理解方式是注意到 **AllReduce** 可以表示为两个其他原语的组合：一个 **ReduceScatter** 和一个 **AllGather**。与 AllReduce 类似，ReduceScatter 在数组上解决部分和，但输出是“分散”或沿给定维度划分的。AllGather 收集所有这些部分，并沿该物理轴“取消划分/取消分片/复制”逻辑轴。

$$\begin{align*}
\textbf{ReduceScatter}_{Y,J} : A[I_X,J] \{U_Y\} \rightarrow &\ A[I_X, J_Y] \\
\textbf{AllGather}_Y : A[I_X, J_Y] \rightarrow &\ A[I_X, J]
\end{align*}$$

**那 ReduceScatter 呢？** 正如 AllGather 会重新组装一个分片数组（移除一个下标），ReduceScatter 会对一个未求和或部分求和的数组进行求和，然后沿着相同的网格轴对不同的逻辑轴进行分片（散播）。$X[F]\\{U_Y\\} \to X[F_Y]$。动画展示了它是如何实现的：请注意，它与 AllGather 非常相似，但与 AllGather 保留每个分片不同，我们是将它们合并在一起。因此，其延迟大致相同，不包括执行求和所需的时间。

{% include figure.liquid path="assets/img/reduce-scatter.gif" class="img-fluid" %}

每跳的通信时间仅仅是每分片字节数 $V / Y$ 除以带宽 $W_\text{ici}$，与 AllGather 的情况相同，因此我们有

$$T_{\text{comms per AllGather or ReduceScatter}} = \frac{V}{W_\text{ici}}$$

$$T_{\text{comms per AllReduce}} = 2 \cdot \frac{V}{W_\text{ici}}$$

其中 $$W_\text{ici}$$ 是双向带宽，只要我们有一个完整的环来减少过载。

### 案例 4：两个乘数在相同轴上沿非收缩维度进行分片

在对张量进行分片时，每个网格维度最多只能出现一次。执行上述规则有时会导致违反此规则的情况，例如：

$$A[I_X, J] \cdot B[J, K_X] \rightarrow C[I_X, K_X]$$

这是无效的，因为沿着维度 **X** 的某个分片，比如 **i**，将拥有 **C** 的 **(i, i)** 分片，即对角线元素。所有分片中没有足够的信息来恢复结果的非对角线元素，因此我们不能允许这种分片方式。

解决这个问题的方法是对某些维度执行 AllGather 操作。在这里，我们有两种选择：

$$\begin{align*}
\textbf{AllGather}_X A[I_X, J] \rightarrow &\ A[I, J] \\
A[I, J] \cdot B[J, K_X] \rightarrow &\ C[I, K_X]
\end{align*}$$

或

$$\begin{align*}
\textbf{AllGather}_X B[J, K_X] \rightarrow &\ B[J, K] \\
A[I_X, J] \cdot B[J, K] \rightarrow &\ C[I_X, K]
\end{align*}$$

无论哪种情况，结果的形状中只会提到 **X** 一次。选择哪一个将取决于后续操作所需的分片方式。

## 对 TPU 通信原语的深入探讨

前 4 个案例介绍了用于执行分片矩阵乘法的几种“核心通信原语”：

1. **AllGather：** 从分片中移除一个下标，将分片进行聚合。
2. **ReduceScatter：** 通过在该轴上对分片求和，移除数组的一个“未归约”后缀，使数组在第二个轴上保持分片状态。
3. **AllReduce：** 移除一个“未归约”后缀，使数组在该轴上不再分片。

还有一种核心通信原语需要提及，它出现在专家混合（MoE）模型和其他计算中：**AllToAll**。

### 我们的最终通信原语：AllToAll

一个最终的基本集合操作，在考虑分片矩阵乘法时不会自然出现，但在实践中却频繁出现，这就是 **AllToAll** 集合操作，更准确地说，是分片转置或重新分片操作的特例。例如：

$$\textbf{AllToAll}_{X, J} A[I_X, J] \rightarrow A[I, J_X]$$

AllToAll 通常用于在分片计算中不同区域之间重新排列不兼容布局方案的分片布局。它们在考虑分片混合专家模型时自然出现。*你可以将 AllToAll 看作是将一个下标从一个轴移动到另一个轴*。由于 AllToAll 不需要将每个分片的所有数据在环中复制，因此实际上 *比 AllGather 更便宜（便宜四分之一）*<d-footnote>。对于大小相等的双向环，每个设备将向右发送 $(N/2 + (N/2-1) + … + 1)$ 块，向左发送 $((N/2-1) + … + 1)$ 块 $= 0.5 \cdot (N / 2) \cdot (N/2 + 1) + 0.5 \cdot (N / 2) \cdot (N/2 - 1) = N^2/4$。每个块（即分片的分片）的大小是 $\text{bytes} / N^2$，因此每个设备的成本是 $(\text{bytes} / N^2) \cdot N^2 / 4 = \text{bytes} / 4$。这一结果随着设备数量的增加而扩展，因为总带宽随着设备数量的增加而扩展。</d-footnote>.

{% include figure.liquid path="assets/img/all-to-all.gif" class="img-fluid" %}

如果我们推广到 ND AllToAll，那么在 AxBxC 网格上，总共有 $V$ 字节（所有设备总和）的数组的整体成本是

$$T_\text{comms per AllToAll} = \frac{V \cdot \max(A, B, C, ...)}{4 \cdot N \cdot W_\text{ici}}$$

其中，如常 $W_\text{ici}$ 是双向 ICI 带宽，$N = A \cdot B \cdot C \cdot \ldots$ 是设备总数。等价地，以每设备字节数 $V / N$ 表示，成本为 $(V / N) \cdot \max(A, B, C, ...) / (4 \cdot W_\text{ici})$。对于一维网格，这简化为 $V / (4 \cdot W_\text{ici})$，其成本是 AllGather 的 1/4。在二维情况下，成本实际上会随着最短轴的尺寸增加而降低。

*旁注：如果你想粗略地推导这个事实，可以从一个一维环 $\mathbb{Z} / N\mathbb{Z}$ 开始。如果我们随机选择一个源节点和目标节点，它们平均相距 N / 4 跳，这给我们带来了 $(V \cdot N) / (4 * N)$ 的成本。现在如果我们考虑一个 ND 环，每个轴基本上是独立的。每个节点有 $1 / N$ 字节，平均需要将其数据跳 $\max(A, B, C, …) / 4$ 次。你也可以从分割带宽推导出这一点：在一个 AllToAll 操作中，网格的一半将它一半的数据（$V / 4$ 字节）发送到另一半。最窄的分割带宽垂直于最长轴，穿过 $2 \cdot N / \max(A, B, …)$ 条链路（两个切割平面，包括环绕），单向带宽为 $N \cdot W_\text{ici} / \max(A, B, …)$。除法后得到上面的公式。*

### 关于 ReduceScatter 的更多内容

ReduceScatter 是一个比其表面看起来更基础的操作，因为它实际上是 AllGather 的导数，反之亦然。即，如果在前向传播中我们有：

$$\textbf{AllGather}_X A[I_X] \rightarrow A[I]$$

然后我们对反向模式导数 **A'**（在各个分片上通常不同）执行 ReduceScatter 操作，以得到分片的 **A'**：

$$\textbf{ReduceScatter}_X A'[I] \{ U_X \} \rightarrow A'[I_X]$$

同样，前向传递中的 $$\text{ReduceScatter}_X(A[I] \{U_X\}) \to A[I_X]$$ 暗示反向传递中的 $$\text{AllGather}_{X}(A'[I_X]) \to A'[I]$$。

{% details 有关 AllGather 和 ReduceScatter 如何相互衍生的详细信息，请点击此处。 %}

这源于广播和归约作为线性算子时互为转置，而 AllGather 和 ReduceScatter 分别是广播和归约的外积（也称为 [Kronecker 积](https://en.wikipedia.org/wiki/Kronecker_product)）。具体来说，如果我们有一个向量 $x \in \mathbb{R}^n$，任意数量的设备 $p \in \mathbb{N}$，并令 $u = (1, \ldots, 1) \in \mathbb{R}^p$，我们可以按以下方式定义广播和归约，这应该与你对它们的直觉理解一致：

$$
\begin{align*}
\text{broadcast} &: \mathbb{R}^n \rightarrow \mathbb{R}^{p n} \\
\text{broadcast} &= u \otimes \mathbf{I}_n \\
\text{reduce} &: \mathbb{R}^{p n} \rightarrow \mathbb{R}^n \\
\text{reduce} &= u^T \otimes \mathbf{I}_n
\end{align*}
$$

让我们通过一个例子来看看这会是什么样子，其中 $n = 1$, $p = 2$。如果 $x = (7)$，我们有 $$\text{broadcast}(x) = \left(\begin{pmatrix} 1 \\ 1 \end{pmatrix} \otimes \begin{pmatrix} 1 \end{pmatrix}\right) x = \begin{pmatrix} 1 \\ 1 \end{pmatrix} x = \begin{pmatrix}  7\\  7  \end{pmatrix} \in \mathbb{R}^{p n}$$。这与我们的预期一致，将 $\mathbb{R}^n$ 中的向量广播到 $\mathbb{R}^{pn}$。现在让 $y = (8, 9)$，我们有 $$\text{reduce}(y) = \left(\begin{pmatrix} 1 & 1 \end{pmatrix} \otimes \begin{pmatrix} 1\end{pmatrix}\right) y = \begin{pmatrix} 1 & 1  \end{pmatrix} \begin{pmatrix}  8 \\ 9  \end{pmatrix} = \begin{pmatrix}   17    \end{pmatrix}$$。这同样与我们的预期一致，将 $\mathbb{R}^{p n}$ 中的向量缩减为 $\mathbb{R}^{n}$ 中的向量。由于对于任何两个矩阵 $A$ 和 $B$ 都有 $(A \otimes B)^T = A^T \otimes B^T$，我们看到 $\text{reduce} = \text{broadcast}^T$。我们将 AllGather 和 ReduceScatter 表示为以下外积：

$$
\begin{align*}
\text{AllGather} &: \mathbb{R}^{p n} \rightarrow \mathbb{R}^{p^2 n} \\
\text{AllGather} &= \text{broadcast} \otimes \mathbf{I}_p \\
\text{ReduceScatter} &= \mathbb{R}^{p^2 n} \rightarrow \mathbb{R}^{p n} \\
\text{ReduceScatter} &= \text{reduce} \otimes \mathbf{I}_p
\end{align*}
$$

此处我们将 $\mathbb{R}^{p^2 n}$ 视为 $\mathbb{R}^{p \times p n}$，因此每个 $\mathbb{R}^{p n}$ 设备对应一个 $p$ 向量。我们建议尝试一些小例子，例如 $n = 2$、$p = 3$，以了解这些运算符作为矩阵时的形态。利用相同的转置性质，我们再次得到 $\text{AllGather}^T = \text{ReduceScatter}$，当然还有 $\text{ReduceScatter}^T = \text{AllGather}$。这种转置将在反向传播过程中出现，因为如果我们对某个线性运算符 $y = Ax$（如 AllGather 或 ReduceScatter）具有 $A$，那么在反向传播过程中我们将获得损失相对于 $y$、$\frac{\partial L}{\partial y}$ 的导数，并得到 $\frac{\partial L}{\partial x}$ 作为 $\frac{\partial L}{\partial x} = A^T \frac{\partial L}{\partial y}$。这表明 AllGather 的导数将是 ReduceScatter，反之亦然。

{% enddetails %}

将 AllReduce 转换为 AllGather 和 ReduceScatter 还具有一个方便的特性，即我们可以将最终的 AllGather 延迟到稍后的某个时刻。非常常见的是，我们更不愿意承担重新组装跨设备复制的完整矩阵乘积的代价。相反，我们希望在这种将两个具有分片收缩维度的乘数结合的情况下，仍然保留一个分片的状态。

$$A[I, J_X] \cdot B[J_X, K] \rightarrow C[I, K_X]$$

在这种情况下，我们也可以执行 ReduceScatter 而不是 AllReduce，然后在稍后的某个时间点选择性地执行 AllGather，即

$$\begin{align*}
A[I, J_X] \cdot_{LOCAL} B[J_X, K] \rightarrow &\ C[I, K] \{ U_X \} \\
\textbf{ReduceScatter}_{X,K} C[I, K] \{ U_X \} \rightarrow &\ C[I, K_X]
\end{align*}$$

请注意，ReduceScatter *引入*了一个分片维度，因此在这种情况下，自然可以在命名维度 **I** 或 **K** 中选择一个进行分片。在使用 ReduceScatter 时，我们通常需要选择 *哪个* 命名维度来引入新的分片（尽管选择通常由更大的建模上下文所决定）。这就是我们使用语法 **ReduceScatter<sub>X,K</sub>** 来指定分片轴的原因。

### 如何将矩阵乘法通信与计算重叠

正如我们在[第1部分](../roofline)中讨论的，我们通常假设如果通信足够快，就可以始终将通信与某些有用的计算重叠。本节中的集合通信通常可以与矩阵乘法计算本身重叠，但要做到这一点并不简单。我们使用的算法称为**集合矩阵乘法**，首次在[Wang 等人](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959)中描述。下面是这种重叠如何实现的简化动画：

{% include figure.liquid path="assets/img/ag_matmul.gif" caption="<b>图：</b> 动画展示了如何将单个分片矩阵-向量乘法与随后的 AllReduce 操作重叠（上述情况 3）。一个完整的矩阵乘法由多个矩阵-向量乘法组成。" %}

简而言之，我们可以在对矩阵的一个块执行矩阵乘法的同时，开始对先前块进行环形归约。在某些情况下，我们还可以在批量维度或矩阵输入维度上进行分块。我们在[第10部分](../jax-stuff)中通过一个简单的JAX实现进行了讲解，[Mosaic文档](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html)也给出了在GPU上的一个良好示例。我们鼓励你在某个时候实现这个版本。

## 我们学到了什么？

* 数组的分片由一个 **Mesh** 指定，该 **Mesh** 命名了我们 TPU 网格的物理硬件轴，以及一个 **Sharding**，该 **Sharding** 将网格轴名称分配给数组的逻辑轴。
  * 例如，**A**[I<sub>XY</sub>, J] 描述了一个抽象数组 **A**，其第一个维度沿两个网格轴 X 和 Y 进行分片。结合 Mesh(mesh_shape=(4, 8), axis_names=('X', 'Y')) 或简写的 Mesh({'X': 4, 'Y': 8})，这告诉我们数组的第一个维度沿该维度被分片为 32 种方式。

* **对分片数组进行算术运算的工作方式与对未分片数组进行算术运算完全相同，除非你沿着分片轴进行收缩操作**。在这种情况下，我们需要引入一些通信。我们考虑四种情况：

1. *两个数组在收缩维度上均未进行分片*：不需要通信。  
2. *其中一个数组在收缩维度上进行了分片（或收缩维度沿不同轴进行分片）*：在执行操作之前，我们对其中一个输入执行 AllGather。  
3. *两个数组在收缩维度上以相同方式分片*：我们本地相乘分片，然后执行 AllReduce 或 ReduceScatter。  
4. *两个数组沿非收缩维度的相同网格轴进行分片*：我们首先对其中一个输入执行 AllGather。

* TPUs 使用大约 **4 个核心通信原语**：
  1. AllGather: $[A_X, B] \to [A, B]$
  2. ReduceScatter: $[A, B] \\{U_X\\} \to [A_X, B]$
  3. AllToAll: $[A, B_X] \to [A_X, B]$
  4. AllReduce: $[A_X, B]\\{U_Y\\} \to [A_X, B]$（严格来说不是一个原语，因为它结合了 ReduceScatter + AllGather）

{% include figure.liquid path="assets/img/all-collectives.png" class="img-fluid" %}

* 这些操作的成本和延迟 **不依赖于轴的大小（只要它们受带宽限制）**，而只依赖于输入数组的大小和链路的带宽。对于单向 AllGather/ReduceScatter：

$$T_{\text{comm per AllGather or ReduceScatter}} = \frac{\text{Data volume}}{\text{bandwidth}} \cdot \frac{\text{Axis} - 1}{\text{Axis}}
\longrightarrow \frac{\text{Data volume}}{\text{bandwidth (bidirectional)}}$$

* AllReduce 由 ReduceScatter 后接 AllGather 组成，因此其成本是上述成本的 2 倍。AllToAll 只需将分片部分传递到环的另一侧，因此其成本仅为 AllGather 的 ¼。以下是总结：

|操作|描述|语法|运行时|
| :---------------- | :----------------------------------------------------------------------------------------------------------------- | :------------------------------- | :----------------------------------------------- ||**AllGather**|沿某一轴合并分片数组的所有分片，移除一个下标。|$[A_X, B] \to [A, B]$|字节 / (双向 ICI 带宽 * num_axes)|
|**ReduceScatter**|沿一个轴对部分求和的数组进行求和，并沿另一个轴进行分片（添加下标）。|$[A, B] \\{U_X\\} \to [A_X, B]$|与 AllGather 相同|
|**AllReduce**|沿轴对部分求和的数组进行求和。移除一个 { U<sub>x</sub> }。结合 AllGather 和 ReduceScatter。|$[A_X, B]\\{U_Y\\} \to [A_X, B]$|2 * AllGather|
|**AllToAll**|沿同一轴聚合（复制）一个轴，并沿同一轴对不同维度进行分片。|$[A, B_X] \to [A_X, B]$|AllGather / 4 对于双向环|

## 一些需要解决的问题

*这里有一些基于本节内容的指导性问题。我们目前不会提供所有答案，但会尽可能编写更多答案。*

**问题 1 [复制分片]**: 一个数组被分片 $A[I_X, J, K, \ldots]$（即仅在 $X$ 上分片），具有一个网格 `Mesh({'X': 4, 'Y': 8, 'Z': 2})`。所有芯片上 $A$ 所占用的总字节数与数组一个副本的大小之比是多少？

{% details 点击此处查看答案。 %}

我们的数组仅沿 X 分片，X 的大小为 4，因此每个分片的有效大小为 $[I / 4, J, K, \ldots] = \text{sizeof}(A) / 4$。由于我们的数组在 Y 和 Z 上进行了复制，总大小为 $Y \cdot Z \cdot \text{sizeof}(A)$，因此总大小与单个芯片大小的比值为 $Y \cdot Z \cdot \text{sizeof}(A) / \text{sizeof}(A) = 16$。

{% enddetails %}

**问题 2 [AllGather 延迟]**: 在 TPU v4p 4x4x4 切片上，若使用 mesh `Mesh({'X': 4, 'Y': 4, 'Z': 4})`，$\text{AllGather}_X([B_X, D_Y])$ 在 $B=1024$ 和 $D=4096$ 为 bfloat16 时应需要多长时间？$$\text{AllGather}_{XY}([B_X, D_Y])$$ 呢？$$\text{AllReduce}_Z([B_X, D_Y] \{U_Z \})$$ 呢？

{% details 点击此处查看答案。 %}

我们在所有轴上都有环绕链接，因为我们有一个完整的 `4x4x4` 立方体，因此我们有 9e10 的双向带宽可供使用。

1. 因为我们只是在一个轴上进行收集，而另一个轴是分片的，因此我们实际上是在一个轴上收集 $2BD / Y$ 字节。*如果你只考虑 Y 轴上的一个分片，那么沿 X 轴的 AllGather 看起来就像一个未分片的 AllGather，字节数为 1 / Y。* 由于我们 TPU v4p 的 ICI 带宽是 9e10 字节/秒（双向），这将需要 $2BD / (\text{9e10} \cdot Y) = 2 \cdot 1024 \cdot 4096 / (\text{9e10} \cdot 4) = 23 \mu s$。

2. 我们的带宽是之前的两倍，但我们正在对完整数组进行 AllGathering，因此 `T = 2BD / (2 * W) = 2*1024*4096 / (2 * 9e10) = 46us`。这远未达到 4 微秒（每跳 1 微秒）的延迟上限，因此我们没有问题。

3. AllReduce 的成本是 AllGather 的两倍。每个分片的大小为 $2BD / (X * Y)$，因此成本约为 $4BD / (X * Y * W)$，或大致为 `4 * 1024 * 4096 / (16 * 9e10) = 11.6us`。

*趣闻:* 部分 (1) 和 (2) 实际上并不是最优的，因为数组也沿着未使用的 Z 轴进行了复制，我们可以利用这些空闲的链接：我们可以先免费重新分片 $[B_X, D_Y] \to [B_{XZ}, D_Y]$（每个设备只需丢弃其分片的一部分），然后执行 $$\text{AllGather}_{XZ}$$（或 $$\text{AllGather}_{XYZ}$$），在更多轴上进行聚合，从而达到相同的最终状态。这将部分 (1) 缩短到 11.5 微秒，部分 (2) 缩短到 31 微秒 — 实际上，只需一开始就沿更多轴进行分片，就可以实现这一点，这也是尽可能精细地对数组进行分片的一个原因。

{% enddetails %}

**问题 3 [延迟受限的 AllGather]**: 假设我们正在执行一个 $\text{AllGather}_X([B_X])$，但 $B$ 非常小（例如 128）。在 TPU v4p 4x4x4 切片上，使用 bfloat16 的 mesh `Mesh({'X': 4, 'Y': 4, 'Z': 4})`，这应该需要多长时间？*提示：你可能受到延迟的限制。*

{% details 点击此处查看答案。 %}

我们的 bfloat16 数组总共仅使用 256 字节，每个设备仅使用 64 字节。由于我们在 TPU v4p 上有一个大小为 4 的轴，我们有一个环回链路，因此可以将数组向两个方向发送。在 `4.5e10` 单向带宽的情况下，每次跳转大约需要 `64 / 4.5e10 ~ 0`，因此我们肯定受到延迟的限制。计算跳转次数，我们仅需 2 次跳转即可完成完整的 gather 操作，因此大约 2 微秒是一个很好的估计。

{% enddetails %}

**问题 4 [矩阵乘法策略]**: 为了执行 $X[B, D] \cdot_D Y[D_X, F] \to Z[B, F]$，在本节中我们告诉你需要执行 $\text{AllGather}_X(Y[D_X, F])$ 并相乘完全复制的矩阵（情况 2，*策略 1*）。相反，你可以像 $X[B, D_X] \cdot_D Y[D_X, F] \to Z[B, F] \\{U_X\\}$ 一样相乘本地分片（情况 3，*策略 2*），然后执行 $\text{AllReduce}_X(Z[B, F] \\{ U_X\\})$。这两种方法各自执行多少 FLOPs 和通信操作？哪种方法更好，为什么？

{% details 点击此处查看答案。 %}

让我们从我们的基线（*策略 1*）开始。正如我们所展示的，AllGather 的成本是 $2DF / W_\text{ici}$。一旦我们有了完全复制的数组，总计算时间是 $2BDF / C$（其中 $C$ 是我们的加速器 FLOPs/s，因为每个 TPU 执行相同的 FLOPs）。因此我们有

$$T_\text{total (Strategy 1)} = \max\left(\frac{2BDF}{C}, \frac{2DF}{W_\text{ici}}\right)$$

相比之下，新策略（策略 2）对 $2BF$ 字节执行一次 AllReduce，这耗费了 $4BF / W_\text{ici}$，但减少了 $1 / X$ 的 FLOPs（因为计算被分片了）。这意味着我们执行了 $2\cdot B\cdot D\cdot F / X$ FLOPs，而生成的 AllReduce 以 bfloat16 格式通信 $$2 \cdot 2 \cdot B \cdot F$$ 字节。因此，我们对 *策略 2*（不使用 AllGather，仅在之后使用一次 AllReduce）的总时间大致为

$$T_\text{total} = \max\left(\frac{2BDF}{X \cdot C}, \frac{4BF}{W_\text{ici}}\right)$$

问题是：*这些之中哪个更大？* 当 $D / (X \cdot C) > 2 / W_\text{ici}$ 或当 $D / 2X > C / W_\text{ici} \approx 2550 \rightarrow X < D / (2 * 2550)$ 时，策略（2）受算力限制。我们可能合理地预期 $D \approx 8k$，这意味着大约 $X < 2$，这不太可能——因此，我们基本上总是受通信限制，使用策略 2。对于基线（策略 1），当 $$B < C / W_\text{ici} = 2550$$ 时，我们受通信限制，这虽然经常成立，但并非总是如此。

因此，如果 $B < 2550$，这两种情况我们都受通信限制。

$$T_\text{comms for Strategy 2} < T_\text{comms for Strategy 1} \Leftrightarrow \frac{4BF}{W_\text{ici}} < \frac{2DF}{W_\text{ici}}$$

当 $D > 2B$ 且 $2B < 5100$ 时，这一说法成立。这通常成立，因此如果我们的批量较小，策略 2 有时会更好。当我们的批量较大（$B > 2550$）时，我们有

$$T_\text{comms for Strategy 2} < T_\text{math for Strategy 1} \Leftrightarrow \frac{4BF}{W_\text{ici}} < \frac{2BDF}{C}$$

当 $2 / W_\text{ici} < D / C$ 时，或者当 $D > 2 * 2550 = 5100$ 时，这通常是大模型的情况。因此，这种替代策略通常更适合大模型，除非 $D$ 很小。

*我们为什么不总是这样做呢？* 实际上，我们有时可能会这样做，但在实践中，通常很少出现其中一个输入的收缩维度沿着另一个输入未被分片的轴进行分片的情况。例如，如果我们使用 FSDP（详见[第 5 章](../training)），我们会沿着数据维度对参数进行分片，而激活值也将_沿着数据维度进行分片_。因此，从这个意义上说，这种情况并不常见。

{% enddetails %}

**问题 5 [最小延迟]**: 假设我想在 TPU v4p 4x4x4 上以最低可能的延迟执行一个 matmul $A[I, J] \cdot_J B[J, K] \to C[I, K]$。假设输入可以任意分片，但结果应完全复制。我的输入应该如何分片？总的 FLOPs 和通信时间是多少？

{% details 点击此处查看（部分）答案。 %}

我们在这里不会提供完整的答案，但我们将首先描述最可能的四个选项：

1. $A[I_{XYZ}, J] \cdot B[J, K]$ + AG 在末尾  
2. $A[I, J] \cdot B[J, K_{XYZ}]$ + AG 在末尾  
3. $A[I, J_{XYZ}] \cdot B[J_{XYZ}, K]$ + AR 在末尾  
4. $A[I, J] \cdot B[J, K]$（完全复制）

我们也可以考虑沿着不同的网格轴对不同的轴进行分片，但这样不太可能改变最终成本。除了（4）之外，每个TPU的总FLOPs是相同的，但每个的通信成本不同。然后我们只需计算每种情况的通信成本，看看哪种最低。简而言之，（1）和（2）同样好。

{% enddetails %}

**问题 6:** 假设我们想在 TPU v5e 4x4 上执行 $A[I_X, J_Y] \cdot_J B[J_Y, K] \to C[I_X, K]$。我们执行哪些通信操作？通信和计算所花费的时间比例是多少？

* 那么 $A[I_X, J] \cdot_J B[J_X, K_Y] \to C[I_X, K_Y]$ 呢？这是训练中最标准的设置，我们在此结合了数据、张量和 ZeRO 分片。  
* 那么 $A[I_X, J] \cdot_J B[J, K_Y] \to C[I_X, K_Y]$ 呢？这是推理的标准设置，我们在此采用纯张量并行（+数据）。

**问题 7:** 一个典型的 Transformer 块包含两个矩阵 $W_\text{in}[D, F]$ 和 $W_\text{out}[F, D]$，其中 $F \gg D$。假设我们有一个批量大小 B。那么完整的块是 $In[B, D] \cdot W_\text{in}[D, F] \cdot W_\text{out}[F, D]$。让我们选择 $D=8192$、$F=32768$ 和 $B=128$，并假设所有内容都使用 bfloat16。假设我们正在使用 TPU v5e 2x2 切片，但让我们假装每个 TPU 只有 300MB 的空闲内存。In、$W_\text{in}$、$W_\text{out}$ 和 Out 应该如何分片，以在不超出内存限制的同时尽量减少总体时间？通信和 FLOPs 各花费多少时间？*提示：最终输出不需要完全复制，但应与输入以相同方式分片，以便“层”可以重复。*

{% details 点击此处查看（部分）答案。 %}

首先让我们考虑内存。我们的两个大矩阵各自使用 `2 * 8192 * 32768 = 536MB`。我们的激活值 `In` 的大小为 `2 * 128 * 8192 = 2MB`（足够小，无需担心）。由于每台设备只有 300MB 的备用内存，我们显然需要对矩阵乘法进行分片。

1. $In[B_X, D] * W_\text{in}[D_{XY}, F] * W_\text{out}[F, D_{XY}] \rightarrow Out[B_X, D]$（这通常称为 FSDP）  
2. $In[B, D_{XY}] * W_\text{in}[D, F_{XY}] * W_\text{out}[F_{XY}, D] \rightarrow Out[B, D_{XY}]$（这称为张量并行）

第一个方法效果很差，因为我们首先需要对大权重或激活值进行 AllGather。第二个方法在开始时需要 AllGather，在结束时需要 ReduceScatter（比 AllReduce 更便宜）。其余的数学计算就留作练习吧。

{% enddetails %}

**问题 8 [挑战]**: 使用上面的简短代码片段作为模板，分配一个分片数组，并使用 pmap 或 shard_map 对 4 个主要通信原语（AllGather、AllReduce、ReduceScatter 和 AllToAll）进行基准测试。你将需要用到 `jax.lax.all_gather`、`jax.lax.psum`、`jax.lax.psum_scatter` 和 `jax.lax.all_to_all`。你理解这些函数的语义吗？它们需要多长时间？

**问题 9 [还有其他分片矩阵乘法的策略吗？]**: [上面](#case-2-one-multiplicand-has-a-sharded-contracting-dimension)我们声称，当矩阵乘法的一个输入沿其收缩维度被分片时，我们应该对分片矩阵执行 AllGather 操作，并在本地执行收缩操作。你可能会想到的另一种策略是执行分片矩阵乘法，然后对结果执行 AllReduce（就好像两个输入都沿收缩维度被分片一样），即 $A[I, J_X] *_J B[J, K] \to C[I, K]$ 通过

1. $C[I, K] \\{ U_X \\} = A[I, J_X] \cdot B[J_X, K]$  
2. $C[I, K] = \text{AllReduce}(C[I, K] \\{ U_X\\})$

回答以下问题：

1. 使用索引明确写出针对矩阵 $A[N, M]$ 和 $B[M, K]$ 的算法，具体说明每个设备上执行的计算内容。假设 $A$ 在 ND 个设备上按 $A[I, J_X]$ 的方式分片，且你希望输出在所有设备上进行复制。  
2. 现在假设你接受最终结果不在每个设备上复制，而是按 N 或 K 维度进行分片。上述算法会如何变化？  
3. 仅从上述策略（第 2 部分，而非第 1 部分）的通信成本来看，该通信成本与先对 A 进行 AllGather，然后再进行矩阵乘法的算法的通信成本相比如何？

{% details 点击此处查看答案。 %}


1. 首先计算外积，将结果存储在 $$O[N, K]: o_{kj} = \sum_i a_{ki} b_{ij}$$ 中。请注意，此处重复的索引并不是被约简的索引，因为我们正在计算外积。此处的求和范围是当前所使用设备上存储的 i 值的集合。例如，如果我们有一个大小为 16 的约简轴，以及 4 个设备，那么在设备 0 上，i 的取值范围是 {0, 1, 2, 3}；在设备 1 上，i 的取值范围是 {4, 5, 6, 7}；在设备 2 上，i 的取值范围是 {8, 9, 10, 11}；在设备 3 上，i 的取值范围是 {12, 13, 14, 15}。然后对每个设备上 $O[N, K]$ 的部分和执行 AllReduce 操作，以形成完整的 $O[N, K]$。

2. 在第 2 步中，我们不必执行 AllReduce 操作，而是可以使用更便宜的 ReduceScatter 操作，沿着任一轴：$[N, K] \\{ U_X \\} \to [N_X, K]$ 或 $[N, K] \\{ U_X \\} \to [N, K_X]$。

3. 如上文主文中所述，当吞吐量受限时，执行 AllGather 的成本与执行 ReduceScatter 的成本相同；它仅由我们处理的完整矩阵的大小决定。因此，在 gather-then-matmul 算法中，这与 $NM$ 成比例（因为我们正在对 $A$ 进行 $\text{AllGather}$）；在 matmul-then-reduce-scatter 算法中，这与 NK 成比例（因为我们正在对 $O$ 进行 reduce-scatter）。因此，这两种算法的通信成本比为 `M/K`。

{% enddetails %}

**问题 10：AllToAll 的趣味性：** 在上表中，我们注意到在吞吐量受限的区间内，执行 AllToAll 所需的时间比执行 AllGather 或 ReduceScatter 所需的时间低 4 倍。在本题中，我们将看到这 4 倍的差异来自何处，并且还将看到，如果我们只拥有单向 ICI 链路，而不是双向 ICI 链路，这一因素将如何变化。

1. 让我们先从单向通信的情况开始。假设我们有一个环形拓扑结构，其中有 *D* 个设备，想要对一个 N x N 矩阵 $A[I_X, J]$（例如 $D$ 整除 $N$ 以简化计算）执行 AllGather 或 ReduceScatter 操作。描述这两个集合通信操作中涉及的通信过程，并计算在整个算法过程中，**单个** ICI 链路上传输的标量（浮点数或整数）总数。

2. 现在让我们考虑 AllToAll 操作，仍然在单向 ICI 的情况下。在这种情况下，该算法与 AllGather 情况有何不同？计算该算法中单个 ICI 链路上传输的标量数量。

3. 你应该发现，第 (a) 部分和第 (b) 部分答案之间的比值是一个很整洁的数字。请用简单的话解释这个因子的来源。

4. 现在我们加入双向通信。这会对 AllGather 情况下的总时间产生什么影响？

5. 添加双向通信会对 AllToAll 情况下的总时间产生什么影响？

6. 现在简单解释在双向环中 AllGather 时间与 AllToAll 时间之间的比例。

{% details 点击此处查看答案。 %}

(1) **解决方案：** 该过程很简单：在算法的每一步中，每个设备会将其矩阵的一个分片“条带”（总共包含 $$\frac{N}{D} \times N$$ 个元素）发送给其最近的邻居。这一过程需要发生 $$D-1$$ 次，因为每个分片需要被发送到除起始设备以外的所有设备上。因此，每个设备总共传输 $$\frac{N^2(D-1)}{D}$$ 个标量，即流经单个 ICI 链路。

**答案：** $$N^2 (1-\frac{1}{D})$$，或在 $$D >> 1$$ 时简称为 $$N^2$$。

(2) **解决方案：** 从通信的角度来看，AllToAll 与 AllGather 的关键区别在于，在 AllToAll 中，特定设备上存储的分片的全部内容不需要传输到其他所有设备。假设存储在特定设备（称为设备 0）上的分片是 $$[A, B, C, D]$$（此处 A、B、C、D 是矩阵，我们假设一个包含 4 个设备的环用于说明）。现在，矩阵 $$A$$ 不需要传输到任何地方，矩阵 $$B$$ 需要传输到设备 1；矩阵 $$C$$ 传输到设备 2；矩阵 $$D$$ 传输到设备 3。因此，在算法的第一步中，我们将 $$B$$、$$C$$ 和 $$D$$ 发送到设备 1；下一步，设备 1 将 $$C$$ 和 $$D$$ 发送到设备 2；最后一步，设备 2 将 $$D$$ 发送到设备 3。在这种情况下，传输的参数总数是 $$(\text{size of A/B/C/D}) * (3 + 2 + 1)$$。A/B/C/D 的大小（现在是一般情况）是 $$\frac{N^2}{D^2}$$，同样在一般情况下，$$(3 + 2 + 1)$$ 项变为 $$((D-1) + (D-2) + … + 1)$$，或 $$\frac{(D)(D-1)}{2}$$。因此，单个 ICI 链路上传输的总字节数是 $$\frac{N^2(D-1)}{D \times 2}$$。

**答案：** $$\frac{N^2}{2}(1-\frac{1}{D})$$，或在 $$D >> 1$$ 时简称为 $$\frac{N^2}{2}$$。

(3) **解决方案：** 该因子仅仅是 $$\frac{1}{2}$$，即在一个单向环形拓扑中，AllToAll 的成本是 all-gather/ReduceScatter 的一半。回顾上述推导，这最终源于这样一个事实：在 all-gather 情况下，我们每次传输大小相同的块 $$(D-1)$$ 次，即我们执行的是求和 $$ \text{tiny block size} * (D + D + D + … + D)$$，而在 AllToAll 情况下，我们执行的是求和 $$\text{tiny block size} * (D + D-1 + D-2 + … + 1)$$。因此，这个两倍的因子本质上来自于 $$1 + 2 + \ldots + n = n(n+1)/2$$ 这一事实。

(4) **解决方案**：现在，任何一条链路需要传输的标量总数减少了 2 倍，因为在双向环中，每个“分片条带”可以同时沿两个方向传输。

(5) **解决方案**：在这种情况下，与单向情况相比，我们获得了 4 倍的提升。这最容易通过考虑单个分片条带中每个大小为 (N2/D2) 的块的命运来理解，例如起始于设备 0 的那个条带。在单向情况下，这些块中的一个需要移动 D-1 的距离，另一个块移动 D-2 的距离，依此类推，直到移动 1 的距离。现在，我们将条带划分为向右或向左移动的块，最大移动距离为 floor(D/2)。因此，相应的总和现在变为 $$D/2 + D/2 - 1 + D/2 - 2 + … = D/2 \cdot (D/2+1)/2$$，或在 $$D$$ 很大时变为 $$D^2/8$$。与单向情况下的 $$D^2/2$$ 相比，我们可以看到我们获得了 4 倍的提升。

(6) **解决方案：** 在单向环中，我们看到 AllToAll 的时间已经比 all-gather 快了两倍；这是因为我们不需要将完整的条带发送到每个设备。随后，当我们添加了双向性后，我们看到 AllToAll 的性能提升了 4 倍，而 all-gather 仅提升了 2 倍。将这些比例综合起来，我们得到了期望的 4 倍提升。

{% enddetails %}

<h3 markdown=1 class="next-section">第三部分就到这里！第四部分（关于Transformer数学）请点击[这里](../transformers)！</h3>
