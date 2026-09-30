---
layout: distill
title: "如何思考 TPUs"
# permalink: /main/
description: "本节将介绍TPU的工作原理，它们如何通过网络连接在一起以实现多芯片训练和推理，以及这对我们喜爱的算法性能的影响。其中也包含一些对GPU用户也有用的信息！"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting

section_number: 2

previous_section_url: "../roofline"
previous_section_name: "Part 1: Rooflines"

next_section_url: ../sharding
next_section_name: "Part 3: Sharding"

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
  - name: What Is a TPU?
  - name: TPU Networking
  - name: Key Takeaways
  - subsections:
    - name: TPU specs
  - name: Worked Problems
  - name: Appendix
  - subsections:
    - name: "Appendix A: More on TPU internals"
    - name: "Appendix B: How does a systolic array work?"

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

<p markdown=1 class="announce">您可能也喜欢阅读关于 NVIDIA GPU 的新[第 12 章](../gpus)！</p>

## 什么是 TPU？

**TPU 基本上是一个专门用于矩阵乘法（称为 TensorCore）的计算核心，连接到一组高速内存（称为高带宽内存或 HBM）<d-cite key="tpu_paper"></d-cite>。** 这是示意图：

{% include figure.liquid path="assets/img/tpu-chip.png" class="img-fluid" caption="<b>图：</b> TPU 芯片的基本组件。TensorCore 是左侧的灰色方框，包含矩阵乘法单元（MXU）、向量单元（VPU）和向量内存（VMEM）。" %}

你可以将 TensorCore 看作基本上只是一个非常出色的矩阵乘法机器，但它还有一些其他值得提及的功能。TensorCore 有三个关键单元：

* **MXU**（矩阵乘法单元）是 TensorCore 的核心。对于大多数 TPU 世代，它每 8 个周期使用 systolic array 执行一次 `bf16[8,128] @ bf16[128,128] -> f32[8,128]` 矩阵乘法 <d-footnote>。TPU v6e（Trillium）具有 256x256 的 MXU，而所有之前的世代均使用 128x128。</d-footnote> 详见 <a href="#appendix-b-how-does-a-systolic-array-work"> 附录 B </a>。  
  * 在 TPU v5e 上，每个 MXU 在 1.5GHz 下的 bf16 FLOPs/s 约为 `5e13`。大多数 TensorCores 配备 2 或 4 个 MXU，因此例如 TPU v5e 的总 bf16 FLOPs/s 为 `2e14`。  
  * TPU 还支持精度更低、吞吐量更高的矩阵乘法（例如，每个 TPU v5e 芯片可执行 `4e14` int8 OPs/s）。

* **VPU**（向量处理单元）执行通用数学运算，例如 ReLU 激活、向量之间的逐点加法或乘法。此处还会执行归约操作（求和）。<a href="#appendix-a-more-on-tpu-internals">Appendix A</a> 提供了更多细节。  
* **VMEM**（向量内存）是位于 TensorCore 中的片上临时存储，靠近计算单元。它的容量远小于 HBM（例如，TPU v5e 上为 128 MiB），但与 MXU 之间的带宽更高。VMEM 的操作方式类似于 CPU 的 L1/L2 缓存，但容量更大且可由程序员控制。在 TensorCore 对 HBM 中的数据进行任何计算之前，需要将数据复制到 VMEM 中。

**TPU 在矩阵乘法方面非常、非常快**。这主要是它们的功能，而且它们做得很好。[TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture) 是迄今为止最强大的 TPU 之一，每秒每核心可以执行 `2.5e14` bf16 FLOPs 或每秒每芯片 `5e14` bf16 FLOPs。由 8960 个芯片组成的一个 pod 可以实现 4 bf16 exaFLOPs/s 的性能。这 *非常多*。这是世界上最强的超级计算机之一。而且谷歌拥有大量这样的 TPU。<d-footnote>TPU，尤其是它们的 systolic arrays，之所以是如此强大的硬件加速器，是因为矩阵乘法是少数几个使用 $O(n^3)$ 算力处理 $O(n^2)$ 字节的算法之一。这使得普通的 ALU 容易受到算力的限制，而不是内存带宽的限制。</d-footnote>

上图还包括其他一些组件，如 SMEM 和标量单元，它们用于控制流处理，并在 <a href="#appendix-a-more-on-tpu-internals"> 附录 A</a> 中简要讨论，但理解它们并非关键。另一方面，HBM 很重要且相对简单：

* **HBM**（High Bandwidth Memory）是一种大容量的高速内存，用于存储 TensorCore 使用的张量。HBM 通常的容量在几十吉字节级别（例如，[TPU v5e 配备 16GiB 的 HBM](https://cloud.google.com/tpu/docs/v5e#system_architecture)）。

* 在需要进行计算时，张量会通过 VMEM（见下文）从 HBM 流式传输到 MXU，结果会从 VMEM 写回 HBM。

* HBM 与 TensorCore 之间的带宽（通过 VMEM）称为“HBM 带宽”（通常约为 1-2TB/s），它限制了在内存受限的工作负载中计算的速度。

**通常，所有 TPU 操作都是流水线化的，并且相互重叠。** 为了执行一个矩阵乘法 $X \cdot A \to Y$，TPU 首先需要将矩阵 $A$ 和 $X$ 的块从 HBM 复制到 VMEM，然后将它们加载到 MXU 中，MXU 会分别对 8x128（针对 $X$）和 128x128（针对 $A$）的块进行乘法运算，然后将结果逐块复制回 HBM。为了高效完成此过程，矩阵乘法是流水线化的，这样从/到 VMEM 的复制操作就可以与 MXU 的工作重叠。这使得 MXU 可以继续工作，而不是等待内存传输，从而保持矩阵乘法算力受限，而非内存受限。

这里是一个如何从 HBM 执行逐元素乘法的示例：

{% include figure.liquid path="assets/img/pointwise-product.gif" caption="<b>图：</b> 一个动画，展示在TPU上执行的逐点乘法，从HBM加载字节。请注意字节是如何以块的形式从内存中流式传输出去的，部分结果在不等待完整数组生成的情况下被流水线式地回传。" %}

一个矩阵乘法操作在外观上几乎完全相同，只是它会将数据加载到 MXU 而不是 VPU/向量单元，而且由于相同的权重块会被用于多个激活块，加载和存储的顺序会有所不同。你可以看到数据块依次流经 VMEM，然后进入 VREGs（向量寄存器），再进入向量单元，然后再返回到 VMEM 和 HBM。正如我们即将看到的，如果从 HBM 到 VMEM 的加载速度慢于向量单元（或 MXU）的 FLOPs，我们将变得“带宽受限”，因为这会导致 VPU 或 MXU 缺乏工作负载。

<p markdown=1 class="takeaway">**关键要点：** TPUs 非常简单。它们从 HBM 将权重加载到 VMEM，然后再从 VMEM 加载到一个每秒可执行约 200 万亿次乘加运算的 systolic 数组中。HBM $\leftrightarrow$ VMEM 和 VMEM $\leftrightarrow$ systolic 数组的带宽设定了 TPU 能够高效执行计算的基本限制。</p>

**VMEM 和算术强度：** VMEM 比 HBM 小得多，但其与 MXU 之间的带宽更高。正如我们在 [第 1 章](../roofline) 中看到的，这意味着如果一个算法能够将所有输入/输出存储在 VMEM 中，它更不容易遇到通信瓶颈。当计算的算术强度较低时，这一点尤其有用：VMEM 带宽大约是 HBM 带宽的 22 倍，这意味着从 VMEM 读取或写入 VMEM 的 MXU 操作只需 10-20 的算术强度即可实现峰值 FLOPs 利用率。这意味着如果我们能将权重存储在 VMEM 而不是 HBM 中，我们的矩阵乘法可以在更小的批量大小下达到 FLOPs 瓶颈。这也意味着那些本质上算术强度较低的算法仍然可以保持高效。由于 VMEM 容量非常有限，这通常是一个挑战。<d-footnote>

我们有时会谈到 VMEM 预取，这指的是提前将权重加载到 VMEM 中，以便我们可以在矩阵乘法中隐藏加载权重的成本。例如，在一个普通的 Transformer 中，我们有时可以在注意力计算期间将较大的前馈权重加载到 VMEM 中，如果我们的内存带宽受限，这可以隐藏权重加载的成本。这要求我们的权重足够小或分片足够细，以便在 VMEM 中留有空间地容纳单个层。</d-footnote>

{% include figure.liquid path="assets/img/tpu-bandwidth.png" class="img-fluid" %}

**一块 TPU 芯片通常（但并非总是）包含两个共享内存的 TPU 核心，可以视为一个大型加速器**，其 FLOPs 是单个核心的两倍（称为“megacore”配置）。这适用于 v4、v5 和 v6 代 TPU（TPU v7 去除了 megacore，改为两个核心之间使用高带宽链路连接）。较早的 TPU 芯片拥有独立的内存，被视为两个独立的加速器（TPU v3 及更早版本）。像 TPU v5e 这样的推理优化芯片每块芯片只包含一个 TPU 核心。

{% include figure.liquid path="assets/img/cores.png" class="img-fluid img-small" %}

**芯片**排列在**每‘托盘’4个**的**组**中，通过**PCIe网络连接到CPU主机**。这是大多数读者熟悉的格式，通过Colab或单个TPU-VM暴露4个芯片（8个核心，通常被视为4个逻辑巨核）。对于像TPU v5e这样的推理芯片，每个主机有2个托盘，而不是1个，但每个芯片只有1个核心，因此8个芯片=8个核心。<d-footnote>在Cloud TPU VM上，每个托盘作为单独VM的一部分暴露出来，因此再次可见4个核心。</d-footnote>

{% include figure.liquid path="assets/img/pcie.png" class="img-fluid" %}

**PCIe 带宽有限：** 与 HBM $\leftrightarrow$ VMEM 链接类似，CPU $\leftrightarrow$ HBM PCIe 连接具有特定的带宽，这限制了从主机内存加载到 HBM 或相反方向的速度。例如，TPU v4 的 PCIe 带宽每个方向为 16GB/秒，几乎是 HBM 的 100 倍慢。我们 *可以* 将数据加载/卸载到主机（CPU）RAM 中，但速度并不快。

## TPU 网络通信

**芯片通过 Pod 中的 ICI 网络相互连接**。在早期的几代产品（TPU v2 和 TPU v3）、推理芯片（例如 TPU v5e）以及 Trillium（TPU v6e）中，ICI（“芯片间互连”）连接 4 个最近的邻居（通过边缘链接形成一个二维环面）。TPU v4 和 TPU v5p 连接到最近的 6 个邻居（形成一个三维环面）。请注意，这些连接**不**经过它们的主机，它们是芯片之间的直接链接。

{% include figure.liquid path="assets/img/ici-wraparound.png" class="img-fluid img-small" %}

环形结构将任意两个节点之间的最大距离从 $N$ 减少到 $N / 2$，从而使通信速度大幅提升。TPUs 还采用了一种“扭曲环形”配置，通过类似莫比乌斯环的拓扑结构包裹环形，以进一步减少节点之间的平均距离。

**TPU 捷豹（通过 ICI 连接）可以变得非常大：** 最大捷豹规模（称为 **超级捷豹**）对于 TPU v4 是 `16x16x16`，对于 TPU v5p 是 `16x20x28`。这些大型捷豹由 `4x4x4` 芯片的可重构立方体组成，这些立方体通过 [光学绕回链路](https://arxiv.org/pdf/2208.10041)<d-footnote> 连接。光学交换机只是一个具有相同 ICI 带宽的可重构连接。它只是让我们在保留绕回链路的同时连接立方体。</d-footnote> 我们可以重新配置以连接非常大的拓扑结构。

{% include figure.liquid path="assets/img/tpu-rack.png" class="img-fluid" %}

较小的拓扑结构（例如 `2x2x1`、`2x2x2`）也可以请求，但不支持绕行。这是一个重要的注意事项，因为它通常会使大多数通信的时间翻倍。任何完整立方体的整数倍（例如 `4x4x4` 或 `4x4x8`）都将通过光开关实现绕行。<d-footnote>请注意，`2x2x4` 不会实现任何绕行，因为绕行是由光开关提供的，而光开关仅在完整立方体上可用。TPU v5e 8x16 _确实_ 在较长轴上会有绕行，因为它不使用可重构的光网络。</d-footnote>

{% include figure.liquid path="assets/img/subslices.png" class="img-fluid" %}

TPU v5e 和 Trillium 模组由一个单一的 `16x16` 2D 扭环组成，沿任何轴方向的尺寸为 16（意味着 `8x16` 在长轴方向上有绕环）。TPU v5e 和 v6e（Trillium）无法扩展到超过 16x16 的扭环，但模组之间仍可通过标准数据中心网络（DCN）进行通信，该网络将 TPU 主机相互连接。同样，也可以在不使用绕环的情况下请求较小的拓扑结构，适用于 $<16$ 维度。

{% include figure.liquid path="assets/img/more-subslices.png" class="img-fluid" %}

**这种最近邻连接方式是TPU和GPU之间的关键区别**。GPU通过一系列交换机进行连接，这些交换机近似地实现了每块GPU之间的点对点连接，而不是像TPU那样使用本地连接。通常，一个节点内的GPU（H100为8块，B200 NVL72最多可达72块）是直接相连的，而更大的拓扑结构则需要每个GPU之间进行O(log(N))次跳转。一方面，这意味着GPU可以在少量跳转内发送任意数据。另一方面，TPU由于NVLink交换机成本较高，因此价格显著更低，连接方式更简单，并且由于每块设备的链接数量和带宽是恒定的，因此可以扩展到更大的拓扑结构。更多信息请参见[此处](../gpus#networking)。

**ICI 相比 DCN 非常快，但仍比 HBM 带宽慢。** 例如，一个 [TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture) 具有：

* 每个芯片提供 `2.8e12` 字节/秒（2.8 TB/s）的 HBM 带宽。  
* 每个轴提供 `9e10` 字节/秒（90 GB/s）的 ICI 带宽，每个芯片有 3 个轴。<d-footnote> 上一页列出的带宽为 100 GB/s，与此处列出的略有不同。TPU ICI 链接的带宽会根据执行的操作略有不同。通常可以放心使用本文档中的数字。</d-footnote>  
* 每个 TPU 提供 `6.25e9` 字节/秒（6.25 GB/s）的 DCN（出站）带宽（通过每台主机上的 1-2 个 NIC）。<d-footnote> TPU v6e 和 TPU7x 提供 12.5e9 字节/秒，而 v5e 提供 3.125e9 字节/秒。</d-footnote>

这意味着当我们把模型分布在多个芯片上时，需要小心避免因跨设备通信速度较慢而成为 MXU 的瓶颈。

**多切片训练：** 一组通过 ICI 连接的 TPUs 被称为一个 **切片**。不同切片之间可以通过 DCN 相互连接，例如将不同机箱中的切片连接起来。由于 DCN 的连接速度远慢于 ICI，我们应该尽量减少计算需要等待 DCN 数据的时间。DCN 是主机到主机的连接，因此要通过 DCN 将缓冲区从 TPU 传输到 TPU，首先需要通过 PCIe 传输到主机，然后通过网络传出，再通过目标主机网络传入，最后通过 PCIe 写入 HBM。

## 关键要点

* TPU 是简单的，大多数情况下可以看作是一个矩阵乘法单元连接到内存（非常快），通过 ICI 连接到其他芯片（较快），并通过 DCN 连接到数据中心其余部分（相对较快）。

* 通信受我们各种网络带宽的限制，按速度排序如下：
  * HBM 带宽：TensorCore 与其相关 HBM 之间的带宽。
  * ICI 带宽：TPU 芯片与其最近的 4 或 6 个邻居之间的带宽。
  * PCIe 带宽：CPU 主机与其相关芯片托盘之间的带宽。
  * DCN 带宽：多个 CPU 主机之间的带宽，通常这些主机未通过 ICI 连接。

* **在一片中，TPU 仅通过 ICI 连接到其最近的邻居。** 这意味着一片中远处芯片之间的 ICI 通信需要先经过中间的芯片。

* **权重矩阵在两个维度上都需要填充到至少 128**（在 TPU v6e 上为 256），以填满 MXU（实际上，较小的维度会被填充到 128）。

* **较低精度的矩阵乘法通常更快。** TPUs 在支持的情况下，可以将 int8 或 int4 操作的速度提高到 bfloat16 FLOPs 的约 2 倍/4 倍。VPU 操作仍然在 fp32 中执行。

* 为了避免限制 TPU 算力单元，我们需要 **确保每条通道上的通信量与其速度成正比**。

### TPU 规格

这是我们芯片的一些具体数字：

|模型|Pod 大小|主机规模|HBM 容量/芯片|HBM 显存带宽/芯片（字节/秒）|FLOPs/s/芯片（bf16）|FLOPs/s/芯片（int8）|
| :----------------------------------------- | :------: | :-------: | :---------------: | :-------------------: | :-----------------: | :-----------------: ||<span class="nowrap-header">TPU v3</span>|32x32|4x2|32GB|9.0e11|1.4e14|1.4e14|
|<span class="nowrap-header">TPU v4p</span>|16x16x16|2x2x1|32GB|1.2e12|2.75e14|2.75e14|
|<span class="nowrap-header">TPU v5p</span>|16x20x28|2x2x1|96GB|2.8e12|4.59e14|9.18e14|
|<span class="nowrap-header">TPU v5e</span>|16x16|4x2|16GB|8.2e11|1.97e14|3.94e14|
|<span class="nowrap-header">TPU v6e</span>|16x16|4x2|32GB|1.6e12|9.20e14|1.84e15|
|<span class="nowrap-header">TPU7x</span>|4x4x576|2x2x1|192GB|7.4e12|2.30e15|4.61e15|

主机规模指的是连接到单个主机的TPU拓扑结构（例如，TPU v5e具有单个CPU主机，通过4x2拓扑连接到8个TPU）。有关最新一代的更多详细信息，请参见[TPU7x文档](https://docs.cloud.google.com/tpu/docs/tpu7x)。以下是互连示意图：

|模型|ICI 带宽/链路（单向，字节/秒）|ICI 带宽/链路（双向，字节/秒）|
| :---------- | :----------------------------: | :-------------------------: ||**TPU v3**|1.0e11|2.0e11|
|**TPU v4p**|4.5e10|9.0e10|
|**TPU v5p**|9.0e10|1.8e11|
|**TPU v5e**|4.5e10|9.0e10|
|**TPU v6e**|9.0e10|1.8e11|
|**TPU7x**|9.0e10|1.8e11|

我们同时包含单向（单向）带宽和双向（双向）带宽，因为单向带宽更贴近硬件，但双向带宽在涉及完整环形结构的方程中更为常见。<d-footnote>我们所说的双向（双向）带宽是指在单条链路上两个方向可以发送的总字节数，或者说，假设我们能高效利用两条链路，从单个TPU沿特定轴方向发出的总字节数。当存在一个功能正常的环形结构，即在特定轴上存在环绕连接时，这种情况成立。这种情况在推理芯片上表现为拥有完整的16轴，在训练芯片（v*p）上表现为某个轴是4的倍数时出现。我们更倾向于使用双向带宽，因为它在涉及双向通信的计算中频繁出现。</d-footnote>

PCIe 带宽通常每块 TPU 约为 `1.6e10` 字节 / 秒（TPU v6e 为 `3.2e10`），而 DCN 带宽通常每块 TPU 约为 `6.25e9` 字节 / 秒（TPU v6e 和 TPU7x 为 `12.5e9`，TPU v5e 为 `3.125e9`）。

## 习题解答

这些数字看起来有些枯燥，但它们能让你对模型性能做出基本的屋顶线估计。我们通过几个例子来说明为什么这很有用。你将在第 3 部分看到更多例子。

**问题 1 [限制大语言模型延迟]:** 假设你想从一个 200B 参数的 bf16 模型中进行采样，该模型分布在 32 个 TPU v4p 上。将所有参数从 HBM 加载到 systolic array 需要多长时间？*提示：使用上面的数字。*

{% details 点击此处查看答案。 %}

**答案：** 我们在 32 个芯片上加载 `sizeof(bf16) * 200e9 = 400e9` 字节，意味着每个芯片加载 12.5e9 字节，每个芯片的 HBM 带宽为 1.23e12。因此，加载耗时约 10 毫秒。

这非常酷，因为*这是从模型中采样延迟的合理下界*。每次采样步骤都需要从 HBM 加载所有参数，因此它不能少于 10 毫秒。在实践中，当批量大小较小时，这几乎可以实现。

{% enddetails %}

**问题 2 [TPU 详情]:** 考虑一个完整的 TPU v5e 机架。总共有多少个 CPU 主机？有多少个 TPU TensorCores？整个机架的总 FLOPs/s 是多少？总共有多少 HBM？对 TPU v5p 机架执行相同的练习。

{% details 点击此处查看答案。 %}

**答案：** 对于 TPU v5e，每个 pod 是 `16x16`，每个主机是一个 4x2 切片，因此我们有 `16*16 / 8 = 32` 个主机。对于 TPU v5e，每个 TPU 只有一个核心，因此我们有 256 个 TensorCores。总的 FLOPs/s 在 bfloat16 中是 `16*16*2e14 = 5.1e16`。每个芯片有 16GB 的 HBM，因此总共有 `256 * 16 = 4TB` 的内存。

对于一个完整的 TPU v5p 机架，我们有 `16x20x28` 个芯片，每个主机是 2x2x1，因此我们有 `(16*20*28) / (2*2) = 2,240` 个主机。对于 TPU v5p，每个 TPU 有两个 TensorCores，因此我们有 `8960 * 2 = 17,920` 个核心。总的 FLOPs/s 为 `8960 * 4.59e14 = 4.1e18`（使用 bfloat16）。每个芯片有 96GB 的 HBM，因此总共有 `8960 * 96 = 860TB` 的内存。

{% enddetails %}

**问题 3 [PCIe 操作强度]:** 假设我们被迫将一个大型权重矩阵 $A$ 类型 $\text{bf16}[D, F]$ 和一批激活值 $x$ 类型 $\text{bf16}[B, D]$ 存储在主机 DRAM 中，并希望在它们上执行矩阵乘法。这运行在单个主机上，并使用连接到它的单个 TPU v6e 芯片。你可以假设 $B \ll D$，以及 $F = 4D$（我们将在后续章节中看到为什么这些是合理的假设）。为了在 PCIe 上保持 FLOPs 限制，我们需要的最小批量大小 $B$ 是多少？假设 PCIe 带宽为 1.6e10 字节 / 秒。

{% details 点击此处查看答案。 %}

**答案：** 我们必须执行 $2BDF$ 次浮点运算，而每个芯片每秒可以执行 `9.2e14` 次浮点运算。因此，这需要 $2BDF / 9.2e14$ 秒来完成。我们需要从 DRAM 读取 $2DF + 2BD$ 字节，并将 $2BF$ 字节写回 DRAM。我们受到 PCIe 传输速度的限制，因此将数据传输到 TPU 和从 TPU 传输出去需要 $2 \cdot (BD + DF + BF) / 1.6e10$ 秒。由于我们希望计算耗时比权重加载更长，假设我们可以将所有权重加载与计算重叠，我们希望 $2BDF / 9.2e14 > 2 \cdot (BD + DF + BF) / 1.6e10$。我们可以利用 $B \ll D$ 以及 $F = 4D$ 的假设来简化，得到

$$\frac{8BD^2}{9.2 \times 10^{14}} > \frac{8D^2}{1.6 \times 10^{10}}$$

或

$$B > \frac{9.2 \times 10^{14}}{1.6 \times 10^{10}} \simeq 57{,}500$$

{% enddetails %}

**问题 4 [通用 matmul 延迟]:** 假设我们想要将一个 int8[16384, 4096] 的权重矩阵与一个大小为 int8[B, 4096] 的激活矩阵相乘，其中 B 是某个未知的批量大小。假设我们首先使用 1 个 TPU v5e。

1. 作为 B 的函数，这个乘法需要多长时间？*提示：计算从 HBM 加载数组需要多长时间以及乘法实际需要多长时间可能会有帮助。哪个步骤是瓶颈？*  
2. 如果我们想在 VMEM 中运行这个操作，作为 B 的函数，需要多长时间？

{% details 点击此处查看答案。 %}

**答案：** (1) 我们需要执行的操作数量为 $2 \cdot 4096 \cdot 16384 \cdot B = 1.3 \times 10^{8} \cdot B$。因此需要 $T_{\text{math}} = (1.3 \times 10^{8} \cdot B) / 3.94 \times 10^{14}$ 秒。我们需要从 HBM 加载 $16384 \cdot 4096 + 4096 \cdot B$ 字节到 VMEM，并从 VMEM 写回 $16384 \cdot B$ 字节到 HBM。这意味着需要 $T_{\text{comms}} = (6.7 \times 10^{7} + 2 \times 10^{4} \cdot B) / 8.2 \times 10^{11}$ 秒。假设通信和计算尽可能重叠，整个乘法操作大约需要

$$\max\{T_{\text{math}}, T_{\text{comms}}\} = \max\left\{\frac{1.3 \times 10^{8} \cdot B}{3.94 \times 10^{14}}, \frac{6.7 \times 10^{7} + 2 \times 10^{4} \cdot B}{8.2 \times 10^{11}}\right\}$$

当 $\frac{6.7 \times 10^{7} + 2 \times 10^{4} \cdot B}{8.2 \times 10^{11}} < \frac{1.3 \times 10^{8} \cdot B}{3.94 \times 10^{14}}$，或等效地，$B > 267$ 时，我们将受到 FLOPs 的限制。这个数值略微大于我们在[第 1 章](../roofline)中推导出的 240，因为我们考虑了 $D$ 和 $F$ 的全部影响。

(2) 如果我们是从 VMEM 加载数据，假设 VMEM 到 MXU 的带宽为 HBM $\leftrightarrow$ VMEM 带宽的 22 倍。这使我们的数据加载分母从 8.2e11 变为 1.80e13，我们得到 $B > 11$。请注意，实际上我们无法将所有 VMEM 带宽都用于加载权重矩阵，因此实际值会更接近 20。

{% enddetails %}

**问题 5 [ICI 带宽]:** 假设我们有一个 TPU v5e `4x4` 切片。假设我们想将类型为 `bf16[8, 128, 8192]` 的数组从 `TPU{0,0}` 发送到 `TPU{3, 3}`。假设 TPU v5e 的每跳延迟为 $1\mu s$。

1. 第一个字节何时能到达目的地？  
2. 整个传输需要多长时间？

{% details 点击此处查看答案。 %}

**答案：** 在 TPU v5e 中，我们具有 2D 连接性。由于我们只有 `4x4` 切片（没有大小为 16 的轴），因此没有环绕连接。因此，我们的目标芯片可以从两个端口接收数据，同样，我们的源芯片也可以从两个端口发送数据。我们需要传输的数据量为 `2 * 8 * 128 * 8192 = 1.7e7` 字节。我们可以同时从两个端口传输（即，将数组的一半向右发送，另一半向下发送），因此每秒可以传输 `2 * 4.5e10 = 9e10` 字节，这意味着传输整个数组大约需要 `1.7e7 / 9e10 = 188us`（假设我们受带宽限制）。在 `4x4` 切片中，芯片 $(0, 0)$ 和 $(3, 3)$ 之间有六个跳步，因为对于芯片数少于 16 的轴，没有环绕链接。由于每个跳步的延迟约为 $1\mu s$，第一个字节将在大约 `6us` 后到达，而整个传输将需要大约 `188 + 6 = 194us`，因为最后一个字节在离开源芯片后也必须依次经过六个跳步（一般来说，延迟和带宽项是相加的，但在这里延迟只是一个微小的修正）。

{% enddetails %}

**问题 6 [综合应用，困难]：** 假设你有一个大矩阵 **A**：`int8[128 * 1024, 128 * 1024]` 均匀分布在 TPU v5e 4x4 切片上，但每个芯片上都卸载到主机 DRAM 中。假设你想将整个数组复制到 TPU{0, 0} 并将其与向量 `bf16[8, 128 * 1024]` 相乘。这将需要多长时间？*提示：使用上面的数字。*

{% details 点击此处查看答案。 %}

**答案：** 让我们首先概述我们需要执行的操作。我们的数组大约为 16GB。根据上表，TPU v5e 主机具有 4x2 的拓扑结构，因此 4x4 需要 2 台主机。因此，由于我们的数组被均匀分片，每台主机实际上包含数组的 1/2，即 8GB。我们需要将这些分片全部复制到 TPU{0,0}，这给我们提供了两个选项：

1. 我们可以复制 DCN，然后通过 PCIe 将整个未分片数组加载到 HBM 中。  
2. 我们可以将分片数组加载到对应的 TPU 上，然后在 ICI 上执行 gather 操作，接着在 TPU{0,0} 上执行矩阵乘法。

应该很明显，选项（2）更优。与 ICI 相比，DCN 较慢，我们更倾向于通过许多 PCIe 链路加载一个大数组，而不是仅通过少数几条（主机 0 上的 8 条）。这是系统的一部分的示意图。如上所述，请注意 TPUs 通过 ICI 连接到其邻居（甚至跨主机），所有 TPUs 都通过 PCIe 连接到其主机 CPU，而主机之间通过 DCN 连接。

{% include figure.liquid path="assets/img/challenge-problem.png" class="img-fluid img-small" caption="每块芯片实际上都有自己的 PCIe 链接到其主机，尽管为了清晰起见，此处仅显示了一条。" %}

现在让我们计算每部分需要多长时间：

1. **PCIe负载**：我们正在通过16条PCIe链路加载16GB的数据块，每条链路的带宽为`1.6e10`字节/秒。因此这将需要大约63毫秒。

2. **ICI复制：** 每个TPU现在有16GB / 16 = 1GB的我们的数组。我们的ICI带宽是每条链路9e10字节/秒，双向传输，从上面的图中可以看出，在这种拓扑结构中，TPU{0,0}只使用了4条ICI链路中的2条。由于TPU{0,0}需要沿着两个轴接收总共15GB的数据，传输速率为`4.5e10`字节/秒/链路，我们可以将时间下限设为`15e9 / (4.5e10 * 2) = 167ms`。在实际中，这可能无法实现，因为负载非常不均衡，但可能在2倍以内。如第3节所示，执行一次完整的AllGather也需要大约`16e9 / (4.5e10 * 2)`，因此这已经接近最优。

3. **HBM $\rightarrow$ MXU 负载：** 为了执行最终的矩阵乘法，我们需要通过 HBM 带宽将这些 16e9 字节加上 bf16[8, 128 \* 1024] 数组（另一个 2MB，可以忽略不计）加载到 MXU 中，这将耗时 `16e9 / 8.2e11 = 20ms`。

4. **FLOPs:** 我们总共执行了 $$2 \cdot 8 \cdot 128 \cdot 1024 \cdot 128 \cdot 1024 = 2.7 \times 10^{11}$$ FLOPs，由于我们每秒可以执行 `1.97e14` bf16 FLOPs，因此耗时 1.4 毫秒。

总时间的上限是所有这些时间的总和，但由于 TPU 通常可以重叠这些操作，我们可以将其视为一个受最慢部分限制的流水线问题。假设这一假设成立，那么答案至少为 167 毫秒，若重叠不完美，可能更接近 200 毫秒。

{% enddetails %}

<h3 markdown=1 class="next-section">第二部分就到这里！第三部分将介绍分区和跨TPU通信，[点击此处](../sharding)。</h3>

## 附录

### 附录 A：更多关于 TPU 内部结构的内容

这里我们将更深入地探讨 TPU 的内部操作。除非另有说明，我们将提供 TPU v5p 的规格。

### VPU

VPU 是 TPU 的向量算术核心。VPU 由一个二维 SIMD 向量机器（**VPU**）组成，该机器执行诸如 vadd（向量加法）或 vmax（逐元素最大值）之类的逐元素算术操作，以及一组称为 **VREGs** 的向量寄存器，用于存储 VPU 和 MXU 的数据。

**VREGs:** 每个 TPU v5p 核心有 64 个 32 位 VREG（TPU v4 为 32 个），使每个核心的 VREG 内存总量约为 `64 * 8 * 128 * 4 = 256kB`（由于每个芯片有两个核心，整个芯片的总量为两倍）。一个 TPU v5p 每个周期可以从 VMEM 加载 3 个寄存器，并将 1 个寄存器写入 VMEM。

**VPU:** VPU 是一个形状为 `(8, 128)` 的二维向量算术单元，其中 128 维被称为通道轴，8 维被称为子通道轴。在 v5 中，每个 (通道, 子通道) 对包含 4 个独立的标准浮点 ALU。VPU 在其每个 ALU 中在一个周期内执行大多数算术指令（如 vadd 或向量加法），延迟为 2 个周期，因此例如在 v5 中，每个周期可以从 VREG 中将 4 对 f32 值相加。一个典型的 VPU 指令可能如下所示 `{v2 = vadd.8x128.f32 v0, v1}`，其中 v0 和 v1 是输入 VREG，v2 是输出 VREG。

所有车道和子车道在每个周期内以纯 SIMD 方式执行相同的程序，但每个 ALU 可以执行不同的操作。因此，例如可以在一个周期内处理 1 个 vadd 和 1 个 vsub，每个操作都作用于两个完整的 VREG，并将结果写入第三个。

**小测验 [计算 VPU 吞吐量]:** 使用上述信息，计算 TPU v5p 能执行多少向量 FLOPs/s。TPU v5p 的时钟频率约为 1.75GHz。

{% details 点击此处查看答案。 %}

*Answer*: 每个周期，每个核心可以在 `8 * 128` ALUs 上执行 4 条向量指令。这使每个核心达到 `8 * 128 * 4` FLOPs/周期，即 `8 * 128 * 4 * 1.75e9 = 7e12 FLOPs/s`。请注意，这个数值与每个核心约 `2e14` 的 MXU FLOPs/s 相比要小得多（大约小 30 倍）。

{% enddetails %}

**缩减操作：** 通常，跨子通道维度的通信或缩减操作比跨通道维度的更容易。例如，VPU 支持一种通道内洗牌操作，可以在大约一个周期内沿大小为 8 的轴进行滚动。这可以用来沿子通道维度高效地执行缩减操作（只需依次洗牌 4、2 和 1，并进行 3 次元素级求和）。

跨车道缩减要困难得多，需要一个称为 XLU（“跨车道单元”）的专用硬件单元，该单元速度较慢且成本较高。

**与 GPU 的对比：** 对于熟悉 NVIDIA GPU 的用户来说，VPU 中的每个 ALU 类似于一个 CUDA 核心，而单个 VPU 通道则类似于“Warp 调度器”，即通常由 32 个 CUDA 核心组成的集合，用于执行 SIMD 算术运算。通道内部的归约操作相对简单，但如果需要跨通道操作，则需要至少通过 VMEM/XLU/SMEM 进行传输，而这些传输速度要慢得多。更多详情请参见[GPU 部分](../gpus)。

### 标量核心

标量核心是 TPU 的控制单元。它获取并分发所有指令，执行从 HBM 到 VMEM 的传输，并可以被编程用于执行标量元数据工作。由于标量核心是单线程的，因此其一个副作用是，每个 TPU 核心每周期只能生成一个 DMA 请求。

为了说明这一点，单个标量核心控制一个VPU（包含4096个ALU）、4个MXU、2个XLUs和多个DMA引擎。每单位算力的控制高度倾斜，这是硬件效率的来源，但也限制了以任何有意义的方式进行数据相关向量化的能力。

### 附录 B： systolic array 是如何工作的？

TPU MXU 的核心是一个 `128x128` 系统阵列（TPU v6e 上的 `256x256`）。当系统阵列完全饱和时，它可以在 8 个时钟周期内执行一次 `bf16[8,128] @ bf16[128,128] -> f32[8,128]`<d-footnote> 如果你不熟悉这种表示法，它的意思是：将一个包含 bfloat16 元素的 `8x128` 矩阵与一个包含 bfloat16 元素的 `128x128` 矩阵相乘，并将结果存储在一个包含 float32 元素的 `8x128` 矩阵中。</d-footnote> 乘法。

* 核心来看，脉动阵列是一个 2D `128x128`（`=16,384`）的 ALU 网格，每个 ALU 都能够执行乘法和加法操作。  
* 权重（**W**，即 `128x128` 输入）从上方（称为 RHS）传递下来，而输入（**X**，即 `8x128` 输入）则从左侧（称为 LHS）传入。

这里是一个简化的动画，展示了将一组权重（蓝色）与一组激活值（绿色）相乘的过程。你会注意到权重（右侧）首先部分地对角线加载，然后激活值也以对角线方式输入。在下面的每一帧中，我们将所有重叠的绿色和蓝色单元相乘，将结果与从上方传入的任何残差相加，然后将结果依次向下传递一个单元。

{% include figure.liquid path="assets/img/systolic-array.gif" %}

这是该动画的更一般版本，展示了计算结果被流式输出：

{% include figure.liquid path="assets/img/systolic-array2.gif" class="img-small" %}

这是展示如何在多个 RHS 和 LHS 数组之间进行流水线处理的示意图：

{% include figure.liquid path="assets/img/systolic-array-pipelining.png" class="img-fluid" %}

在权重（RHS）和激活值（LHS）被加载时，初始阶段会出现一个流水线气泡。在初始气泡之后，新的输入和权重可以被加载而不会产生额外的气泡。

这是 bf16[2, 3] x bf16[3, 3] 矩阵乘法的一个较差的动画演示，你可以想象成一个 2x3 权重矩阵与一个批量大小为 1、尺寸为 3 的输入激活值的矩阵乘法。与之前的幻灯片相比，这个动画的方向发生了旋转，输入流向右侧而不是下方，但你可以大致看到结构。

{% include figure.liquid path="assets/img/systolic-array-bad.gif" class="img-small" %}

我们可以高效地进行流水线处理，以乘法大矩阵，而不会产生过大的流水线气泡。话虽如此，重要的是我们的矩阵形状要大于 MXU 的边长维度，通常为 128x128。一些 TPUs（自 TPU v3 起）拥有多个 MXUs，TPU v3 为 2 个，TPU v4/5 为 4 个，因此我们需要确保分块维度大于 128 * MXU 的数量。[这里](https://www.youtube.com/watch?v=sJltBQ4MOHA)有一个很好的动画演示。

Trillium（TPU v6e）拥有 `256x256` 系统阵列，这意味着它可以每周期执行 4 倍的 FLOPs。这也意味着你的张量维度需要大两倍，才能充分利用 MXU。

[这篇博客文章](https://fleetwood.dev/posts/domain-specific-architectures#google-tpu)还有另一个关于固定权重矩阵的 systolic array 乘法的优秀动画。
