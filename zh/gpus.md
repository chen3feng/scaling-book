---
layout: distill
title: "如何思考 GPU"
description: "我们在 Google 非常喜欢 TPUs，但 GPUs 同样也很出色。本章将深入探讨 GPUs 的世界——每块芯片的工作原理、它们是如何连接在一起的，以及这对 LLM 的意义，特别是与 TPUs 相比的情况。虽然 NVIDIA、AMD、Intel 等公司提供了多种 GPU 架构，但在这里我们将专注于 NVIDIA 的 GPU。本节建立在 <a href='https://jax-ml.github.io/scaling-book/tpus/'>Chapter 2</a> 和 <a href='https://jax-ml.github.io/scaling-book/training'>Chapter 5</a> 的基础上，因此建议您先阅读它们。"
date: 2025-08-18
future: true
htmlwidgets: true
hidden: false

section_number: 12

previous_section_url: "../conclusion"
previous_section_name: "Part 11: Conclusion"

next_section_url:
next_section_name: "The End"

bibliography: main.bib

giscus_comments: true

authors:
  - name: Jacob Austin<sup>†</sup>
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: <sup>†</sup>Google DeepMind
  - name: Swapnil Patil<sup>†</sup>
    url: "https://www.linkedin.com/in/swapnil-patil-5b47a068"
  - name:  Adam Paszke<sup>†</sup>
    url: https://x.com/apaszke
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope
    affiliations:
      name: <sup>*</sup>MatX

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: What Is a GPU?
  - subsections:
    - name: Memory
    - name: "Summary of GPU specs"
    - name: GPUs vs. TPUs at the chip level
    - name: "Quiz 1: GPU hardware"
  - name: Networking
  - subsections:
    - name: At the node level
    - name: "Quiz 2: GPU nodes"
    - name: Beyond the node level
    - name: "Quiz 3: Beyond the node level"
  - name: How Do Collectives Work on GPUs?
  - subsections:
    - name: Intra-node collectives
    - name: Cross-node collectives
    - name: "Quiz 4: Collectives"
  - name: "Rooflines for LLM Scaling on GPUs"
  - subsections:
    - name: "Data Parallelism"
    - name: "Tensor Parallelism"
    - name: "Expert Parallelism"
    - name: "Pipeline Parallelism"
    - name: "Examples"
    - name: "TLDR of LLM scaling on GPUs"
    - name: "Quiz 5: LLM rooflines"
  - name: "Acknowledgements and Further Reading"
  - name: "Appendix"
  - subsections:
    - name: "Appendix A: How does this change with GB200?"
    - name: "Appendix B: More networking details"

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

## 什么是 GPU？

现代 ML GPU（例如 H100、B200）基本上是由多个专门用于矩阵乘法（称为 **Streaming Multiprocessors** 或 **SMs**）的计算核心连接到一根高速内存（称为 **HBM**）组成。以下是一张示意图：

{% include figure.liquid path="assets/gpu/gpu-diagram.png" class="img-fluid" link="true" caption="<b>图：</b> 一张展示 H100 或 B200 GPU 抽象布局的示意图。H100 有 132 个 SM，而 B200 有 148 个。我们使用“Warp Scheduler”这一术语较为宽泛地描述一组 32 个 CUDA SIMD 核心 <i>和</i> 以及将工作分配给它们的调度器。请注意，这看起来多么像一个 TPU！" %}

每个 SM（类似于 TPU 的 Tensor Core）都有一个专用的矩阵乘法核心（不幸地也被称为 **Tensor Core**<d-footnote>GPU 的 Tensor Core 是 SM 的矩阵乘法子单元，而 TPU 的 Tensor Core 是包含 MXU、VPU 和其他组件的上层单元。</d-footnote>），一个向量算术单元（称为 **Warp Scheduler**<d-footnote>NVIDIA 没有为这个单元起一个好名字，因此我们只能在几个不那么糟糕的选项中选择最好的一个。Warp Scheduler 主要负责将工作分配给一组 CUDA 核心，但在这里我们用它来描述控制单元及其所控制的一组核心。</d-footnote>），以及一个快速的片上缓存（称为 **SMEM**）。与 TPU 最多只有 2 个独立的“Tensor Core”不同，现代 GPU 拥有超过 100 个 SM（H100 上有 132 个）。每个 SM 的性能远不如 TPU 的 Tensor Core，但整体系统更加灵活。每个 SM 几乎完全独立，因此 GPU 可以同时执行数百个独立任务。<d-footnote>尽管 SM 是独立的，但为了达到最佳性能，它们通常必须进行协调，因为它们共享一个容量有限的 L2 缓存。</d-footnote>

让我们更详细地看一下 H100 SM：

{% include figure.liquid path="assets/gpu/blackwell-sm.png" class="img-small" link="true" caption="<b>图：</b> 一张 H100 SM（<a href='https://wccftech.com/nvidia-hopper-gh100-gpu-official-5nm-process-worlds-fastest-hpc-chip-80-billion-transistors-hbm3-memory/'>来源</a>）的示意图，展示了 4 个 <i>子分区</i>，每个子分区包含一个 Tensor Core、Warp 调度器、寄存器文件以及不同精度的 CUDA 核心组。底部附近的 'L1 Data Cache' 是 256kB 的 SMEM 单元。B200 的结构类似，但增加了大量 Tensor 内存（TMEM），用于为庞大的 Tensor Core 提供数据。" %}

每个 SM 被划分为 4 个相同的象限，NVIDIA 将其称为 **SM 子分区**，每个子分区包含一个 Tensor Core、16k 个 32 位寄存器，以及一个称为 Warp Scheduler 的 SIMD/SIMT 向量算术单元，其通道（ALUs）被 NVIDIA 称为 **CUDA 核心**。每个分区的核心组件可以说是 Tensor Core，它执行矩阵乘法并构成了其 FLOPs/s 的绝大多数，但它并不是唯一值得关注的组件。

* **CUDA 核心：** 每个子分区包含一组称为 CUDA 核心的 ALU，用于执行 SIMD/SIMT 向量算术。每个 ALU 通常每个周期可以执行一个算术操作，例如 f32.add.<d-footnote>较新的 GPU 支持 FMA（融合乘加）指令，从技术上讲，每个周期可以执行两个 FLOPs，NVIDIA 毫不吝啬地利用这一点来使他们的规格翻倍。</d-footnote> 每个子分区包含 32 个 fp32 核心（以及少量的 int32 和 fp64 核心），它们在每个周期内都执行相同的指令。与 TPU 的 VPU 类似，CUDA 核心负责 ReLU、逐点向量操作和归约（求和）。<d-footnote>历史上，在 Tensor Core 引入之前，CUDA 核心是 GPU 的主要组件，用于渲染，包括光线-三角形相交和着色。在当今的游戏 GPU 上，它们仍然承担了大部分的渲染工作，而 TensorCores 用于上采样（DLSS），这使得 GPU 可以在较低的分辨率（像素更少 = 工作量更少）下进行渲染，并使用机器学习进行上采样。</d-footnote>

* **张量核心（Tensor Core，TC）：** 每个子分片都有自己的张量核心，它是一种专用的矩阵乘法单元，类似于 TPU 的 MXU。张量核心代表了 GPU 的绝大多数 FLOPs/s（例如，在 H100 上，我们有 990 bf16 TC TFLOP/s，而 CUDA 核心仅有 66 TFLOPs/s）。
  * 132 个 SM 以 1.76GHz 运行时，[990 bf16 TFLOPs/s](https://www.nvidia.com/en-us/data-center/h100/) 意味着每个 H100 TC 每周期可以执行 `7.5e12 / 1.76e9 / 4 ~ 1024` bf16 FLOPs，大致相当于 8x8x8 的矩阵乘法。<d-footnote>NVIDIA 并未公开许多 TC 硬件细节，因此这更像是一个猜测而非确凿事实 – 当然，它并未说明 TC 的具体实现方式。我们知道 V100 每个 TC 每周期可以执行 256 FLOPs，A100 可以执行 512，H100 可以执行 1024，而 B200 的细节尚未公布，但考虑到 `2250e12 / (148 * 4 * 1.86e9)` 大约为 2048，B200 很可能可以执行约 2048 FLOPs/TC/cycle。一些更多细节在此处 <a href='https://forums.developer.nvidia.com/t/how-to-calculate-the-tensor-core-fp16-performance-of-h100/244727'>here</a> 得到了确认。</d-footnote>
  * 与 TPU 类似，GPU 可以在更高吞吐量下执行更低精度的矩阵乘法（例如，H100 的 fp8 FLOPs/s 是 fp16 的 2 倍）。低精度训练或服务可以显著加快速度。
  * 自 Volta 以来的每一代 GPU 都在前一代的基础上增加了 TC 的规模（[有关此内容的好文章](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)）。在 B200 上，TC 已经变得如此庞大，以至于无法将输入存储在 SMEM 中，因此 B200 引入了一种新的内存空间，称为 TMEM。<d-footnote>在 Ampere 中，张量核心可以从单个 warp 获取输入，而在 Hopper 中则需要完整的 SM（warpgroup），在 Blackwell 中则从 2 个 SM 获取输入。在 Blackwell 中，矩阵乘法也变得如此庞大，以至于参数（特别是累加器）不再能放入寄存器内存/SMEM 中，因此 Blackwell 增加了 TMEM 来应对这一问题。</d-footnote>

**CUDA 核心比 TPU 的 VPU 更加灵活：** 自 V100 以来，GPU 的 CUDA 核心使用所谓的 SIMT（*Single Instruction Multiple Threads*）编程模型，而 TPU 使用的是 SIMD（*Single Instruction Multiple Data*）模型。与 TPU 的 VPU 中的 ALU 类似，子分区内的每个 CUDA 核心在每个周期内必须执行相同的操作（例如，如果一个核心在执行两个浮点数的加法，那么子分区中的每个其他 CUDA 核心也必须执行相同的操作）。然而，与 VPU 不同的是，每个 CUDA 核心（或 CUDA 编程模型中的“线程”）都有自己的指令指针，并且可以独立地被 _编程_。当同一个 warp 中的两个线程被指示执行不同的操作时，实际上会 _同时_ 执行这两个操作，屏蔽掉不需要执行发散操作的核。

{% include figure.liquid path="assets/gpu/warp-divergence.png" class="img-fluid" caption="<b>图：</b> 线程组内 warp 分歧的一个例子（<a href='https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf'>来源</a>）。白色区域表示至少部分物理 CUDA 核心的停顿" %}

这使得在线程级别能够进行灵活的编程，但代价是如果线程发散过于频繁，性能会悄然下降。线程在可访问的内存方面也更加灵活；虽然 VPU 只能对连续的内存块进行操作，但 CUDA 核心可以访问共享寄存器中的单个浮点数，并维护每个线程的状态。

**CUDA 核心调度也更加灵活：** SM 的运行方式有点像多线程 CPU，它们可以“调度”许多程序（**线程束**）并行执行（每个 SM 最多可同时执行 64 个线程束），但每个 _线程束调度器_ 在每个时钟周期内只执行一个程序。<d-footnote> 被调度到某个 SM 上的线程束称为“驻留”线程束。</d-footnote> 线程束调度器会自动在活跃的线程束之间切换，以隐藏诸如内存加载之类的 I/O 操作。与之相比，TPU 通常是单线程的。

### 内存

除了算力单元之外，GPU 还具有层次化的内存结构，最大的是 HBM（主 GPU 内存），然后是一系列较小的缓存（L2、L1/SMEM、TMEM、寄存器内存）。

* **寄存器：** 每个子分区都有自己的寄存器文件，包含 16,384 个 32 位字（`4 * 16384 * 4 = 256kiB` 每个 SM），可被 CUDA 核心访问。  
  * 每个 CUDA 核心一次只能访问最多 256 个寄存器，因此尽管我们可以在每个 SM 上调度多达 64 个“驻留线程束”，但如果每个线程使用 256 个寄存器，一次只能容纳 8 个（`256 * 1024 / (4 * 32 * 256)`）。

* **SMEM（L1 缓存）：** 每个 SM 都有自己的 256kB 片上缓存，称为 SMEM，既可以由程序员控制为“共享内存”，也可以被硬件用作片上缓存。SMEM 用于存储激活值和 TC 矩阵乘法的输入。

* **L2 缓存：** 所有 SM 共享 <d-footnote>从技术上讲，L2 缓存被分为两部分，因此在 H100 上，一半的 SM 可以分别访问 25MB 的缓存。这两部分之间有一个连接，但带宽较低。</d-footnote> 这是一个相对较大的约 50MB L2 缓存，用于减少对主内存的访问。
  * 这与 TPU 的 VMEM 大小相似，但它**慢得多**，而且不是程序员可控的。这导致了一种“远距离幽灵作用”，程序员需要修改内存访问模式以确保 L2 缓存被充分利用。<d-footnote> 由于 L2 缓存被所有 SM 共享，这实际上迫使程序员以一种相当协调的方式运行 SM，尽管从原则上讲，它们是独立的单元。</d-footnote>
  * NVIDIA 没有公布其芯片的 L2 带宽，但已被 [测量](https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth) 约为 5.5TB/s。这大约是 HBM 带宽的 1.6 倍，而且是全双工的，因此有效双向带宽接近 3 倍。相比之下，TPU 的 VMEM 是其 2 倍大 *且* 带宽更高（约 40TB/s）。

* **HBM:** GPU 的主要显存，用于存储模型权重、梯度、激活值等。
  * HBM 容量从 Volta 的 32GB 增加到 Blackwell（B200）的 192GB。
  * 从 HBM 到 CUDA Tensor Core 的带宽称为 HBM 带宽或显存带宽，在 H100 上约为 3.35TB/s，在 B200 上为 9TB/s。

### GPU 规格摘要

以下是近期 GPU 模型的规格摘要。同一款 GPU 的不同变体在 SM 数量、时钟频率和 FLOPs 方面存在一些差异。以下是内存容量数据：

|GPU|生成|时钟频率|SMs/芯片|SMEM容量/SM|L2 容量/芯片|HBM 容量/芯片|
| :---: | :--------: | :-------------: | :------: | :--------------: | :--------------: | :---------------: ||V100|Volta|1.25GHz/1.38GHz|80|96kB|6MB|32GB|
|A100|安培|1.10GHz/1.41GHz|108|192kB|40MB|80GB|
|H100|Hopper|1.59GHz/1.98GHz|132|256kB|50MB|80GB|
|H200|Hopper|1.59GHz/1.98GHz|132|256kB|50MB|141GB|
|B200|Blackwell|?|148|256kB|126MB|192GB|

所有生成的芯片每个SM都有256kB的寄存器内存。Blackwell每个SM还增加了256kB的TMEM。以下是每颗芯片的FLOPs和带宽数据：

|GPU|生成|HBM 显存带宽/芯片|FLOPs/s/芯片（bf16/fp16）|FLOPs/s/芯片（fp8/int8）|FLOPs/s/芯片（fp4）|
| :---: | :--------: | :---------: | :----------------------: | :---------------------: | :----------------: ||V100|Volta|9.0e11|—|—|—|
|A100|安培|2.0e12|3.1e14|6.2e14|—|
|H100|Hopper|3.4e12|9.9e14|2.0e15|—|
|H200|Hopper|4.8e12|9.9e14|2.0e15|—|
|B200|Blackwell|8.0e12|2.3e15|4.5e15|9.0e15|

我们排除 B100，因为它并未大规模生产。<d-footnote> 虽然 NVIDIA 生产了 B100 一代，但它们仅短暂销售和生产，据称是由于设计缺陷，使其无法接近其宣称的规格。由于热量和功耗问题，它们在不降频的情况下难以达到峰值 FLOPs。</d-footnote> 某些规格会略微依赖于 GPU 的具体版本，因为 NVIDIA GPU 并不像 TPU 那样标准化。

这里是一份有助于比较 GPU 和 TPU 组件的快速参考表：

|GPU|TPU|这是什么？|
| :---------------------------: | :---------: | :-----------------------------------: ||流式多处理器 (SM)|张量核心|核心“单元”包含其他组件|
|Warp 调度器|VPU|SIMD向量算术单元|
|CUDA 核心|VPU ALU|SIMD ALU|
|SMEM（L1 缓存）|VMEM|高速片上缓存内存|
|张量核心|MXU|矩阵乘法单元|
|HBM（又称 GMEM）|HBM|高带宽大容量内存|

### 芯片层面的 GPU 与 TPU 对比

GPU 最初用于渲染视频游戏，但自 2010 年代深度学习兴起以来，它们越来越像专用的矩阵乘法机器，换句话说，越来越像 TPU。<d-footnote>在深度学习热潮之前，GPU（图形处理单元）确实处理图形，主要是用于视频游戏。视频游戏用数以百万计的小三角形来表示物体，游戏会将这些三角形渲染（或“光栅化”）成一个 2D 图像，并以每秒 30-60 次的频率显示在屏幕上（这个频率称为帧率）。光栅化涉及将这些三角形投影到相机的坐标系中，并计算哪些三角形覆盖了哪些像素，每秒需要进行数十亿次这样的计算。可想而知，这非常昂贵，而这仅仅是开始。接下来你还需要通过将可能多个半透明三角形的颜色组合起来，为每个像素着色。GPU 被设计为能极快地执行这些操作，注重多功能性；你需要同时运行许多不同的 GPU 工作负载（称为“着色器”），没有任何一个操作占据主导地位。因此，面向消费级图形的 GPU 可以执行矩阵乘法，但这并不是它们的主要功能。</d-footnote>在某种程度上，这段历史解释了为什么现代 GPU 看起来是现在的样子。它们并不是专门为大语言模型或机器学习模型设计的，而是作为通用加速器设计的，硬件旨在实现一定程度的“通用性”，这既是优势也是劣势。与 TPU 相比，GPU 在应用于新任务时“通常都能正常工作”，对良好编译器的依赖也更少。但这也使它们更难进行推理或发挥出顶线性能，因为许多编译器特性都可能导致瓶颈。

**GPU 更加模块化。** TPU 有 1-2 个大型 Tensor Core，而 GPU 有数百个小型 SM。同样，每个 TC 都包含一个由 4 个独立可编程的 8x128 单元组成的大型 VPU（总共 4096 个 ALU）；相比之下，H100 有 132 * 4 = 528 个独立的 SIMD 单元，每个单元宽 32（总共 16k 个 ALU）。以下是对 GPU 与 TPU 的 1:1 对比，突出了这一点：

|GPU|TPU|H100 #|TPU v5p #|
| :---------------------------: | :----------------------: | :----: | :-------: ||SM（流式多处理器）|张量核心|132|2|
|Warp 调度器|VPU 插槽|528|8|
|SMEM（L1 缓存）|VMEM|32MB|128MB|
|寄存器|向量寄存器（VRegs）|32MB|256kB|
|张量核心|MXU|528|8|

这种模块性上的差异一方面使 TPUs 更加便宜且更容易理解，但另一方面也给编译器带来了更大的负担，要求它必须正确地完成各项工作。由于 TPUs 只有一个控制线程，并且仅支持向量化 VPU 级别的指令，编译器需要手动对所有内存加载和 MXU/VPU 工作进行流水线处理，以避免停滞。GPU 程序员只需启动数十个不同的内核，每个内核都在一个完全独立的 SM 上运行。然而，这些内核可能会因为频繁访问 L2 缓存或无法合并内存加载而导致性能极差；由于硬件控制了运行时的许多方面，使得理解幕后发生的情况变得困难。因此，TPUs 通常只需较少的工作就能更接近峰值性能。

**历史上，单个 GPU 的性能（和价格）比同等的 TPU 更强（更贵）：** 一块 H200 的 FLOPs/s 接近 TPU v5p 的 2 倍，HBM 为 TPU v5p 的 1.5 倍。同时，Google Cloud 上 TPU v5p 的标价约为 \\$10/hour for an H200 compared to \\$4/小时。TPU 通常比 GPU 更依赖将多个芯片通过网络连接在一起。

**TPU 拥有更多的高速缓存内存。** TPU 还拥有比 GPU 的 SMEM（+TMEM）更多的 VMEM，而且这种内存可以以一种让权重和激活值被极快加载和使用的方式进行存储。如果你能持续地将模型权重存储或预取到 VMEM 中，这可能会使它们在 LLM 推理时更快。

### Quiz 1: GPU 硬件

以下是一些需要解决的问题，用于测试上述内容的一些知识点。提供了答案，但可能在查看答案之前，最好动手尝试回答这些问题。

**问题 1 [CUDA 核心]:** H100 有多少个 fp32 CUDA 核心（ALUs）？B200 有多少个？这与 TPU v5p 中独立 ALUs 的数量相比如何？

{% details 点击此处查看答案。 %}

**答案：** 一个 H100 拥有 132 个 SM，每个 SM 包含 4 个子分区，每个子分区包含 32 个 fp32 CUDA 核心，因此我们 `132 * 4 * 32 = 16896` CUDA 核心。一个 B200 拥有 `148` SM，因此总共有 `18944`。一个 TPU v5p 拥有 2 个 TensorCores（通常通过 Megacore 连接），每个 TensorCore 包含一个 VPU，具有 (8, 128) 条车道，每条车道有 4 个独立的 ALU，因此 `2 * 4 * 8 * 128 = 8192` ALU。这大约是 H100 向量车道数量的一半，运行频率大致相同。

{% enddetails %}

**问题 2 [向量 FLOPs 计算]**: 一个 H100 拥有 132 个 SM，运行频率为 1.59GHz（最高可提升至 1.98GHz）。假设每个周期每个 ALU 可执行一个向量操作。每秒可以执行多少个向量 fp32 FLOPs？在频率提升的情况下呢？这与 matmul FLOPs 相比如何？

{% details 点击此处查看答案。 %}

**答案：** `132 * 4 * 32 * 1.59e9 = 26.9TFLOPs/s`。通过提升其 33.5 TFLOPs/s。这只有 [规格表](https://www.nvidia.com/en-us/data-center/h100/) 中报告的一半，因为从技术上讲，我们可以在一个周期内执行一个 FMA（融合乘加），这算作两个 FLOPs，但在大多数情况下这并不实用。我们可以实现 990 bfloat16 矩阵乘法 TFLOPs/s，因此不考虑 FMA 的话，Tensor Cores 的 FLOPs/s 约为 30 倍。

{% enddetails %}

**问题 3 [GPU 矩阵乘法强度]:** H100 上 fp16 矩阵乘法的峰值强度是多少？B200 上呢？fp8 呢？*此处强度指的是矩阵乘法 FLOPs/s 与内存带宽的比值。*

{% details 点击此处查看答案。 %}

**答案：** 对于 H100，我们有峰值 990e12 fp16 FLOPs 和 3.35e12 字节 / 秒的带宽。因此，关键强度是 `990e12 / 3.35e12 = 295`，与 TPU 的 240 相当。对于 B200，其关键强度是 `2250e12 / 8e12 = 281`，也非常相似。这意味着，与 TPU 类似，我们需要大约 280 的批量大小，才能在矩阵乘法中达到算力限制。

对于 H100 和 B200，我们恰好拥有 2 倍的 fp8 FLOPs，因此峰值强度也分别增加到 590 和 562，尽管从某种意义上说，如果我们考虑到权重很可能也会以 fp8 格式加载，那么它实际上保持不变。

{% enddetails %}

**问题 4 [矩阵乘法运行时间]:** 根据问题 3 的答案，你预计在单个 B200 上执行 `fp16[64, 4096] * fp16[4096, 8192]` 矩阵乘法需要多长时间？`fp16[512, 4096] * fp16[4096, 8192]` 呢？

{% details 点击此处查看答案。 %}

从以上内容可知，当批量大小低于 281 个词元时，我们将受到通信的限制。因此，第一个阶段纯粹是带宽限制的。我们读取或写入 $2BD + 2DF + 2BF$ 字节（`2*64*4096 + 2*4096*8192 + 2*64*8192=69e6`），带宽为 `8e12` 字节/秒，因此大约需要 `69e6 / 8e12 = 8.6us`。在实际中，我们可能只能使用总带宽的一小部分，因此可能需要接近 10-12 微秒。当我们增加批量大小时，我们将完全受到算力的限制，因此我们预计 `T=2*512*4096*8192/2.3e15=15us`。我们同样只能使用总 FLOPs 的一小部分，因此可能需要接近 20 微秒。

{% enddetails %}

**问题 5 [L1 缓存容量]:** H100 的 L1/SMEM 总容量是多少？寄存器内存呢？这与 TPU 的 VMEM 容量相比如何？

{% details 点击此处查看答案。 %}

**答案：** 每个 SM 有 256kB 的 SMEM 和 256kB 的寄存器内存，因此每个 SM 约有 33MB（`132 * 256kB`）。两者合计约为 66MB。这大约是现代 TPU 的 VMEM（120MB）的一半，尽管 TPU 总共只有 256kB 的寄存器内存！TPU 的 VMEM 延迟低于 SMEM 延迟，这也是为什么 TPU 上的寄存器内存并不那么关键（向 VMEM 泄漏和填充的成本较低）。

{% enddetails %}

**问题 6 [计算 B200 时钟频率]:** NVIDIA 在此 [here](https://resources.nvidia.com/en-us-blackwell-architecture) 报告称，B200 可以每秒执行 80TFLOPs 的向量 fp32 计算。已知每个 CUDA 核心在 FMA（融合乘加）操作中每周期可以执行 2 FLOPs，估算峰值时钟周期。

{% details 点击此处查看答案。 %}

**答案：** 我们知道我们有 148 * 4 * 32 = 18944 个 CUDA 核心，因此我们可以执行 `18944 * 2 = 37888 FLOPs / cycle`。因此 `80e12 / 37888 = 2.1GHz`，一个较高但合理的峰值时钟频率。B200 通常采用液冷方式，因此更高的时钟周期更为合理。

{% enddetails %}

**问题 7 [估算 H100 的添加运行时间]:** 使用上述数据，计算在单个 H100 上将两个 `fp32[N]` 向量相加所需的时间。分别计算 $T_\text{math}$ 和 $T_\text{comms}$。该操作的算术强度是多少？如果你有条件，请尝试在 PyTorch 或 JAX 中运行此操作，以获得 `N = 1024` 和 `N=1024 * 1024 * 1024` 的结果。这两者相比如何？

{% details 点击此处查看答案。 %}

**答案：** 首先，将两个 `fp32[N]` 向量相加需要 N FLOPs，并需要加载 `4 * N * 2` 字节，写回 4 * N 字节，总共为 `3 * 4 * N = 12N`。计算它们的比率，我们得到 `total FLOPs / total bytes = N / 12N = 1 / 12`，这非常糟糕。

正如我们上面所计算的，如果我们忽略 FMA，我们可以实现大约 33.5 TFLOPs/s 的性能提升。这只有在使用所有 CUDA 核心的情况下才能实现。对于 `N = 1024`，我们最多只能使用 1024 个 CUDA 核心或 8 个 SM，这将需要更长时间（假设我们是算力受限的，大约是原来的 16 倍）。我们还有 3.35e12 字节/s 的显存带宽。因此，我们的峰值硬件强度是 `33.5e12 / 3.35e12 = 10`.<d-footnote>值得注意的是，这种强度在最近几代 GPU 中保持不变。对于 H100，它是 33.5 / 3.5，对于 B200，它是 80 / 8。为什么会出现这种情况尚不清楚，但这是一个有趣的观察结果。</d-footnote> 因此，我们将受到通信的严重限制。因此，我们的运行时间仅仅是

$$T = \max(T_\text{comms}, T_\text{math}) = \frac{12 \cdot N}{\text{3.35e12}} = \frac{N}{\text{2.8e11}}$$

对于 `N = 65,536`，这大约是 0.23 微秒。在实践中，我们在 JAX 中看到的运行时间约为 1.5 微秒，这没有问题，因为我们预计在这里会受到极低延迟的限制。对于 `N = 1024 * 1024 * 1024`，我们有一个约 3.84 毫秒的性能上限，实际测得为 4.1 毫秒，这是很好的结果！

{% enddetails %}

## 网络

网络是 GPU 和 TPU 差异最大的领域之一。正如我们所看到的，TPU 通过二维或三维环形拓扑连接，每个 TPU 只连接到其相邻的 TPU。这意味着在两个 TPU 之间发送消息必须经过所有中间的 TPU，并迫使我们在网格上仅使用统一的通信模式。虽然在某些方面不太方便，但这也意味着每个 TPU 的连接数量是固定的，我们可以扩展到任意大的 TPU “机组”而不会损失带宽。

另一方面，GPU 使用更传统的基于层次树结构的交换网络。被称为 **节点**（在 GB200<d-footnote> 中最多可达 72 个）的 8 个 GPU 集合通过称为 NVLinks 的高带宽互连链路在 1 跳内相互连接。这些节点通过每个 GPU 上的 NIC（网络接口卡）使用带宽较低的 InfiniBand（IB）或以太网网络连接到更大的单元（称为 **SUs** 或可扩展单元）。这些单元随后可以通过更高层级的交换机连接成任意大的单元。</d-footnote>

{% include figure.liquid path="assets/gpu/superpod-diagram.png" class="img-fluid" caption="<b>图：</b> 一张显示典型 H100 网络的示意图。8 个 GPU 组成一个节点或 NVLink 域，通过 NVSwitches（也称为 NVLink 交换机）连接，这些节点通过交换式 InfiniBand 网络相互连接。每个 H100 在 NVLink 域中具有约 450GB/s 的出站带宽，每个节点在 IB 网络中具有 400GB/s 的出站带宽。" %}

### 在节点层面

一个 GPU 节点是一个小型单元，通常包含 8 个 GPU（GB200 最多可达 72 个），通过全互联、高带宽、低延迟的 NVLink 互连连接在一起。<d-footnote>NVLink 被描述为一种增强版的 PCIe 连接，具有低延迟和协议开销，但并未设计用于可扩展性/容错性，而 InfiniBand 更类似于 Ethernet，适用于更大规模的有损网络。</d-footnote> 每个节点包含多个高带宽的 NVSwitch，用于在所有本地 GPU 之间交换数据包。实际的节点级拓扑结构随时间发生了很大变化，包括每个节点的交换机数量，但对于 H100，每个节点有 4 个 NVSwitch，GPU 通过 `5 + 4 + 4 + 5` 连接模式连接到它们，如图所示：

{% include figure.liquid path="assets/gpu/nvlink-nodes.png" class="img-fluid" caption="<b>图：</b>节点即NVLink域示意图，自Pascal（P100）起。自Volta（V100）起，节点内所有GPU通过一组交换机实现了全连接。H100节点配有4个NVSwitch，通过25GB/s链路连接所有8个GPU。" %}

对于 Hopper 一代（NVLink 4.0），每条 NVLink 链路具有 25GB/s 的全双工<d-footnote>此处的全双工表示每个方向均为 25GB/s，且两个方向相互独立。您可以在链路上总共发送 50GB/s 的数据，但每个方向最多为 25GB/s.</d-footnote> 带宽（B200 的带宽为 50GB/s），使我们从每个 GPU 到网络获得 `18 * 25=450GB/s` 的全双工带宽。巨大的 NVSwitches 最多具有 64 个 NVLink 端口，这意味着一个 8xH100 节点搭配 4 个交换机可处理高达 `64 * 25e9 * 4=6.4TB/s` 的带宽。以下是这些数值随 GPU 一代变化的概述：

|NVLink 一代|NVSwitch 一代|GPU 生成|NVLink 带宽 (GB/s，全双工)|NVLink 端口 / GPU|节点 GPU 到 GPU 带宽（GB/s 全双工）|节点规模（NVLink 域）|每节点 NVSwitch 数量|
| :--------: | :----------: | :------------: | :----------------------------------: | :----------------: | :------------------------------------------: | :-----------------------: | :-----------------: ||**3.0**|**2.0**|安培|25|12|300|8|6|
|**4.0**|**3.0**|Hopper|25|18|450|8|4|
|**5.0**|**4.0**|Blackwell|50|18|900|8/72|2/18|

Blackwell (B200) 采用 8 个 GPU 的节点。GB200NVL72 支持更大规模的 NVLink 域，包含 72 个 GPU。我们分别展示了 8 个和 72 个 GPU 系统的详细信息。

### Quiz 2: GPU 节点

这里有一些关于网络的更多问答问题。我发现做这些题目特别有用，因为它们迫使你去实际处理通信模式。

**问题 1 [H100 节点的总带宽]:** 在一个包含 4 个交换机的 8xH100 节点中，每个节点有多少总带宽？*提示:* 考虑 NVLink 和 NVSwitch 的带宽。

{% details 点击此处查看答案。 %}

**答案：** 我们有 Gen4 4 个 NVSwitch，每个具有 `64 * 25e9=1.6TB/s` 的单向带宽。这将使我们在交换机级别拥有 `4 * 1.6e12=6.4e12` 的带宽。然而请注意，每个 GPU 只能处理 450GB/s 的单向带宽，这意味着我们最多只有 `450e9 * 8 = 3.6TB/s` 的带宽。由于这个数值更小，因此峰值带宽为 3.6TB/s。

{% enddetails %}

**问题 2 [二分带宽]**: 二分带宽被定义为网络任意偶数划分之间可用的最小带宽。换句话说，如果我们把网络分成两个相等的部分，两个部分之间有多少带宽？你能计算出 8x H100 节点的二分带宽吗？*提示:* 二分带宽通常包括两个方向的流量。

{% details 点击此处查看答案。 %}

**答案：** 任何偶数划分都会使每半部分有 4 个 GPU，每个 GPU 都可以向另一半传输 `4 * 450GB/s`。双向流量计算，这会带来 `8 * 450GB/s` 字节穿过划分，即 3.6TB/s 的双工带宽。这就是 NVIDIA 在例如 [此处](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf) 所报告的数据。

{% enddetails %}

**问题 3 [AllGather 成本]**: 给定一个大小为 B 字节的数组，在 8xH100 节点上，一个（受吞吐量限制的）AllGather 操作需要多长时间？对 bf16[D<sub>X</sub>, F] 进行计算，其中 `D=4096`, `F=65,536`。*在回答此问题之前，建议阅读 TPU 集合通信的[章节](https://jax-ml.github.io/scaling-book/sharding/)。在此处仔细思考这个问题，但接下来我们会更详细地讨论集合通信。*

{% details 点击此处查看答案。 %}

**答案：** 每块 GPU 可以输出 450GB/s，每块 GPU 有 $B / N$ 字节（其中 `N=8` 为节点大小）。我们可以想象每个节点依次将其字节发送到其他 $N - 1$ 个节点，导致每个节点进行 (N - 1) 轮传输，每轮传输 $T_\text{comms} = (B / (N * W_\text{unidirectional}))$，总计为 $T_\text{comms} = (N - 1) * B / (N * W_\text{unidirectional})$。这大约为 $B / W_\text{uni}$ 或 $B / \text{450e9}$。

对于给定的数组，我们有 `B = 4096 * 65536 * 2 = 536e6` 字节，因此总时间是 `536e6 * (8 - 1) / (8 * 450e9) = 1.04ms`（或使用近似值为 `536e6 / 450e9 = 1.19ms`）。这可能是延迟限制的，因此在实际中可能需要更长的时间（实际中大约需要 1.5 毫秒）。

{% enddetails %}

## 超越节点级别

在节点层面之外，GPU 网络的拓扑结构标准化程度较低。NVIDIA 发布了一种 [参考 DGX SuperPod 架构](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/network-fabrics.html)，该架构使用 InfiniBand 连接比单个节点更多的 GPU，但客户和数据中心提供商可以自由根据自身需求进行定制。<d-footnote>例如，Meta 在一个与上述描述差异显著的数据中心网络上训练了 LLaMA-3，该网络使用以太网、三层交换结构以及在顶层使用了过载的交换机。</d-footnote>

这里是一张参考用的 1024 GPU H100 系统的示意图，其中底部每一格代表一个包含 8 张 H100 显卡、8 张 400Gbps CX7 网卡（每张显卡一张）和 4 张 NVSwitch 的节点。

{% include figure.liquid path="assets/gpu/h100-superpod.png" class="img-fluid" caption="<b>图：</b> 参考 1024 H100 DGX SuperPod 的示意图，包含 128 个节点（有时为 127 个），每个节点配备 8 个 H100 GPU，并通过 InfiniBand 扩展网络连接。32 个节点（256 个 GPU）的集合称为“可扩展单元”（Scalable Units）或 SUs。叶脊结构的 InfiniBand 交换机为节点之间提供足够的带宽，实现完整的双工带宽。" %}

**可扩展单元：** 每组 32 个节点称为一个“可扩展单元”（或 SU），由一组 8 个叶 InfiniBand 交换机支持。该 SU 每个节点配备 4 个 NVSwitches，共 256 个 GPU 和 8 个 InfiniBand 叶交换机。图中所示的所有布线均为 InfiniBand NDR（50GB/s 全双工），使用 64 端口 NDR IB 交换机（每个端口也为 50GB/s）。*请注意，IB 交换机的带宽是 NVSwitches 的两倍（64 个端口，每个端口 400 Gbps 链路）。*

**SuperPod：** 整个 SuperPod 然后通过 16 个顶层“脊柱”IB 交换机将 4 个这样的 SUs 连接起来，使我们拥有 1024 个 GPU、512 个节点级 NVSwitch、32 个叶 IB 交换机和 16 个脊柱 IB 交换机，总计 512 + 32 + 16 = 560 个交换机。叶交换机以 32 个节点为一组连接到节点，因此每组 256 个 GPU 有 8 个叶交换机。所有叶交换机都连接到所有脊柱交换机。

**我们有多少带宽？** InfiniBand 网络（称为“扩展网络”）的整体拓扑结构是 **胖树**，电缆和交换机在节点级别以上保证了完整的双分割带宽（此处为 400GB/s）。这意味着如果我们把节点分成两半，每个节点可以同时以 400GB/s 的速度向另一分区中的节点发送数据。更关键的是，这意味着在扩展网络中，AllReduce 带宽应该大致保持恒定！虽然可能不是这样实现的，但你可以想象在扩展网络中对任意数量的节点执行环形归约，因为你可以构建一个包含所有节点的环。

|级别|GPU|每单位开关数|切换类型|单位带宽（TB/s，全双工）|GPU 到 GPU 带宽（GB/s，全双工）|Fat Tree 带宽（GB/s，全双工）|
| :---: | :------------: | :-------------------------: | :---------: | :------------------------------------------: | :--------------------------------------: | :---: ||节点|8|4|NVL|3.6|450|450
|叶|256|8|IB|12.8|50|400|
|脊柱|1024|16|IB|51.2|50|400|

相比之下，每个 TPU v5p 的链路出站带宽约为 90GB/s，沿着 3D 扭环的所有轴方向总出站带宽为 540GB/s。这并不是点对点的，因此只能用于受限的、均匀的通信模式，但它仍然为我们提供了显著更高的 TPU 到 TPU 带宽，并可以扩展到任意大的拓扑结构（至少达到 8960 个 TPU）。

理论上可以通过增加额外的交换机或间接层来扩展 GPU 交换结构至任意规模，但会付出额外延迟和昂贵网络交换机的代价。

<p markdown=1 class="takeaway">**要点**: 在一个 H100 节点内，每个 GPU 具有完整的胖树带宽 450GB/s，而节点之外，节点间的带宽降至 400GB/s。这将对通信原语至关重要。</p>

**GB200 NVL72s:** NVIDIA 最近开始生产新的 GB200 NVL72 GPU 集群，该集群将 72 个 GPU 集成在一个 NVLink 域中，GPU 之间的带宽达到完整的 900GB/s。这些域随后可以连接到更大的 SuperPod 中，其 IB 脂肪树带宽相应提高（9 倍）。以下是该拓扑结构的示意图：

{% include figure.liquid path="assets/gpu/gb200-superpod.png" class="img-fluid" caption="<b>图：</b>展示了一个包含576个GPU的GB200 DGX SuperPod的示意图。最底层的每个机架包含72个GB200 GPU。" %}

统计单个节点的出带宽（上方的橙色线条），我们有 `4 * 18 * 400 / 8 = 3.6TB/s` 的带宽到达叶节点级别，这比 H100 高出 9 倍（正如该节点包含的 GPU 数量是 H100 的 9 倍）。这意味着关键节点的出带宽要高得多，_多得多_，而我们的跨节点集合带宽实际上可以_更低_于节点内部的带宽。  
更多讨论请参见[附录 A](#appendix-a-how-does-this-change-with-gb200)。

|节点类型|每节点 GPU 数量|GPU 出带宽|节点出站带宽|
| :---------: | :-----------: | :------------------: | :-------------------: ||H100|8|450e9|400e9|
|B200|8|900e9|400e9|
|GB200 NVL72|72|900e9|3600e9|

<p markdown=1 class="takeaway">**要点**: GB200 NVL72 SuperPods 显著增加了单个节点的节点规模和出站带宽，这显著改变了我们的性能上限。</p>

### Quiz 3: 超越节点级别

**问题 1 [胖树拓扑]：** 使用上述 DGX H100 图表，计算整个 1024 GPU 机架在节点级别的带宽。证明每条链路的带宽选择是为了确保完全的带宽。*提示：确保同时计算链路带宽和交换机带宽。*

{% details 点击此处查看答案。 %}

**答案：** 我们逐个组件来处理：

* 首先，每个节点通过 8 条 400Gbps NDR IB 电缆连接到叶交换机，为每个节点提供 `8 * 400 / 8 = 400 GB/s` 的带宽到叶交换机。我们有 8 个叶交换机，每个交换机有 3.2TB/s（64 条 400GBps 链路），但我们只能使用 64 个端口中的一部分来从 SU 进入，因此是 `32 * 400 / 8 = 12.8TB/s` 用于 32 个节点，再次正好是 400GB/s。  
* 然后在脊柱层，我们有 `8 * 16 * 2` 条 400Gbps NDR IB 电缆将每个 SU 连接到脊柱，为每个 SU 提供 `8 * 16 * 2 * 400 / 8 = 12.8 TB/s` 的带宽到叶交换机。同样，每个节点的带宽是 400GB/s。我们有 16 个脊柱交换机，每个交换机有 3.2TB/s，这给我们提供了 `16 * 3.2 = 51.2 TB/s`，加上 128 个节点，再次正好是 400GB/s。

因此，如果我们以任何方式将节点一分为二，节点之间将有 400GB/s 的带宽。每个组件都具备确保胖树结构所需的带宽。

{% enddetails %}

**问题 2 [扩展到更大的 DGX 机架]：** 假设我们想使用 2048 个 GPU 而不是 1024 个进行训练。修改上述 DGX 拓扑以处理这一需求的最简单/最佳方式是什么？如果是 4096 个呢？*提示：没有唯一的正确答案，但请尽量降低成本。注意链路容量。[此](https://docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf)文档可能会有帮助。*

{% details 点击此处查看答案。 %}

**答案：** 一种选择是保持 SU 结构不变（8 个交换机下有 32 个节点），只需添加更多 SU 并使用更多顶层交换机。我们需要增加 2 倍的脊柱交换机，这样我们会有 8 个 SU，共 32 个脊柱交换机，从而提供足够的带宽。

这里的一个问题是，每台叶交换机只有 64 个端口，而上面的图中我们已经用完了所有端口。但相反，我们很容易为每个脊交换机使用 1 条 400 Gbps NDR 电缆，而不是 2 条，这样可以提供相同的总带宽，同时节省一些端口。

对于 4096 个 GPU，我们实际上会用尽端口，因此需要增加另一层间接性，也就是说，在层次结构中增加另一层。NVIDIA 将这些称为“核心交换机”，并使用 128 个脊椎交换机和 64 个核心交换机构建了一个 4096 个 GPU 的集群。你可以进行计算以证明这提供了足够的带宽。

{% enddetails %}

## 集合通信在 GPU 上是如何工作的？

GPU 可以执行与 TPU 相同的所有集合通信操作：ReduceScatters、AllGathers、AllReduces 和 AllToAlls。与 TPU 不同的是，这些操作的实现方式取决于它们是在节点级别（通过 NVLink）还是在更高层次（通过 InfiniBand）上执行。这些集合通信操作由 NVIDIA 在 [NVSHMEM](https://developer.nvidia.com/nvshmem) 和 [NCCL](https://developer.nvidia.com/nccl)（发音为 "nickel"）库中实现。NCCL 的开源代码可在 [此处](https://github.com/NVIDIA/nccl) 获取。虽然 NCCL 根据延迟要求/拓扑结构使用多种实现方式（[详情](https://github.com/NVIDIA/nccl/issues/1415#issuecomment-2310650081)），但从现在开始，我们将讨论在交换树结构上理论上最优的模型。

### 节点内集合通信

**AllGather 或 ReduceScatter：** 对于节点级别的 AllGather 或 ReduceScatter，你可以像 TPU 一样围绕一个环进行操作，每一步都使用完整的 GPU 到 GPU 带宽。可以任意排列 GPU，并使用完整的 GPU 到 GPU 带宽将数组的一部分沿着环发送。<d-footnote> 你也可以认为每个 GPU 将其大小为 $\text{bytes} / N$ 的数据块发送到其他 $N - 1$ 个 GPU，总共通信 $(N - 1) * N * bytes / N$ 字节，这给出了相同的结果。</d-footnote> 每一步的代价是 $T_\text{hop} = \text{bytes} / (N * \text{GPU egress bandwidth})$，因此总体代价是

$$T_\text{AG or RS comms} = \frac{\text{bytes} \cdot (N - 1)}{N \cdot \text{GPU egress bandwidth}} \rightarrow \frac{\text{bytes}}{\text{GPU egress bandwidth}}$$

你会发现这与在 TPU 上的情况完全相同。对于 AllReduce，你可以像平常一样将 RS + AG 组合在一起，但成本会增加一倍。

{% include figure.liquid path="assets/gpu/all-gather.gif" class="img-fluid" caption="<b>图：</b> 带宽最优的1D环形AllGather算法。对于B字节，这会通过顶层交换机发送B / X字节X - 1次。" %}

如果你担心延迟（例如，如果数组非常小），可以执行树状归约，即在 2 个一组、然后是 4 个一组、再是 8 个一组中进行 AllReduce，总共需要 $\log(N)$ 次通信，而不是 $N - 1$ 次，尽管总成本仍然相同。

<p markdown=1 class="takeaway">**要点：** 在单个节点内对一个 B 字节的数组执行 AllGather 或 ReduceScatter 的成本约为 $T_\text{comms} = B * (8 - 1) / (8 * W_\text{GPU egress}) \approx B / W_\text{GPU egress}$。理论上在 H100 上约为 $B  / \text{450e9}$，在 B200 上约为 $B / \text{900e9}$。除非启用了网络内归约，否则 AllReduce 的成本是这个值的 2 倍。</p>

<b markdown=1 style="color: #57cf57;">小测验 1 [AllGather 时间]:</b> 使用 8xH100 节点和 450 GB/s 全双工带宽，AllGather(bf16[B<sub>X</sub>, F]) 需要多长时间？让 $B=1024$, $F=16,384$。

{% details 点击此处查看答案。 %}

**答案：** 我们总共有 $2 \cdot B \cdot F$ 字节，单向带宽为 450e9。这将需要大约 $T_\text{comms} = (2 \cdot B \cdot F) / \text{450e9}$，更精确地说是 $(2 \cdot B \cdot F \cdot (8 - 1)) / (8 \cdot \text{450e9})$。使用提供的数值，这给我们大约 $(2 \cdot 1024 \cdot 16384) / \text{450e9} = \text{75us}$，更精确地说是 $\text{65us}$。

{% enddetails %}

**AllToAlls:** 节点内的 GPU 之间具有全连接性，这使得 AllToAlls 非常容易实现。每个 GPU 只需直接发送到目标节点。在一个节点内，对于 B 字节，每个 GPU 有 $B / N$ 字节，并发送 $(B / N^2)$ 字节到 $N - 1$ 个目标节点，总共有

$$T_\text{AllToAll comms} = \frac{B \cdot (N - 1)}{W \cdot N^2} \approx \frac{B}{W \cdot N}$$

与 TPU 相比，其成本为 $B / (4W)$。因此，在单个节点内，我们获得了 2 倍的理论运行时速提升（$B / 4W$ vs. $B / 8W$）。

对于混合专家（MoE）模型，我们经常需要执行一种 *稀疏或不规则的 AllToAll*，其中我们保证输出维度上最多有 $k$ 个 $N$ 分片是非零的，也就是说 $T_\text{AllToAll} \rightarrow K[B, N]$ 每个轴上最多有 $k$ 个 $N$ 元素是非零的。这种操作的成本降低了 $k/N$，总成本约为 $\min(k/N, 1) \cdot B / (W \cdot N)$。对于 MoE 模型，我们通常随机独立地选择非零值，因此存在一些概率使得非零值少于 $k$，从而给我们大约 $(N-1)/N \cdot \min(k/N, 1) \cdot B / (W \cdot N)$.<d-footnote> 实际成本实际上是 $$(1 - \left(\frac{Z - 1}{Z}\right)^K) \cdot \frac{Z - 1}{Z}$$ 次 $K$ 次骰子投掷中不同结果的期望数量，但它与给定的近似值非常接近。更多细节请参见附录。</d-footnote>

<b markdown=1 style="color: #c55404ff;">小测验 2 [AllToAll 时间]:</b> 使用一个 8xH100 节点，单向带宽为 450 GB/s，AllToAll<sub>X->N</sub>(bf16[B<sub>X</sub>, N]) 需要多长时间？如果我们知道其中只有 4 个条目是非零的，又会怎样？

{% details 点击此处查看答案。 %}

**答案：** 请注意，此处 $B$ 是数组的批量维度，因此数组总大小为 $V = 2 \cdot B \cdot N$ 字节。从上面可知，在密集情况下，成本为 $V \cdot (N-1) / (W \cdot N^2)$，即大约 $V / (W \cdot N)$。如果我们知道只有 $\frac{1}{2}$ 的条目是非填充的，我们可以发送 $V \cdot k/N / (W \cdot N) = V / (2 \cdot W \cdot N)$，即总成本的大约一半。

{% enddetails %}

<p markdown=1 class="takeaway">**要点:** 在单个节点的 GPU 上，对 $B$ 字节的数组执行 AllToAll 的成本约为 $T_\text{comms} = (B \cdot (8 - 1)) / (8^2 \cdot W_\text{GPU egress}) \approx B / (8 \cdot W_\text{GPU egress})$。对于不规则（top-$k$）的 AllToAll，这一成本进一步降低到 $(B \cdot k) / (64 \cdot W_\text{GPU egress})$.</p>

**经验测量：** 这里是对 8xH100 节点上 AllReduce 带宽的经验测量。Algo BW 是测得的带宽（字节 / 运行时间），Bus BW 计算方式为 $2 \cdot W \cdot (8 - 1) / 8$，理论上是对实际链路带宽的度量。你会发现我们确实达到了接近 370GB/s，虽然低于 450GB/s，但差距合理，不过每设备仅约 10GB。这意味着虽然这些估计在理论上是正确的，但需要大消息量才能实现。

{% include figure.liquid path="assets/gpu/gpu-all-reduce-bw.png" class="img-fluid" caption="<b>图：</b> 禁用 SHARP 的 8xH100 节点的 AllReduce 吞吐量。蓝色曲线是根据 $2 * \text{bytes} * (N - 1) / (N * \text{runtime})$ 的实测数据计算出的经验链路带宽。请注意，即使使用了大规模的 10GB 数组，我们也没有特别接近所声称的 450GB/s 带宽。" %}

这是一个真实的问题，因为它显著复杂化了我们所能做出的任何理论主张，例如，即使是对一个合理大小的数组（如 LLaMA-3 70B 的 MLP（大小为 `bf16[8192, 28672]`，或在 8 路模型分片下为 `bf16[8192, 3584] = 58MB`））进行 AllReduce，也只能达到约 150GB/s，远低于峰值 450GB/s。相比之下，TPUs 在更低的消息大小下即可达到峰值带宽（见附录 B）。

<p markdown=1 class="takeaway">**要点：** 尽管 NVIDIA 宣称 H100 NVLink 的带宽约为 450GB/s，但实际上很难超过 370GB/s，因此请相应调整上述估计值。</p>

**在网络缩减中：** 自 Hopper 世代起，NVIDIA 交换机支持 ["SHARP"（Scalable Hierarchical Aggregation and Reduction Protocol）](https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/)，这使得可以实现“网络内缩减”。这意味着 *网络交换机本身* 可以执行缩减操作，并将结果多路复用或“MultiCast”到多个目标 GPU：

{% include figure.liquid path="assets/gpu/sharp-algorithm.png" class="img-fluid" caption="<b>图：</b> 一个没有 SHARP 的 AllReduce 操作的理论成本是原来的 2 倍，因为它必须经过每个 GPU 两次。实际上，加速效果仅为约 30%（来自 NCCL 2.27.5）。" %}

理论上，这几乎将 AllReduce 的成本降低了一半，因为这意味着每个 GPU 可以将其数据发送到顶级交换机，该交换机本身执行归约操作并将结果广播到每个 GPU，而无需每个 GPU 两次传出数据，同时还能减少网络延迟。

$$T_\text{SHARP AR comms} = \frac{\text{bytes}}{\text{GPU egress bandwidth}}$$

请注意，这与 $1/N$ 的倍数无关，是精确的，因为每个 GPU 首先传出 $B \cdot (N - 1) / N$，然后接收其本地分片的部分缩减版本（接收 $B/N$），完成缩减后，再次传出 $B/N$，然后接收完全缩减的结果（接收 $B \cdot (N - 1) / N$），最终导致恰好接收 $B$ 字节。

然而，实际上在启用 SHARP 后，带宽仅增加了约 30%，远低于预测的 75%。这使我们仅达到约 480GB/s 的有效集合带宽，远未达到 2 倍的水平。

{% include figure.liquid path="assets/gpu/sharp-all-reduce-cost.png" class="img-fluid" caption="<b>Figure:</b> 在节点内启用和未启用 NVIDIA SHARP 的 AllReduce 算法带宽的实测数据。即使从算法上讲它应该能够实现接近 75% 的增益，但在峰值时增益约为 30% 的吞吐量提升。" %}

<p markdown=1 class="takeaway">**要点：**理论上，NVIDIA SHARP（大多数 NVIDIA 交换机上均可使用）应将 $B$ 字节的 AllReduce 成本从约 $2 * B / W$ 降低到 $B / W$。然而，实际上我们仅看到带宽提升了约 30%。由于纯 AllReduce 在大语言模型中较为罕见，因此这一改进并不特别有用。</p>

### 跨节点集合通信

当我们超越节点级别时，成本会更加微妙一些。在对树进行归约时，可以认为是从下往上进行归约，首先在节点内部进行归约，然后在叶子级别进行归约，接着在脊椎级别进行归约，每个级别都使用正常的算法。对于 AllReduce 特别来说，可以看到这使我们能够整体上减少通信的数据量，因为在节点级别完成 AllReduce 之后，我们只需要将 $B$ 字节传出到叶子级别，而不是 $B * N$。

**这会带来多大的成本？** 首先近似来看，由于我们拥有完整的二分带宽，AllGather 或 ReduceScatter 的成本大致等于缓冲区大小（以字节为单位）除以节点出带宽（H100 上为 400GB/s），*无论树减少的任何细节如何*。

$$T_\text{AG or RS comms} = \frac{\text{bytes}}{W_\text{node egress}} \underset{H100}{=} \frac{\text{bytes}}{\text{400e9}}$$

其中 $W_\text{node}$ 的出站带宽通常为上述 H100 网络的 400GB/s（每个节点有 8 条 400Gbps 的 IB 链路出站）。最清晰的想象方式是设想对集群中的 *每个节点* 进行环形归约。由于胖树拓扑结构，我们总可以在任意两个节点之间构建一个具有 $W_\text{node}$ 出站的环，并进行正常的归约。节点级归约几乎不会成为瓶颈，因为它具有更高的整体带宽和更优的延迟，尽管一般来说成本是

$$T_\text{total} = \max(T_\text{comms at node}, T_\text{comms in scale-out network}) = \max\left[\frac{\text{bytes}}{W_\text{GPU egress}}, \frac{\text{bytes}}{W_\text{node egress}}\right]$$

{% details 你可以在更精确的推导中看到这里。 %}

我们可以更精确地指出，我们在网络的每一层实际上是在执行环形归约操作，这些操作我们可以大部分重叠，因此我们有：

$$T_\text{AG or RS comms} = \text{bytes} \cdot max_\text{depth i}\left[\frac{D_i - 1}{D_i \cdot W_\text{link i}}\right]$$

其中 $D_i$ 是深度 $i$ 处的度数（深度 $i$ 处的子节点数量），$W_\text{link i}$ 是连接每个子节点到节点 $i$ 的链路带宽。

利用这一点，我们可以计算给定拓扑结构下可用的 AllGather/AllReduce 带宽为 $min_\text{depth i}(D_i * W_\text{link i} / (D_i - 1))$。在上述情况下，我们有：

* **Node:** $D_\text{node}$ = 8，因为我们在一个节点中拥有 8 个 GPU，Wlink i = 450GB/s。因此我们的 AG 带宽为 `450e9 * 8 / (8 - 1) = 514GB/s`。  
* **Leaf:** $D_\text{leaf}$ = 32，因为我们在一个 SU 中拥有 32 个节点，Wlink i = 400GB/s（8 个 400Gbps IB 链接）。因此我们的带宽为 `400e9 * 32 / (32 - 1) = 413GB/s`。  
* **Spine:** $D_\text{spine}$ = 4，因为我们有 4 个 SUs，$W_\text{link i}$ = 12.8TB/s（来自上述 `8 * 16 * 2 * 400Gbps` 链接）。我们的带宽为 `12.8e12 * 4 / (4 - 1) = 17.1TB/s`。

因此，我们在叶子层的总体 AG 或 RS 带宽为 `min(514GB/s, 413GB/s, 17.1TB/s) = 413GB/s`，所以在实际中为 $T_\text{AG or RS comms} = B / \text{413GB/s}$，即我们在最高层级仍拥有约 413GB/s 的 AllReduce 带宽。对于使用 SHARP 的 AllReduce，其带宽会略低于这个数值（约为 400GB/s），因为我们没有 $(N - 1) / N$ 因素。不过，450GB/s 和 400GB/s 差不多，可以作为近似值使用。

{% enddetails %}

**其他集合通信操作：** 除非启用 SHARP，否则 AllReduces 的成本仍然是上述成本的 2 倍。NVIDIA 也销售支持 SHARP 的 IB 交换机，尽管并非所有供应商都提供此类产品。AllToAll 在跨节点时变化较大，因为它们不像 AllReduces 那样具有“层次结构”。如果我们希望将数据从每个 GPU 发送到其他所有 GPU，就无法在节点级别充分利用二分带宽。这意味着，如果有一个跨越 $M = N / 8$ 个节点的 N 路 AllToAll，其成本是

$$T_\text{AllToAll comms} = \frac{B \cdot (M - 1)}{M^2 \cdot W_\text{node egress}} \approx \frac{B}{M \cdot W_\text{node egress}}$$

这实际上只有 50GB/s 的带宽，而不是 400GB/s。我们从 $B / (8 * \text{450e9})$ 单个 H100 节点上的性能下降到 $B / (2 \cdot \text{400e9})$，当扩展到 2 个节点时，性能下降超过 4 倍。

这里是 1024-GPU DGX H100 SuperPod 架构的概要：

|级别|GPU 数量|度（子节点数）|交换带宽（全双工，TB/s）|电缆带宽（全双工，TB/s）|集合带宽 (GB/s)|
| :-------: | :------------: | :-----------------: | :----------------------------------: | :---------------------------------: | :-------------------------: ||节点|8|8|6.4|3.6|450|
|叶（SU）|256|32|25.6|12.8|400|
|脊柱|1024|4|51.2|51.2|400|

我们使用“集合带宽（Collective Bandwidth）”这一术语来描述我们能够从 GPU 或节点中传出数据的有效带宽。它也是 $\text{bisection bandwidth} * 2 / N$。

<p markdown=1 class="takeaway">**要点：** 除了节点级别之外，对 B 字节执行 AllGather 或 ReduceScatter 的成本大致为 $B / W_\text{node egress}$，这在 H100 DGX SuperPod 上为 $B / \text{400e9}$，而 AllReduce 的成本是其两倍，除非启用了 SHARP。整体拓扑结构是胖树，旨在在任意两个节点之间提供恒定的带宽。</p>

**当数组在单独轴上进行分片时的缩减操作：** 考虑像这样的缩减操作的成本

$$\text{AllReduce}_X(A[I_Y, J]\ \{ U_X \})$$

我们在对一个本身沿另一个轴被分片的数组进行 AllReduce 操作 $Y$。在 TPU 上，与未分片版本相比，该操作的整体成本降低了 $1 / Y$ 倍，因为我们每个轴发送的数据量减少了 $1 / Y$。在 GPU 上，成本取决于哪个轴是“内部”轴（节点内 vs. 节点间）以及每个分片是否跨越多个节点。假设 $Y$ 是内部轴，且数组总共有 $\text{bytes}$ 字节，整体成本将有效降低 $Y$，但前提是 $Y$ 跨越多个节点：

$$T_\text{comms at node} = \frac{\text{bytes}}{W_\text{GPU egress}} \cdot \frac{1}{\min(Y, D_\text{node})}$$

$$T_\text{comms in scale-out network} = \frac{\text{bytes}}{W_\text{node egress}} \cdot \frac{D_\text{node}}{\max(D_\text{node}, Y)}$$

$$T_\text{total} = \max(T_\text{comms at node}, T_\text{comms in scale-out network})$$

其中 N 是 GPU 的数量，$D_\text{node}$ 是节点中 GPU 的数量（节点的度数）。如你所见，如果 $Y < D_\text{node}$，我们在节点层面会有所收益，但通常不会看到整体运行时间的减少，而如果 $Y > D_\text{node}$，我们会看到与所跨越节点数量成比例的速度提升。

如果我们想精确地描述环形归约，树状 AllGather<sub>X</sub>(A<sub>Y</sub> { U<sub>X</sub> }) 的一般规则（假设 Y 是内部轴）是

$$T_\text{AR or RS comms} = \text{bytes} \cdot \max_{\text{depth } i}\left[\frac{D_i - 1}{D_i \cdot \max(Y, S_{i-1}) \cdot W_{\text{link } i}}\right]$$

其中 $S_i$ 是 M * N * …，即树中第 i 层以下子节点的规模。这大致意味着我们所跨越的 GPU 或节点越多，可用带宽就越大，但仅限于该节点内部。

**小测验 3 [沿两个轴分片]:** 假设我们想要执行 $\text{AllGather}_X(\text{bf16}[D_X, F_Y])$，其中 $Y$ 是单个 SU（256 个芯片）上的内轴。这将需要多长时间，作为 $D$、$F$ 和 $Y$ 的函数？

{% details 点击此处查看答案。 %}

**答案：** 我们可以将其分为两种情况，即 Y <= 8 和 Y > 8。当 $Y <= 8$ 时，我们仍受叶交换机限制，因此答案与往常一样，为 $T_\text{comms} = 2 * D * F * (32 - 1) / (32 * 400e9)$。当 Y > 8 时，我们从上面得出，大致为

$$T_\text{comms} = \frac{2 \cdot D \cdot F \cdot 256}{Y \cdot \text{12.8e12}} = \frac{2DF}{Y \cdot \text{50GB/s}}$$

对于 `D = 8192`、`F = 32,768`，我们有：

{% include figure.liquid path="assets/gpu/sharded-all-gather-cost.png" class="img-fluid" caption="<b>图：</b> 分片 AllGather 的理论成本，当内部轴跨越更多节点时。" %}

请注意，如果我们精确地进行 8 路模型并行，确实会将节点级归约的成本降低 8 倍，但整体成本保持不变，因此这是免费的，但无助于提高整体带宽。

{% enddetails %}

<p markdown=1 class="takeaway">**要点：** 当我们有多个分片轴时，外层归约的成本会减少内层轴所跨越的节点数的倍数。</p>

### Quiz 4: 集合通信

**问题 1 [AllGather]:** 仅考虑一个 SU，其中包含 M 个节点，每个节点有 N 个 GPU。在 AllGather 过程中，节点级交换机的入流量和出流量各为多少字节？顶层交换机的入流量和出流量各为多少字节？

{% details 点击此处查看答案。 %}

**答案：** 我们逐步进行，处理缩减的各个组成部分：

1. 每个 GPU 向交换机发送 $B / MN$ 字节，总入流量为 $NB / MN = B / M$ 字节。  
2. 我们将全部 $B / M$ 字节传输到脊柱交换机。  
3. 我们从脊柱交换机接收 $B * (M - 1) / M$ 字节。  
4. 我们将 $B - B / MN$ 字节传输 $N$ 次，总传输量为 $N * (B - B / MN) = NB - B / M$。

总共有 $B$ 入流量和 $BN$ 出流量，因此我们应受出流量限制，总时间将是 $T_\text{AllGather} = BN / W_\text{node} = B / \text{450e9}$。

对于脊椎交换机，数学实际上更简单。我们必须有 $B / M$ 字节被 ingress $B$ 次（总共 $B (M - 1) / M$ 字节），然后被 egress $M$ 次，总共 $B * (M - 1)$ 字节流出。由于这个数值显著更大，成本是 $T_\text{AllGather} = B \cdot (M - 1) / (M \cdot W_\text{node}) = B \cdot (M - 1) / (M \cdot \text{400e9})$。

{% enddetails %}

**问题 2 [单节点 SHARP AR]:** 考虑一个节点，每个节点有 N 个 GPU。在使用 SHARP（网络内归约）进行 AllReduce 时，交换机的 ingress 和 egress 的字节数精确是多少？

{% details 点击此处查看答案。 %}

**答案：** 与之前一样，让我们逐步进行。

1. 每个 GPU 发送 $B * (N - 1) / N$ 字节，因此我们接收了 $N * B * (N - 1) / N = B * (N - 1)$ 字节。  
2. 我们累积部分和，然后将 $B / N$ 字节发送回每个 GPU，因此发送了 $N * B / N = B$ 字节。  
3. 我们在本地对残差进行部分求和，然后将结果发送回交换机。总共接收了 $N * B / N = B$ 字节。  
4. 我们捕获所有分片并进行组播，将 $B * (N - 1) / N$ 发送到 $N$ 个目的地，总共发送了 $B * (N - 1) / N * N = B * (N - 1)$ 字节。

因此总共有 $B * (N - 1) + B = BN$ 字节被传入和传出。这支持整体吞吐量正好为 $B / W_\text{egress}$。

{% enddetails %}

**问题 3 [跨节点 SHARP AR]:** 考虑一个 bf16[D<sub>X</sub>, F<sub>Y</sub>] 数组，该数组在一个包含 N 个 GPU 的单个节点上进行分片。AllReduce(bf16[D, F<sub>Y</sub>] { U<sub>X</sub> }) 需要多长时间？你可以假设我们进行了网络内归约。解释如果节点数量超过一个，情况会有什么不同？

{% details 点击此处查看答案。 %}

**答案：** 我们可以尝试修改上面问题的答案。基本上，我们首先从每个 GPU 传出 $B * (X - 1) / XY$ 字节，然后将 $B / XY$ 发送回每个 GPU，然后将相同数量的数据再发送回交换机，然后将 $B * (X - 1) / XY$ 发送回每个 GPU。总共有 $NB / Y$ 的入流量和出流量，因此总时间为 $T_\text{comms} = NB / (Y * N * W_\text{link}) = N * 2DF / (Y * N * W_\text{link}) = 2 * D * F / (Y * W_\text{link})$，因此总时间会随着 $Y$ 的增加而减少。

如果我们超越单个节点，我们可以大致实现与上述相同的减少，但当我们离开节点级交换机时，我们需要发送全部 B 字节，而不仅仅是 $B / Y$。这是因为我们需要保持每个分片独立。

{% enddetails %}

**问题 4 [脊柱层级 AR 成本]:** 考虑与上述相同的设置，但使用 $Y = 256$（因此 AR 在脊柱层级发生）。AllReduce 需要多长时间？同样，可以自由假设网络内减少。

{% details 点击此处查看答案。 %}

**答案：** 这使我们能够利用脊柱层级相当惊人的带宽。在 4 个 SU 上，脊柱层级的带宽达到 51.2TB/s，即每个 SU 为 12.8TB/s。使用 SHARP，这可能仅需 `2 * D * F / 12.8e12` 秒。

{% enddetails %}

**问题 5 [2 节点 AllGather 成本]:** 计算 $B$ 字节在恰好 2 个节点上进行 AllGather 的精确成本。*确保计算精确成本而非近似值，并考虑节点内和节点间的成本。*

{% details 点击此处查看答案。 %}

**答案：** 在节点级别，我们有 $T_\text{comms} = B * 7 / (8 * \text{450e9}) = B / \text{514e9}$，而更高层次我们实际上有 $T_\text{comms} = B * (2 - 1) / (2 * \text{400e9}) = B / \text{800e9}$。因此，我们实际上受到节点级别缩减的限制，而不是叶子级别！这促使了例如 DeepSeek v3 的出现，它采用 2 路数据并行。

{% enddetails %}

## LLM 在 GPU 上缩放的屋顶线

现在让我们看看所有这些内容最终指向的是什么：理解在 GPU 上进行大语言模型缩放的屋顶线。这旨在补充 TPU 训练章节[此处](../training)。正如我们在那里所做的那样，这里的目的是查看不同并行策略的总 $T_\text{math}$ 和 $T_\text{comms}$，并了解 $T_\text{comms} > T_\text{math}$ 的临界点。与之前一样，我们只考虑具有这些操作的 MLP 模块。

$$\text{MLP}(x) \equiv x[B, D] *_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D]$$

其中 $B$ 是以词元为单位的全局批量大小（即 $B = \text{batch size} \cdot \text{sequence length}$）。

这里我们将复现上面的表格，展示在 GPU 和节点层面的有效带宽：

|节点类型|每节点 GPU 数量|GPU 出带宽|节点出站带宽|
| :---------: | :-----------: | :------------------: | :-------------------: ||H100|8|450e9|400e9|
|B200|8|900e9|400e9|
|GB200 NVL72|72|900e9|3600e9|

**注意：** GPU 和节点出站带宽共同决定了我们大语言模型的屋顶线。我们将使用术语 $W_\text{collective}$ 来描述 GPU 或节点带宽，具体取决于我们是在节点级别以下还是以上运行。

让我们像之前对TPU所做的那样，来看一下数据并行、张量并行、流水线并行、专家并行以及它们的组合的算力通信屋顶线。在本节的其余部分，我们将专注于H100针对特定计算的屋顶线。GB200-NVL72具有相同的总体屋顶线，但由于我们拥有更大的节点出口带宽，有时会在节点级别遇到瓶颈。

### 数据并行

如前所述，DP 和 ZeRO 分片在反向传播过程中涉及一个权重 AllReduce 或一个 ReduceScatter + AllGather。由于这两者的成本相同，为了在纯数据并行或 FSDP *不使用网络内约简* 时达到算力限制，每层在反向传播过程中，对于大小为 X 的轴：

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{X \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF}{W_\text{collective}}$$

因此，对于 $T_\text{math} > T_\text{comms}$，我们需要 $B / (XC) > 1 / W_\text{collective}$ 或

$$\frac{B}{X} > \frac{C}{W_\text{collective}}$$

其中 $W_\text{collective}$ 是 GPU 或节点级别的出站带宽，具体取决于我们是在节点内还是跨节点进行分片。因此：

* **在单个节点内**，我们只需确保每块 GPU 的 **词元** 批量大小 > $\text{990e12} / \text{450e9} = 2200$。  
* **在单个 SU 内或在脊柱层级**，BS > $\text{990e12} / \text{400e9} = 2475$。

这比在 TPU 上的数值要高得多，在 TPU 上三个轴的数值为 850。例如，LLaMA-3 在 16000 张 H100 上训练，需要至少 40M 个词元的批量大小（作为参考，他们使用了 16M）。DeepSeek v3 在 2048 张 H800 GPU 上训练，带宽较低，为 300GB/s（而不是 H100 的 450GB/s），每张 GPU 需要 $\text{990e12} / \text{300e9} = 3300$ 个词元，大约为 6.7M（实际上，他们使用了 4M）。

启用网络内缩减并使用纯数据并行时，理论上我们拥有 2 倍的 AllReduce 带宽，这将使这两个数字都减半。然而，实际上收益更接近 30%，这仅能部分弥补我们通常难以达到报告数值的事实。此外，由于纯数据并行很少有用，这在实践中基本上无关紧要。

**MoE 模型：** 对于一个混合专家（MoE）模型，其中我们有 E 个专家，每个词元有 k 个专家，这会增加到

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot k \cdot BDF}{X \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot EDF}{W_\text{collective}}$$

这将每个 GPU 的词元批量大小按 $E/k$ 的比例扩大，即

$$\frac{B}{X} > \frac{E}{k} \frac{C}{W_\text{collective}}$$

例如，新的 OpenAI 开源模型 $k=4$ 和 $E=128$，这使得跨节点的 `32 * 2475  = 79,200` 增加到一个荒谬地高的数字。

**当 X 较小时会发生什么？** 当我们只进行例如 2 节点的数据并行时，我们能从 $(X - 1) / X$ 缩放中获益，这为我们提供了

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{N * C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF \cdot (X-1)}{X \cdot W_\text{collective}}$$

其中 X 是节点数量和 $N = 8 \cdot X$。对于密集模型，我们有 $B / N > \alpha \cdot (X - 1) / X$，或例如 $B / N > \text{1237}$，为上述值的一半。你会因为这个原因经常看到 2-way 数据并行。

<p markdown=1 class="takeaway">**要点：** 在假设完美重叠和 FLOPs 利用率的情况下，数据并行和 ZeRO 分片在 H100 或 B200 上达到算力限制时，每个 GPU 的批量大小约为 2500 个词元。对于 MoE 模型，这一数值会乘以 $E / k$，即总参数与激活参数的比值。当进行少量的数据并行时，临界批量大小会减小。</p>

### 张量并行

张量并行需要对激活值进行 AllGather 和 ReduceScatter 操作，我们需要将这些操作与 MLP 的 FLOPs 重叠。换句话说，在前向传播过程中，我们有

$$T_\text{math} = \frac{2\cdot 2 \cdot BDF}{Y \cdot C}$$

$$T_\text{comms} = \frac{2\cdot 2 \cdot BD}{W_\text{collective}}$$

哪一种受算力限制给出了规则

$$Y < \frac{F \cdot W_\text{collective}}{C}$$

在单个节点内，这使我们能够实现约 $F / 2200$ 或 $F / 2475$ 的规模，超出单个节点的限制。对于像 LLaMA-3 这样的 $F=\text{28000}$，这大约是 11 路张量并行（或向下取整为约 8 路，这正是一个节点的规模）。与上述情况类似，当我们跨越恰好 2 个节点时，可以额外获得 2 倍的带宽，因此我们通常可以实现 16 路张量并行（$F > 2475 \cdot (Y - 8)$），理论上可达到最多 19 路模型并行。

<p markdown=1 class="takeaway">**要点：** 在大小为 Y 的轴上进行张量并行，且前馈维度为 F 时，当 $Y > F / 2475$ 时，通信将成为瓶颈，这通常将我们限制为仅节点内 TP 或最多 2 节点 TP。</p>

### 专家并行

正如我们之前已经指出的，混合专家（MoE）模型在仅增加 k 倍 FLOPs 的情况下，带来了 E 倍更多的模型权重，这使得数据并行变得更加困难。我们可以通过沿专家维度对权重进行分片，即 W<sub>in</sub>[E<sub>Z</sub>, D, F]，在一定程度上缓解这一问题。为了执行 MLP 模块，我们需要引入 2 次 AllToAll 操作，将激活值发送到对应的专家。

如上所述，如果这个 AllToAll<sub>Z->k</sub>([B, D, k]) 跨越多个节点，其成本大约为 $T_\text{AllToAll} = 2 \cdot B \cdot D \cdot (Z-8)/Z \min(8 * k / Z, 1)$，因此对于纯粹的专家并行，我们需要

$$T_\text{math} = \frac{4 \cdot B \cdot k \cdot D \cdot F}{Z \cdot C}$$

$$T_\text{comms} = \frac{4 \cdot B \cdot D \cdot (Z-8)}{W \cdot Z} \cdot \min\left(\frac{8 \cdot k}{Z}, 1\right)$$

我们要么需要 $K > Z/8$ 配合 $F > \alpha \cdot (Z - 8)/k$，要么需要 $Z \gg K$ 和 $F > 8 \cdot \alpha$，其中 $\alpha = C/W$。这为你提供了两个可以实现专家并行的领域，一个具有少量专家并行（大约 2 节点）和较小的 $F$，另一个具有较大的 $F$ 和 $Z$，可以任意增大（最多达到 E 路专家并行）。

在实践中，你将看到这两种情况，一种是专家并行程度较低的情况（例如 DeepSeek v3，其 F 非常小，且跨节点的专家并行程度也较小且受限），另一种是 F 较大的模型，在这种情况下，我们可以同时实现显著的跨节点专家并行和张量并行。

<p markdown=1 class="takeaway">**要点：** 如果 $F < 8 * C / W_\text{node}$，专家并行可以在 1-2 个节点上进行，成本与 TP 相似（略低）；如果 $F > 8 * C / W_\text{node}$，我们可以以相对较低的成本进行大量专家并行（最多 $E$ 个节点）。</p>

### 流水线并行

流水线并行在节点之间分配层，通信成本极低，因为我们只需每隔几层发送少量激活值的微批量。历史上，流水线技术曾因“流水线气泡”而受到限制，但随着新的零气泡流水线方法的出现，通常可以避免这一问题。

流水线的整体通信成本非常小：使用 $N_\text{MB}$ 微批量和 $N_\text{stages}$，我们有 $T_\text{comms per hop} = 2 \cdot B \cdot D / (W \cdot N_\text{MB})$ 和 $N_\text{MB} + N_\text{stages} - 2$ 跳步，因此大致

$$T_\text{total PP comms} = \frac{2BD}{W \cdot N_\text{MB}} \cdot (N_\text{MB} + N_\text{stages} - 2)$$

$$T_\text{per-layer comms} \approx 1.5 \cdot \frac{2BD}{W \cdot N_\text{layers}}$$

由于我们是在除以 $N_\text{layers}$，因此这比其他任何成本都小得多。换句话说，从通信的角度来看，流水线处理基本上是免费的。那么，我们为什么不直接使用流水线处理呢？有几个原因：

(1) **代码复杂性：** 流水线技术不如其他方法那样很好地适配自动并行框架（如 XLA 的 GSPMD）。因为它引入了微批量以隐藏流水线气泡，从而改变了程序的结构，而自定义的零气泡流水线调度方案则通过要求前向和反向传播的复杂交错进一步加剧了这一问题。

(2) **流水线并行使数据并行和 FSDP 变得困难：** 可能不采用流水线并行的最重要原因在于它与 FSDP 和数据并行不兼容。特别是 ZeRO-3 的分片方式效果很差，因为它要求我们在每个微批量上执行 AllGather 操作，而当我们只有 $B / N_\text{microbatches}$ 个词元来分摊 AllGather 的成本时，这种方式无法奏效。此外，在反向传播过程中，*我们必须等到最后一个微批量通过某个阶段后，才能执行 AllReduce 或 ReduceScatter 梯度操作，这意味着我们会有显著的无法重叠的通信时间。*

{% include figure.liquid path="assets/gpu/pipeline-bubble.png" class="img-fluid" caption="<b>图：</b> 一个两阶段、两微批量的流水线示例。F 表示一个阶段的前向传递，B 表示一个阶段的反向传递（成本是前者的两倍）。G 表示数据并行的 AllReduces，其耗时可能显著长于单个微批量的时间。" %}

(3) **流水线气泡与步骤不平衡：** 从上面（糟糕的）流水线调度图中可以看出，在一个天真的流水线调度中，很容易出现显著的气泡（意味着浪费的算力）。上面的例子中，第二阶段在第 0 步处于空闲状态，第一阶段从第 2 步到第 3 步处于空闲状态，第二阶段在最后一步再次处于空闲状态。虽然我们可以通过仔细的调度来避免这些气泡，但仍然经常存在一些气泡。我们还需要在关键路径上将激活值从一个阶段传递到下一个阶段，这会增加开销：

{% include figure.liquid path="assets/gpu/pipeline-transfer.png" class="img-fluid" caption="<b>图：</b> 一个示例流水线，用红色表示转移成本。这会改变各阶段之间的相对位置，并增加流水线气泡的开销。" %}

这些问题都有变通方法，但它们通常难以实现且难以维护；与其它方法相比，流水线技术的通信成本较低。

**关于延迟的注意事项：** 如前所述，即使消息规模较大，GPU 也难以实现完整的 AllReduce 带宽。这意味着，即使理论上我们可以将例如专家并行的 AllToAll 操作扩展到多个节点，我们可能也难以达到总带宽的 50%。因此，我们尝试将 TP 或 EP 限制在较少的节点内，以最小化延迟开销。

### 示例

**DeepSeek 是做什么的？** 作为参考，[DeepSeek V3](https://arxiv.org/abs/2412.19437) 使用 2048 个 H800 GPU 进行训练，具体配置为：

* 64 专家并行（EP）跨 8 个节点  
* 16 管道并行（PP）  
* 2 路 ZeRO-1 数据并行（DP）

他们采用了一个稳定的批量大小为 `4096 * 15360 = 62,914,560` 个词元，即每个 GPU 30k 个词元。你可以看到这个批量大小已经相当大了，但他们的模型也非常稀疏（k=8，E=256），因此需要较大的批量大小。你可以看到，使用 64 路 EP 和 16 路 PP，我们最终得到了总共 1024 路的模型并行，这意味着 AllReduce 是在 spine 层进行的，由于只有 2 路，实际带宽达到了 $2 / (2 - 1) = 2$ 倍。这也帮助降低了最终数据并行 AllReduce 与最终流水线阶段重叠时的成本。

**LLaMA-3 做了什么？** LLaMA-3 使用 16k GPU 上的 16M 词元进行训练，每块 GPU 约为 1k 词元。他们做了以下事情：

* 节点内 8-way 张量并行（TP）  
* 16-way 流水线并行（PP）  
* 128-way ZeRO-1 数据并行

这同样是一个密集模型，因此一般来说这些事情都相当简单。16 路流水线并行将数据并行 AllReduce 的成本降低了 16 倍，这有助于我们减少关键批量大小。

### LLM 在 GPU 上缩放的要点

让我们后退一步，对到目前为止学到的内容做一个总体的总结：

* **数据并行或 FSDP（ZeRO-1/3）要求每个 GPU 的本地批量大小约为 2500 个词元**，尽管理论上通过网络内归约 + 纯数据并行可以在一定程度上降低这一数值。  
* **张量并行在最多约 8 路时受算力限制**，但由于带宽不足，在扩展到更多路之前会很快变为通信受限。这使我们主要局限于单个 NVLink 域（即单节点或需要使用 GB200NVL72，最多 72 个 GPU）。  
* **任何跨越多个节点的模型并行形式都可以进一步降低 FSDP 的成本**，因此我们通常希望混合使用 PP + EP + TP，以跨越多个节点并减少 FSDP 的成本。  
* **如果能够处理零气泡流水线的代码复杂性，并保持较大的批量大小以避免数据并行瓶颈，流水线并行效果良好**。流水线通常会使 ZeRO-3 不可行（因为每个流水线阶段都需要 AllGather），但可以改用 ZeRO-1。

**从高层次来看，这为我们提供了一个在 GPU 上对大型模型进行分片的方案：**

* 对于相对较小的密集模型，如果具备足够的批量大小，激进的 FSDP 效果很好，必要时还可以结合一定量的流水线并行或张量并行。
* 对于较大的密集模型，1-2 节点的张量并行加上许多节点的流水线并行以及纯数据并行的组合效果较好。
* 对于 MoE 模型，上述规则同样适用，但我们还可以进行专家并行，通常我们更倾向于使用张量并行。如果 $F > 8 * C / W_\text{node}$，我们可以进行大量多节点的专家并行，否则我们只能限制在大约 2 节点的专家并行。

### Quiz 5: LLM 屋顶线

**问题 1 [B200 屋顶线]:** 一个 B200 DGX SuperPod（**不是 GB200 NVL72**）在节点内部具有两倍的带宽（900GB/s 出口）但在 scale-out 网络中具有相同数量的带宽（400GB/s）([来源](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-b200/latest/network-fabrics.html))。上述已报告了总 FLOPs。这会如何改变模型和数据并行的 rooflines？

{% details 点击此处查看答案。 %}

**答案：** 我们的 bfloat16 FLOPs/s 从 990 提升至 2250 TFLOPs，提升了 2.25 倍。在带宽翻倍的情况下，单个节点内的性能上限基本保持不变。以 TP 为例，关键强度上升至 `2250e12 / 900e9 = 2500`，因此我们有一个限制为 $Y < F / 2500$，仅略微提高（除非节点规模增加，否则这对我们没有帮助）。

然而，超出一个节点后，额外带宽的缺乏实际上使我们更难达到算力限制！例如，对于数据并行，我们的关键批量大小增加到 `2250e12 / 400e9 = 5625`，因为我们的 GPU 在相同带宽下可以执行显著更多的 FLOPs。

GB200 SuperPods 通过增加更多的出站带宽改变了这一情况（[来源](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)）。

{% enddetails %}

**问题 2 [如何分片 LLaMA-3 70B]:** 考虑在 bfloat16 精度下训练 LLaMA-3 70B，使用 fp32 优化器状态和 Adam 优化器。

1. 至少，我们需要多少块 H100 显卡才能仅仅存储权重和优化器？  
2. 假设我们想在 4096 块 H100 GPU 上训练 15T 词元，假设我们实现了 45% 的 MFU（模型 FLOPs 利用率）。训练需要多长时间？  
3. LLaMA-3 70B 具有 `F = 28,672`，并使用约 4M 词元的批量大小进行训练。在不因通信受限的情况下，我们最多可以实现多少模型并行？结合纯数据并行，我们能否在 4k 张芯片上训练 LLaMA-3 并保持算力受限？如果是 ZeRO-3 呢？如果是 8 路流水线并行呢？*注意：考虑通信成本和 GPU 显存使用情况。*

{% details 点击此处查看答案。 %}

1. 我们需要 2 字节用于权重，8 字节用于优化器状态，因此至少需要 700GB。在拥有 80GB DRAM 的情况下，我们至少需要 9 块 GPU，或者（向上取整）至少 2 个 8xH100 节点。这将需要非常长的时间进行训练，而且无法保存梯度检查点，但这只是一个下限。

2. 这将总共需要 `6 * 70e9 * 15e12 = 6.3e24 bf16 FLOPs`。每块 GPU 可以执行 `990e12` FLOPs，因此在 45% 的 MFU 下，我们可以达到 1.8e18 FLOPs/s。因此，整个训练过程将需要 3.5e6 秒，即 40 天。

3. 在一个节点内，我们有 450GB/s 的带宽，因此限制大约为 `F / 1995 = 28672 / 1995 = 14.372`。由于这不会跨越两个节点，实际上这意味着我们最多可以达到 8 路模型并行。

   1. 这将要求我们进行 512 路 DP。首先，我们需要查看是否有足够的内存。由于我们的模型仅被分片为 8 路，这意味着 `700GB / 8 = 87.5GB / GPU`，这无法容纳，所以不行！

   2. 使用 ZeRO-3 和 8 路 TP，我们将进行 512 路 ZeRO-3。由于我们积极地对一切进行分片，因此内存方面不会有问题。每块 GPU 的批量大小将是 `4e6 / 4096 = 976`。这个数值相当低，甚至低于我们纯 DP 的限制，而且由于我们需要移动权重，这个数值是该限制的两倍。所以也不行。

   3. 使用 8 路流水线后，每个模型并行分片现在跨越 8 个节点。正如我们所看到的，这将叶级 AllGathers 的成本降低了 8 倍，因此那里的整体 AllReduce/AllGather 带宽将从 400GB/s 变为 `8 * 400GB/s = 3200GB/s`。此时的性能上限为 `990e12 / 3200e9 = 309`，因此我们应该没问题！我们只需要高效地实现流水线即可。

{% enddetails %}

**问题 3 [Megatron-LM 超参数]:** 请考虑来自 [Megatron-LM 仓库](https://github.com/NVIDIA/Megatron-LM) 的这张图，该图突出了他们较高的 MFU 数值。

{% include figure.liquid path="assets/gpu/megatron-hparams.png" class="img-fluid" %}

请注意，它们的序列长度在所有地方都是 4096。对于 16B、70B 和 314B 模型，每个 GPU 的词元批量大小是多少？假设数据并行是外层轴，并假设使用 bfloat16 的归约操作，判断这些模型在理论上是算力受限还是通信受限，并判断是否存在更优的配置？

{% details 点击此处查看答案。 %}

**答案：** 让我们从每块 GPU 的批量大小开始。

* **16B**: 每个 GPU `192 * 4096 / 192 = 4096` 词元  
* **70B**: 每个 GPU `384 * 4096 / 768 = 2048` 词元  
* **314B**: 每个 GPU `1536 * 4096 / 3072 = 2048` 词元

这意味着，除了第一个情况外，这些都在每批约 2k 词元附近波动，这明显接近我们为 FSDP 计算出的关键阈值。我们根据脊柱级别的缩减计算出该限制为每 GPU 2,472 词元，这应大致在此处起作用。然而，对于 70B 和 314B 模型，由于我们分别采用了 16 和 64 路模型（PP + TP）分片，我们在脊柱级别分别获得了 2 倍和 8 倍的吞吐量提升，这意味着在每步约 1k 和 300 词元的水平上，我们应分别受到算力的限制。

{% enddetails %}

## 致谢与进一步阅读

本章在许多有见识的 GPU 专家的帮助下完成，包括：

* Adam Paszke，他帮助解释了在 GPU 上进行内核编程的实际情况。  
* Swapnil Patil，他首次解释了 GPU 网络的工作原理。  
* Stas Bekman，他指出 GPU 的实际性能往往与标称规格存在差异。  
* Reiner Pope，他帮助澄清了在硬件层面上 GPU 和 TPU 的比较。  
* Frédéric Bastien，他对芯片层面的叙述提供了详细的反馈。  
* Nouamane Tazi，他在 GPU 上进行大语言模型训练的经验帮助改进了顶线部分。  
* Sanford Miller，他帮助我理解了 GPU 的网络连接方式，以及 NVIDIA 的规格与实际部署情况之间的比较。

关于 GPU 有很多优秀的阅读材料，但我的一些最爱包括：

* [SemiAnalysis' History of the NVIDIA Tensor Core](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)：一篇精彩的文章，描述了 GPU 如何从视频游戏引擎转变为 ML 加速器。  
* [SemiAnalysis' Analysis of Blackwell Performance](https://semianalysis.com/2024/04/10/nvidia-blackwell-perf-tco-analysis/)：值得一读，以了解下一代 NVIDIA GPU 的性能。  
* [H100 DGX SuperPod Reference](https://docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf)：一篇枯燥但有用的读物，介绍了如何将更大的 GPU 集群进行网络连接。[这里](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576) 是一篇关于 GB200 系统的类似文档。  
* [Hot Chips Talk about the NVLink Switch](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf)：一篇有趣的读物，介绍了 NVLink 和 NCCL 集合通信，特别是包括网络内归约操作。  
* [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)：一篇大型半开源 LLM 训练报告的优秀示例，描述了他们如何选择分片设置。  
* [How to Optimize a CUDA Matmul](https://siboehm.com/articles/22/CUDA-MMM)：一篇优秀的博客，介绍了如何使用 CUDA 核心实现高效的矩阵乘法，并关注 GPU 上的缓存一致性。  
* [HuggingFace Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)：一份关于在 GPU 上进行 LLM 并行的指南，部分启发了本章内容。  
* [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)：一份更侧重于 GPU 和 PyTorch 的教程，介绍了 LLM 的性能极限和性能工程。  
* [Cornell Understanding GPU Architecture site](https://cvw.cac.cornell.edu/gpu-architecture)：一份与本书类似的指南，更具体地比较了 GPU 和 CPU 的内部结构。

## 附录 A：这会随着 GB200 如何变化？

Blackwell 引入了一系列重大的网络改进，包括 NVLink 5，其整体 NVLink 带宽是之前的两倍（900GB/s）。B200 仍然采用 8-GPU 节点，与 H100 相同，但 GB200 系统（将 B200 GPU 与 Grace CPU 结合）引入了更大规模的 NVLink 域（NVL72 中包含 72 个 GPU，理论上最多可达 576 个）。这种更大规模的 NVLink 域还有效提高了节点出口带宽，从而在节点级别以上降低了集合通信的成本。

{% include figure.liquid path="assets/gpu/b200-node.png" class="img-small" caption="<b>图：</b> 展示了一个 GB200 NVL72 单元的结构图，包含 18 个交换机和 72 个 GPU。" %}

在单个节点内，这种增加的带宽（从 450GB/s 提升到 900GB/s）并没有带来太大的变化，因为我们同时也将每块 GPU 的总 FLOPs/s 提高了一倍。我们的性能上限基本保持不变，尽管 NVLink 具有更高的带宽，使得专家并行变得更加容易。

超出一个节点后，情况会变得更加复杂。这里是从[此处](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)获取的 SuperPod 示意图。

{% include figure.liquid path="assets/gpu/gb200-superpod.png" class="img-fluid" caption="<b>图：</b>展示一个包含576个GPU的GB200 DGX SuperPod的示意图。" %}

如你所见，每节点的出带宽增加到 `4 * 18 * 400 / 8 = 3.6TB/s`，相比 H100 的 400GB/s 有所提升。由于我们的 FLOPs/芯片也翻了一番，这使得有效跨节点上限提高了约 4 倍。现在我们可能开始担心是否在节点层面而非扩展层面出现了瓶颈。

**Grace Hopper:** NVIDIA 还销售 GH200 和 GB200 系统，这些系统将一定数量的 GPU 与 Grace CPU 配对。例如，GH200 包含 1 个 H200 和 1 个 Grace CPU，而 GB200 系统包含 2 个 B200 和 1 个 Grace CPU。这种系统的优点在于，CPU 通过全带宽 NVLink 连接（称为 NVLink C2C）连接到 GPU，因此 CPU 到 GPU 的带宽非常高，适用于将参数卸载到主机内存。换句话说，对于任何给定的 GPU，访问主机内存的带宽与访问另一个 GPU 的 HBM 的带宽是相同的。

## 附录 B：更多网络细节

这是 NVLink 4 交换机的示意图。总共有 64 个 NVLink4 端口（每个端口使用 2 个物理通道），以及一个大型交叉开关，用于处理通道间的切换。相比之下，TPU 使用的是光学交换机，其镜面可以动态重新配置。

{% include figure.liquid path="assets/gpu/nvlink4.png" class="img-fluid" caption="<b>图：</b> 单个 NVLink4 交换机的更低层次视图。" %}

在每一层级，我们可能会受到可用链路带宽或总交换机带宽的限制。

* **节点级别：** 在节点级别，我们拥有 4 × 1.6TB/s = 6.4TB/s 的 NVSwitch 带宽，但每个 GPU 只能以 450GB/s 的速率向交换机输出数据，这意味着节点内部的实际峰值带宽为 450e9 × 8 = 3.6TB/s（全双工）。
* **SU/叶节点级别：** 在 SU 级别，我们使用 8 个交换机以全连接方式连接 32 个节点，使用 1×400 Gbps 的 InfiniBand。这为我们提供了 8 × 32 × 400 / 8 = 12.8TB/s 的节点输出带宽，交换机级别也有 8 × 1.6TB/s = 12.8TB/s，两者完全一致。
* **脊椎交换机级别：** 在脊椎交换机级别，我们使用 16 个交换机通过 2×400 Gbps 的链路连接 32 个叶交换机，因此我们拥有 32 × 16 × 400 × 2 / 8 = 51.2TB/s 的输出带宽。与叶交换机不同，脊椎交换机的所有 64 个端口都朝下，因此每个交换机可以处理 64 × 400 / 8 = 3.2TB/s 的流量，从而在交换机级别实现 16 × 3.2TB/s = 51.2TB/s 的带宽，两者再次完全一致。

每块 GPU 上，这使我们在节点级别实现了 450GB/s 的 GPU 到 GPU 带宽，在 SU 和脊椎级别实现了 50GB/s 的带宽。

**GPU 经验性 AR 带宽：**

{% include figure.liquid path="assets/gpu/gpu-all-reduce-bw.png" class="img-fluid" caption="<b>图：</b> 8xH100 集群上的 AllReduce 带宽（节点内，SHARP 禁用）。" %}

TPU v5p 带宽（1 轴）：

{% include figure.liquid path="assets/gpu/tpu-all-reduce-bw.png" class="img-fluid" caption="<b>图：</b> TPU v5p 4x4x4 集群（沿一个轴）的 AllReduce 带宽。" %}

这里还有 AllGather 带宽：

{% include figure.liquid path="assets/gpu/gpu-all-gather-bw.png" class="img-fluid" caption="<b>图：</b> 8xH100 集群（节点内）的 AllGather 带宽。" %}

{% include figure.liquid path="assets/gpu/tpu-all-gather-bw.png" class="img-fluid" caption="<b>图：</b> TPU v5e 8x16 集群（沿一个轴）的 AllGather 带宽。" %}

**关于 AllToAll 成本的更多内容：**

此处我们可以将近似值 $\min(K / Z) * (Z - 1) / Z$ 与 $(1 - ((Z - 1) / Z) ** K) * (Z - 1) / Z$ 的真实值进行比较。它们在 $Z$ 的小值时相似。

{% include figure.liquid path="assets/gpu/all-to-all-approx.png" class="img-fluid" caption="<b>图：</b> 不同分片数量下，粗糙 AllToAll 的近似成本与真实成本的对比。" %}
