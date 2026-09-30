---
layout: distill
title: "如何扩展您的模型"
subtitle: "基于TPU的大语言模型系统视图"
# permalink: /main/
description: "训练大型语言模型（LLMs）常常感觉像是炼金术，但理解和优化模型的性能并不需要如此。本书旨在揭示语言模型缩放的科学原理：TPUs（以及GPUs）的工作原理及其相互之间的通信方式，LLMs如何在真实硬件上运行，以及如何在训练和推理过程中对模型进行并行化处理，从而在大规模场景下高效运行。如果你曾经疑惑“训练这个LLM需要多昂贵”或“我需要多少内存才能自己部署这个模型”或“什么是AllGather”，希望本书对你有所帮助。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

giscus_comments: true

section_number: 0

previous_section_url: ""
previous_section_name: "Part 0: Intro"

next_section_url: roofline
next_section_name: "Part 1: Rooflines"

bibliography: main.bib

citation: true

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
  - name: High-Level Outline
  - name: Links to Sections

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
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

{% include figure.liquid path="assets/img/dragon.png" class="img-fluid" %}

深度学习的许多方面仍然类似于一种黑魔法，但优化模型的性能并不一定如此——即使在大规模场景下也是如此！相对简单的原理适用于所有情况——从单个加速器到数万个——理解这些原理可以让你完成许多有用的事情：

- 大致估算模型各部分与理论最优值的接近程度。  
- 在不同规模下（如何在多个设备间分配计算），就不同的并行方案做出有根据的选择。  
- 估算训练和运行大型 Transformer 模型所需的成本和时间。  
- 设计能够利用 [特定](https://arxiv.org/abs/2205.14135) [硬件](https://arxiv.org/abs/1911.02150) [特性](https://arxiv.org/abs/2007.00072) 的算法。  
- 设计硬件，基于对当前算法性能限制的明确理解。

**预期背景：** 我们假设你对大型语言模型（LLMs）和 Transformer 架构有基本的了解，但不一定了解它们在大规模下的运作方式。你应该了解大型语言模型训练的基础知识，并且最好对 JAX 有基本的熟悉。一些有用的背景阅读材料可能包括关于 Transformer 架构的[这篇博客文章](https://jalammar.github.io/illustrated-transformer/)和[原始 Transformer 论文](https://arxiv.org/abs/1706.03762)。此外，可以查看[这份列表](conclusion#further-reading)获取更多有用的同步和未来阅读材料。

**目标与反馈：** 最终，你应该能够自信地估算出在给定硬件平台上 Transformer 模型的最佳并行方案，以及训练和推理大致需要多长时间。如果做不到，请给我们发邮件或留下评论！我们非常想知道如何让这些内容更清晰易懂。

<p markdown=1 class="announce">您可能还喜欢阅读关于 NVIDIA GPU 的新[第 12 章](gpus)！</p>

### 你为什么应该关心？

三到四年前，我认为大多数机器学习研究人员都不需要理解本书中的任何内容。但如今，即使是“小型”模型也接近硬件极限，进行新颖的研究需要你思考如何在大规模下实现效率。<d-footnote>历史上，机器学习研究在系统创新和软件改进之间遵循某种“tick-tock”循环。Alex Krizhevsky 必须编写一些难以置信的 CUDA 代码才能让卷积神经网络运行得更快，但几年后，像 Theano 和 TensorFlow 这样的库出现，使得你不再需要这样做。也许这里也会发生类似的事情，几年后本书中的所有内容都可能被抽象掉。但缩放定律将我们的模型不断推向硬件的最前沿，而且在可预见的未来，进行前沿研究似乎不可避免地要与如何高效地将模型扩展到大型硬件拓扑结构的理解紧密联系在一起。</d-footnote> **在基准测试中获得 20% 的提升，如果是以 20% 的屋顶效率损失为代价，那毫无意义。** 有前景的模型架构经常失败，要么是因为它们 _无法_ 在大规模下高效运行，要么是因为没有人付出努力去实现这一点。

**“模型缩放”的目标是能够在增加用于训练或推理的芯片数量的同时，实现吞吐量的成比例、线性增长。** 这被称为“*强缩放*”。虽然增加额外的芯片（“并行性”）通常会减少计算时间，但也伴随着芯片之间通信增加的成本。当通信时间超过计算时间时，我们就会变得“受通信限制”，无法实现强缩放。<d-footnote>随着计算时间的减少，你通常还会在单个芯片层面面临瓶颈。你崭新的TPU或GPU可能被标称每秒可以执行500万亿次操作，但如果你不小心，它也可能因为内存中移动参数而只完成其中的十分之一。每个芯片的计算能力、内存带宽和总内存之间的相互作用对缩放故事至关重要。</d-footnote>如果我们充分理解硬件，能够预见这些瓶颈将出现在哪里，就可以设计或重新配置模型以避免它们。<d-footnote>硬件设计师面临的是相反的问题：构建能够为我们的算法提供足够算力、带宽和内存的硬件，同时尽量降低成本。你可以想象这个“协同设计”问题有多么令人压力山大：你必须押注在第一批芯片实际可用时算法会是什么样子，而第一批芯片通常要2到3年后才会面世。TPU的故事在这个领域取得了巨大成功。矩阵乘法是一种独特的算法，因为它每字节内存使用的FLOPs远高于几乎所有其他算法（N FLOPs每字节），而早期TPU及其 systolic array 架构在它们推出时的性能/美元比方面远优于当时的GPU。TPU是为机器学习工作负载设计的，而GPU凭借其Tensor Core也在迅速改变，以填补这一领域。但你可以想象，如果神经网络没有兴起，或者以某种根本性的方式发生了变化，而TPU（本质上不如GPU灵活）无法应对，那将会多么昂贵。</d-footnote>

*本书的目标是解释 TPU（以及 GPU）硬件的工作原理，以及 Transformer 架构是如何演进以在当前硬件上表现良好的。我们希望这既对设计新架构的研究人员有用，也对让当前一代大语言模型运行得更快的工程师有所帮助。*

## 高层次概述

本书的整体结构如下：

[第 1 章](roofline)解释了屋顶线分析以及哪些因素可能限制我们扩展能力（通信、计算和内存）。[第 2 章](tpus)和[第 3 章](sharding)详细介绍了 TPU 的工作原理，既作为单个芯片，又作为互联系统，其中芯片间的链路带宽和延迟有限。我们将回答如下问题：

* 某一特定规模的矩阵乘法需要多长时间？在什么情况下会受到算力、内存或通信带宽的限制？  
* TPU 如何连接在一起形成训练集群？系统各部分具有多少带宽？  
* 在多个 TPU 之间收集、分散或重新分配数组需要多长时间？  
* 如何高效地乘以在不同设备上分布的矩阵？

{% include figure.liquid path="assets/img/pointwise-product.gif" class="img-small" caption="<b>图：</b> 来自 <a href='tpus'>第2节</a> 的一张图表，展示了TPU如何执行逐元素乘法。根据数组的大小和各种链路的带宽，我们可能会遇到算力受限（充分利用硬件算力）或内存受限（受内存加载瓶颈限制）的情况。" %}

五年之前，机器学习领域有着丰富多彩的架构图景——卷积网络（ConvNets）、长短期记忆网络（LSTMs）、多层感知机（MLPs）、Transformer——但现在我们基本上只使用Transformer<d-cite key="transformers"></d-cite>。我们坚信理解Transformer架构的每一个细节都是非常有价值的：每个矩阵的确切尺寸、归一化发生在何处、参数和FLOPs<d-footnote>FLoating point OPs，基本上就是所需的加法和乘法的总次数。虽然许多资料将FLOPs理解为“每秒操作次数”，但我们使用FLOPs/s来明确表示这一点。</d-footnote>包含在每个部分中。[第4节](transformers)详细介绍了这种“Transformer数学”，展示了如何计算训练和推理时的参数和FLOPs。这告诉我们模型将使用多少内存、在计算或通信上将花费多少时间，以及注意力机制相对于前馈块何时变得重要。

{% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid" caption="<b>图：</b> 一个标准的 Transformer 层，每个矩阵乘法（matmul）以圆圈内的点表示。所有参数（不包括归一化层）以紫色显示。<a href='transformers'>第4节</a> 对此图进行了更详细的讲解。" %}

[第5章：训练](training)和[第7章：推理](inference)是本书的核心，我们在其中讨论一个基本问题：给定某种规模的模型和一定数量的芯片，如何将模型并行化以保持在“强缩放”区间？这是一个简单的问题，但答案却出人意料地复杂。从高层次来看，有四种主要的并行化技术用于将模型拆分到多个芯片上（**数据**、**张量**、**流水线**和**专家**），还有一些其他技术用于减少内存需求（**重计算**、**优化器/模型分片（又称 ZeRO）**、**主机卸载**、**梯度累积**）。我们在本章中将讨论其中的许多技术。

希望在完成这些章节后，你能够自行在新的架构或设置中选择适用的方法。[第6章](applied-training)和[第8章](applied-inference)是实践教程，将这些概念应用于LLaMA 3这一流行的开源模型。

最后，[第 9 章](profiling)和[第 10 章](jax-stuff)将探讨如何在 JAX 中实现这些想法，以及在出现问题时如何对代码进行性能分析和调试。[第 12 章](gpus)是一个新章节，深入介绍了 GPU 的相关内容。

在本书中，我们尽量为你提供需要自行解决的问题。请不必感到有压力，需要阅读所有章节或按顺序阅读。也请给予反馈。目前，这只是一个草稿，将持续修订。感谢！

*我们想感谢 James Bradbury 和 Blake Hechtman，他们提出了本书中的许多想法。*

<h3 markdown=1 class="next-section">不加赘述，[这里是第1节](roofline)关于TPU性能极限。</h3>

## 章节链接

*这个系列可能比必要的要长一些，但我们希望这不会阻止你阅读。前三个章节是预备知识，如果你已经熟悉相关内容，可以跳过，不过它们会介绍后面会用到的符号。最后三个部分可能最有实际用处，因为它们解释了如何与实际模型进行交互。*

**第一部分：预备知识**

* [**第 1 章：屋顶线分析简介**](roofline)。算法受到三个方面限制：算力、通信和内存。我们可以利用这些因素来估算算法的运行速度。

* [**第 2 章：如何理解 TPUs**](tpus)。TPUs 是如何工作的？这会影响我们可以训练和部署哪些模型？

* [**第 3 章：分片矩阵及其相乘方法**](sharding)。在这里，我们通过我们最喜欢的运算：（分片）矩阵乘法，来解释模型分片和多 TPU 并行。

**第二部分：Transformer**

* [**第 4 章：你需要了解的所有 Transformer 数学知识**](transformers)。Transformer 在其前向和反向传播中使用了多少 FLOPs？你能计算出参数的数量吗？它的 KV 缓存的大小？我们在这里详细推导这些数学内容。

* [**第 5 章：如何为训练并行化 Transformer**](training)。FSDP。Megatron 分片。流水线并行。给定一定数量的芯片，如何尽可能高效地训练给定规模和批量大小的模型？

* [**第 6 章：在 TPUs 上训练 LLaMA 3**](applied-training)。我们如何在 TPUs 上训练 LLaMA 3？需要多长时间？成本是多少？

* [**第 7 章：Transformer 推理全解析**](inference)。一旦我们训练好一个模型，就必须对其进行服务。推理引入了新的考量——延迟——并改变了内存的使用情况。我们将讨论分散式服务的工作原理以及如何思考 KV 缓存。

* [**第 8 章：在 TPU 上部署 LLaMA 3**](applied-inference)。在 TPU v5e 上部署 LLaMA 3 的成本是多少？延迟/吞吐量之间有哪些权衡？

**第三部分：实践教程**

* [**第 9 章：如何分析 TPU 代码**](profiling)。真实的大型语言模型从不会像上述理论那样简单。在这里，我们解释 JAX + XLA 栈以及如何使用 JAX/TensorBoard 分析器来调试和解决实际问题。

* [**第 10 章：使用 JAX 编程 TPUs**](jax-stuff)。JAX 提供了一系列用于并行化计算的神奇 API，但你需要知道如何使用它们。有趣示例和实际问题解析。

**第 4 部分：结论与附加内容**

* [**第 11 章：结论与进一步阅读**](conclusion). 对 TPU 和大语言模型的总结思考及进一步阅读资料。

* [**第12章：如何理解GPU**](gpus)。关于GPU的附加章节，介绍它们的工作原理、网络连接方式，以及它们的性能上限与TPU的不同之处。

