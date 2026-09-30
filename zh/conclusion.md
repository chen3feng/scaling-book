---
layout: distill
title: "结论与进一步阅读"
# permalink: /main/
description: "感谢您的阅读！在这里我们将再列出一些参考资料供进一步学习。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 11

previous_section_url: "../jax-stuff"
previous_section_name: "Part 10: JAX"

next_section_url: "../gpus"
next_section_name: "Part 12: GPUs"

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
  - name: "Acknowledgments"
  - name: "Further Reading"
  - name: "Feedback"

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
  .algorithm {
    padding: 10px;
    margin-top: 5px;
    margin-bottom: 5px;
    border-style: dashed;
    background-color: #fffaf2;
  }

  .algorithm li {
    margin-bottom: 0px;
  }
---

**感谢您读完整个内容，并祝贺您一路读到结尾。** 在我们结束之前，先做几点致谢：

## 致谢

这份文档凝聚了 Google DeepMind 许多员工的重要集体投入，我们想简要致谢！

- James Bradbury、Reiner Pope、Noam Shazeer 和 Blake Hechtman 最早提出了本文稿中的许多想法，并率先理解了 Transformer 的系统视角。  
- Sholto Douglas 撰写了这份文档的初稿，并负责启动该项目。他是所有作者中对这份文档整体叙述负责的人。  
- Jacob Austin 主导了将这份初稿从草稿转化为更加完善和全面的成果的工作。他完成了大部分的编辑、格式化和发布工作，并协调了其他作者的贡献。  
- 大多数图表和动画由 Anselm Levskaya 和 Charlie Chen 制作。  
- Charlie Chen 撰写了推理部分，并绘制了大部分推理相关的图表。  
- Roy Frostig 在出版、编辑以及旅程中的许多其他步骤上提供了帮助。

我们还想感谢在过程中给予关键反馈的许多人，特别是 Zak Stone、Nikhil Sethi、Caitlin Stanton、Alek Dimitriev、Sridhar Lakshmanamurthy、Albert Magyar、Diwakar Gupta、Jeff Dean、Corry Wang、Matt Johnson、Peter Hawkins 和许多其他人士。感谢 Ruiqi Gao 在 HTML 格式方面的帮助。

**谢谢大家！**

<p markdown=1 class="announce">在您离开之前，您可能还喜欢阅读关于 NVIDIA GPU 的新 [第 12 部分](../gpus)！</p>

## 进一步阅读

有关这方面的内容已有大量相关论述，包括以下内容：

- [**TPU Deep Dive**](https://henryhmko.github.io/posts/tpu/tpu.html): 一本深入探讨 TPU 架构的精彩书籍，体现了本书的精神。  
- [**Domain specific architectures for AI inference**](https://fleetwood.dev/posts/domain-specific-architectures): 一本在本书精神指导下深入探讨硬件和模型的书籍。  
- [**A Domain-Specific Supercomputer for Training Deep Neural Networks**](https://dl.acm.org/doi/pdf/10.1145/3360307): 这是最早的 TPU 论文之一，其中包含大量关于 Google TPU 计划的精彩细节，这些内容在本书中并未涉及。  
- [**Making Deep Learning Go Brrrr From First Principles**](https://horace.io/brrr_intro.html): 一本更侧重于 GPU 和 PyTorch 的教程，介绍了 LLM 的性能边界和性能工程。  
- [**Writing TPU Kernels with Pallas**](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html): 越来越多的 TPU 编程涉及使用 Pallas 编写自定义内核。该系列文章讨论了如何编写内核以及许多在本书中未提及的底层 TPU 细节。  
- [**How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog**](https://siboehm.com/articles/22/CUDA-MMM): 虽然内容特定于 GPU 和 CUDA，但这是一篇优秀的博客文章，展示了如何在 CUDA 中优化矩阵乘法内核。这可能是一个深入了解 TPU 和 GPU 差异的好方法。  
- [**Distributed arrays and automatic parallelization**](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html): 这是一篇关于 JAX 中并行 API 的优秀指南，是学习如何实际实现我们在此讨论的一些想法的好方法。  
- [**Rafi Witten's High Performance LLMs 2024 Class**](https://github.com/rwitten/HighPerfLLMs2024): 我们的前同事 Rafi 曾讲授一门关于 TPU 性能工程的精彩课程，所有幻灯片都发布在 GitHub 上。该课程对许多内容进行了更深入的讲解。  
- [**\[2211.05102\] Efficiently Scaling Transformer Inference**](https://arxiv.org/abs/2211.05102): 一篇详细探讨 Transformer 推理数学原理的论文。这是本书很多内容的灵感来源。  
- [**Huggingface Ultra-Scale Playbook**](https://huggingface.co/spaces/nanotron/ultrascale-playbook): 本书的 GPU 类比，深入探讨了 PyTorch 如何在训练过程中实现并行化技术和内存节省技术。  
- [**Transformer Inference Arithmetic**](https://kipp.ly/transformer-inference-arithmetic/): 一篇包含与本书类似想法的博客，并配有出色的插图。  
- [**Stanford CS336 Slides and Videos**](https://stanford-cs336.github.io/spring2025/index.html#coursework): 一门斯坦福大学的精彩课程，涵盖大量关于 LLM 训练和部署的细节，并附有有用的练习。第一和第二项作业尤其相关。  
- [**Stas Bekman's ML Engineering Handbook**](https://github.com/stas00/ml-engineering): 一本高度实用的机器学习基础设施指南，涵盖本书未涉及的内容，如如何与云服务提供商谈判、集群管理和 GPU 吞吐量的实证测量。  
- [**ezyang's blog**](https://blog.ezyang.com/2026/01/computing-sharding-with-einsum/): 一篇由 PyTorch 负责人撰写的博客，涵盖所有关于分片和 PyTorch 的内容，包括一篇 [PyTorch 内部机制指南](https://blog.ezyang.com/2019/05/pytorch-internals/) 和一篇 [分片矩阵乘法的说明](https://blog.ezyang.com/2026/01/computing-sharding-with-einsum/)。这里还有许多其他优秀内容。  
- [**The Anatomy of Collective Communication**](https://www.aleksagordic.com/blog/collective-operations): 一篇在本书精神指导下对 GPU 和 TPU 集合通信的精彩讲解。对 N-D 和 GPU 集合通信的描述比本书更详细。

在这一领域，仍有大量空间可以进行全面的写作，因此我们希望这篇手稿能够鼓励更多此类写作！我们还认为，这是一个富有成果的研究领域。在许多情况下，即使没有许多硬件加速器也可以进行研究。

## 反馈

请留下评论或问题，以便我们进一步改进。您可以通过以下方式联系我们对应的作者 Jacob Austin：jacobaustin123 [at] gmail [dot] com，或通过在 [GitHub](https://github.com/jax-ml/scaling-book) 上发布问题、拉取请求或讨论来提出修改建议。
