---
layout: distill
title: "在 TPUs 上训练 LLaMA 3"
# permalink: /main/
description: "让我们仔细看看如何利用上一节学到的知识，在TPU v5p上训练LLaMA 3模型。它们有多大？在不同配置下训练的费用如何？它们是如何分片的？让我们对前几节内容如何映射到实际模型做一些粗略估计。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 6

previous_section_url: "../training"
previous_section_name: "Part 5: Training"

next_section_url: ../inference
next_section_name: "Part 7: Inference"

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
  - name: "What does LLaMA 3 look like?"
  - name: "Counting parameters and FLOPs"
  - name: "How to shard LLaMA 3-70B for training"
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

本节的目标是将上一节的结果应用于一个非常实际的问题：训练LLaMA 3模型家族（群体）。与前面的章节不同，我们希望你自己完成大部分工作。因此，我们隐藏了每节的答案，以便你先尝试回答。试着拿起笔，手动完成它！

### LLaMA 3 看起来是什么样子？

LLaMA-3 模型系列<d-cite key="llama3"></d-cite> 包括 3 个主要模型：LLaMA 3 8B、70B 和 405B。我们将主要关注 70B，而将 8B 和 405B 留给你在最后的问题部分进行探索。以下是 LLaMA 3-70B 的架构，摘自 LLaMA [HuggingFace 页面](https://huggingface.co/NousResearch/Meta-Llama-3-70B/blob/main/config.json)。

|**超参数**|**价值函数**|
| --------------------------- | --------- ||$$n_\text{layers}$$ (L)|80|
|$$d_\text{model}$$ (D)|8,192|
|$$d_{ff}$$ (F)|28,672|
|$$n_\text{heads}$$ (N)|64|
|$$n_\text{kv_heads}$$ (K)|8|
|$$d_\text{qkv}$$ (H)|128|
|$$n_\text{embeddings}$$ (V)|128,256|

为了突出显示这有多么容易找到，这里是配置本身，以及一个映射：

{% include figure.liquid path="assets/img/llama-json.png" class="img-fluid" %}

为许多不同的开源大语言模型制作一个包含这些数字的大表格是有用的，这样你可以快速比较它们所做出的设计决策。

### 统计参数和 FLOPs

**问题：** 从这张表中，我们能计算出 LLaMA 3-70B 的参数量吗？🤫 让我们应用[第 4 章](../transformers)的内容，看看是否能得到 70B！

|参数|公式|数量|
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ ||前馈网络参数|d_model * d_ff * 3（用于 SwiGLU 门控、上投影和下投影）* n_layers|8,192 * 8,192 * 3.5 * 3 * 80 = **56.3e9**|
|词元参数|2（输入和输出嵌入）* n_embeddings * d_model|2 * 128,256 * 8,192 = **2.1e9**|
|注意力参数|n_layers * [ 2（用于 q 嵌入和拼接输出投影）* d_model * n_heads * d_qkv + 2（用于 k 和 v）* d_model * n_kv_heads * d_qkv]|80 * (2 * 8,192 * 64 * 128 + 2 * 8,192 * 8 * 128) = **12e9**|
|||56.3e9 + 2.1e9 + 12e9 = **70.4e9**|

太好了！我们得到了预期的数字。正如预期的那样，你会注意到FFW参数完全主导了总体参数量，尽管注意力部分并非微不足道。

<p markdown=1 class="takeaway">**要点**: MLP 模块中的三个大权重矩阵比 Transformer 中的所有其他数组大得多，因此在考虑模型内存或 FLOPs 时，通常可以忽略所有其他参数。对于 LLaMA 3-70B，它们占 70B 参数中的 56B。</p>

现在让我们来看 FLOPs！*请记住[第4节](../transformers)中提到的训练通用规则。*

**问题：** 每个训练步中，LLaMA-3 每个词元执行多少 FLOPs？_这有助于我们确定整个训练过程的费用。_

{% details 思考过后，点击此处查看答案！ %}

**答案**：如[第 4 章](../transformers)所示，我们每词元大约执行 $$6 \cdot \text{param count}$$ FLOPs，因此这里大约是 `6 * 70e9 = 4.2e11` FLOPs / 词元。这大约是每词元每步 0.5 TFLOP。假设我们受算力限制，这在单个 TPU v5p 芯片上应需要大约 `4.2e11 / 4.59E+14 = 1ms`，假设 FLOPs 利用率为完美。

{% enddetails %}

**问题：** LLaMA 3 是使用大约 15 万亿个词元进行训练的。总共是多少 FLOPs？

{% details 思考过后，点击此处查看答案！ %}

**答案**：这很简单，就是 `4.2e11 * 15e12 = 6.3e24 FLOPs` 总共的算力。6.3 太 FLOPs。这很多！在单个 TPU 上这将需要 `6.3e24 / 4.59E+14 = 435 years`。这也很多！

{% enddetails %}

**问题：** 假设我们想在由 16x20x28 = 8960 个芯片组成的完整 TPU v5p 集群上进行训练，如果在 bfloat16 精度下以 40% 的 MFU 进行训练，并且假设我们受算力限制，这需要多长时间？

{% details 思考过后，点击此处查看答案！ %}

**答案**：我们知道每个 TPU v5p 每秒可以执行 4.59e14 FLOPs。在 40% MFU 的情况下，这将需要大约 `T = 6.3e24 / (8960 * 4.59e14 * 0.4) = 3.8e6 seconds`。**这大约是 44 天！** 假设我们实际上可以实现 40% MFU，这在合理范围内。

{% enddetails %}

**问题：** LLaMA 3-70B 使用约 4M 词元的批量大小进行预训练。要使用这个批量大小进行训练，我们至少需要多少个 TPUs？_你可以假设参数为 bfloat16，优化器状态为 float32，并且你在每一层开始时和注意力之后两次检查点激活值。_

{% details 思考过后，点击此处查看答案！ %}

**答案**：这个问题主要询问内存使用情况，因为这是对可用算力的唯一严格约束。在训练过程中，我们有三种主要的 HBM 使用方式：模型参数、优化器状态和激活检查点。如果我们假设使用 bfloat16 权重、float32 优化器状态以及一个非常保守的激活检查点方案（每层两次），那么我们有：

| **参数** | 2 * 70GB | ~140GB |
| **优化器状态** | 8 * 70GB | ~560GB |
| **激活检查点** | 2 * 8192 * 4e6 * 2 * 80 | ~10.5TB |
| **总计**                |                         | ~11.2TB |

此处的总容量约为 11.2TB。你注意到激活值检查点机制在内存占用中占据主导地位，即使采用非常保守的检查点方案也是如此。从技术上讲，我们可以做到每层一个检查点，或者进行微批量处理，但当前的画面是合理的。基于这些假设，由于每个 TPU v5p 有 96GB 的 HBM，我们需要 `11.2e12 / 96e9 = 117` 个 TPUs。实际上，这个数量并不算多！

*我们为什么不这么做呢？* 好吧，因为这将需要我们 `44 days * 8960 / 117 = 3369 days` 的时间来训练。这几乎是十年。**这太多了。** 然而，这清楚地表明，我们使用这些大型集群并不是因为受内存限制，而是因为我们需要额外的 FLOPs。此外，由于我们很少进行检查点记录，我们实际上执行了接近 8ND FLOPs 的计算，而不是 6ND。
{% enddetails %}

**问题：** 在与上述问题相同的假设下，如果我们使用 8960 个 TPU v5p 芯片，每个芯片将使用多少内存？

{% details 思考过后，点击此处查看答案！ %}

**答案**：我们的总内存仍然是约 11.2TB，因此每块芯片上将使用约 1.3GB 内存，这几乎可以忽略不计。即使我们进行更激进的检查点机制，例如每层设置 12 个检查点，每块芯片的内存使用量也仅为 8GB。在这些规模下，训练过程中我们远未达到内存限制。

{% enddetails %}

<p markdown=1 class="takeaway">**要点**: 从技术上讲，即使在非常小的拓扑结构上训练非常大的模型也是可行的，但需要注意的是，这很可能会耗费大量时间。能够计算一次训练任务的总 FLOPs，使我们可以通过假设一个适中的 MFU 和已知的拓扑结构，大致估算其训练时间。</p>

### 如何对 LLaMA 3-70B 进行分片以进行训练

让我们继续沿用上面的设置，并说我们希望使用 4M 词元批量大小（每批量 1024 个长度为 4096 的序列）在由 8960 个芯片组成的 TPU v5p 集群上训练 LLaMA 3-70B。让我们讨论该模型的最佳分片策略是什么。

**问题：** 在上述假设下，我们能否仅使用 FSDP 来训练模型？首先，假设我们无法进行任何序列/上下文并行。_这应该是你首先想到的方案，因为它简单，且如果有效的话不会引入额外的通信开销。_

{% details 思考过后，点击此处查看答案！ %}

**答案**：这个答案会有点过于讲究。如上所述，LLaMA 3-70B 最初是使用长度为 4K 的序列进行训练的，因此 4M 词元的批量大小给我们一个 *序列批量大小* 为 1024。这意味着我们只能在最多 1024 个芯片上真正实现纯粹的数据并行/FSDP，_因为数据并行必须在这么多序列上进行_。因此，从“完全数据并行且没有任何额外通信”这个简单意义上来说，答案是否定的。下一个问题将回答一个稍微不那么讲究的版本。

{% enddetails %}

**问题：** 让我们放松不进行任何序列分片的要求。如果我们允许自己在批量和序列轴上都进行 FSDP，是否仅使用 FSDP 就能在 8960 个芯片上训练 LLaMA 3-70B？

{% details 思考过后，点击此处查看答案！ %}

**答案**：现在我们允许自己进行序列/上下文并行，我们可以扩展得更多。首先让我们计算每设备的批量大小。如果我们进行 8960-way FSDP，最终每 TPU 的批量大小为 `4 * 1024 * 1024 / 8960 = 468 tokens`。从上一节我们知道，当 $$\text{per device batch size} < 2550 / M_X$$ 时，FSDP 会使我们受到 ICI 的限制。由于我们可以在这里使用一个完整的 3D pod 专用 3 个轴，这将给我们一个下限为 850，而我们远低于这个下限。**所以答案是否定的，即使有 3 个轴，我们仍然会受到通信的限制。**

{% enddetails %}

**问题：** 现在让我们来看混合张量并行和 FSDP。是否存在某种组合方式，使我们能够保持算力受限？如果存在，我们应该采用多少 FSDP 和张量并行？

{% details 思考过后，点击此处查看答案！ %}

**答案**：首先让我们检查一下这是否甚至能够适应。我们知道，如果每芯片的批量大小小于 $2550^2 / 2F = 113$，我们将受到通信的限制。正如我们上面看到的，我们略微超过了这个数值。这很好！现在，为了选择最佳的 FSDP 数量，我们可以使用以下公式

$$X_{opt} = \sqrt{\frac{2BN}{F}} = \sqrt{\frac{2 \cdot 4.19e6 \cdot 8960}{28672}} = 1618$$

四舍五入到 2 的合理倍数，这给我们大约 2048 个 FSDP 和 4 个张量并行。这应该能很好地工作！

{% enddetails %}

<p markdown=1 class="takeaway">**要点**: 我们可以在一个完整的 TPU v5p 芯片组上，使用数据并行（1024 路）、序列并行（2 路）和张量并行（4 路）的混合方式，以 4M 词元的批量大小训练 LLaMA-3，而不会受到通信的限制。如果我们尝试仅使用 FSDP 或 FSDP + 序列并行，就会受到通信的限制。我们在上一节中推导出的公式非常实用。</p>

## 习题解答

**问题 1 [将 LLaMA 70B 扩展到更多芯片]:** 假设我们想在 4 个 pod 上以相同的批量大小训练 LLaMA 3-70B。我们会使用哪种并行方案？我们会受到算力还是通信的限制？大致需要多长时间来训练？*确保使用正确的 roofline 限制。*

**问题 2 [LLaMA 405B]:**

(a) 使用 LLaMA 3-405B [配置](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-405B/blob/main/config.json)（这是一个门控模型，因此您可能需要登录并申请访问权限才能查看它），编写一个包含上述所有关键超参数的表格。该模型总共有多少参数？每个训练步有多少 FLOPs？如果我们训练 15T 个词元，总共执行多少 FLOPs？

(b) 假设我们想在 8 个 TPU v5p 节点上进行训练。我们会使用哪种并行方案？训练需要多长时间？我们会受到算力还是通信的限制？

<h3 markdown=1 class="next-section">第6节的内容到此结束。关于第7节，有关Transformer推理的内容，请点击[这里](../inference)。</h3>