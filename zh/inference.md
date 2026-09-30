---
layout: distill
title: "关于 Transformer 推理的一切"
# permalink: /main/
description: "对 Transformer 进行推理与训练可能有很大不同。部分原因在于推理引入了一个新的需要考虑的因素：延迟。在本节中，我们将从模型中采样一个新词元，一直到作为推理引擎的一部分，高效地将大型 Transformer 扩展到多个加速器切片上。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 7

previous_section_url: "../applied-training"
previous_section_name: "Part 6: Training LLaMA"

next_section_url: ../applied-inference
next_section_name: "Part 8: Serving LLaMA"

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
  - name: "The Basics of Transformer Inference"
  - subsections:
    - name: "What do we actually want to optimize?"
    - name: "Linear operations: what bottlenecks us?"
    - name: "What about attention?"
    - name: "Theoretical estimates for LLM latency and throughput"
    - name: "What about memory?"
    - name: "Modeling throughput and latency for LLaMA 2-13B"
  - name: "Tricks for Improving Generation Throughput and Latency"
  - name: "Distributing Inference Over Multiple Accelerators"
  - subsections:
    - name: "Prefill"
    - name: "Generation"
    - name: "Sharding the KV cache"
  - name: "Designing an Effective Inference Engine"
  - subsections:
    - name: "Continuous batching"
    - name: "Prefix caching"
    - name: "Let's look at an implementation: JetStream"
  - name: "Worked Problems"
  - name: "Appendix"

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

## Transformer 推理基础

所以你已经训练了一个 Transformer，现在想要用它来生成一些新的序列。_归根结底，基准分数的上升和损失曲线的下降只是代理指标，它们并不能说明一旦真正应用时是否会有一些有趣的事情发生！_<d-footnote>历史上，你可以在不接触推理的情况下对 Transformer 做大量研究——基于评分的多项选择基准可以高效运行，而无需完善的 KV 缓存或生成循环实现。这意味着，特别是在研究代码库中，推理代码路径中常常存在大量低垂的果实。</d-footnote>

采样在概念上很简单。我们将一个序列输入，我们最喜欢的 Transformer 会输出 $$\log p(\text{next token}_i \vert \text{previous tokens})$$，即所有可能的下一个词元的对数概率。我们可以从这个分布中进行采样，得到一个新的词元。将这个词元追加到序列中并重复这个过程，我们就能得到一个词元序列，它是对提示的延续。

{% include figure.liquid path="assets/img/naive-inference.png" class="img-fluid" caption="<b>图：</b> 从Transformer进行朴素采样。蓝色的logits给出了下一个词元的分布，我们可以从中进行采样。请注意，每一步都会重新处理整个前缀，导致算法的$\Theta(n^2)$运行时间。" %}

我们刚刚描述了 Transformer 采样的朴素实现方式，虽然它能运行，**但在实际中我们从不会这么做**，因为我们每次生成一个词元时都要重新处理整个序列。该算法在 FFW 上的效率是 $$O(n^2)$$，在注意力机制上的效率是 $$O(n^3)$$，仅能生成 $$n$$ 个词元！

**我们如何避免这种情况？** 每次都不必进行完整的前向传递，实际上，我们可以保存每次前向传递中的一些中间激活值，从而避免重新处理之前的词元。具体来说，由于在点积注意力过程中，一个给定的词元只会关注之前的词元，我们可以将每个词元的键和值投影写入一种称为 **KV 缓存** 的新数据结构中。一旦我们保存了过去词元的这些键/值投影，未来的词元就可以直接计算它们的 $$q_i \cdot k_j$$ 乘积，而无需对之前的词元执行任何新的 FLOPs。太棒了！

考虑到这一点，推理有两个关键部分：

* <b style="color: red;">Prefill</b>：给定一个长提示，我们同时处理提示中的所有词元，并将得到的激活值（具体为键值投影）保存在**“KV 缓存”**中。我们还会保存最后一个词元的 logit。
* <b style="color: blue;">Generation</b>：给定 KV 缓存和之前的 logit，我们从 logit 中逐步采样一个词元，将该词元反馈给 Transformer，并为下一步生成一组新的 logit。我们还会将该新词元的 KV 激活值追加到 KV 缓存中。我们重复这一过程，直到遇到一个特殊的 `<EOS>` 词元或达到某个最大长度限制。

这是使用 KV 缓存进行采样的示意图：

{% include figure.liquid path="assets/img/cached-inference.png" class="img-fluid" caption="<b>图：</b>高效Transformer采样与KV缓存的示意图。<b style=\"color: red;\">Prefill</b>处理我们的提示，并将每个词元的键值激活值保存在缓存中。<b style=\"color: blue;\">Generation</b>使用该缓存（以及最后一个词元的logit），采样一个新的词元，并将该新词元通过模型，注意KV缓存，并将新词元的键值投影保存回缓存。这是一个$O(n)$算法在MLP块中。" %}

通过使用 KV 缓存进行采样，我们已将生成 $n$ 个词元的时间复杂度降低到 $$O(n)$$（在 FFW 上）和 $$O(n^2)$$（在注意力上），因为我们不会重新处理之前的词元。然而，生成一个序列仍然需要许多前向传递——这就是当你查询 Gemini 或 ChatGPT 时，结果会逐步返回给你时发生的情况。每个词元（通常）都是对一个大规模模型的单独（但部分缓存）Transformer 调用。

我们很快就会看到 <b style="color: red;">prefill</b> 和 <b style="color: blue;">generation</b> 是截然不同的概念 —— Transformer 推理实际上是两个任务的伪装！与训练相比，KV 缓存也是新增的一个显著复杂来源。

### 我们实际上想要优化什么？

在我们继续之前，有必要强调推理中一个全新的方面：延迟。在训练过程中，我们只关心吞吐量（每秒每块芯片处理的总词元数），而在推理过程中，我们必须关注生成词元的速度（包括**首次生成词元时间（TTFT）**和**每个词元的延迟**）。例如：

* **离线批量推理** 仅关注推理的总体成本，对单个样本的延迟不敏感。  
* **聊天界面/流式任务** 需要在大规模下以低成本运行，同时具有较低的首次令牌延迟（TTFT），并能以足够快的速度生成令牌，以超过人类的阅读速度。  
* **边缘推理**（例如 `llama.cpp` 在你的笔记本电脑上）只需以最低可能的延迟为一个用户服务，可能面临严格的硬件限制。

最大化硬件利用率仍然至关重要，有助于降低成本和首次生成时间（TTFT），但与训练不同，它在所有情境下并不*必然*转化为用户更好的体验。许多在加速器、系统和模型架构层面的优化在延迟、吞吐量、上下文长度甚至模型质量之间做出权衡。

### Transformer 的更细致视图

到目前为止，我们基本上将 Transformer 视为一系列前馈块的堆叠。虽然从 FLOPs 和内存的角度来看，这通常是合理的，但它不足以正确建模推理。<d-footnote> 你将在本节中注意到的一个特点是，推理对训练的容忍度要低得多。我们通常拥有的 FLOPs 更少，批量处理的机会更少，并且对延迟的敏感性要高得多。KV 缓存也会显著增加推理的复杂性。</d-footnote> 如我们在[第 4 部分](../transformers)中所看到的，Transformer 前向传播的主要组成部分包括：

1. **一系列线性操作**，包括 MLP（$W_{in}$, $W_{out}$）以及注意力 QKV 投影和输出投影（$W_Q$, $W_K$, $W_V$, 和 $W_O$）。这些操作都涉及从 HBM 读取参数和一批激活值，执行一些 FLOPs，然后将结果写回 HBM。  
2. **点积注意力**。我们需要从 HBM 读取一批键值投影和一批查询激活值，执行一些内积运算和 softmax 操作，然后将注意力结果写回 HBM。  
3. **其他所有操作**，包括应用层归一化、激活函数、词元采样、更新 KV 缓存和位置嵌入。这些操作确实需要一些 FLOPs，但会被上述操作主导或融合到其中。

接下来的几个小节中，我们将分别在预填充和生成的背景下探讨这些问题，并询问哪些因素最可能成为性能的瓶颈。在一个加速器内部，我们是受算力限制还是受内存限制？我们想强调，预填充和生成的答案会有很大不同。

### 线性操作：什么限制了我们？

我们所有的线性操作在概念上都是相同的，无论它们位于 MLP 模块还是注意力模块中。它们的算术强度取决于批量大小。我们在[第 1 章](../roofline)中做过这个计算，但值得再次重复。让我们来看一个 $\text{bf16[B, D]}$ 批量与 $\text{bf16[D, F]}$ 矩阵的单个矩阵乘法。这可能是大的 MLP 模块（$W_\text{in}$ 或 $W_\text{out}$）或较小的注意力投影（$W_Q$、$W_K$、$W_V$、$W_O$）。为了执行这个矩阵乘法，我们需要将这两个数组从 HBM 加载到 MXU 中，进行乘法运算，然后将结果写回 HBM。和之前一样，我们有：

$$T_\text{math} = \frac{\text{Computation FLOPs}}{\text{Accelerator FLOPs/s}} = \frac{2BDF}{\text{Accelerator FLOPs/s}}$$

$$T_\text{comms} = \frac{\text{Communication Bytes}}{\text{Bandwidth Bytes/s}} = \frac{2BD + 2FD + 2BF}{\text{Bandwidth Bytes/s}}$$

一个 TPU 或 GPU 可以在加载的同时进行计算，从而重叠这些操作，因此要成为算力受限，我们需要 $$T_\text{math} \geq T_\text{comms}$$，或者：

$$\frac{2BDF}{2BD + 2DF + 2BF} \geq \frac{\text{Accelerator FLOPs/s}}{\text{Bandwidth Bytes/s}} \underset{\text{TPU v5e}}{=} \frac{1.97E+14}{8.20E+11} = 240$$

其中 RHS 是我们硬件的算术强度。现在假设 $D$ 和 $F$ 相比 $B$ 非常大（通常我们的批量最多为 500，而 $D$ 和 $F > 10k$），我们可以利用 $\small{2BD + 2DF + 2BF \approx 2DF}$ 这一事实来简化分母，从而得到

$$\begin{align*}
\frac{2BDF}{2BD + 2DF + 2BF} \approx \frac{2BDF}{2DF} \geq \frac{\text{Accelerator FLOPs/s}}{\text{Bandwidth Bytes/s}} \\
\underset{\text{TPU v5e}}{=} \frac{1.97E+14}{8.20E+11} \implies B \geq 240 = B_{\text{crit}}
\end{align*}$$

如果我们对权重进行量化或使用较低精度的 FLOPs 进行矩阵乘法，这个关键的批量大小会发生变化。例如，如果我们把权重量化为 int8 或 fp8，$B_\text{crit}$ 会减少 2 倍。如果我们用 int8 或 fp8 进行 FLOPs 计算，$B_\text{crit}$ 会增加 2 倍。因此，如果我们让 $\beta = \text{bits per param} / \text{bits per activation}$ 和 $\alpha_\text{hbm} = C / W_\text{hbm}$，我们的关键批量大小实际上是 $B_\text{crit} = \beta \alpha_\text{hbm}$。

<p markdown=1 class="takeaway">**要点：** Transformer 的矩阵乘法在计算上受限制 *iff* 每个副本的 **词元** 批量大小大于 $B_\text{crit} = C / W_\text{hbm} \cdot (\text{bits per param} / \text{bits per activation}) = \beta \cdot \alpha_\text{hbm}$。在 TPU v5e 上使用 bf16 激活值时，这个值是 240 个词元。在 H100 上，这个值约为 280 个词元。</p>

在训练过程中，由于我们在非常大的批量中重复使用相同的权重，所有矩阵乘法都将具有很高的计算强度。**这种高的算术强度会延续到预填充阶段，因为用户提示通常有数百甚至上千个词元长。** 正如我们之前看到的，TPUv5e 的硬件算术强度为 240，因此如果将长度超过 240 个词元的序列输入到运行在该硬件上的 bf16 精度的密集模型中，我们预计会受到算力限制，一切正常。理论上，较短的提示可以一起批量处理以提高利用率，但通常并不需要这样做。

<p markdown=1 class="takeaway">**要点：** 在预填充阶段，所有的矩阵乘法基本上始终是算力受限的。因此，只需最大化硬件利用率或模型 FLOPs 利用率（MFU）即可最大化每芯片（成本）的吞吐量和延迟（以 TTFT 的形式）。除非提示极其简短，否则在每个提示级别进行批处理只会略微提高预填充吞吐量，却会增加延迟。</p>

然而，在生成过程中，对于每个请求，我们必须一次只处理一个词元，因为步骤之间存在顺序依赖！因此，我们只能通过将多个请求批量处理，沿批量维度进行并行化，才能（轻松地）实现较高的利用率。我们之后会更详细地讨论这一点，但实际上，在不增加延迟的情况下将大量并发请求批量处理是非常困难的。因此，**通过生成来充分利用硬件 FLOPs 要困难得多。**

<p markdown=1 class="takeaway">**要点：** 在生成过程中，总词元批量大小必须大于 $B_{\text{crit}}$，才能在线性/前馈操作上达到算力限制（TPU v5e 上 bf16 参数为 240）。由于生成是按词元逐个进行的，这要求我们必须将多个请求合并成一批处理，这非常困难！</p>

*值得注意的是，这个规模非常大！* 生成批量大小为 240 意味着同时有 240 个并发请求进行生成，且对于密集模型需要维护 240 个独立的 KV 缓存。这意味着在实践中很难实现，除非在某些批量推理设置中。相比之下，在预填充过程中推送超过 240 个词元是非常常规的操作，尽管随着稀疏性增加，需要一定的注意。

**请注意，这个确切数值会因量化方式和硬件的不同而有所变化。** 加速器通常可以在较低精度下提供更多的算术运算能力。例如，如果我们使用 int8 参数但用 bf16 进行计算，关键批量大小会降至 120。使用 int8 激活值和 int8 参数时，由于 TPUv5e 可以提供 400 TOPs/s 的 int8 x int8 运算能力，关键批量大小会回升至 240。

### 关于注意力呢？

当我们审视点积注意力操作时，事情会变得更加复杂，尤其是因为我们必须考虑 KV 缓存。让我们仅看一个具有纯多头注意力的注意力头。在一次 Flash Attention 融合中，我们<d-footnote>我们在这里通过忽略应用 softmax、掩码等操作中的非矩阵乘法 FLOPs，大大简化了问题。这些操作应该与计算或 HBM 读取重叠，但在某些 TPU 世代上实现这一点可能并不简单。虽然这些细节不会改变主要信息，即 KV 缓存通常受内存限制，但它们值得关注。</d-footnote>:

1. 从 HBM 读取形状为 $\text{bf16[B, T, D]}$ 的 $Q$ 激活值。  
2. 从 HBM 读取 $KV$ 缓存，该缓存是一对 $\text{bf16[B, S, D]}$ 张量。  
3. 在 $$QK$$ 矩阵乘法中执行 $2BSTD$ FLOPs。使用 Flash Attention 时，无需将 $\text{bf16[B, S, T]}$ 注意力矩阵写回 HBM。  
4. 在注意力 $$AV$$ 矩阵乘法中执行 $2BSTD$。  
5. 将生成的 $\text{bf16[B, T, D]}$ 张量写回 HBM。

综合以上内容，我们得到：

$$\text{Multiheaded Attention Arithmetic Intensity} = \frac{4BSTD}{4BSD + 4BTD} = \frac{ST}{S+T}$$

对于预填充，$S=T$ 因为我们正在执行自注意力，因此这简化为 $T^2 / 2T = T / 2$。这很好，因为它意味着 **预填充期间注意力的算术强度是 $\Theta(T)$**。这意味着注意力很容易达到算力限制。只要我们的序列长度足够大，我们就没问题！

但由于生成操作具有平凡的序列维度，且 $B$ 和 $D$ 维度相互抵消，我们可以进行如下近似：

$$S \gg T = 1 \implies \frac{ST}{S+T} \approx 1$$

这很糟糕，因为这意味着我们无法在生成过程中采取任何措施来提高注意力的算术强度。我们在加载大量 KV 缓存时只执行了极少的 FLOPs。**因此，在注意力计算过程中，我们基本上始终受到显存带宽的限制！**

<p markdown=1 class="takeaway">**要点：** 在预填充阶段，对于任何合理的序列长度（大约 $\gt 480$ 个词元），注意力机制通常受算力限制，而在生成阶段，我们的算术强度较低且恒定，因此我们始终受显存带宽限制。</p>

*从概念上讲，为什么会这样？* 主要是因为在模型的线性部分，我们受到算力的限制，因为参数（占用大量显存带宽的组件）会被多个批量项重复使用。然而，每个批量项都有自己的 KV 缓存，因此更大的批量大小意味着更多的 KV 缓存。除非架构被积极调整，否则我们在这里几乎*总是*会受到内存的限制。

这也意味着，一旦参数内存与 KV 缓存内存变得相当，增大批量大小对吞吐量的提升效果将逐渐减弱。这种逐渐减弱的效果对你造成的影响程度，取决于单个序列中参数与 KV 缓存字节的比例，即大致为比例 $2DF / SHK$。自 $HK\approx D$ 以来，这一比例大致取决于 $F$ 与 $S$（序列长度）的比例。这还取决于使 KV 缓存更小的架构修改（我们稍后会进一步说明）。

### 大语言模型延迟和吞吐量的理论估计

从这个数学推导中，我们可以在优化时得到对应该采取的步时的良好界限。**(注意：如果本章有哪一点希望读者记住，那就是以下内容。)** 在生成过程中（这很常见），对于小批量大小，我们可以通过假设在注意力模块和MLP模块中都受显存带宽限制，来对每步延迟进行下界估计：

$$\begin{equation*}
\text{Theoretical Min Step Time} = \frac{\text{Batch Size} \times \text{KV Cache Size} + \text{Parameter Size}}{\text{Total Memory Bandwidth}}
\end{equation*}$$

同样地，对于吞吐量：

$$\begin{equation*}
\text{Theoretical Max Tokens/s} = \frac{\text{Batch Size} \times \text{Total Memory Bandwidth}}{\text{Batch Size} \times \text{KV Cache Size} + \text{Parameter Size}}
\end{equation*}$$

最终，随着批量大小的增加，FLOPs 开始主导参数加载，因此在实践中我们有更一般的公式：

$$\begin{align}
\tiny \text{Theoretical Step Time (General)} = \underbrace{\frac{\text{Batch Size} \times \text{KV Cache Size}}{\tiny \text{Total Memory Bandwidth}}}_{\text{Attention (always bandwidth-bound)}} + \underbrace{\max\left(\frac{2 \times \text{Batch Size} \times \text{Parameter Count}}{\text{Total FLOPs/s}}, \frac{\text{Parameter Size}}{\text{Total Memory Bandwidth}}\right)}_{\tiny \text{MLP (can be compute-bound)}}
\end{align}$$

其中注意力组件（左）从不受到算力限制，因此不需要 FLOPs 屋顶线。这些对于粗略估算非常有用，例如

<b markdown=1 style="color: #57cf57;">小测验：</b> 假设我们想使用一个 30B 参数的密集模型，在 TPU v5e 4x4 切片上以 int8 精度和 bf16 FLOPs 运行，批大小为 4 个词元，上下文长度为 8192，每个词元的 KV 缓存为 100 kB，进行一次生成步骤。这次操作的延迟合理下限是多少？如果我们想生成一个包含 256 个词元的批次，延迟又会是多少？

{% details 点击此处查看答案。 %}

**答案：** 在 int8 精度下，我们的参数将使用 30e9 字节，根据给定的规格，我们的 KV 缓存将每个使用 `100e3 * 8192 = 819MB`。我们有 16 个芯片，每个芯片具有 `8.2e11` 字节/秒的带宽和 `1.97e14` bf16 FLOPs/秒。根据上述公式，由于我们使用了较小的批量大小，我们预计每步时间至少为 `(4 * 819e6 + 30e9) / (16 * 8.2e11) = 2.5 ms`。在 256 个词元的情况下，我们的 MLP 模块将进入计算受限区间，因此每步时间约为 `(256 * 819e6) / (16 * 8.2e11) + (2 * 256 * 30e9) / (16 * 1.97e14) = 21ms`。

{% enddetails %}

如你所见，这里存在吞吐量和延迟之间的明显权衡。小批量处理速度快，但无法充分利用硬件。大批量处理速度慢，但效率高。这是为一些较早的 PaLM 模型计算出的延迟-吞吐量帕累托前沿（来自论文[ESTI](https://arxiv.org/pdf/2211.05102)<d-cite key="esti"></d-cite>）：

{% include figure.liquid path="assets/img/latency-cost.png" class="img-fluid" caption="<b>Figure:</b> 几个 PaLM 模型的成本（读作：吞吐量）与延迟的帕累托前沿。请注意芯片数量（C）和批量大小（B）如何沿帕累托前沿移动，除了绿色点（PaLM 540B：C:32 B:16）外，该点由于可用内存限制了设置支持良好的批量大小，导致吞吐量下降。请注意，批量大小达到 240 后，吞吐量通常趋于平稳。int8 权重提供了更好的延迟-吞吐量帕累托最优，但并未提供更好的最大吞吐量。" %}

我们不仅通过批量大小作为旋钮来权衡延迟和吞吐量，还可能更倾向于选择较大的拓扑结构而非较小的拓扑结构，以便在受到HBM限制时容纳更大的批量。[下一节](../applied-inference)将对此进行更详细的探讨。

<p markdown=1 class="takeaway">**要点：** 如果你关注生成吞吐量，请使用芯片上尽可能大的批量大小。任何超过 TPU 算术强度（$B_\text{crit}$，通常为 120 或 240）的芯片级批量大小都将最大化吞吐量。你可能需要增加拓扑结构以实现这一点。较小的批量大小可以在牺牲吞吐量的情况下改善延迟。</p>

{% details 从硬件角度来看，这一点有一些需要注意的地方。点击此处查看一些细节。 %}

这在理论上听起来很完美。但在实践中，我们往往由于以下几个原因，并没有看到如此明显的性能上限：

* 我们假设 HBM 读取将与 FLOPs 完全重叠是不现实的，因为我们的编译器（XLA）并非完美无误。  
* 对于分片模型，XLA 通常也无法高效地将模型分片矩阵乘法的 ICI 通信与 FLOPs 本身重叠，因此我们通常会在 $$\text{BS}=32$$ 上的线性层开始出现延迟损失。  
* 由于重叠不完美，即使批量大小超过理论上限，吞吐量仍会有所提升，但这只是一个良好的启发式方法。

{% enddetails %}

### 内存如何呢？

我们花了一些时间研究带宽和 FLOPs，但还没有研究内存。由于我们新的数据结构 KV 缓存，推理期的内存情况看起来大不相同。为了本节内容，让我们选择一个实际的模型（LLaMA 2-13B）来演示不同之处：

|超参数|价值|
| ------------------ | ------ ||L（num_layers）|40|
|D（d_model）|5,120|
|F (ffw_dimension)|13,824|
|N（num_heads）|40|
|K（num_kv_heads）|40|
|H (qkv_dim)|128|
|V (num_embeddings)|32,000|

推理期间占用内存的是什么？很明显，是我们的参数。统计这些，我们有：

|参数|公式|大小（以字节为单位）|
| ---------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- ||前馈网络参数|d_model<sup>2</sup> x ffw_multiplier x 3（用于 SwiGLU 门控、上投影和下投影）x n_layers|5,120 x 5,120 x 2.7 x 3 x 40 = **8.5e9**|
|词元参数|2（输入和输出嵌入）x n_embeddings x d_model|2 x 32,000 x 5,120 = **0.3e9**|
|注意力参数|[2（*q 和 output*）x d_model x n_heads x d_qkv + 2（*for k and v*）x d_model x n_kv_heads x d_qkv] x n_layers|(2 × 5,120 × 40 × 128 + 2 × 5,120 × 40 × 128) × 40 = **4.2e9**|

将这些参数相加，我们得到 8.5e9 + 4.2e9 + 0.3e9 = **13e9 总参数**，正如预期的那样。正如我们在前面几节中看到的，在训练过程中，我们可能会将参数存储为 bfloat16，并将优化器状态存储为 float32。这可能占用大约 100GB 的内存。与我们的梯度检查点相比，这显得微不足道，梯度检查点可能会使用数 TB 的内存。

**推理有什么不同？** 在推理过程中，我们只存储一份参数副本，比如使用 bfloat16。这需要 26GB 内存 —— 而且在实践中，通过量化我们通常可以做得比这更好。不需要跟踪优化器状态或梯度。因为我们不进行检查点（不需要为反向传播保留激活值），所以预填充<d-footnote>期间的激活值占用的内存可以忽略不计，特别是得益于 Flash Attention，它避免了显式生成注意力矩阵</d-footnote>和生成过程。如果我们预填充 8k 个词元，单个激活值仅占用大约 `8,192 x 5,120 x 2 bytes = 80MB` 的内存。更长的预填充可以分解为许多更小的前向传递，因此对于更长的上下文也不会成为问题。生成过程使用的词元甚至更少，因此激活值可以忽略不计。

**主要区别在于 KV 缓存**。这些是所有过去词元的键和值投影，其大小仅受允许的最大序列长度限制。$$T$$ 词元的总大小是

$$\text{KV cache size} = 2 \cdot \text{bytes per float} \cdot H \cdot K \cdot L \cdot T$$

其中 $$H$$ 是每个注意力头的维度，$$K$$ 是 KV 头的数量，$$L$$ 是层数，而 2 来源于同时存储键和值。

**这可能会迅速变得很大**，即使使用适度的批量大小和上下文长度。对于 LLaMA-13B，单个 8192 序列在 bf16 下的 KV 缓存是

$$8192\ (T) \times 40\ (K) \times 128\ (H) \times 40\ (L) \times 2\ (\text{bytes}) \times 2 = 6.7 \text{GB}$$

**这四个的内存使用量超过了我们的参数！** 为了明确起见，LLaMA 2 并未针对更长上下文的 KV 缓存大小进行优化（通常 $K$ 要小得多，如在 LLaMA-3 中所示，因此情况并不总是如此糟糕），但这一点仍然具有说明性。我们在内存或延迟估计中不能忽视这些因素。

### 对 LLaMA 2-13B 的吞吐量和延迟进行建模

让我们看看在 8xTPU v5es 上，以不同批量大小尝试以最高效率进行生成时会发生什么，直到达到之前推导出的最大理论吞吐量对应的临界批量大小（240）。

|批量大小|1|8|16|32|64|240|
| :-------------------------------- | -----: | -----: | -----: | -----: | -----: | -----: ||KV 缓存内存（GiB）|6.7|53.6|107.2|214.4|428.8|1608|
|总内存 (GiB)|32.7|79.6|133.2|240.4|454.8|1634|
|理论步时（ms）|4.98|12.13|20.30|36.65|69.33|249.09|
|理论吞吐量（词元/秒）|200.61|659.30|787.99|873.21|923.13|963.53|

8 个 TPU v5e 提供了 128GiB 的 HBM、6.5TiB/s 的 HBM 带宽（每个 0.82TiB/s）和 1600TF/s 的算力。

对于此模型，增加批量大小确实能提高吞吐量，但我们很快就会遇到收益递减的情况。在批量大小超过 16 后，我们会遇到内存不足的问题，需要多出一个数量级的内存才能接近 240。更大的拓扑结构可以改善延迟，但我们在每块芯片的吞吐量上已经遇到了瓶颈。

假设我们保持参数总数不变，但神奇地将 KV 缓存缩小到原来的 1/5（例如，使用 1:5 [GMQA](#tricks-for-improving-generation-throughput-and-latency)，这意味着我们在 40 个 Q 头上共享 8 个 KV 头 —— 更多细节请参见下一节）。

|批量大小|1|8|16|32|64|240|
| :-------------------------------- | -----: | -------: | -------: | -------: | -------: | -------: ||KV 缓存内存（GiB）|1.34|10.72|21.44|42.88|85.76|321.6|
|总内存 (GiB)|27.34|36.72|47.44|68.88|111.76|347.6|
|理论步时（ms）|4.17|5.60|7.23|10.50|17.04|52.99|
|理论吞吐量（词元/秒）|239.94|1,429.19|2,212.48|3,047.62|3,756.62|4,529.34|

KV 缓存更小时，我们仍然存在边际效益递减的现象，但每块芯片的理论吞吐量仍可继续扩展到批量大小 240。我们可以容纳更大的批量，达到 64，且在所有批量大小下延迟也始终表现更好。延迟、最大吞吐量和最大批量大小都显著提升！事实上，后续的 LLaMA 版本也使用了这一确切优化 —— LLaMA-3 8B 有 32 个查询头和 8 个 KV 头（[来源](https://huggingface.co/MaziyarPanahi/Llama-3-13B-Instruct-v0.1/blob/dfdeb40bdb2c149dfa399ea2be0d56eb120f0831/config.json)）。

<p markdown=1 class="takeaway">**要点：** 除了参数数量外，KV 缓存的大小对模型最终的推理性能有很大影响。我们希望通过架构决策和运行时优化来控制它。</p>

## 提高生成吞吐量和延迟的技巧

自原始论文[Attention is All You Need](https://arxiv.org/abs/1706.03762)以来，许多技术已被开发出来，以提高模型的效率，通常专门针对KV缓存。一般来说，较小的KV缓存使得在不损害延迟的情况下更容易增加生成步骤的批量大小和上下文长度，并使围绕Transformer的系统（如请求缓存）更加便捷。忽略对质量的影响，我们可能会看到：

**分组多查询注意力（又称 GMQA、GQA）：** 我们可以减少键值（KV）头的数量，并在注意力机制中将它们与多个查询（Q）头共享。在极端情况下，可以将单个 KV 头与所有 Q 头共享。这使得与纯多头注意力（MHA）相比，键值缓存减少了 Q:KV 比例的倍数，并且观察到模型的性能对此变化相对不敏感。

{% include figure.liquid path="assets/img/gmqa.png" class="img-fluid" %}

这也有效提高了注意力计算的算术强度（参见[第4节](../transformers)中的问题4）。

**混合一些局部注意力层：** 局部注意力将上下文限制为较小到中等长度的最大长度。在训练时间和预填充时间，这涉及将注意力矩阵掩码为对角条带，而不是三角形。这有效地限制了局部层的 KV 缓存的最大长度的大小。通过将一些局部层与一些全局层混合到模型中，在上下文长度超过局部窗口时，KV 缓存的大小会大大减少。

**跨层共享 KVs：** 模型可以学习以某种模式在各层之间共享相同的 KV 缓存。虽然这会减少 KV 缓存的大小，并在增加批量大小、缓存、离线存储等方面带来好处，但共享的 KV 缓存可能需要多次从 HBM 读取，*因此并不一定改善步时。*

{% include figure.liquid path="assets/img/kv-sharing.png" class="img-fluid" caption="<b>左：</b> 多层纯全局注意力。 <b>右：</b> 一个全局/局部交错模式的示例，与相邻层共享。来源：<a href=\"https://research.character.ai/optimizing-inference/?ref=blog.character.ai\">Character.ai 博客</a>。"%}

**量化：** 推理通常对参数和 KVs 的精度不太敏感。通过将参数和 KV 缓存量化（例如量化为 int8、int4、`fp8` 等），我们可以在两者上节省显存带宽，减少达到算力前沿所需的批量大小，并节省内存以运行更大的批量大小。量化还有一个额外的优势，即使模型没有使用量化进行训练，通常也可以在训练后应用量化。

**使用不规则 HBM 读取和分页注意力：** 在上述计算中，我们为每个 KV 缓存分配了 8k 的上下文，但通常没有必要从内存中读取整个 KV 缓存 —— 请求的长度分布范围很广，且不会使用模型的最大上下文，因此我们通常可以实现仅读取 KV 缓存非填充部分的内核（例如 Flash Attention 的变体）。

Paged Attention<d-cite key="paged"></d-cite> 是对此的改进，它将 KV 缓存存储在类似操作系统风格的页表中，并基本避免对 KV 缓存进行填充。这增加了许多复杂性，但意味着每个批次仅使用其所需的内存。这是一种运行时优化，因此同样与架构无关。

{% include figure.liquid path="assets/img/paged-attention.png" class="img-fluid img-small" caption="<b>图：</b>在生成过程中，单个词元（"forth"）会关注多个 KV 缓存块/页。通过分页 KV 缓存，我们避免加载或存储超出需要的内存。摘自 <a href=\"https://arxiv.org/pdf/2309.06180\">PagedAttention 论文</a>。" %}

<p markdown=1 class="takeaway">**宏观视角：** 总体而言，这些 KV 缓存优化措施可以使 KV 缓存的大小相比标准 MHA Transformer 减少一个数量级以上。这可以带来 Transformer 整体成本一个数量级的改善。</p>

## 在多个加速器上分布推理

到目前为止，我们只是粗略地说明了如何在单个芯片之外进行扩展。遵循[第5节](../training)，让我们探讨可用的不同策略及其权衡。一如既往，我们将分别查看预填充和生成。

### 预填充

从屋顶线的角度来看，**预填充几乎等同于训练**，几乎所有相同的技巧和权衡都适用——模型（Megatron）并行、序列分片（针对足够长的上下文）、流水线，甚至FSDP都是可行的！你只需要确保KVs持续存在，以便之后进行生成。与训练一样，增加芯片数量可以让我们获得更多的FLOPs/s（可能降低TTFT），但也会增加通信开销（可能降低每块芯片的吞吐量）。

**预填充分片的一般规则：** 这里是一组预填充的一般规则。我们假设只对单个序列进行预填充（没有批量维度）：

1. *模型分片：* 我们通常首先进行一定程度的模型并行，直到达到ICI限制。如[第5节](../training)所示，对于一个轴（通常为4到8路分片），这大约在$F / 2200$处。
2. *序列并行：* 超过这个阶段后，我们进行序列并行（类似于数据并行，但是在序列维度上进行分片）。虽然序列并行在注意力机制中引入了一些额外的通信，但在较长的上下文中，这种通信通常非常小。与训练类似，我们可以将通信和计算重叠（Megatron使用集合矩阵乘法，环形注意力分别使用相应的方法）。

<p markdown=1 class="takeaway">**要点：** 在预填充阶段，几乎所有在训练期间可以工作的分片方式都可以正常工作。先进行到 ICI 限制的模型并行，然后再进行序列并行。</p>

### 生成

生成比预填充更复杂。一方面，由于需要将许多请求合并为一个批量，因此更难获得较大的批量大小。延迟目标更低。综上所述，这意味着我们通常更受内存限制，并且对通信开销更敏感，这限制了我们的分片策略：

1. **FSDP 是不可能的：** 由于我们在从 HBM 将参数和 KV 缓存加载到 MXU 时受到内存限制，我们不希望通过 ICI 移动它们，因为 ICI 比 HBM 慢好几个数量级。*我们希望移动激活值而不是权重。* 这意味着类似于 FSDP 的方法通常对于生成任务来说完全不可行。<d-footnote>训练后不小心将其开启是一种常见且容易导致性能下降几个数量级的方式。</d-footnote>

2. **没有必要进行数据并行：** 纯数据并行没有帮助，因为它会复制我们的参数，而无法帮助我们更快地加载参数。你还不如启动多个模型副本。<d-footnote>我们的意思是，启动多个服务器，每个服务器运行模型的副本，但使用较小的批量大小。模型级别的数据并行严格来说更差。</d-footnote>

3. **没有序列 = 没有序列分片。** 祝你好运，序列分片。

这使我们主要剩下密集模型生成的模型分片变体。与预填充类似，我们能做的最简单的事情是简单的模型并行（激活值完全复制，MLP 的权重在隐藏维度上完全分片），在受到 ICI 限制时最多可达到 4-8 种方式。然而，由于我们经常受到显存带宽的限制，实际上我们可以超越这一限制以提高延迟！

**关于生成阶段 ICI 限制的说明：** 在训练期间，我们希望处于算力限制状态，因此我们的性能上限关注的是 ICI 通信时间超过 FLOPs 时间的情况。然而，在生成阶段，如果由于参数加载而受到显存带宽的限制，我们可以在这一点之后增加模型分片，以最小的吞吐量代价（以 tokens/sec/chip 为单位）来改善延迟。更多的模型分片可以让我们利用更多的 HBM 来加载权重，而 FLOPs 的影响则变得无关紧要。<d-footnote> 从这个意义上说，FLOPs 时间并不是我们的瓶颈，我们需要关注的是 ICI 时间超过参数加载时间。</d-footnote> 让我们看看在模型并行达到瓶颈之前，我们能进行多少并行。

$$\begin{align*}T_\text{HBM comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{ICI comms} = \frac{2BD}{W_\text{ici}}\end{align*}$$

$$T_\text{ICI comms} > T_\text{HBM comms} \rightarrow \frac{W_\text{hbm}}{W_\text{ici}} > \frac{F}{Y \cdot B} \rightarrow Y > F / (B \cdot \beta)$$

其中 $\beta = W_\text{hbm} / W_\text{ici}$。这个数值对于 TPU v5e 和 TPU v6e 通常约为 8。这意味着例如，如果 $F$ 是 16,384，而 $B$ 是 32，理论上我们可以将模型并行化到 `16384 / (32 * 8) = 64` 种方式，而不会对吞吐量造成显著影响。这假设我们可以将 KV 缓存完全分片为 64 种方式，这具有挑战性：我们将在下文讨论这一点。

对于注意力层，我们还按照 Megatron 风格对头进行分片，建模 shard attention $$W_Q$$ 和 $$W_O$$。KV 权重非常小，复制它们通常比超过 $K$-way 分片的分片更便宜。

<p markdown=1 class="takeaway">**要点：** 在生成过程中，我们唯一的选择是模型并行的各种变体。我们旨在移动激活值，而不是更大规模的 KV 缓存或参数。当我们的批量大小较大时，我们会将模型并行扩展到 FLOPs-ICI 限制（$F / \alpha$）。当我们的批量大小较小时，我们可以通过更细致的模型分片来提高延迟（以适度的吞吐量代价为代价）。当我们希望以比 KV 头数量更多的方式对模型进行分片时，我们也可以沿着批量维度对 KV 进行分片。</p>

### 对 KV 缓存进行分片

**我们还有一个需要进行分片的额外数据结构——KV 缓存。** 同样，我们几乎总是倾向于避免复制缓存，因为它是一般注意力延迟的主要来源。为此，我们首先沿注意力头维度对 KVs 进行 Megatron 分片。这种方式仅限于 $K$ 路分片，因此对于头数量较少的模型，我们尽可能多地对头维度进行分片，然后沿批量维度进行分片，即 $\text{KV}[2, B_Z, S, K_Y, H]$。这意味着 KV 缓存是完全分布式的。

{% include figure.liquid path="assets/img/esta-figure.png" class="img-fluid" caption="<b>Figure:</b> 注意力机制与（a）纯模型分片的多头注意力和（b）KV缓存批量分片的多查询注意力的对比。请注意，我们需要两个额外的AllToAll操作，将激活值从模型分片转移到批量分片，以便它们能够作用于KV缓存。" %}

这样做的成本是每个注意力层需要两次 AllToAll 操作 —— 一次将 Q 激活值转移到批量分片中以便我们使用批量分片计算注意力，另一次将批量分片的注意力输出转移回纯模型分片。

{% details 这就是完整的算法！ %}

在这里，我们将写出完整的注意力算法，该算法在 $Y$ 和 $Z$ 上均实现了模型并行。对于同时使用 $K$ 作为键张量和 KV 头维度，我表示歉意。让 $M=N/K$。

<div markdown=1 class="algorithm">

1. X[B, D] = ...（现有激活值，未分片，来自前一层）  
2. K[B<sub>Z</sub>, S, K<sub>Y</sub>, H], V[B<sub>Z</sub>, S, K<sub>Y</sub>, H] = ...（现有 KV 缓存，按批次分片）  
3. Q[B, N<sub>YZ</sub>, H] = X[B, D] \* W<sub>Q</sub>[D, N<sub>YZ</sub>, H]  
4. Q[B<sub>Z</sub>, N<sub>Y</sub>, H] = **AllToAll**<sub>Z->B</sub>(Q[B, N<sub>YZ</sub>, H])  
5. Q[B<sub>Z</sub>, K<sub>Y</sub>, M, H] = **Reshape**(Q[B<sub>Z</sub>, N<sub>Y</sub>, H])  
6. O[B<sub>Z</sub>, S, K<sub>Y</sub>, M] = Q[B<sub>Z</sub>, K<sub>Y</sub>, M, H] \*<sub>H</sub> K[B<sub>Z</sub>, S, K<sub>Y</sub>, H]  
7. O[B<sub>Z</sub>, S, K<sub>Y</sub>, M] = **Softmax**<sub>S</sub>(O[B<sub>Z</sub>, S, K<sub>Y</sub>, M])  
8. O[B<sub>Z</sub>, K<sub>Y</sub>, M, H] = O[B<sub>Z</sub>, S, K<sub>Y</sub>, M] \*<sub>S</sub> V[B<sub>Z</sub>, S, K<sub>Y</sub>, H]  
9. O[B, K<sub>Y</sub>, M<sub>Z</sub>, H] = **AllToAll**<sub>Z->M</sub>(O[B<sub>Z</sub>, K<sub>Y</sub>, M, H])  
10. O[B, N<sub>YZ</sub>, H] = **Reshape**(O[B, K<sub>Y</sub>, M<sub>Z</sub>, H])  
11. X[B, D] {U<sub>YZ</sub>} = W<sub>O</sub>[N<sub>YZ</sub>, H, D] \*<sub>N,H</sub> O[B, N<sub>YZ</sub>, H]  
12. X[B, D] = **AllReduce**(X[B, D] { U<sub>YZ</sub>})

这相当复杂，但你可以大致看到它是如何运作的。新的通信方式由于基于我们较小的激活值，因此成本相对较高，但作为回报，我们在加载 KV（它们是静态的）时节省了大量显存带宽。

</div>

{% enddetails %}

* **序列分片：** 如果批量大小过小，或上下文过长，我们可以对 KV 缓存进行序列分片。同样地，这里在跨分片累积注意力时需要支付一次集合通信的代价。首先我们需要 AllGather Q 激活值，然后以类似 Flash Attention 的方式累积 KVs。

## 设计一个高效的推理引擎

到目前为止，我们已经探讨了如何在隔离的情况下高效地优化和分片单个预填充和生成操作。要实际有效地使用它们，我们需要设计一个推理引擎，能够在延迟/吞吐量的帕累托前沿上选择性地将这两个操作馈入。

最简单的方法是先运行一批预填充，然后运行一批生成：

{% include figure.liquid path="assets/img/batched-prefill.png" class="img-fluid" caption="<b>图：</b>在最简单的设置中，请求会被聚合，服务器在运行一批预填充和调用生成函数之间交替进行，直到所有序列完成。" %}

这易于实现，也是大多数代码库中的第一个推理设置，但它存在多个缺点：

1. **延迟非常糟糕。** 我们将预填充和生成的批量大小耦合在一起。在较大的预填充批量大小下，首次生成 token 的时间（TTFT）非常糟糕 —— 必须完成所有预填充操作后，用户才能看到任何 token。在较小的批量大小下，生成吞吐量也非常糟糕。  
2. **较短的生成会被较长的生成阻塞。** 许多序列会在其他序列之前完成，导致生成过程中出现空的批量槽位，进一步降低生成吞吐量。随着批量大小和生成长度的增加，这个问题会更加严重。  
3. **预填充进行了填充。** 预填充会被填充到最长序列的长度，导致大量算力浪费。虽然有一些解决方案，但历史上 XLA 使得跳过这些 FLOPs 非常困难。同样，随着批量大小和预填充序列长度的增加，这个问题会变得更加严重。  
4. **我们被迫在预填充和生成之间共享分片。** 预填充和生成都位于同一块切片上，这意味着两者使用相同的拓扑结构和分片方式（除非你保留两份权重副本），这通常对性能没有帮助，例如生成过程需要更多的模型分片。

因此，此方法仅推荐用于边缘应用（通常只关注为单个用户提供服务并使用每字节 FLOPs 较少的硬件）以及 Transformer 代码库生命周期早期的快速迭代（由于其简单性）。

一种稍好的方法是在批量大小为 1 的情况下执行预填充（此时计算受限但延迟合理），但在生成期间将多个请求合并成批量处理：

{% include figure.liquid path="assets/img/interleaving.png" class="img-fluid" %}

这将避免在批量预填充时浪费 TTFT，同时保持生成吞吐量较高。我们将这种配置称为 **交错** 配置，因为我们“交错”了预填充和生成步骤。这对于以吞吐量为主要目标的批量生成应用（如评估）来说非常强大。调度器可以配置为在任何生成槽位一打开就优先进行预填充，从而即使在非常大的生成批量大小下也能实现高利用率。我们还可以避免将预填充填充到最大长度，因为它不会与其他请求一起批量处理。

主要缺点是当服务器执行预填充时，所有其他请求的生成都会暂停，因为所有计算资源都将被预填充消耗。用户 A 的响应正在忙于解码，会被用户 B 的预填充所阻塞。这意味着尽管 TTFT 得到了改善，但平均而言，词元生成会变得抖动且缓慢，这对许多应用来说不是一个良好的用户体验——其他用户的预填充位于请求整体延迟的关键路径上。

为了解决这个问题，我们将解码和预填充分开处理。虽然 Transformer 推理可以在一台服务器上完成，但从延迟的角度来看，通常将这两个不同的任务在两组 TPUs/GPUs 上执行会更好。预填充服务器生成 KV 缓存，并通过网络将这些缓存发送到生成服务器，生成服务器将多个缓存合并成批，并为每个缓存生成词元。我们将这种 **"disaggregated"** 服务方式称为“分散式”服务。

{% include figure.liquid path="assets/img/disaggregation.png" class="img-fluid" %}

这提供了几个优势：

1. **大规模下的低延迟**：用户的请求不会因其他用户的请求而被阻塞，除非预填充容量不足。请求应立即进行预填充，然后发送到生成服务器，并立即分配到生成缓冲区。如果我们预计会有大量并发请求，可以独立扩展预填充服务器的数量，与生成服务器的数量无关，以避免用户长时间停留在预填充队列中。

2. **专业化：** 很多时候，预填充和生成阶段的延迟最优参数分片策略/硬件拓扑是截然不同的（例如，生成阶段更需要模型并行，而预填充阶段则不需要）。将这两个操作限制为使用相同的分片策略会损害两者的性能，同时维护两组权重会占用内存。此外，将预填充任务转移到独立的服务器上，它无需保留除当前处理之外的任何 KV 缓存。这意味着我们可以释放更多内存用于历史缓存（见下一节）或优化预填充延迟。

一个缺点是现在需要在网络中对 KV 缓存进行移位。这通常是可接受的，但再次为减小 KV 缓存规模提供了动机。

<p markdown=1 class="takeaway">**要点：** 对于延迟敏感、高吞吐量的部署服务，我们通常需要将预填充和生成分离到不同的服务器上，预填充在批量大小为 1 的情况下运行，而生成则将许多并发请求合并成批量一起处理。</p>

### 连续批处理

上述问题（2）促使我们提出了**连续批处理**的概念。我们对以下内容进行优化和编译：

* 一个预填充函数，能够处理可变上下文长度，并将结果插入到具有最大批量大小和上下文长度/页数的 KV 缓冲区中。  
* 一个生成函数，接收 KV 缓存，并为所有当前活跃的请求执行生成步骤。

我们随后使用一个协调器将这些功能组合在一起，该协调器对传入的请求进行排队，根据可用的生成槽位调用预填充和生成，处理历史缓存（见下一部分），并逐个输出词元。

{% include figure.liquid path="assets/img/continuous-batching.gif" class="img-fluid" %}

### 前缀缓存

由于预填充成本高昂且受算力限制（给我们留出的余地更少），减少其成本的最佳方法之一是减少预填充的次数。由于大语言模型是自回归的，查询 ["I", "like", "dogs"] 和 ["I", "like", "cats"] 在前两个词元上生成的 KV 缓存是相同的。这意味着，原则上，如果我们先计算 "I like dogs" 的缓存，然后再计算 "I like cats" 的缓存，只需要进行 1/3 的计算。通过重用缓存，我们可以节省大部分工作。这在某些特定情况下尤其强大：

1. **聊天机器人**：大多数聊天机器人的对话涉及一个来回的对话，严格地追加到自身。这意味着如果我们能保存每次对话轮次的 KV 缓存，就可以跳过除最新词元以外的所有计算。  
2. **少样本提示**：如果我们有任何类型的少样本提示，都可以保存并免费复用。系统指令通常也具有这种形式。

唯一难以实现这一点的原因是内存限制。正如我们所看到的，KV 缓存占用空间很大（通常为多个 GB），为了使缓存有用，我们需要在后续查询到达之前一直保留它们。通常，预填充服务器上任何未使用的 HBM 都可以用于本地缓存系统。此外，加速器的 CPU 主机上通常有大量内存（例如，一个 8xTPUv5e 服务器有 128GiB 的 HBM，但约有 450GiB 的 Host DRAM）。这种内存比 HBM 慢得多——通常太慢而无法执行生成步骤——但足以用于缓存读取。在实践中：

* 由于 KV 缓存仅对处理初始请求的 TPU 集合本地可见，我们需要某种形式的亲和路由，以确保后续查询到达相同的副本。这可能会对负载均衡造成影响。  
* 更小的 KV 缓存（再次）是有帮助的——它使我们能够在相同的空间中保存更多的 KV 缓存，并减少读取时间。  
* KV 缓存及其查找可以很自然地存储在树或字典树中。驱逐操作可以基于 LRU 原则进行。

{% include figure.liquid path="assets/img/prefix-caching-trie.png" class="img-fluid" caption="<b>图：</b> 作为 LRU trie 实现的 KV 前缀缓存。通过共享前缀，我们可以避免重复 KV 内存。来源：<a href=\"https://research.character.ai/optimizing-inference/?ref=blog.character.ai\">Character.ai 博客</a>。" %}

### 让我们看看一个实现：JetStream

Google 已开源了一个名为 [JetStream](https://github.com/google/JetStream) 的库来实现这一逻辑。该服务器拥有一组“预填充引擎”和“生成引擎”，通常位于不同的 TPU 切片上，由一个控制器进行协调。预填充在“[预填充线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L499)”中进行，而生成在“[生成线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L629)”中进行。我们还有一个“[传输线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L592)”，用于协调将 KV 缓存从预填充切片复制到生成切片。

Engine 接口（实现见 [此处](https://github.com/google/JetStream/blob/445f1aa8e857d0a09d72618e365daf80723bdf4c/jetstream/engine/engine_api.py#L138)）是一个通用接口，任何大型语言模型（LLM）都必须提供。关键方法包括：

* **预填充:** 接收一组输入词元并生成一个 KV 缓存。  
* **插入:** 接收一个 KV 缓存并将其插入到正在生成的 KV 缓存批次中。  
* **生成:** 接收一组批处理的 KV 缓存，并为每个批次条目生成一个词元，将每个词元的 KV 缓存追加到对应的解码状态中。

我们还有一个 JetStream 的 PyTorch 版本可用 [此处](https://github.com/google/jetstream-pytorch)。

## 习题解答

我将为本节基于 LLaMA-2 13B 构建一个新模型。以下是详细信息：

|超参数|价值|
| :----------------- | :----- ||L（num_layers）|64|
|D（d_model）|4,096|
|F (ffw_dimension)|16,384|
|N（num_heads）|32|
|K（num_kv_heads）|8|
|H (qkv_dim)|256|
|V (num_embeddings)|32,128|

**问题 1:** 上述模型有多少个参数？每个词元的 KV 缓存大小在 int8 下是多少？*你可以假设我们共享输入和输出投影矩阵。*

{% details 点击此处查看答案。 %}

**参数量:**

* MLP 参数量：$L * D * F * 3$  
* 注意力参数量：$L * 2 * D * H * (N + K)$  
* 词汇参数：$D * V$（因为我们共享这些矩阵）

因此，我们的总参数量为 $L * D * (3F + 2H * (N + K)) + D * V$。代入上述数值，我们得到 `64 * 4096 * (3*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 18.4e9`。因此，该模型约有 184 亿个参数。

KV 缓存每个词元为 $2 * L * K * H$（int8），每个词元为 `2 * 64 * 8 * 256 = 262kB`。

{% enddetails %}

**问题 2:** 假设我们想在 TPUv5e 4x4 切片上部署该模型，并且可以将 KV 缓存完全分片到该拓扑结构上。如果我们使用 int8 格式，并希望支持 128k 序列，那么可以容纳的最大批量大小是多少？如果我们减少 KV 头的数量到 1，结果又会如何？

{% details 点击此处查看答案。 %}

我们的 KV 缓存每个词元在 int8 下的大小为 $2 \cdot L \cdot K \cdot H$，或 `2 * 64 * 8 * 256 = 262kB`。对于 128k 序列，这意味着每个批量条目为 `262e3 * 128e3 = 33.5GB`。由于每个 TPU 有 16GB 的 HBM，包括我们的参数，我们能容纳的最大批量大小为 `(16 * 16e9 - 18.4e9) / 33.5e9 = 7`。如果我们有 $K=1$，我们将拥有这个数值的 8 倍，即大约 56。

{% enddetails %}

**问题 3:** 假设参数在 TPU v5e 4x4 切片上被完全分片，将所有参数从 HBM 加载到 MXU 需要多长时间？假设参数为 int8。*这是每步延迟的一个很好的下界。*

{% details 点击此处查看答案。 %}

我们总共有 184 亿个参数，或以 int8 格式表示为 18.4e9 字节。每块芯片有 8.2e11 HBM 带宽，因此假设我们可以充分利用 HBM 带宽，大约需要 `18e9 / (8.2e11 * 16) = 1.4ms`。

{% enddetails %}

**问题 4:** 假设我们想使用 int8 FLOPs 和参数/激活值在 TPUv5e 4x4 切片上部署此模型。我们该如何对其进行分片，以同时支持预填充和解码？*提示：也许先回答这些问题：*

1. ICI 在 4x4 上是什么样子？
2. 张量并行的屋顶线界限是什么？
3. 我们如何分片 KV 缓存？

对于这种分片方式，生成的每步延迟大致是多少？

**问题 5：** 假设上述模型实际上是一个 MoE 模型。MoE 模型本质上是一个密集模型，其中包含 E 个 FFW 模块的副本。每个词元会通过其中的 k 个 FFW 模块，这些 `k` 的结果会被平均以生成输出。让我们使用 `E=16` 和 `k=2`，并采用上述设置。

1. 它总共有多少参数，其中有多少参数被激活？*激活表示被任意给定词元使用。*
2. 需要多大的批量大小才能在 TPU v5e 上达到 FLOPs 限制？
3. 每个词元的 KV 缓存有多大？
4. 使用 T 个词元进行一次前向传递涉及多少 FLOPs？

{% details 点击此处查看答案。 %}

(1) 作为一种 MoE，每个 MLP 模块现在有 $3 * E * D * F$ 个参数，比密集版本增加了 $E$。因此，它现在总共有 $L * D * (3EF + 2H * (N + K)) + D * V$ 或 `64 * 4096 * (3*16*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 212e9` 个参数，增加了约 12 倍。对于激活的参数，我们有 $k$ 而不是 $E$ 个激活参数，总数为 `64 * 4096 * (3*2*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 31.2e9`，比密集版本增加了不到 2 倍。

(2) 因为我们仅拥有 $k$ 倍更多的 FLOPs，却拥有 $E$ 倍更多的参数，我们的 HBM 顶线提高了 $E/k$ 倍。这意味着在 TPU v5e 上，我们需要大约 `240 * (16 / 2) = 1920` 个词元。

(3) KV 缓存大小保持不变，因为 MoE 字符不会对注意力机制产生任何影响。

(4) 这仍然是 $2 \cdot \text{activated params} \cdot T$。因此这是 $2 * \text{31.2e9} * T$。

{% enddetails %}

**问题 6:** 使用 MoE 时，我们可以进行“专家分片”，即将专家分布在网格的一个轴上。按照我们的标准表示法，我们的第一个 FFW 权重形状为 `[E, D, F]`，我们将其分片为 [E<sub>Z</sub>, D<sub>X</sub>, F<sub>Y</sub>]，其中 `X` 仅在训练期间使用，作为我们的 FSDP 维度。假设我们想要在 TPU v5e 上进行推理：

1. 上述模型在 TPU v5e 8x16 切片上，Y=8，Z=16 时的 HBM 权重加载时间是多少？每个 TPU 上有多少可用的空闲 HBM？
2. 我们能将模型放入的最小切片是什么？

**问题 7 [2D 模型分片]:** 在这里，我们将推导[ESTI 论文](https://arxiv.org/pdf/2211.05102)所称的 2D 权重静止分片的数学原理。我们在附录 B 中简要描述了这一点，但请先尝试解决这个问题，看看你是否能够推导出数学公式。2D 权重静止分片的基本思想是沿着 $D$ 和 $F$ 轴对权重进行分片，使得每个分片大致呈正方形。这可以减少通信负载，并使我们能够稍微扩展规模。

这是 2D 权重驻留的算法：

<div markdown=1 class="algorithm">

1.  In[B, D<sub>X</sub>] = **AllGather**<sub>YZ</sub>(In[B, D<sub>XYZ</sub>])  
2.  Tmp[B, F<sub>YZ</sub>] {U<sub>X</sub>} = In[B, D<sub>X</sub>] \*<sub>D</sub> W<sub>in</sub>[D<sub>X</sub>, F<sub>YZ</sub>]  
3.  Tmp[B, F<sub>YZ</sub>] = **AllReduce**<sub>X</sub>(Tmp[B, F<sub>YZ</sub>] {U<sub>X</sub>})  
4.  Out[B, D<sub>X</sub>] {U<sub>YZ</sub>} = Tmp[B, F<sub>YZ</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>YZ</sub>, D<sub>X</sub>]  
5.  Out[B, D<sub>XYZ</sub>] = **ReduceScatter**<sub>YZ</sub>(Out[B, D<sub>X</sub>] {U<sub>YZ</sub>})  
</div>

你的目标是推导出该算法的 $T_\text{math}$ 和 $T_\text{comms}$，并找出它何时会优于传统的 3D 模型分片方法？

{% details 点击此处查看答案！ %}

让我们计算 $T_\text{math}$ 和 $T_\text{comms}$。我们所有的 FLOPs 都是完全分片的，因此和之前一样我们有 $T_\text{math} = 4BDF / (N \cdot C)$，但我们的通信现在

$$\begin{align*}
T_\text{2D comms} = \frac{2BD}{2X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}} + \frac{2BD}{2X \cdot W_\text{ici}} = \frac{2BD}{X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}}
\end{align*}$$

其中我们注意到 AllReduce 的开销是两倍，并且我们根据每个操作执行的轴数来缩放我们的通信。假设我们有自由选择拓扑结构，并假设 $F=4D$（如 LLaMA-2 所示），我们声称（通过一些基本的微积分）$X$、$Y$ 和 $Z$ 的最优值分别为 $X = \sqrt{N / 8}$、$YZ = \sqrt{8N}$，因此总的通信量为

$$T_\text{2D comms} = \frac{2B}{W_\text{ici}} \left(\frac{D}{X} + \frac{8D}{YZ}\right) = \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \approx \frac{11.3 BD}{\sqrt{N} \cdot W_\text{ici}}$$

首先，从上面复制过来，普通的 1D 模型并行会具有 $T_\text{model parallel comms} = 4BD / (3 \cdot W_\text{ici})$，那么在什么情况下新的通信会更小？我们有

$$\begin{align*}
T_\text{model parallel comms} > T_\text{2D comms} \iff \frac{4BD}{3 \cdot W_\text{ici}} > \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \\
\iff N > 128 \cdot \left(\frac{3}{4}\right)^2 = 72
\end{align*}$$

对于一般的 $F$，我们声称这一条件是

$$N > 32 \cdot \left(\frac{F}{D}\right) \cdot \left(\frac{3}{4}\right)^2$$

这告诉我们，如果我们拥有超过 72 块芯片，使用这种新方案会更划算。现在这个结果有点奇怪，因为我们过去通常在大约 ~20 路张量并行时受到 ICI 的限制。但在这里，即使我们受到通信限制，总的通信量仍然随着芯片总数的增加而减少！这告诉我们，我们可以继续增加芯片数量，增加批量大小，进行更多的参数缩放，并看到延迟的降低。

{% enddetails %}

<h3 markdown=1 class="next-section">第7部分就到这里！第8部分将探讨如何在TPU上部署LLaMA 3，点击[这里](../applied-inference)查看。</h3>

## 附录

### 附录 A：批大小 > 240 的规则有多真实？

我们上面提供的简单规则，即批量大小必须大于 240 词元才能成为算力限制因素，大致上是正确的，但忽略了 TPU 在其他操作未完全使用所有可用 HBM 时预取权重的能力，例如在进行设备间通信时。

这是一个小 Transformer 的层时间（以微秒为单位）的经验图表，其中 d<sub>model</sub> 为 8192，d<sub>ff</sub> 为 32768，每层仅有 2 个矩阵乘法。该图表来自 [这个 Colab 笔记本](https://colab.sandbox.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing)。你会看到，直到大约第 240 个批次之前，步时间增加得非常缓慢，之后则呈线性增长。

{% include figure.liquid path="assets/img/batch-scaling-latency.png" class="img-fluid img-small" %}

这里是实际的吞吐量，单位为词元 / 微秒。这使得论点变得相当明确。由于我们的层在这里大约有 600M 参数，并以 4 种方式分片，我们预计最低延迟约为 365 微秒。

{% include figure.liquid path="assets/img/batch-scaling-throughput.png" class="img-fluid img-small" %}

因此，至少在这个模型中，我们确实观察到吞吐量在每个数据并行分片约 BS240 之前呈增长趋势。

### 附录 B：2D 权重驻留分片

随着拓扑结构的增长，如果我们能够访问更高维的网格（如TPU的网格），就可以通过引入第二个分片轴，进一步使用“**2D Weight Sharding**”来细化这一过程。我们将这种方法称为“**2D Weight Stationary**”，在[《高效扩展Transformer推理》论文](https://arxiv.org/abs/2211.05102)中对此进行了更详细的描述。

由于我们在 Megatron 中仅对隐藏 $$F$$ 维度进行分片，当芯片数量随着 1D 分片的增加而变得很大时，它可能会显著小于 $$E$$（$$d_\text{model}$$ 维度）。这意味着在更大的批量大小下，在应用 MLP 的第一层之后，对隐藏维度的一部分集合通信进行操作可能更加经济。

{% include figure.liquid path="assets/img/2d-weight-stationary.png" class="img-fluid img-small" %}

此图显示：

1. 1D 权重静态分片，又称纯 Megatron 分片，其中在 AllGather 之后激活值被完全复制，权重在隐藏维度 F 上被完全分片。  
2. 2D 权重静态分片，其中权重在隐藏维度 F 和约简维度 E 上均被分片，激活值在 E 维度上被分片。我们在第一层之前在 (yz) 轴上执行 AllGather，然后在 (x) 轴上执行 ReduceScatter。

对于注意力层，当芯片数量较少时，Megatron 风格的分片也相对简单。然而，Megatron 是在 $$n_\text{heads}$$ 维度上进行的，这限制了可能的分片程度。通过修改注意力的 2D 分片方式（不是对隐藏维度进行分片，而是对 $$n_\text{heads}$$ 维度进行分片），我们可以进一步扩展规模。

### 附录 C：延迟限制通信

作为回顾，在[第3节](../sharding)中，我们推导了在1D环形拓扑上，使用全双工带宽为WICI且延迟为Tmin的X个芯片，将数据AllGather到大小为B的张量中所需的时间。

$$T_{total} = \max\left(\frac{T_{min} \cdot |X|}{2}, \frac{B}{W_{ICI}}\right)$$

对于较大的 B，墙钟时间相对保持恒定，因为当你向系统中添加更多芯片时，所需的数据移动量和可用的总带宽同时增加。

{% include figure.liquid path="assets/img/all-gather.gif" class="img-fluid" %}

由于在延迟优化的推理过程中移动的数据量相对较低，激活值上的集合通信通常受延迟项的限制（尤其是小批量大小时）。通过计算在完成之前需要完成的跳数，可以很容易地可视化延迟。

在 TPU 上，如果通信中与张量大小相关的部分每跳（一跳是指两个相邻设备之间的通信）小于 1 微秒，我们可能会受到实际调度集合通信的固定开销的限制。具有 `4.5e10` 单向 ICI 带宽时，当 $$(\text{bytes} / n_\text{shards}) / 4.5e10 < 1e-6$$ 时，ICI 通信将变为延迟受限。对于 8 路 Megatron 分片，这发生在 `buffer_size < 360kB` 时。**实际上，在推理期间这个数值并不小：** 使用 `BS=16` 和 `D=8192` 以 int8 格式时，我们的激活值将使用 `16*8192=131kB`，因此我们已经处于延迟受限状态。

<p markdown=1 class="takeaway">**要点：** 当 $$\text{total bytes} < W_{ICI} \times 1e-6$$ 时，我们的通信会受到延迟的限制。例如，在 $$Y$$ 上使用模型并行时，当 $$Y > BD / 45,000$$ 时，我们会受到 int8 的限制。</p>

这里可以与算力屋顶图进行类比——我们正在承担一些小操作的固定成本（通信的延迟、矩阵乘法的显存带宽）。

### 附录 D：推测采样

当我们*真的*关心端到端延迟时，我们可以使用一种称为推测采样的额外技巧<d-cite key="spec1"></d-cite><d-cite key="spec2"></d-cite>。简要回顾一下，我们通常逐个生成来自大型Transformer的词元：

{% include figure.liquid path="assets/img/spec-sampling1.png" class="img-fluid" %}

使用推测采样时，我们使用一个更小、更便宜的模型来生成词元，然后用大模型检查结果。这通过 *贪心解码* 最容易理解：

{% include figure.liquid path="assets/img/spec-sampling2.png" class="img-fluid" %}

1. 我们从某个较小且成本较低的模型中贪婪采样。理想情况下，我们使用一个经过训练以匹配大模型的模型，例如通过蒸馏，但也可以简单到仅使用 n-gram 或匹配一小段文本的词元。
2. 在我们生成了 K 个词元后，我们使用大模型计算到目前为止生成的所有词元的下一个词元的 logit。
3. 由于我们是贪婪解码，只需检查小模型生成的词元是否在所有可能的词元中具有最高概率。如果其中一个词元错误，我们取最长的正确前缀，并将第一个错误的词元替换为正确的词元，然后返回到步骤 (1)。如果所有词元都正确，我们可以使用最后一个正确的 logit 生成一个额外的词元，然后再返回到步骤 (1)。

**为什么这能降低延迟？** 这种方案仍然需要我们为每个词元执行一次通过大模型的等效 FLOPs 的前向传递，但由于我们可以将许多词元批量处理在一起，我们可以在一次前向传递中完成所有这些 FLOPs，并利用我们 *不是* *算力受限* 的事实，免费处理更多词元。

每个被接受的词元在平均 FLOPs 方面都变得更加昂贵（因为有些词元会被拒绝，我们需要调用一个草稿模型），但我们从硬件中榨取了更多的 FLOPs，而小模型成本低廉，因此总体上我们仍然占优。我们还可以在多个步骤之间共享 KV 缓存负载，因此 **推测性解码也可以在长上下文中实现吞吐量提升。** 由于所有内容都已由大模型检查过，我们完全不会改变采样分布（尽管对于非贪心策略，确切的轨迹会有所不同）。

传统上，推测解码依赖于存在一个与目标模型具有相似采样分布的较小模型，例如为 LLaMA-2 70B 提供的 LLaMA-2 2B，但这种情况通常并不存在。即使存在这样的模型，如果接受率较低，较小的起草者仍然可能成本过高。相反，可以在主模型中嵌入一个起草者，例如通过在基础模型的后期层中添加一个专用的起草者头<d-cite key="eagle"></d-cite><d-cite key="medusa"></d-cite><d-cite key="DeepSeek3"></d-cite>。由于这个头与主模型共享了大部分参数，因此运行速度更快，并且更紧密地匹配采样分布。

对于普通的自回归采样，词元/步的时间与步时间相同。我们仍然受制于此处“算术强度”部分所描述的理论最小步时间（事实上，推测采样的步时间通常比普通的自回归采样慢很多，但由于平均每步可以得到超过 1 个词元，因此可以实现更高的词元/秒）。

{% include figure.liquid path="assets/img/spec-sampling3.png" class="img-fluid" caption="<b>Figure:</b> 该图展示了 Chinchilla（DeepMind 的一个 70B 模型）在使用 4B 参数的 drafter（小模型）时的每步延迟和推测成功率。对于 XSum（一个自然语言数据集），理想的推测量约为 3-4 个词元，而 HumanEval（一个代码数据集）更具可预测性，更激进的推测能带来更好的效果。" %}

**非贪心解码情况下，这种方法如何运作？** 这要复杂一些，但本质上可以归结为一种受 Metropolis-Hastings 算法启发的算法，其中 $$P_{\text{draft model}}(\text{chosen token})$$ 和 $$P_{\text{target model}}(\text{chosen token})$$ 是从 logit 中推导出来的，并且如果这两个概率的比值小于某个阈值，则以一定概率拒绝所选的 token。

这两篇[论文](https://arxiv.org/abs/2211.17192)和[论文](https://arxiv.org/abs/2302.01318)同时得出了这一结论，并提供了该方法在实践中如何运作的良好示例。

<p markdown=1 class="takeaway">**要点：** 投机采样是又一个强大的杠杆，可用于以牺牲吞吐量为代价换取更优的每词元延迟。然而，在批量大小受限的场景（例如硬件规模较小或KV缓存较大）中，它则成为一个双赢的策略。</p>
