---
layout: distill
title: "如何分析 TPU 程序"
# permalink: /main/
description: "到目前为止，这一系列内容完全停留在理论层面：基于硬件性能极限的粗略估算。这种理解能带你走得很远，但很多优化都取决于实际细节：XLA 编译器的工作原理，以及如何使用 JAX/TensorBoard Profiler 等分析工具，在它失败时找出该怎么做。我们在这里讨论这些问题。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 9

previous_section_url: "../applied-inference"
previous_section_name: "Part 8: Serving LLaMA"

next_section_url: ../jax-stuff
next_section_name: "Part 10: JAX"

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
  - name: "A Thousand-Foot View of the TPU Software Stack"
  - name: "The JAX Profiler: A Multi-Purpose TPU Profiler"
  - subsections:
    - name: "Trace Viewer"
    - name: "How to read an XLA op"
    - name: "Graph Viewer"
    - name: "Looking at a real(ish) example profile"
    - name: "Memory Profile"
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

## TPU 软件栈的全景概览

谷歌公开了一系列用于编程TPU的API，从高级的JAX代码到低级的Pallas或HLO。大多数程序员仅编写JAX代码，这使你能够用类似NumPy风格的抽象线性代数程序进行编写，这些程序会自动编译以在TPU上高效运行。

这是一个简单的示例，一个使用 JAX 编写的程序，用于将两个矩阵相乘：

```py
import jax
import jax.numpy as jnp

def multiply(x, y):
  return jnp.einsum('bf,fd->db', x, y)

y = jax.jit(multiply)(jnp.ones((128, 256)), jnp.ones((256, 16), dtype=jnp.bfloat16))
```

通过调用 `jax.jit`，我们告诉 JAX 对该函数进行追踪，并生成一种称为 [StableHLO](https://openxla.org/stablehlo) 的更低层次的中间表示（IR），这是一种与平台无关的 ML 计算中间表示，随后由 XLA 编译器将其降低为 HLO。编译器运行许多阶段以确定融合、布局和其他因素，这些因素最终生成在 JAX 分析文件中可见的 HLO。这种 HLO 以类似 LLVM 的图视图方式表示 JAX 代码中的所有核心线性代数操作（矩阵乘法、逐点操作、卷积等）。例如，这里是上述程序的 HLO 简化版本：<d-footnote>。要获取此 HLO，可以运行 `jax.jit(f).lower(*args, **kwargs).compile().as_text()`.</d-footnote>：

```c
ENTRY %main.5 (Arg_0.1: f32[128,256], Arg_1.2: bf16[256,16]) -> f32[16,128] {
  %Arg_1.2 = bf16[256,16]{1,0} parameter(1), metadata={op_name="y"}
  %convert.3 = f32[256,16]{1,0} convert(bf16[256,16]{1,0} %Arg_1.2),
  %Arg_0.1 = f32[128,256]{1,0} parameter(0), metadata={op_name="x"}
  ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1},
}
```

我们将在片刻后解释 HLO 的语法，但目前只需注意它实际上与上面的 JAX 代码匹配得相当好。例如，

```c
ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1}
```

是沿 0 和 1 维度分别相乘两个 f32 矩阵的实际矩阵乘法。

**要将此 HLO 转换为可在 TPU 上执行的代码，XLA 编译器首先将其降低到 LLO**（低级优化器）IR。LLO 直接对 TPU 进行编程，安排内存之间的复制，将数组推送到 systolic 数组等。LLO 代码包含将缓冲区推送到 systolic 数组、从其中拉取结果以及安排在不同 TPU 内存部分之间通信的 DMA 的原语。一旦降低到 LLO，它将被编译为机器代码，并加载到 TPU 的 IMEM 中执行。

当程序运行速度不如我们期望时，我们主要在 JAX 层进行性能优化。然而，这样做通常需要我们理解一些 HLO 的语义以及代码在 TPU 上的实际运行方式。当更低层次出现问题时，我们会再启用另一个逃生口，使用 [Pallas](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html) 编写自定义内核。为了查看程序的 HLO 以及其运行时统计信息，我们使用 JAX 分析器。

## JAX 分析器：一种多功能 TPU 分析器

JAX 提供了一个多功能的 TPU 分析器，包含许多有用的工具，用于理解程序在 TPU 上运行时发生了什么。你可以使用 `jax.profiler` 模块来跟踪程序的运行过程，并记录每个子组件的持续时间、每个程序的 HLO、内存使用情况等信息。例如，这段代码会将跟踪信息输出到 `/tmp/tensorboard` 中的文件，可以在 TensorBoard 中查看（[这里](https://docs.jax.dev/en/latest/profiling.html#tensorboard-profiling) 是逐步指南）。

```py
import jax
with jax.profiler.trace("/tmp/tensorboard"):
  key = jax.random.key(0)
  x = jax.random.normal(key, (1024, 1024))
  y = x @ x
  y.block_until_ready()

# Now you can load TensorBoard in a Google Colab with
#
# !pip install -U xprof
# !pip install -U protobuf
# %load_ext tensorboard
# %tensorboard --logdir=/tmp/tensorboard
#
# or externally with
#
# > tensorboard --logdir=/tmp/tensorboard
#
```

以下是性能分析器中可以执行的操作概览：

{% include figure.liquid path="assets/img/xprof-overview.png" class="img-fluid" %}

一旦进入 TensorBoard，性能分析工具有几个关键标签页，可以帮助你理解你的程序：

1. **Trace Viewer** 显示 TPU 上实际发生事件的详细时间线。  
2. **Graph Viewer** 显示 HLO 图，让您看到程序中哪些部分相互连接以及如何进行分片。  
3. **Memory Profile and Memory Viewer**：这些工具显示您的程序使用了多少内存。

虽然分享配置文件略显困难，[这里](https://ui.perfetto.dev/#!/?s=fa9f13b487bde622707c1a503f9227c34594760a)是一个 Perfetto 链接，其中至少包含用于简单 Transformer 的 Trace Viewer 组件。[这个 Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing) 可以让你生成完整的 JAX/TensorBoard 跟踪记录，并对其进行操作。

### 跟踪查看器

**Trace Viewer 可能是性能分析器最有用的部分。** 下面的示例展示了一个简单的 Transformer，其中某些部分进行了标注。名称来自代码中提供的标签。

{% include figure.liquid path="assets/img/trace-viewer.png" class="img-fluid" %}

Trace Viewer 显示了每个 TPU 核上所有操作的按时间顺序排列的时间线。这里我们只查看 TPU:0，因为通常所有 TPU 都执行相同的指令。一些关键说明：

1. 第一行（XLA Ops）显示了实际的 TPU 操作（名称为 HLO 名称）。其余部分是基于 `jax.named_scope`、`jax.named_call` 和 Python 堆栈跟踪的近似追踪。  
2. 通过注意到重复的模块，我们可以在此处隔离出单个层。我们还可以看到（通过查看代码/理解 Transformer 的工作原理），哪些部分是注意力机制，哪些部分是 MLP。  
3. 通过点击一个 XLA 操作，我们可以查看它在代码中的来源（有助于理解追踪），并看到链接到图查看器。

<p markdown=1 class="takeaway">**提示：** 您可以使用类似“视频游戏”的控制方式在 Trace Viewer 中进行导航，A/D 键用于左右平移，W/S 键用于缩放。这些控制方式使导航变得更加容易。</p>

### 如何读取一个 XLA 操作

HLO 实际上并不难阅读，它对于理解上面跟踪中某一部分所对应的含义非常有帮助。这里有一个名为 fusion.3 的示例操作。

```c
%fusion.3 = bf16[32,32,4096]{2,1,0:T(8,128)(2,1)S(1)} fusion(bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32), kind=kCustom, calls=%all-reduce-scatter.3
```

让我们将其分解为各个部分。

* **操作名称**: fusion.3  
  * 点积或融合操作是一组操作，最多包含一个矩阵乘法，可能还包含一些相关的逐点 VPU 操作。  
* **形状**: `bf16[32,32,4096]`  
  * 这是操作的输出形状。我们可以看到数据类型是 bf16（每个元素 2 字节），`[32,32,4096]` 是形状。  
* **布局**: `{2,1,0:T(8,128)(2,1)}`  
  * `{2,1,0:T(8,128)(2,1)}` 告诉我们内存中轴的顺序（列优先、行优先等）以及数组的填充方式。更多内容见下文。  
* **内存位置**: S(1)  
  * S(1) 告诉我们这个数组位于 VMEM 中。S(0)（有时省略）是 HBM。S(2) 和 S(3) 是其他内存空间。  
* **参数**: `bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32`  
  * 这个操作有一个输入，是一个名为 fusion.32 的 bf16 数组，具有特定的形状。这告诉我们哪个函数输入到这个操作中。

让我们试着更深入地理解这个符号表示。我们以这个简单的例子来说明：

`f32[3,5]{1,0:T(2,2)}`

这再次告诉我们，这个 Op 返回一个形状为 `[3, 5]` 的 float32 数组，并具有特定的分块方式 `{1,0:T(2,2)}`。虽然分块方式并不*太*重要，简而言之，分块方式告诉我们一个 N 维数组在内存中是如何按顺序排列的。下图展示了这个数组的排列方式：

{% include figure.liquid path="assets/img/tiling.png" class="img-fluid" %}

在 `{1,0:T(2,2)}` 中，`1,0` 部分告诉我们物理内存中数组维度的排列顺序，从最次要到最主要。你可以从右到左阅读这部分内容，并从 `f32[3,5]` 中挑选出对应的维度，以确定数组的物理布局。在此示例中，物理布局为 `[3,5]`，与逻辑形状相同。  
之后，`T(2,2)` 告诉我们数组以 `(2, 2)` 的块进行分块，其中在每个块内，数组首先按行排列（**行优先**），然后是列，即 `(0, 0)` 被 `(0, 1)` 紧随其后，然后是 `(1, 0)` 和 `(1, 1)`。由于 `T(2, 2)` 的分块方式，数组被填充到 `[4, 6]`，使其内存使用量增加了约 1.6 倍。对于上面给出的大 bf16 数组 `bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)}`，我们执行 `T(8,128)(2,1)`，它告诉我们数组具有两层分块，外层为 `(8, 128)` 分块，内层为 `(2, 1)` 分块（用于 bf16，因此我们的加载量始终是 4 字节的倍数）。例如，这里是 `bf16[4,8]{1,0:T(2,4)(2,1)}`（颜色为 (2,4) 分块，红色框为 (2,1) 分块）：

{% include figure.liquid path="assets/img/tiling2.png" class="img-fluid img-small" %}

分块可能会影响张量块加载到 VMEM 的效率，XLA 有时会引入一些复制操作，对张量在程序内部进行“重新分块”或“重新布局”，有时会产生非微不足道的开销。<d-footnote>JAX 提供了 <a href="https://docs.jax.dev/en/latest/notebooks/layout.html">一个实验性功能</a> 来解决这个问题，允许 XLA 计算程序输入的“首选”布局。当你使用 `jax.jit` 对程序进行“即时编译”时，通常会传入一些“模拟”输入，告诉 JAX 预期的形状和数据类型。这些输入通常也携带了可能不是最优的分块信息。相反，你可以将输入布局指定为 AUTO，`jax.jit` 会返回被编译程序偏好的布局。然后你可以显式地以该布局加载张量，以避免在程序内部引发复制操作。</d-footnote>

### 图查看器

虽然上述的一些融合操作看起来可能比较复杂，但 XLA 图查看器使它们更容易解析。例如，这里是一个相对复杂的融合操作的视图：

{% include figure.liquid path="assets/img/graph-viewer.png" class="img-fluid" %}

盯着一大堆 HLO 图并尝试将 HLO 操作映射到你正在分析的代码上，这真的很有帮助。将鼠标悬停在一个框上时，你通常可以看到定义该函数的代码行。

### 查看一个真实（ish）的示例配置文件

[此 Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing) 提供了一个假 Transformer 的示例性能分析。[这里](https://ui.perfetto.dev/#!/?s=fa9f13b487bde622707c1a503f9227c34594760a) 是一个 Perfetto 链接，至少可以查看 Trace Viewer，如果你时间紧迫的话。我比平时更努力地对跟踪信息进行了注释，使用了 `jax.named_scope` 调用，以便你能识别发生了什么。

{% include figure.liquid path="assets/img/transformer-xprof.png" class="img-fluid" %}

看一下这个配置文件，试着真正理解每一部分在做什么。让我们逐步分解，从 FFW 模块开始：

{% include figure.liquid path="assets/img/transformer-ffw.png" class="img-fluid" %}

此处我们放大到了FFW块。你会看到上投影操作（Op）是一个融合（matmul），输入为`bf16[8, 1024, 8192]`和`bf16[8192, 16384]`，输出为`bf16[8, 1024, 16384]`。我知道（因为我写了这段代码），这是4路DP、2路MP分片矩阵乘法的局部视图，因此我们实际上正在执行

**X:** `bf16[32, 1024, 8192]` \* **W<sub>in</sub>**: `bf16[8192, 32768]` -> **Tmp**: `bf16[32, 1024, 32768]`

**我们预计这需要多长时间？** 首先，每个数据并行分片的批量大小为 `8 * 1024 = 8192`，因此我们应处于算力受限的状态。这是在 8 个 TPU v2 核心上运行，因此我们预计需要大约 `2 * 32 * 1024 * 8192 * 32768 / (23e12 * 8) = 95.6ms`，这几乎正好就是实际所需的时间（96 毫秒）。太棒了！这意味着我们获得了极佳的 FLOPs 利用率！

请注意，Google Colab 已不再提供 TPU v2-8 的切片。若想获得真正的 8 核切片以进行后续操作，可以使用 [Kaggle](https://www.kaggle.com/)，它仍然免费提供此类资源，或者在 GCP 上配置一个 8 核切片。<d-footnote> 如果你只想在虚构的问题上尝试分片，也可以使用 `import jax; jax.config.update("jax_num_cpu_devices", 8)`（需要 jax >= 0.4.27ish）在 CPU 上模拟 8 个设备，然后使用 `print(jax.devices())`。这仅适用于玩具问题，无法反映实际性能。</d-footnote>

**关于通信呢？** 你会注意到第二个矩阵乘法末尾隐藏的小型融合。如果我们点击它，将会看到

```c
%fusion.1 = bf16[8,1024,4096]{2,1,0:T(8,128)(2,1)} fusion(bf16[8,1024,8192]{2,1,0:T(8,128)(2,1)} %fusion.31), kind=kCustom, calls=%all-reduce-scatter.1
```

这基本上是一个小的 ReduceScatter（这里是图查看器）；

{% include figure.liquid path="assets/img/reduce-scatter-xprof.png" class="img-fluid" %}

我们预计这需要多长时间？嗯，我们正在对 TPU v2 4x2 执行 ReduceScatter 操作，这在 1.2e11 双向带宽下只需要一次跳转。数组的大小为 `2*32*1024*8192`，批量轴被分片为 4 份，因此每个分片的大小为 `2*8*1024*8192=128MB`。因此，这应该需要大约 1.1 毫秒。**实际需要多长时间？** 个人资料中报告为 1.13 毫秒。因此，我们非常接近理论极限！

**我们也来看看注意力机制！** 这是注意力组件的性能分析：

{% include figure.liquid path="assets/img/attn-xprof.png" class="img-fluid" %}

我点击了 Q 投影操作，它使用了一个形状为 [d<sub>model</sub> = 8192, n<sub>heads</sub> = 32, d<sub>qkv</sub> = 256] 的矩阵 $$W_Q$$。我们沿着头维度进行 Megatron 分片。试着做同样的练习，计算这些操作应该需要多长时间。

### 内存概况

内存分析功能使你可以轻松查看程序内存随时间变化的情况。这对于调试内存不足（OOM）问题很有帮助。在这里你可以看到约 7.5GB 内存被分配给模型参数，约 8.5GB 内存处于空闲状态。因此，我们还可以将更多内容加载到内存中。

{% include figure.liquid path="assets/img/memory-viewer.png" class="img-fluid" %}

## 习题解答

**问题 1**：查看 [这个](https://colab.research.google.com/drive/1LfLO3OTr-_MWFPxUN36KJ3cqH0BcAoli?usp=sharing) Colab/性能分析，找出哪些部分看起来可疑，以及这里发生了什么。你能准确告诉我正在进行哪些计算，每个操作在做什么吗？每个矩阵的真实形状是什么，它们是如何分片的？*先尝试不看代码，直接查看性能分析。*

{% include figure.liquid path="assets/img/all-reduce-profile.png" class="img-fluid" %}

{% details 点击此处查看答案。 %}

这是两次矩阵乘法，即具体如下：

```py
def matmul(w1, w2, x):
  return jnp.einsum('wf,bf->bw', w2, jnp.einsum('fw,bw->bf', w1, x))
```

你可以看到一个 reduce、两次大的融合以及一个 all-reduce。第一次大的融合是：

```%fusion.1 = bf16[4096]{0:T(1024)(128)(2,1)} fusion(bf16[4096,8192]{1,0:T(8,128)(2,1)} %param.1, bf16[8192]{0:T(1024)(128)(2,1)} %reduce.6), kind=kLoop, calls=%fused_computation.1```

这告诉我们每个分片的形状是 `bf16[8192] * bf16[4096, 8192] -> bf16[4096]`（沿 8192 维度）。通过观察最终的 AllReduce 及其 {% raw %}`replica_groups={{0,16,32,48,64,80,96,112}, ...}`{% endraw %}，我们可以看出正在进行 8 路模型并行，因此真实的形状是 `bf16[8, 8192] * bf16[32768, 8192] -> bf16[8, 32768]`。

{% enddetails %}

**问题 2:** [之前提供的 Transformer Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing) 实现了一个简单的模拟 Transformer。由于 Colab 不再提供 TPU v2-8 切片，你将需要在 [Kaggle](https://www.kaggle.com/) 或一个 8 核 GCP 切片上运行此代码以继续操作。按照 Colab 中的说明进行操作，并获取使用 GSPMD 分区的朴素 Transformer 的基准测试结果。每个部分需要多长时间？应该需要多长时间？使用了哪种分片方式？尝试修复分片方式！*提示：使用 `jax.lax.with_sharding_constraint` 来约束行为。通过此修复，你能达到的最佳 MFU 是多少？*

作为参考，初始版本每层大约耗时 184ms，而优化后的剖面每层为 67ms。完成这些之后，试着仔细查看剖面，看看能否仅凭剖面就回答以下问题：

- 这是什么分片策略？  
- 批量大小是多少，$$d_\text{model}$$，$$d_\text{ff}$$？  
- 多少时间用于注意力计算，多少时间用于 MLP 模块？  
- 在屋顶线模型中，每个操作应分配多少时间比例？

**注意：** 自从这个问题编写以来，XLA 编译器已经有所改进。初始版本现在大约为每层 90 毫秒，优化后的配置仅比初始版本好约 10 毫秒（即每层 80 毫秒）。尽管如此，仍然值得一试，看看是否能做得更好。

<h3 markdown=1 class="next-section">第 9 部分到此结束。要进入第 10 部分、深入了解 JAX 并行机制，请点击[此处](../jax-stuff)。</h3>
