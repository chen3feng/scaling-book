---
layout: distill
title: "用 JAX 编程 TPUs"
# permalink: /main/
description: "如何高效地使用 JAX 编程 TPUs！本节的大部分内容取自 <a href='https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html'>here</a>。你可以通过 <a href='https://colab.sandbox.google.com/'>Google Colab</a> 上的免费 TPU 运行本节中的代码示例。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 10

previous_section_url: "../profiling"
previous_section_name: "Part 9: Profiling"

next_section_url: ../conclusion
next_section_name: "Part 11: Conclusions"

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
  - name: Yash Katariya
    url: https://x.com/yashk2810
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "How Does Parallelism Work in JAX?"
  - subsections:
    - name: "Auto sharding mode"
    - name: "Explicit sharding mode"
    - name: "Manual sharding mode via shard_map"
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

## JAX 中的并行性是如何工作的？

JAX 支持三种多设备编程方法：

1. **编译器，接管控制权！** 让 XLA 编译器自动对数组进行分区，并决定需要添加哪些通信以支持给定的程序。这使你可以将原本在单个设备上运行的程序，无需任何修改即可自动扩展到数千个设备上运行。  
2. **JAX，接管控制权！** 自动并行化非常方便，但有时编译器会做出一些奇怪的决策。显式分片允许你像往常一样编写单设备代码，但由 JAX 负责分片传播（而不是编译器）。这意味着当 JAX 不确定你想要什么时，它可以向你请求澄清。  
3. **让我按我想要的方式编写代码，别管我！** 虽然编译器很方便，但有时它们会做错误的事情，添加你并不想执行的通信。有时我们希望明确指定我们想要执行的确切通信。

|模式|视图？|显式分片？|显式集合通信？|
|:---:|:---:|:---:|:---:||自动|全球|❌|❌|
|显式|全球|✅|❌|
|手册|每设备|✅|✅|

相应地，JAX 为每种模式提供了相应的 API：

1. `jax.jit`（带有 `Auto` 网格轴）允许你对任何现有的 JAX 函数进行调用，并使用分片输入。JAX 随后会使用 XLA 的 [Shardy](https://openxla.org/shardy) 编译器，该编译器会自动并行化程序。XLA 会在需要时为你添加通信操作（如 AllGathers、ReduceScatters、AllReduces 等），以支持现有操作。虽然它并不完美，但通常能够很好地在不修改代码的情况下，将你的程序自动扩展到任意数量的芯片上。

2. `jax.jit` 带有 `Explicit` 网格轴的用法与（1）类似，但让 JAX 负责分片传播，而不是 XLA。这意味着数组的分片实际上是 JAX 类型系统的一部分，当 JAX 检测到通信模糊时，可以报错并让用户自行解决。

3. `jax.shard_map` 是更手动的对应方式。你将获得程序在设备本地的视图，并需要显式编写任何所需的通信操作。如果你有一个分片数组，并希望每个设备上都有完整的数组？添加一个 `jax.lax.all_gather`。如果你想在所有设备上对数组求和？添加一个 `jax.lax.psum`（一个 AllReduce）。编程难度更高，但发生你不希望的操作的可能性要小得多。

<h3 id="auto-sharding-mode">自动分片模式</h3>

`jax.jit` 在 JAX 中扮演两个角色。顾名思义，它将 Python 函数“即时”编译为字节码（通过 XLA/HLO/LLO），从而使其运行得更快。但如果输入是分片的，或者用户指定了 `in_sharding` 或 `out_sharding`，它还会让 XLA 将计算分布在多个设备上，并根据需要添加通信。例如，下面是使用 `jax.jit` 编写分片矩阵乘法的示例：

```py
import jax
import jax.numpy as jnp

Auto = jax.sharding.AxisType.Auto

# This creates a fake set of 8 CPU devices so you can run this on a CPU without TPUs.
jax.config.update("jax_num_cpu_devices", 8)

# This creates a 2D 4x2 mesh with axis names X and Y that JAX uses by default.
# We explicitly tell JAX to let the XLA compiler infer sharding along these axes.
mesh = jax.make_mesh(axis_sizes=(4, 2), axis_names=('X', 'Y'), axis_types=(Auto, Auto))
jax.set_mesh(mesh)

# We create a matrix W and input activations In sharded across our devices.
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('X', 'Y')))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('Y', None)))

def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

# We can explicitly compile the sharded matmul function here. This adds all the
# necessary comms (e.g. an AllReduce after the matmul).
jit_matmul = jax.jit(matmul_square, out_shardings=jax.P('X', None)).lower(In, W).compile()

out = jit_matmul(In, W)
```

这将自动运行，并与任何分片配合使用，将计算分布在我们的设备上。**但硬件层面实际上发生了什么？**

1. 首先，我们在设备之间创建 In 和 W 的分片<d-footnote>注意我们是如何做到这一点的。这是创建具有特定分片方式的数组的一种方法（即通过向创建函数添加 device 参数）。另一种方法是使用 `jnp.array(....)` 正常创建数组，然后执行例如 `jax.device_put(..., jax.P('X', 'Y'))`。还有一种方法是编写一个创建你想要的数组的函数，并使用 `out_shardings` 进行 jit 编译，其中 `out_shardings` 是你想要的分片方式。</d-footnote>。W 沿着收缩维度进行 2 路分片，而 In 进行 8 路分片：输入维度进行 4 路分片，收缩维度进行 2 路分片。这对应于分片 W[D<sub>Y</sub>, F] 和 In[B<sub>X</sub>, D<sub>Y</sub>]，即一种模型并行和数据并行的组合。

2. 如果我们本地运行（即在一个设备上），`matmul_square` 会简单地对输入进行平方并执行一个简单的矩阵乘法。但由于我们指定了 `out_shardings` 为 `P('X', None)`，输出将在 batch 维度上进行分片，而在模型维度上进行复制，并需要一个 AllReduce 来计算。

使用我们之前章节中的符号表示，这可能会做一些类似的事情

1. Out[B<sub>X</sub>, F] { U<sub>Y</sub> } = In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F]  
2. Out[B<sub>X</sub>, F] = **AllReduce**(Out[B<sub>X</sub>, F] { U<sub>Y</sub> })

`jax.jit` 会为我们自动添加！实际上，我们可以使用 `jit_matmul.as_text()` 打印 HLO，并看到如下 HLO（大幅简略）：

```py
# This fusion is the actual matmul of the sharded inputs and matrix
%fusion = bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} fusion(bf16[2,1024]{1,0:T(4,128)(2,1)} %param, bf16[8192,1024]{1,0:T(8,128)(2,1)S(1)} %copy-done)

# We reduce the partially summed results across devices
ROOT %AllReduce = bf16[2,8192]{1,0:T(4,128)(2,1)} AllReduce(bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} %fusion)
```

我们可以看到上面的 matmul（融合操作）和 AllReduce。特别注意形状。`bf16[2, 1024]` 是激活值的局部视图，因为我们的 `batch_size=8` 被分片到 4 个设备上，而我们的 `d_model=2048` 也被分片为 2 部分。

**这非常神奇！** 无论我们的程序有多复杂，[Shardy](https://openxla.org/shardy) 和 jit 都会尝试为所有中间激活值找到分片方式，并根据需要添加通信。话虽如此，Shardy 也有其缺陷。它可能会犯错误。有时你会查看一个性能分析结果，发现某些地方出了问题。一个巨大的 AllGather 占用了性能分析的 80%，而实际上并不需要。当这种情况发生时，我们可以通过显式地使用 `jax.lax.with_sharding_constraint` 注解中间张量来纠正编译器。例如，对于两个矩阵乘法，我可以强制中间激活值沿着 `y` 维度进行分片（尽管这可能不是一个好主意），具体如下：

```py
import jax
import jax.numpy as jnp

Auto = jax.sharding.AxisType.Auto

mesh = jax.make_mesh((4, 2), ('X', 'Y'), (Auto, Auto))
jax.set_mesh(mesh)

def matmul(x, W_in, W_out):
  hidden = jnp.einsum('bd,df->bf', x, W_in)
  hidden = jax.lax.with_sharding_constraint(hidden, jax.P('X', 'Y'))
  return jnp.einsum('bf,df->bd', hidden, W_out)
```

这约占 JAX 自动分片世界中并行编程的 60%，在该世界中，你可以通过 `jax.lax.with_sharding_constraint` 控制中间分片。但“编译器调教”显然不是一个有趣的编程模型。你可能会为每个中间变量添加注解，但仍无法确定是否能获得正确的结果。那么，如果 JAX 本身能够处理和控制分片传播会怎样呢？

<h3 id="explicit-sharding-mode">显式分片模式</h3>

显式分片（或“按类型分片”）看起来与自动分片非常相似，但分片传播发生在 JAX 层级！每个 JAX 操作都有一个分片规则，该规则根据操作参数的分片生成操作结果的分片。你可以使用 `jax.typeof` 查看生成的分片：

```py
import jax
import jax.numpy as jnp
import numpy as np

Explicit = jax.sharding.AxisType.Explicit

# Running on a TPU v5e 2x2. This assigns names to the two physical axes of the hardware.
mesh = jax.make_mesh(axis_sizes=(2, 2), axis_names=('X', 'Y'), axis_types=(Explicit, Explicit))

# This tells JAX to use this mesh for all operations, so you can just specify the PartitionSpec P.
jax.set_mesh(mesh)

x = jax.device_put(np.arange(16, dtype=np.float32).reshape(8, 2), jax.P('X', 'Y'))

@jax.jit
def f(x):
  print(jax.typeof(x))  # float32[8@X,2@Y]
  out = x * 2
  print(jax.typeof(out))  # float32[8@X,2@Y]
  return out

f(x)
```

如你所见，JAX 将分片从输入（`x`）传播到了输出（`out`），这些信息可以在跟踪时通过 `jax.typeof` 进行检查。对于大多数操作，这些规则是简单且显而易见的，因为只有一种合理的选择（例如，逐元素操作保留相同的分片）。但对于某些操作，如何对结果进行分片是模糊的，在这种情况下，JAX 会在跟踪时抛出错误，并要求程序员显式提供一个 `out_sharding` 参数（例如 jnp.einsum、jnp.reshape 等）。让我们再看一个存在冲突的例子：

```py
# We create a matrix W and input activations In sharded across our devices.
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, out_sharding=jax.P('X', 'Y'))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, out_sharding=jax.P('Y', None))

@jax.jit
def matmul_square(In, W):
  print(jax.typeof(In))  # bfloat16[8@X, 2048@Y]
  print(jax.typeof(W))  # bfloat16[2048@Y, 8192]
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

matmul_square(In, W)  # This will error
```

此代码报错为：

```
Contracting dimensions are sharded and it is ambiguous how the output should be sharded.
Please specify the output sharding via the `out_sharding` parameter.
Got lhs_contracting_spec=('Y',) and rhs_contracting_spec=('Y',)
```

这非常棒，因为 einsum 的输出应该如何分片是模糊的。输出分片可以是：  
* P('X', 'Y')，这将引发一个 ReduceScatter 或  
* P('X', None)，这将引发一个 AllReduce

与 Auto 模式不同，显式模式在检测到模糊通信时会报错，并要求用户解决。因此，你可以在这里执行以下操作：

```py
@jax.jit
def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W, out_sharding=jax.P('X', 'Y'))

out = matmul_square(In, W)
print(jax.typeof(out))  # bfloat16[8@X,8192@Y]
```

Auto 模式和 Explicit 模式可以通过 `jax.sharding.auto_axes` 和 `jax.sharding.explicit_axes` API 组合使用。更多信息可参见[这篇很好的文档](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html)。

<h3 id="manual-sharding-mode-via-shard-map">通过 shard_map 实现的手动分片模式</h3>

虽然 Shardy 是“编译器接管控制权”的模式，但 jax [shard_map](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html) 将一切交由你掌控。你像在 jax.jit 中一样指定输入的分片方式，但随后需要显式编写所有通信。而 `jax.jit` 为你提供程序的全局跨设备视图，`shard_map` 则为你提供每个设备的局部视图。

这是一个例子。试着推理这个函数的作用：<d-footnote>如果你想通过在 colab 中模拟网格来亲自尝试，可以使用以下单元格 `import jax; jax.config.update('jax_num_cpu_devices', 8)`</d-footnote>

```py
import jax
import jax.numpy as jnp

Explicit = jax.sharding.AxisType.Explicit

mesh = jax.make_mesh((2, 4), ('x', 'y'), (Explicit, Explicit))
jax.set_mesh(mesh)

x = jnp.arange(0, 512, dtype=jnp.int32, out_sharding=jax.P(('x', 'y')))

# This function will operate on 1/8th of the array.
@jax.shard_map(in_specs=jax.P(('x', 'y')), out_specs=jax.P())
def slice_and_average(x):
  assert x.shape == (512 // 8,)
  return jax.lax.pmean(x[:4], axis_name=('x', 'y'))

out = slice_and_average(x)
assert out.shape == (4,)
```

**这是做什么的？** `slice_and_average` 在每个 TPU 上运行，使用数组的 1/8，从中我们取出前 4 个元素并在整个网格上对它们进行平均。这意味着我们实际上是在执行 `mean(x[:4], x[64:68], x[128:132], …)`。这非常酷，因为在 JAX 中，否则很难表达这种操作。

**为什么要使用这种方法而不是 jax.jit？** 如果我们使用了 `jax.jit`，`slice_and_average` 将会看到数组的全局视图（完整的 `[512,]` 数组）。我们必须切出这个非均匀切片，然后进行平均，而 XLA 必须正确解释这一操作。XLA 可能会添加错误的通信或产生混淆。在这里，我们看到的是局部视图，并且只编写我们需要的通信。

**示例 [集合矩阵乘法]:** 为了举一个更现实的例子，假设我们想要实现模型并行，其中激活值最初是按模型分片的，即 A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D, F<sub>Y</sub>] -> Out[B<sub>X</sub>, F<sub>Y</sub>]。直觉上，我们会先对 A 进行 AllGather 操作，然后进行本地矩阵乘法：

1. A[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(A[B<sub>X</sub>, D<sub>Y</sub>])  
2. Out[B<sub>X</sub>, F<sub>Y</sub>] = A[B<sub>X</sub>, D] *<sub>D</sub> W[D, F<sub>Y</sub>]

令人遗憾的是，这很不好，因为它不允许我们将通信与计算重叠。通过使用如 [Wang 等人 2023](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959) 中所述的“集体矩阵乘法”可以实现它们的重叠。算法基本上如下：

* 对于每个 Y 分片，将 A 的本地块与 W 的本地块进行矩阵乘法运算，生成形状为 `[B / X, F / Y]` 的结果。同时，对 A 进行重排，以在本地获取下一个块，执行矩阵乘法，并将结果相加。

我们可以很轻松地通过 `jax.shard_map` 实现这一点：

```py
import functools

import jax
import jax.numpy as jnp
import numpy as np

Explicit = jax.sharding.AxisType.Explicit

# This is intended to run on a TPU v5e-8 runtime. If you can't get this,
# try setting jax.config.update('jax_num_cpu_devices', 8).
#
mesh = jax.make_mesh(axis_sizes=(2, 4), axis_names=('X', 'Y'), axis_types=(Explicit, Explicit))
jax.set_mesh(mesh)

B, D, F = 1024, 2048, 8192
A = jnp.arange(np.prod((B, D))).reshape((B, D))
W = jnp.arange(np.prod((D, F))).reshape((D, F))

A = jax.device_put(A, jax.P('X', 'Y'))
W = jax.device_put(W, jax.P(None, 'Y'))

@functools.partial(jax.jit, out_shardings=jax.P('X', 'Y'))
def matmul(lhs, rhs):
  return lhs @ rhs

def collective_matmul_allgather_lhs_contracting(lhs, rhs):
  # lhs is the looped operand; rhs is the local operand
  axis_size = jax.lax.axis_size('Y')  # axis_size = 4 for this example
  idx = jax.lax.axis_index('Y')

  chunk_size = lhs.shape[1]
  assert rhs.shape[0] % chunk_size == 0

  def f(i, carrys):
    accum, lhs = carrys
    rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
    # Matmul for a chunk
    update = lhs @ rhs_chunk
    # Circular shift to the left
    lhs = jax.lax.ppermute(
        lhs,
        axis_name='Y',
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)]
    )
    return accum + update, lhs

  accum = jnp.zeros((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype)
  accum = jax.lax.pcast(accum, ('X', 'Y'), to='varying')
  accum, lhs = jax.lax.fori_loop(0, axis_size - 1, f, (accum, lhs), unroll=True)

  # Compute the last chunk after the final permute to leave lhs in the state we found it
  i = axis_size - 1
  rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
  update = lhs @ rhs_chunk
  return accum + update

jit_sharded_f = jax.jit(jax.shard_map(
  collective_matmul_allgather_lhs_contracting,
  in_specs=(jax.P('X', 'Y'), jax.P(None, 'Y')), out_specs=jax.P('X', 'Y')))

shmapped_out = jit_sharded_f(A, W)
expected_out = matmul(A, W)

np.testing.assert_array_equal(shmapped_out, expected_out)
```

这非常不错！我们可以进行基准测试，发现它也快得多！[这里是](https://imgur.com/a/e9I6SrM) 使用默认 jit 矩阵乘法的性能分析，初始时有一个较大的阻塞 AllGather，耗时 311 微秒：

{% include figure.liquid path="assets/img/not-overlapped.png" class="img-fluid" %}

这里有一个耗时 244 微秒的版本[here's](https://imgur.com/a/21iy0Sv)。你可以看到性能分析中没有 AllGather。所有工作都是有用的！我们的 FLOPs 利用率也高得多。

{% include figure.liquid path="assets/img/overlapped.png" class="img-fluid" %}

还有一点值得注意的是，在收缩维度上不进行分片时，矩阵乘法的时间为 [224us](https://imgur.com/a/i3gNKfq)，因此在此处我们与不分片的基线非常接近。这是一个你可能会为了提高 TPU 利用率而进行的性能工程的很好例子。有关 `shard_map` 的更多示例，请参见 [这篇笔记](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html#example-1-all-gather-on-one-side)。

现在这里有几个有用的练习题，可以尝试使用 `jax.jit` 或 `shard_map` 来实现！

## 习题解答

这里有一些与 JAX 相关的随机问题。我之后还会添加更多。对于所有这些问题，你都需要一定数量的 TPU。Colab 不再提供 TPU v2-8 的切片，因此请使用 [Kaggle](https://www.kaggle.com/)（它仍然免费提供这些资源）或一个 8 核的 GCP 切片。<d-footnote> 如果你只是想在虚构的问题上模拟一个网格，你也可以使用 `import jax; jax.config.update('jax_num_cpu_devices', 8)`（需要 jax >= 0.4.27ish）在 CPU 上模拟 8 个设备，尽管这不会反映实际性能。</d-footnote> 从现在开始，我们将假设你有 N 个设备可用。

**问题 1:** 设 **A** 是一个形状为 float32[S<sub>X</sub>, D<sub>Y</sub>] 的激活数组，其中 `X * Y = N`。请执行以下操作：

1. 用 JAX 编写一个函数，计算每个 `(X, Y)` 分片内的平均值，即它返回一个大小为 [X, Y] 的数组，其中 `arr[i, j]` 是分片 `(i, j)` 的平均值。分别用 `jax.jit` 和 `shard_map` 实现。对每种方法进行性能分析，查看它们各耗时多久。是否添加了任何通信？*提示：不应该添加，但有时 XLA 会添加。*

2. 用 JAX 编写一个函数，使其在沿 X 的每个分片内进行某种平移时返回 `roll(x, shift, axis=0) - x`。我不会强迫你用 jax.jit 来实现这个功能，所以只需用 `shard_map` 来实现即可。

{% details 点击此处查看答案。 %}

**第一部分：** 这是第一部分的解决方案。请注意，为了实现 `jax.jit` 解决方案，我们必须进行相当复杂的形状重塑。

```py
import numpy as np

import jax
import jax.numpy as jnp

Auto = jax.sharding.AxisType.Auto

mesh = jax.make_mesh((4, 2), ('X','Y'), (Auto, Auto))

average_shmap = jax.shard_map(
    lambda x: x.mean(keepdims=True),
    mesh=mesh,
    in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
)

def average(x):
  X, Y = mesh.axis_sizes
  return x.reshape(X, x.shape[0] // X, Y, x.shape[1] // Y).mean(axis=(1, 3))

average_jit = jax.jit(average, out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))

x = jnp.arange(8 * 64 * 8, dtype=jnp.float32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))

y1 = average_shmap(x)
y2 = average_jit(x)

np.testing.assert_array_equal(y1, y2)
```

**第二部分：** 这是与第二部分类似的解决方案。`shard_map` 版本是您所要求的版本，但为了比较也包含了 `jax.jit` 版本。

```py
import numpy as np

import jax
import jax.numpy as jnp

import functools

Auto = jax.sharding.AxisType.Auto

mesh = jax.make_mesh((4, 2), ('X','Y'), (Auto, Auto))

def shift_shmap(x, shift: int):
  shmapped = jax.shard_map(
      lambda x: jnp.roll(x, shift, axis=0) - x,
      mesh=mesh,
      in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
  )
  return shmapped(x)

@functools.partial(jax.jit, static_argnames=['shift'], out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))
def shift_jit(x, shift: int):
  X, Y = mesh.axis_sizes
  reshaped = x.reshape(X, x.shape[0] // X, -1)
  return (jnp.roll(reshaped, shift, axis=1) - reshaped).reshape(x.shape[0], x.shape[1])

x = jnp.arange(8 * 64 * 8, dtype=jnp.float32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))

y1 = shift_shmap(x, 5)
y2 = shift_jit(x, 5)

np.testing.assert_array_equal(y1, y2)
```

{% enddetails %}

**问题 2:** 这里我们将一起构建一个基本的“混合专家”模型。令 **W**: float32[E<sub>X</sub>, D, F] 为一组 E 个“专家”矩阵。令 **A**: float32[S<sub>X</sub>, D]（我们的激活值），并令 **B**: int32[S<sub>X</sub>] 为一组“路由分配”，其中 B[i] 是一个在 `[0, E)` 范围内的整数，告诉我们想要用哪个矩阵来处理该激活值。我们希望用 JAX 编写一个函数，返回 `Out[i] = A[i] @ W[B[i]]`。

1. 让我们先完全忽略分片。使所有这些张量足够小，以便它们能放入一个设备中。编写该函数的本地实现。*确保你不要生成形状为 `[S, D, F]` 的数组！提示：有几种方法可以做到这一点。最简单的方法是遍历所有专家，将所有的 **A** 乘以 `W[e]`，并屏蔽未路由到专家 `e` 的词元。或者，你可以按专家对词元进行排序，放入形状为 `[E, S, D]` 的填充缓冲区中，进行一次批量矩阵乘法，屏蔽填充部分（为什么第二个维度需要大小为 S？）。`jax.lax.ragged_dot` 实现了不带填充的排序版本。*

2. 如果你只是 `jax.jit` 了上述方法，某些事情会发生。进行性能分析，看看它决定执行什么通信操作。需要多长时间？

3. 你会发现上述方法存在一个问题，即它很可能会将完整的激活值集合 **A** 本地收集起来，即 AllGather<sub>X</sub>([S<sub>X</sub>, D])。这不仅在通信方面代价高昂，如果无法在本地存储完整的激活值集合，还会在内存方面带来极大的开销。使用 `shard_map` 和显式通信来实现上述方法。

1. 首次尝试时，使用 `jax.lax.all_gather` 并按步骤 1 的方式重新排序可能最为简便。

2. 进行第二次尝试时，尽量避免生成大小为 `[E, S, D]` 的数组，即尝试使用 `jax.lax.all_to_all` 内部的 `jax.lax.while_loop` 以不规则的方式执行计算。这样可以避免生成完整的激活值，并减少对填充部分的算力浪费。与原始实现相比，这种方法能快多少？

4. 大多数 MoE 会路由到多个（k）专家，然后对结果进行平均。请重构上述内容以实现这一点。在此情况下，令 **B**: int32[S<sub>X</sub>, k] 表示路由到的 k 个专家。

{% details 点击此处查看（部分）答案。 %}

**第一部分：** 你在这里有很多选择。这里是最简单的选项，它只是通过掩码对专家进行迭代，且从不排序词元：

```py
def moe_local(W: jnp.ndarray, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    S, _ = A.shape
    E, _, F = W.shape

    def expert_forward(carry, e):
        output = carry  # [S, F]
        mask = (B == e)[:, None]  # [S, 1]
        expert_result = A @ W[e]  # [S, F] - this expert's transform of ALL tokens
        output = output + expert_result * mask  # Only keep results for assigned tokens
        return output, None

    output = jnp.zeros((S, F))
    output, _ = jax.lax.scan(expert_forward, output, jnp.arange(E))

    return output
```

这会浪费算力，因为每个专家都会处理每个词元。更高效的选择是按专家对词元进行排序，然后使用 `jax.lax.ragged_dot`，它会利用排序后的激活值以及每个专家的词元数量，在没有任何填充或掩码的情况下执行矩阵乘法。

**第三部分：** 我在这里只将伪代码草图展示出来（如果你有清晰的解决方案，欢迎添加）：

```py
chunk_size = 128
def matmul(W, x, B):
  i = 0
  outs = []
  x = # sort x according to assignments
  while (chunk := x[i:i+chunk_size]).any():
     chunk = all_to_all(chunk)
     outs.append(matmul_local(W, chunk))
     i += chunk_size
  return concat(outs)
```

基本思想是对数组的块进行迭代，对它们进行排序并执行 all_to_all 操作，然后进行本地 FLOPs 计算。

{% enddetails %}

**问题 3:** 上述集体矩阵乘法示例实际上对真实的大型语言模型（LLM）非常相关。让我们调整示例以实现完整的 Transformer 栈。

1. 作为练习，让我们从实现一个 AllReduce 集合矩阵乘法开始，即 A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F] -> Out[B<sub>X</sub>, F]。请注意，输出仅沿 X 分片，即在 Y 方向上是复制的。上述讨论了朴素算法，基本上只是本地矩阵乘法后接一个 AllReduce。尝试制作一个通信重叠的“集合”版本的此操作。*提示：在输出维度上进行分块，并可以自由使用 `jax.lax.psum`（即 AllReduce）*。*注意：由于 XLA 处理方式的原因，它可能实际上并不比基线更快。*

2. 上述 AllReduce 集合矩阵乘法的补充是 ReduceScatter 集合矩阵乘法，例如 Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W2[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]。这发生在 Transformer 中的下投影矩阵中。在 JAX 中实现一个集合、重叠版本的此操作。注意只传递你需要的最小数据量。*提示：在累积结果时尝试对其进行置换。*

3. 将主文中 AllGather 集合矩阵乘法与（2）中的 ReduceScatter 集合矩阵乘法组合成一个端到端的 Transformer 块，执行 In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]，并实现通信重叠。<d-footnote> 与之前一样，由于此处省略了一个非线性操作，我们不能先执行 $W_{in} \cdot W_{out}$。</d-footnote> 这比 `jax.jit` 实现快多少？

**问题 4:** 上述所有集合矩阵乘法都是单向的：它们只在一个方向上进行排列。将集合 AllReduce 矩阵乘法和集合 ReduceScatter 矩阵乘法重写为使用双向通信。这些方法能快多少？

### 第十部分到此结束。基本上就是这样！如需最终结论和进一步阅读，点击[这里](../conclusion)。
