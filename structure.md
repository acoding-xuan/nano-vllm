# Nano-vLLM 代码学习笔记

Nano-vLLM 是一个从零实现的轻量级 vLLM 推理框架。它的目标不是覆盖完整 vLLM 的所有工程能力，而是用较少代码把离线推理里最关键的机制串起来：请求调度、PagedAttention 风格的 KV cache、prefix cache、Tensor Parallelism、CUDA Graph、FlashAttention 以及采样。

这份笔记按照当前代码库重新整理，重点解释代码中真实存在的模块、数据流和关键实现。

## 1. 总体架构

Nano-VLLM 的主流程可以理解为：

```text
用户 prompts
  -> LLM.generate
  -> LLMEngine.add_request
  -> tokenizer 编码
  -> Sequence
  -> Scheduler.schedule
  -> ModelRunner.run
  -> Qwen3ForCausalLM.forward
  -> Attention / KV cache / FlashAttention
  -> lm_head 得到 logits
  -> Sampler 采样 token
  -> Scheduler.postprocess
  -> tokenizer 解码
  -> 返回生成结果
```

核心模块如下：

- `LLMEngine`：推理入口，负责创建配置、tokenizer、scheduler、model runner，并在 `generate` 中驱动整个推理循环。
- `Sequence`：单条请求的状态对象，保存 prompt token、已生成 token、调度状态、KV cache block table 等信息。
- `Scheduler`：调度器，维护 `waiting` 和 `running` 两个队列，决定每一步执行 prefill 还是 decode。
- `BlockManager`：KV cache 块管理器，负责 block 分配、释放、prefix cache 命中和引用计数。
- `ModelRunner`：模型执行器，负责模型加载、KV cache 分配、prefill/decode 输入准备、CUDA Graph 捕获与执行、TP 多进程通信。
- `layers`：模型基础层，包括 attention、linear、embedding、sampler、RMSNorm、RoPE 等。
- `models/qwen3.py`：当前主要模型结构实现，包含 Qwen3 attention、MLP、decoder layer、model 和 causal LM wrapper。

架构图：

<p align="center">
<img src="./assets/overall_architecture.jpg">
</p>

## 2. 配置与启动

配置定义在 `nanovllm/config.py`：

```python
@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    kvcache_block_size: int = 256
```

几个重要参数：

- `max_num_batched_tokens`：单次 prefill 最多处理多少 token。它限制 prompt 阶段的批量大小，也支持 chunked prefill。
- `max_num_seqs`：最多同时处理多少条序列。
- `max_model_len`：模型允许的最大序列长度，初始化时会和 HuggingFace 配置里的 `max_position_embeddings` 取较小值。
- `gpu_memory_utilization`：用于估算 KV cache 可占用的显存比例。
- `tensor_parallel_size`：张量并行规模，范围限制为 1 到 8。
- `enforce_eager`：是否强制 eager 执行。为 `False` 时会尝试捕获 CUDA Graph。
- `kvcache_block_size`：KV cache 的块大小，代码中要求是 256 的倍数。

`LLM` 本身只是继承 `LLMEngine`：

```python
class LLM(LLMEngine):
    pass
```

因此主要逻辑都在 `LLMEngine` 中。

## 3. LLMEngine：用户接口与推理循环

`LLMEngine.__init__` 做了几件事：

1. 从传入参数中筛选出 `Config` 支持的字段，创建配置对象。
2. 将 `Sequence.block_size` 设置为 `config.kvcache_block_size`。
3. 如果 `tensor_parallel_size > 1`，用 `multiprocessing.spawn` 启动额外 rank 的 `ModelRunner`。
4. 在主进程创建 rank 0 的 `ModelRunner`。
5. 加载 tokenizer，并把 tokenizer 的 eos token 写入配置。
6. 创建 `Scheduler`。
7. 注册退出清理逻辑。

`generate` 是对外主要接口。它先把每个 prompt 加入 scheduler，然后不断执行 `step`，直到 scheduler 中没有 waiting/running 请求。

```text
generate
  -> add_request
  -> while not is_finished:
       step
       收集完成序列
  -> 按 seq_id 排序
  -> tokenizer.decode
```

`step` 的职责更像一次推理 tick：

1. `scheduler.schedule()` 选出当前要执行的序列，并判断是 prefill 还是 decode。
2. `model_runner.call("run", seqs, is_prefill)` 执行模型前向和采样。
3. `scheduler.postprocess(...)` 更新序列状态、追加 token、释放已完成请求的 KV cache。
4. 返回已经完成的请求输出。

这里的关键点是：`LLMEngine` 不直接关心 block 如何分配，也不直接关心模型如何组织输入，它只是连接 scheduler 和 model runner。

## 4. Sequence：请求状态对象

`Sequence` 表示一条生成请求，保存从 prompt 到 completion 的完整状态。

重要字段：

- `seq_id`：请求编号，用于最后按原始顺序恢复输出。
- `status`：请求状态，可能是 `WAITING`、`RUNNING`、`FINISHED`。
- `token_ids`：当前完整 token 序列，包含 prompt 和已生成 token。
- `last_token`：最近一个 token，decode 阶段只需要喂入这个 token。
- `num_prompt_tokens`：prompt 长度。
- `num_cached_tokens`：已经写入 KV cache 的 token 数。
- `block_table`：该序列占用的 KV cache block id 列表。
- `temperature`、`max_tokens`、`ignore_eos`：采样和停止条件。

几个属性很重要：

- `num_completion_tokens`：当前已经生成的 token 数。
- `completion_token_ids`：只取生成部分，不包含 prompt。
- `num_blocks`：当前序列需要多少个 KV cache block。
- `last_block_num_tokens`：最后一个 block 中实际有多少 token。

`block(i)` 会返回第 `i` 个 block 对应的 token 片段。BlockManager 正是通过这个方法计算 block 内容的 hash，从而做 prefix cache。

## 5. Scheduler：prefill 优先的调度策略

Scheduler 内部维护两个队列：

- `waiting`：尚未完成 prefill，或被抢占后需要重新进入调度的请求。
- `running`：已经完成 prefill，正在 decode 的请求。

调度入口是 `schedule()`。它分两段：

### 5.1 Prefill 阶段

调度器会优先从 `waiting` 队列中取请求做 prefill。

核心逻辑：

1. 只要还有 waiting 请求，并且当前 batch 序列数没超过 `max_num_seqs`，就尝试调度。
2. 通过 `max_num_batched_tokens` 限制本轮最多 prefill 的 token 数。
3. 如果请求还没有 block table，说明是首次调度，需要调用 `BlockManager.can_allocate/allocate`。
4. 如果能命中 prefix cache，则 `num_cached_tokens` 会提前增加，本轮只需要计算 cache miss 的后续 token。
5. 若剩余 token 数超过本轮容量，允许第一个请求做 chunked prefill。
6. 当一个请求的 prompt 全部 prefill 完成后，把它从 `waiting` 移到 `running`。

只要本轮成功调度了任何 prefill 请求，`schedule()` 就会直接返回 `(scheduled_seqs, True)`。也就是说代码采用的是 prefill-first 策略。

### 5.2 Decode 阶段

当没有 waiting 请求可 prefill 时，调度器才进入 decode。

decode 每条 running 序列本轮只调度 1 个 token：

```python
seq.num_scheduled_tokens = 1
seq.is_prefill = False
```

decode 前会调用 `block_manager.can_append(seq)` 判断如果追加一个 token 是否需要新 block，以及当前是否有空闲 block。若资源不足，会执行 preempt：

```python
def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    seq.is_prefill = True
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
```

抢占的含义是：

- 把请求从 running 退回 waiting；
- 释放它当前占用的 KV cache；
- 后续再重新 prefill；
- 如果前缀 block 已经被 hash 记录，则之后可能通过 prefix cache 找回部分 KV cache。

### 5.3 Postprocess

模型返回 token 后，`postprocess` 会：

1. 调用 `block_manager.hash_blocks(seq)` 记录可缓存 block 的 hash。
2. 增加 `seq.num_cached_tokens`。
3. 如果是 chunked prefill 且 prompt 还没 prefill 完，暂时不追加新 token。
4. 否则把采样得到的 token 追加到序列。
5. 如果 token 是 eos，或生成长度达到 `max_tokens`，则标记 `FINISHED` 并释放 KV cache。

注意：当前工作区里的 `scheduler.py` 和 `block_manager.py` 存在一处接口不一致。`scheduler.py` 中把 `can_allocate(seq)` 的返回值当作 `num_cached_blocks` 使用，并调用 `allocate(seq, num_cached_blocks)` 与 `hash_blocks(seq)`；但当前 `block_manager.py` 里 `can_allocate(seq)` 返回的是布尔值，`allocate` 只接收 `seq`，也没有单独的 `hash_blocks` 方法。这说明这两个文件可能来自不同修改阶段。阅读调度逻辑时，可以按 `scheduler.py` 理解“期望接口”，阅读 prefix cache 细节时，可以按当前 `block_manager.py` 理解“已有实现”。

## 6. BlockManager：PagedAttention 与 Prefix Cache

KV cache 被切成固定大小的 block。每条序列不直接持有连续 KV tensor，而是持有一个 `block_table`：

```text
seq.block_table = [block_id_0, block_id_1, block_id_2, ...]
```

真正的 KV cache tensor 在 `ModelRunner.allocate_kv_cache` 中分配，block table 只记录逻辑 token block 到物理 KV block 的映射。

### 6.1 Block 数据结构

每个 block 保存：

- `block_id`：物理 block 编号。
- `ref_count`：引用计数，用于 prefix cache 共享。
- `hash`：该 block 内容的 hash。
- `token_ids`：该 block 对应的 token 内容，用于校验 hash 命中是否真实有效。

### 6.2 链式 hash

BlockManager 使用链式 hash：

```python
hash(current_block_tokens, previous_block_hash)
```

这样即使两个 block 的 token 内容相同，只要前缀不同，hash 也不同。它表达的是“从序列开头到当前 block 的完整前缀状态”，而不是单独一个 block 的局部内容。

只有完整 block 才适合加入 prefix cache。不完整 block 的 hash 会设为 `-1`，因为它后续还会继续写入 token。

### 6.3 Prefix cache 命中

分配 block 时，如果某个完整 block 的链式 hash 已经存在，并且保存的 `token_ids` 也完全一致，就可以认为命中 prefix cache：

- 命中后不需要重新计算该 block 对应 token 的 KV；
- 如果 block 正在被其他序列使用，则增加引用计数；
- 如果 block 当前空闲，也可以重新分配出来使用；
- `seq.num_cached_tokens` 会增加一个 block size。

一旦遇到 cache miss，后续 block 都必须重新分配。原因是链式 hash 依赖前缀，只要某个位置 miss，后面的 KV 也不能直接复用。

### 6.4 Decode 时追加 block

decode 每次追加一个 token。BlockManager 需要判断是否跨越 block 边界：

- 如果新 token 会开启一个新 block，需要提前分配空闲 block。
- 如果追加后刚好填满最后一个 block，就计算该 block 的 hash，加入 prefix cache 映射。
- 如果最后一个 block 仍未满，则继续使用原 block，不加入 prefix cache。

## 7. ModelRunner：模型执行与多进程 TP

`ModelRunner` 是实际跑模型的模块。初始化流程：

1. 初始化 `torch.distributed`，后端是 `nccl`。
2. 设置当前 CUDA device 为 rank id。
3. 设置默认 dtype 和默认 device。
4. 创建 `Qwen3ForCausalLM`。
5. 调用 `load_model` 加载 safetensors 权重。
6. 创建 `Sampler`。
7. `warmup_model()` 预热模型，估算显存峰值。
8. `allocate_kv_cache()` 分配 KV cache。
9. 如果未设置 `enforce_eager`，调用 `capture_cudagraph()`。
10. 如果 TP 大于 1，rank 0 创建共享内存，其他 rank 进入 `loop()` 等待调用。

### 7.1 TP 多进程调用

当 `tensor_parallel_size > 1` 时，每个 rank 都有一个 ModelRunner。

- rank 0 在主进程中接收 Scheduler 传来的 `seqs`。
- rank 0 通过 `SharedMemory` 写入方法名和参数，例如 `("run", seqs, is_prefill)`。
- 其他 rank 通过 `Event` 被唤醒，从共享内存读取任务。
- 每个 rank 都执行相同方法。
- 只有 rank 0 返回采样 token。

这样做的好处是：上层 LLMEngine 只需要和 rank 0 通信，张量并行细节被封装在 ModelRunner 和各并行层内部。

### 7.2 KV cache 显存分配

`allocate_kv_cache` 会根据当前 GPU 空闲显存估算可分配多少 KV block：

```python
block_bytes = (
    2
    * num_hidden_layers
    * block_size
    * num_kv_heads_per_rank
    * head_dim
    * dtype.itemsize
)
```

KV cache 形状是：

```text
[
  2,                    # K 和 V
  num_hidden_layers,     # 层数
  num_kvcache_blocks,    # 物理 block 数
  block_size,            # 每个 block 的 token 数
  num_kv_heads_per_rank, # 当前 rank 负责的 KV heads
  head_dim
]
```

分配完成后，ModelRunner 会遍历模型模块，把每一层 Attention 中的 `k_cache` 和 `v_cache` 指向对应层的 KV cache 切片。

## 8. Prefill 与 Decode 的输入准备

ModelRunner 的 `run` 会根据阶段选择不同准备逻辑：

```python
input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
```

### 8.1 Prefill 输入

`prepare_prefill` 会构造：

- `input_ids`：本轮需要 prefill 的 token。
- `positions`：每个 token 的位置编码下标。
- `cu_seqlens_q`：FlashAttention varlen q 的累积长度。
- `cu_seqlens_k`：FlashAttention varlen k 的累积长度。
- `max_seqlen_q`：本 batch 中最大 query 长度。
- `max_seqlen_k`：本 batch 中最大 key 长度。
- `slot_mapping`：每个 token 应该写入 KV cache 的物理 slot。
- `block_tables`：prefix cache 场景下提供历史 KV block 映射。

如果 `cu_seqlens_k[-1] > cu_seqlens_q[-1]`，说明存在 prefix cache：key/value 的总上下文比本轮 query 更长，此时 attention 需要从 KV cache 中读取已缓存前缀。

### 8.2 Decode 输入

decode 阶段每条序列只输入最后一个 token：

- `input_ids.append(seq.last_token)`
- `positions.append(len(seq) - 1)`
- `context_lens.append(len(seq))`
- `slot_mapping` 指向新 token 写入 KV cache 的物理位置。
- `block_tables` 提供每条序列完整上下文对应的 block 映射。

decode 的输入形状和执行模式更稳定，因此 CUDA Graph 主要用于 decode。

## 9. Context：把调度信息传给 Attention

`ModelRunner.prepare_prefill/prepare_decode` 最后都会调用 `set_context(...)`。

这个 context 不是模型参数，而是每次前向时 attention 层需要的一组运行时信息。Attention 层通过 `get_context()` 读取：

- 当前是 prefill 还是 decode；
- varlen attention 的累积长度；
- KV cache 写入位置 `slot_mapping`；
- decode 的 `context_lens`；
- 每条序列的 `block_tables`。

这种设计避免了把大量调度参数一层层传进模型 forward。模型结构仍然只接收 `input_ids` 和 `positions`，attention 层内部再读取全局 context。

## 10. Attention 与 KV cache 写入

Attention 实现在 `nanovllm/layers/attention.py`。

每次 attention 前向时，如果该层已经绑定了 KV cache，就会先调用 `store_kvcache`：

```python
store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
```

`store_kvcache` 使用 Triton kernel，把当前 token 的 K/V 写入物理 cache slot。`slot_mapping` 中为 `-1` 的位置会被跳过，这也配合了 CUDA Graph 中预分配最大 batch 后只填有效位置的做法。

prefill 和 decode 使用不同 FlashAttention 接口：

- prefill：`flash_attn_varlen_func`
- decode：`flash_attn_with_kvcache`

prefill 时，如果有 prefix cache，`k` 和 `v` 会直接切换为 `k_cache` 和 `v_cache`，并通过 `block_table` 让 FlashAttention 找到历史 KV。

decode 时，query 只有当前 token，但 key/value 来自完整 KV cache，因此使用 `flash_attn_with_kvcache`。

## 11. CUDA Graph：为什么只用于 decode

CUDA Graph 的作用是减少大量小 kernel launch 的 CPU overhead。它会先捕获一次固定计算图，之后相同形状和相同内存地址的计算可以直接 replay。

Nano-VLLM 中由 `capture_cudagraph()` 捕获 decode 图。

关键点：

- `max_bs = min(max_num_seqs, 512)`，最多捕获到 batch size 512。
- `graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))`。
- 对每个 batch size 都准备一份 graph。
- 捕获时预先创建固定地址的 tensor：`input_ids`、`positions`、`slot_mapping`、`context_lens`、`block_tables`、`outputs`。
- replay 前把真实输入 copy 到这些固定 tensor 中。

实际运行时：

```python
graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
graph_vars["input_ids"][:bs] = input_ids
graph_vars["positions"][:bs] = positions
...
graph.replay()
```

也就是说，真实 batch size 如果是 23，会选择已捕获的 32。多出来的位置通过 `slot_mapping.fill_(-1)` 等方式避免产生实际写入影响。

为什么主要用于 decode：

- decode 每条序列每步只处理一个 token，计算图形状更稳定；
- prefill 的 prompt 长度差异大，varlen attention 参数变化大；
- prefill 通常更偏计算密集，kernel launch overhead 占比相对较小；
- decode 是大量小步循环，CUDA Graph 收益更明显。

为什么倒序捕获：

- 大 batch 的图通常需要更多显存；
- 第一次捕获后保存 `graph_pool`；
- 后续较小 batch 的图复用同一个 graph memory pool；
- 这样可以减少捕获过程中重复分配显存的开销，并让显存布局更稳定。

## 12. Sampler：temperature 与 Gumbel-max

采样逻辑在 `nanovllm/layers/sampler.py`：

```python
logits = logits.float().div_(temperatures.unsqueeze(dim=1))
probs = torch.softmax(logits, dim=-1)
sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
```

这里不是直接 `torch.multinomial`，而是使用 Gumbel-max 等价采样技巧：

- 先根据 temperature 缩放 logits；
- softmax 得到概率；
- 用指数分布噪声做变换；
- 取 argmax 得到采样 token。

`Sampler.forward` 上有 `@torch.compile`，说明采样部分也希望通过 PyTorch 编译降低开销。

## 13. Qwen3 模型结构

模型入口是 `Qwen3ForCausalLM`：

```text
Qwen3ForCausalLM
  -> Qwen3Model
       -> VocabParallelEmbedding
       -> 多层 Qwen3DecoderLayer
       -> RMSNorm
  -> ParallelLMHead
```

每个 DecoderLayer：

```text
input_layernorm
  -> self_attn
  -> post_attention_layernorm
  -> mlp
```

Attention：

```text
hidden_states
  -> qkv_proj
  -> split q/k/v
  -> q_norm/k_norm
  -> rotary_emb
  -> Attention(q, k, v)
  -> o_proj
```

MLP：

```text
x
  -> gate_up_proj
  -> SiluAndMul
  -> down_proj
```

代码里把 `gate_proj` 和 `up_proj` 合并成 `MergedColumnParallelLinear`，减少一次线性层调度与权重管理复杂度。

权重加载时，`Qwen3ForCausalLM.packed_modules_mapping` 负责把 HuggingFace 权重名映射到合并后的模块：

```python
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}
```

## 14. Tensor Parallelism：列切分、行切分与通信

Nano-VLLM 的 TP 实现主要在 `layers/linear.py` 和 `layers/embed_head.py`。

### 14.1 ColumnParallelLinear

`ColumnParallelLinear` 按输出维度切分权重：

```text
完整权重 W: [out_features, in_features]
每个 rank: [out_features / tp_size, in_features]
```

每个 rank 计算一部分输出通道，不需要立即通信。

典型使用：

- attention 的 `qkv_proj`
- MLP 的 `gate_up_proj`

### 14.2 RowParallelLinear

`RowParallelLinear` 按输入维度切分权重：

```text
完整权重 W: [out_features, in_features]
每个 rank: [out_features, in_features / tp_size]
```

每个 rank 得到部分矩阵乘结果，最后需要 `dist.all_reduce(y)` 把各 rank 的部分结果相加。

典型使用：

- attention 的 `o_proj`
- MLP 的 `down_proj`

### 14.3 为什么先列切再行切

连续两层 Linear 在 TP 下通常成对设计：

```text
ColumnParallelLinear -> RowParallelLinear
```

第一层按输出维度切分，每个 rank 得到局部 hidden/intermediate 表示。第二层按输入维度切分，刚好消费这个局部表示。这样中间激活不需要 all-gather，只在第二层输出时做一次 all-reduce。

这就是 Megatron-LM 中常见的做法：升维层列并行，降维层行并行，把通信推迟到 block 输出位置。

### 14.4 QKVParallelLinear

`QKVParallelLinear` 是列并行的特殊形式。它把 q、k、v 三个投影合并到一个线性层中，同时按 head 维度切分：

```text
output_size = (num_q_heads + 2 * num_kv_heads) * head_dim
```

每个 rank 负责一部分 q heads 和 kv heads。权重加载时根据 `loaded_shard_id` 区分 q/k/v，并拷贝到合并参数的对应位置。

### 14.5 词表并行

`VocabParallelEmbedding` 按 vocab 维度切分 embedding：

```text
rank 0: vocab [0, N/tp)
rank 1: vocab [N/tp, 2N/tp)
...
```

forward 时：

1. 每个 rank 判断输入 token 是否落在自己的 vocab 范围。
2. 不属于当前 rank 的 token 被 mask 掉。
3. 当前 rank 查自己的 embedding 表。
4. 多个 rank 的 embedding 结果通过 `all_reduce` 相加。

`ParallelLMHead` 复用这一套词表切分。不同的是输出 logits 时：

- 每个 rank 只算自己 vocab shard 的 logits；
- rank 0 gather 所有 rank 的 logits；
- 拼接后得到完整 vocab logits；
- 只有 rank 0 做采样并返回 token。

## 15. 权重加载

权重加载逻辑在 `nanovllm/utils/loader.py`。整体思路是：

1. 遍历模型目录中的 safetensors 文件。
2. 读取每个权重名和 tensor。
3. 根据模型的 `packed_modules_mapping` 判断是否需要加载到合并模块。
4. 找到目标参数上的 `weight_loader`。
5. 由各层自己的 `weight_loader` 负责切分和拷贝。

这种设计的好处是：loader 不需要知道每种并行层怎么切权重，切分规则封装在参数所属模块里。

## 16. 一次请求的完整生命周期

下面用一条请求串起所有模块：

1. 用户调用 `llm.generate(["hello"], sampling_params)`。
2. `LLMEngine.add_request` 使用 tokenizer 把 prompt 编码成 token ids。
3. 创建 `Sequence`，初始状态是 `WAITING`。
4. Scheduler 把 sequence 放入 `waiting`。
5. 第一次 `step` 时，Scheduler 进入 prefill。
6. BlockManager 为 sequence 分配 block，并尝试 prefix cache 命中。
7. ModelRunner 准备 prefill 输入，构造 varlen attention context。
8. 模型 forward 时，每层 attention 把 K/V 写入 KV cache。
9. lm_head 得到 logits，Sampler 采样出第一个生成 token。
10. Scheduler postprocess，把 token append 到 sequence。
11. 后续没有 waiting 请求时，Scheduler 进入 decode。
12. 每次 decode 只输入 `last_token`，attention 通过 block table 读取完整历史 KV。
13. 如果启用 CUDA Graph，decode 会选择合适 batch size 的 graph replay。
14. 每步采样一个 token 并 append。
15. 遇到 eos 或达到 `max_tokens` 后，Sequence 标记为 `FINISHED`。
16. BlockManager 释放该 sequence 的 block 引用。
17. LLMEngine 收集 completion token ids，按 seq_id 排序并 decode 成字符串。

## 17. 代码阅读建议

建议按下面顺序读代码：

1. `nanovllm/llm.py`：确认 `LLM` 只是 `LLMEngine` 的别名式封装。
2. `nanovllm/engine/llm_engine.py`：理解用户请求如何进入系统，以及主循环如何驱动 scheduler/model runner。
3. `nanovllm/engine/sequence.py`：理解请求状态和 block table。
4. `nanovllm/engine/scheduler.py`：理解 prefill-first、chunked prefill、decode、preempt。
5. `nanovllm/engine/block_manager.py`：理解 KV block、prefix cache、引用计数。
6. `nanovllm/engine/model_runner.py`：理解输入准备、KV cache 分配、CUDA Graph、TP 多进程。
7. `nanovllm/layers/attention.py`：理解 K/V 写 cache、prefill/decode 两套 FlashAttention。
8. `nanovllm/layers/linear.py`：理解 TP 下列并行和行并行。
9. `nanovllm/layers/embed_head.py`：理解 vocab parallel embedding 和 logits gather。
10. `nanovllm/models/qwen3.py`：把模型结构和上面的并行层对应起来。

## 18. 关键理解总结

- Nano-VLLM 的本质是一套围绕 KV cache 组织的推理调度系统。
- Prefill 负责把 prompt 写入 KV cache，decode 负责基于 KV cache 一步步生成。
- Sequence 保存逻辑状态，BlockManager 保存逻辑 block 到物理 block 的管理关系，ModelRunner 保存真正的 KV tensor。
- Prefix cache 的关键是完整 block 的链式 hash，以及 block 引用计数。
- Scheduler 采用 prefill-first 策略，有 waiting 请求时优先 prefill，没有 waiting 请求时才 decode。
- Decode 阶段形状稳定，因此更适合 CUDA Graph。
- Tensor Parallelism 中，升维投影用列并行，降维投影用行并行，中间不 all-gather，只在需要合并输出时 all-reduce。
- rank 0 是控制面入口，其他 rank 通过共享内存接收任务；模型层内部用 `torch.distributed` 完成张量通信。
- Attention 层通过全局 context 获取调度信息，从而让模型 forward 接口保持简洁。
