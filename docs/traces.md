# Trace 的含义、格式和局限

## 已实际验证的数据

`scripts/prepare_longbench.py` 从官方 `zai-org/LongBench` 的固定 revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672` 下载 archive，seed=42，选择 qasper:163、gov_report:163、repobench-p:327。原始 context 分别为 2,733 / 4,372 / 7,477 个 Qwen tokenizer tokens。

`model_trace.py capture` 使用预训练 Qwen2.5-0.5B-Instruct，常规完整 prefill，再按 dense greedy 轨迹 decode。默认采集层 2、7、15、23，每请求 8 步。为固定 trace 长度，采集器执行指定步数，不以 EOS 提前截断；运行引擎的正常生成则遵守 EOS。

记录 post-RoPE Q/K、V、新 token 的位置、模型 revision、dataset revision/row、原始 context hash、最终 prompt token IDs/hash 和 head+tail 截断。第一轮约 1K prompt；第二轮提高到 4K 上限，qasper 不再截断，其他两类仍保留显式截断记录。原始文本和 tensor trace 默认留在 `results/`，不作为代码分发的一部分；数据各来源的使用条件仍适用。

## 三种选择不能混称“真实 DSA top-k”

| 选择来源 | 使用什么信息 | 用途与边界 |
|---|---|---|
| `mean` | 页均值 K 和合并后的 Q | 初版近似基准，可能出现正负抵消 |
| `quest` | 页内 K min/max + 每头 Q | 可在 GPU 常驻小索引上计算；是 Quest 风格 GQA 适配，额外保留 sink/tail |
| `oracle_token` | 完整实际 QK score | 离线质量上界对照，读取完整 K 有成本，不能用作免费线上 indexer |
| `native` | 模型 indexer 实际导出的逻辑位置 | 不重新评分；目前仅验证了导入语义和构造测试，没有原生 DSA 权重采集结果 |

当前实测属于“真实 dense 模型激活 + 显式稀疏选择”。真实文本、RoPE、GQA、层间差异和时间复用都存在，但 learned DSA 的 indexer 分布、MLA 压缩布局与跨层 index sharing 不在该等价范围内。

## Tensor trace v1

使用 `torch.save` 保存只含 tensor、dict/list 和标量的对象，使用 `torch.load(weights_only=True)` 读取：

```python
{
    "schema_version": 1,
    "provenance": {"family": "dense_qwen_teacher", "native_sparse": False, ...},
    "prefills": [
        {"request": "r", "layer": 2, "k": K, "v": V},
        # K/V: [prefix_tokens, kv_heads, head_dim]
    ],
    "events": [
        {"request": "r", "layer": 2, "position": prefix_tokens,
         "q": Q, "k": new_K, "v": new_V},
        # Q: [query_heads, head_dim]; new_K/V: [1, kv_heads, head_dim]
    ],
}
```

每个请求每层的位置必须连续，所有被采集层每个 step 必须完整。当前 replay 要求同构 GQA head shape/dtype 和相同采样层集合。每个 query 的可见长度为 `position + 1`，包含已经追加的新 KV。

原生逻辑 top-k 可以加入 event：

```python
event["native_topk"] = positions.detach().cpu()
# [K] / [1,K]：共享选择；[query_heads,K]：每头独立选择
# -1 padding 被忽略，重复逻辑位置去重，其他非法或未来位置拒绝。
```

随后用 `scripts/replay_trace.py --selector native`。**导入 token ID 不等于支持 DeepSeek 的完整 MLA kernel**：数值回放仍要求上述 GQA Q/K/V 布局。压缩 MLA 的不同 K/V 维度、FP8 和特殊 attention 形式需要额外 backend，不能把原始 compressed payload 直接塞入当前格式。

回放现在默认 `--cache-layout hybrid`，按 trace 中最大的逐头 token 并集配置 `max_union_rows` 和固定 hot 容量。显式传入过小的 GPU 页数会因准入下限而拒绝；要重现前两轮 32 页的分窗结果，请加 `--cache-layout pages`。`scripts/compare_layouts.py` 对两条路径使用同一 GPU block 数、同一 frozen selection，并在计时前预热全部实际 row-kernel 形状。

在线接口还可直接传入 GPU `LogicalTopK(positions)`（共享 `[K]`），不经过 trace 的 CPU 校验/转换。resolver 在设备上去重和屏蔽非法位置，生成物理 row IDs；`Selection` 的多头结果则通过显式 union/head-columns 适配。这个接口是原生 top-k 的消费边界，不是新增了 learned indexer 权重。

## 如何公平比较存储策略

`prepare()` 在计时外冻结所有 selection 与 oracle 输出。`replay()` 将不同请求按 step 交错执行，每层追加对应的新 K/V，改变的只有 cache 容量和策略。必须对完全相同的 selected token 集合检查结果，不能让某策略通过漏掉 miss 或扩大选择来取得“加速”。

报告区分：

- **选择误差**：retained attention mass、相对于完整 attention 的输出 L2、同预算 oracle token recall。
- **存储误差**：paged output 与同一 selection oracle 的差异。
- **访问形态**：相邻 decode 步的每头 token Jaccard、物理页面并集 / 每头预算。
- **缓存行为**：attention 命中率、append 缺页、H2D bytes；二者分开统计。
- **层级时延**：append + attention 的串行回放时间，排除 selection/oracle；不是 LLM tokens/s。

固定 teacher 轨迹消除了策略间输入差异，也消除了稀疏输出对未来 query 的反馈。因此这些数字不能证明完整生成质量、原生 DSA 分布或在线调度吞吐。采样只有三个文本、四个层、八步，跨模型/长输出结论仍需要更大样本。
