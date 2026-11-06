# 文档导航 / Documentation

从 [项目介绍](../README.zh-CN.md) 或 [English README](../README.md) 开始。中文教程按设计 → 代码 → 运行 → 测评组织；[Architecture](design.md) 和 [Validation](validation.md) 提供英文概览。

**独立面试答卷：[在线设计文档](partial-residency-design.md) · [PDF 版本](partial-residency-design.pdf)。** 从约束、页状态、共享池配额到 decode 时序与效率优化自成体系，以同一容量示例串联七张图和短伪代码，无需先阅读项目文档。

## 第一次阅读：按这条路线

| 顺序 | 文档 | 读完能回答什么 |
|---:|---|---|
| 1 | [图解设计](visual-guide.md) | page / block / row 是什么？clean 回收、hot 分配、tail 和 stream 如何协作？ |
| 2 | [代码调用链](code-walkthrough.md) | 从 Engine 到 resolver/kernel，实际执行经过哪些函数？ |
| 3 | [三个可运行例子](examples.md) | 怎样检查索引、观察多请求竞争、用真实 Qwen 询问本地文档？ |
| 4 | [当前 baseline 对比](benchmarks.md) | 与全驻留/页缓存相比何时获益，何时更慢，结果怎样复现？ |

如果时间有限：面试讲解先看 [三问详答、进展条件与改进评估](interview.md)；调试地址先跑 [inspect_indices.py](../examples/inspect_indices.py)；检查数值与生命周期看 [Validation](validation.md)。

## 按问题查设计细节

| Document | Contents |
|---|---|
| [Architecture](design.md) | Page/block/row identities, address translation, ownership and execution order |
| [Validation](validation.md) | Current reproducible checks, recorded hardware and numerical results |
| [Hot 容量敏感性](hot-capacity.md) | 固定总容量比较 1U / 1.5U / 2U，解释最低工作集与性能取舍 |
| [HiSparse alignment](hybrid-alignment.md) | Hot block leases, GPU row replacement and differences from vLLM |
| [Residency alignment](residency-alignment.md) | Clean-page unpin/reuse, hot triggers and two active tail pages |
| [Trace format](traces.md) | Dataset provenance, model activation capture, selection semantics and replay |
| [Page-based reference](page-cache.md) | The optional `--cache-layout pages` path |
| [Source notes](upstream.md) | Pinned vLLM, SGLang, nano-vLLM and Quest references |
| [Refactor and migration](refactor.md) | Composition, runtime/reference separation and 0.2 API changes |

## Reports and development history

Small JSON reports live in [reports/](reports/). Current comparisons are [layout-quest.json](reports/layout-quest.json) and [layout-token.json](reports/layout-token.json); [assets/](assets/) contains the generated figure. Dataset text, weights, large traces and local experiment outputs stay in the ignored `results/` directory.

- [Initial validation](history/initial-validation.md): the first page-based implementation.
- [Round 1 review](history/round1-review.md): real model traces and exact selection semantics.
- [Round 2 review](history/round2-review.md): page allocation, policy comparisons and lifetime fixes.

Historical reports describe the code and configurations used at the time. In particular, the early hybrid reports use one tail page; the current implementation uses two. Their block counts and timings must not be mixed.
