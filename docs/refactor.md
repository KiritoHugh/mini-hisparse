# 0.2：默认运行路径与参考实验分离

旧实现中，`HybridKVCache(PagedKVCache)` 复用了 host backing 和事件管理，也继承了不属于 hybrid 的页级替换与窗口策略。模型 decode 还经由 `paged_attention()` 的类型判断转到 row attention。现在这两个依赖已移除。

## 当前结构

| 位置 | 负责什么 | 不负责什么 |
|---|---|---|
| `mini_hisparse/storage.py` / `HostStore` | 逻辑页、host credit、索引计费、增量镜像、事件和延迟回收 | 选择 victim、hot 容量、attention 窗口 |
| `mini_hisparse/allocator.py` / `DeviceAllocator` | 唯一 GPU tensor、block 所有权、generation、unpin/reuse | 页级 LRU/公平策略、选择哪些 token |
| `mini_hisparse/hybrid.py` / `HybridKVCache` | clean 回收、hot 租约、两页 tail、resident/source tables 与 row 生命周期 | 旧页级分窗和软配额策略 |
| `mini_hisparse/row_index.py` | logical token → resident/hot physical row，tags 与 row LRU | 申请 GPU blocks |
| `mini_hisparse/attention.py` | 等待 miss gather 后，直接消费物理 rows | 判断缓存具体类型并转发到其他实现 |
| `reference/` | 独立的页级缓存、窗口规划、分窗 attention 和 kernel | 作为 hybrid 的父类或默认依赖 |
| `experiments/` | benchmark、trace 采集/分析、同 selection 回放 | 正常模型执行 |

`HybridKVCache` 与 `reference.PagedKVCache` 是两个独立类，各自显式组合 `HostStore` 和 `DeviceAllocator`。`model.decode()` 只调用缓存的 `attention()` 接口；默认路径不需要导入 reference。CLI 为兼容现有实验保留 `--cache-layout pages` 和 `bench`，仅在显式选择时加载对应包。

HostStore 的 append 接口接受 `prepare_page` 与 `after_write` 两个明确回调：前者提供驻留位置，后者在镜像提交后运行策略。存储层不反向调用某个子类的隐式重写方法。

请求退休时，host 页和 caller-owned row tables 使用同一完成事件：`retire(keepalive=...)` 保持元数据存活，直到释放相应索引预算。这样物理对象生命周期与计费生命周期保持一致。

## 代码规模

按 Python 物理行数统计，含空行和注释，不含测试、脚本、文档和 JSON：

| 范围 | 重构前 | 重构后 |
|---|---:|---:|
| 默认主包 `mini_hisparse/` | 3,167 | 2,216 |
| 独立 `reference/` | 混在主包 | 498 |
| 独立 `experiments/` | 混在主包 | 595 |
| 所有实现合计 | 3,167 | 3,309 |

主包缩小约 30%，默认阅读路径更集中。合计略有增加，主要来自显式组件接口、独立参考配置与验证；没有把“移动目录”描述成删除了同等数量的实现。原有数值、异步生命周期和对照实验仍保留。

## Python API 迁移

常用入口保持简短：

```python
from mini_hisparse import CacheConfig, HotConfig, HybridKVCache, LogicalTopK
from mini_hisparse.attention import row_attention

cache = HybridKVCache(CacheConfig(device="cpu"), HotConfig(max_union_rows=16))
```

以下是 0.2 的模块和状态访问变化：

| 旧接口 | 新接口 |
|---|---|
| `mini_hisparse.cache.CacheConfig` | `mini_hisparse.config.CacheConfig`，或包顶层导入 |
| `mini_hisparse.cache.PagedKVCache` | `reference.cache.PagedKVCache` |
| 页级 `CacheConfig(chunk_pages=..., overlap=..., eviction_policy=...)` | `reference.cache.PagedConfig(...)` |
| `mini_hisparse.hybrid.row_attention` | `mini_hisparse.attention.row_attention` |
| `mini_hisparse.attention.paged_attention` | `reference.attention.paged_attention` |
| `mini_hisparse.attention.dense_attention / select_pages` | `mini_hisparse.selection` 中的同名函数 |
| `mini_hisparse.replay / traces / benchmark` | `experiments` 包中的同名模块 |
| `cache.pages / requests / host / stats / retired` | `cache.storage` 中的同名字段 |
| `cache.host_kv() / synchronize()` | `cache.storage` 中的同名方法 |

`cache.write/reserve/collect/release_request/snapshot` 仍在缓存对象上。Hybrid 的 `cache.page_state()` 继续提供额外 residency 状态；通用诊断位于 `cache.storage.page_state()`。原 `mini_hisparse.cache` 模块已移除，没有用兼容导入重新引入旧路径。

## 验证

- 原有 93 项 CPU/CUDA 测试保留并通过；新增依赖边界测试后，共 **94 项通过**。
- [依赖边界测试](../tests/test_runtime_boundary.py) 在子进程中禁止导入 `reference` 和 `experiments`，默认 CLI 仍完成一次 prefill 和两步 decode。
- Trace 单元测试同时覆盖独立 reference 与 hybrid，对 Quest、mean、token oracle 使用相同选择校验输出。
- [重构后的真实 Qwen 验证](reports/refactor-pretrained-parity.json)：32 个 token 与 Hugging Face 完全一致，prefill=1；resident/hot hits、1,536 row misses 和 786,432 B H2D 与重构前一致。
- [重构后的真实 trace 回放](reports/refactor-replay.json)：96 个事件、最大 BF16 存储误差 0.001953125；resident/hot hits、3,696 row misses 和 1,892,352 B H2D 与重构前一致。保留原四页 warmup，避免改变初始 hot 内容后把命中率变化误归因于重构。

实验仍受 [验证文档](validation.md) 中的范围限制。本轮不增加 batching、DSA/MLA、chunked prefill 或 fused resolver，也不根据单次回放时间宣称性能提升。
