# Documentation

Start with [the project README](../README.md) or [中文说明](../README.zh-CN.md). The detailed design notes are primarily in Chinese; the architecture overview is in English.

## Current implementation

| Document | Contents |
|---|---|
| [Refactor and migration](refactor.md) | Composition, runtime/reference separation and 0.2 API changes |
| [Architecture](design.md) | Page/block/row identities, address translation, ownership and execution order |
| [Validation](validation.md) | Current reproducible checks, recorded hardware and numerical results |
| [HiSparse alignment](hybrid-alignment.md) | Hot block leases, GPU row replacement and differences from vLLM |
| [Residency alignment](residency-alignment.md) | Clean-page unpin/reuse, hot triggers and two active tail pages |
| [Trace format](traces.md) | Dataset provenance, model activation capture, selection semantics and replay |
| [Page-based reference](page-cache.md) | The optional `--cache-layout pages` path |
| [Source notes](upstream.md) | Pinned vLLM, SGLang, nano-vLLM and Quest references |
| [Interview walkthrough](interview.md) | A guided explanation of the original partial-residency problem |

## Reports and development history

Small JSON reports live in [reports/](reports/). Dataset text, weights, large traces and new experiment outputs stay in the ignored `results/` directory.

- [Initial validation](history/initial-validation.md): the first page-based implementation.
- [Round 1 review](history/round1-review.md): real model traces and exact selection semantics.
- [Round 2 review](history/round2-review.md): page allocation, policy comparisons and lifetime fixes.

Historical reports describe the code and configurations used at the time. In particular, the early hybrid reports use one tail page; the current implementation uses two. Their block counts and timings must not be mixed.
