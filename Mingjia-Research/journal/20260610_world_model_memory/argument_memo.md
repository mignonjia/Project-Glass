# World-Model Memory Argument

Date: 2026-06-10

## Bottom line

`73 TB/s` aggregate all-to-all is not the headline advantage. Current NVIDIA
NVL72 systems already advertise `130 TB/s` aggregate NVLink bandwidth. The
publishable claim should be:

> Glass does not beat NVL72 by making all-to-all faster; it wins when extra
> memory makes all-to-all unnecessary for long-horizon world-model state.

For world models and long video generation, the bottleneck is often not just
how fast one collective runs. The real bottleneck is how much rollout state can
stay resident near compute before the system is forced into sequence/context
parallelism, offload, recomputation, KV compression, or state dropping.

## Current NVIDIA reference points

As of June 2026, public NVIDIA pages give these relevant numbers:

| System | Relevant public number | Implication |
|---|---:|---|
| H100 SXM | `80 GB`, `3.35 TB/s` HBM, `900 GB/s` NVLink | Strong compute, but single-GPU memory is far below long-video/world-model state needs. |
| H200 SXM | `141 GB`, `4.8 TB/s` HBM, `900 GB/s` NVLink | Better memory, still fundamentally GPU-attached memory. |
| DGX B200 | `1.44 TB` total GPU memory, `64 TB/s` HBM bandwidth, `14.4 TB/s` aggregate NVLink | Strong 8-GPU node, not a memory-capacity platform for many long rollouts. |
| GB200 NVL72 | `72` Blackwell GPUs, `13.4 TB` HBM3E, `576 TB/s` GPU memory bandwidth, `130 TB/s` NVLink | This is the key baseline. Its all-to-all fabric is faster than `73 TB/s`. |
| GB300 NVL72 | `37 TB` fast memory, `20 TB` GPU memory, `576 TB/s` GPU memory bandwidth, `130 TB/s` NVLink | Current stronger NVL72-class reference; still GPU-attached memory. |

So the glass argument must not be "73 TB/s is faster." A safer comparison:

- `73 TB/s / 72 = ~1.0 TB/s` average per GPU if interpreted as rack-aggregate
  injection bandwidth.
- NVL72 advertises `130 TB/s / 72 = ~1.8 TB/s` per GPU, matching the public
  `1.8 TB/s GPU-to-GPU interconnect` language.
- Therefore 73 TB/s is only about `56%` of NVL72 aggregate NVLink bandwidth.

This is fine if the architecture changes the traffic pattern. It is weak if
the benchmark is raw all-to-all.

## What extra memory changes

World-model and long-video workloads have state that scales with horizon:

- video tokens / latent frames,
- rolling KV/history memory,
- diffusion denoising intermediates,
- activations for long-context training,
- teacher/student states for distillation,
- many sampled trajectories for RL or planning,
- multiple hypotheses for agentic search.

Current GPU systems cope by shortening horizon, compressing memory, using
SP/CP, checkpointing, offloading, or reducing rollout batch size. Those are
not just implementation details; they change model behavior and serving
latency.

A dual/triple memory-panel glass system can target a different point:

1. Keep more video history resident instead of compressing or dropping it.
2. Keep more rollout branches alive in parallel.
3. Keep teacher, student, reward, and policy/world-model state colocated for
   post-training.
4. Reduce memory-forced SP/CP, so fewer attention/history exchanges are on the
   per-frame or per-denoising-step critical path.
5. Shift from "shard state because HBM is full" to "place state once and reuse
   stable routes."

## What current NVIDIA systems cannot cleanly achieve

### 1. Long-horizon interactive world models without aggressive state reduction

Current systems can generate video, but long interactive horizons push memory
systems hard. NVIDIA's own GB300 page says a single five-second 720p60 Cosmos
video sequence processes `4 million tokens` and takes nearly `90 seconds` on
Hopper, with GB300 giving a projected `30x` improvement. That is strong
evidence that video/world-model inference is a first-class NVIDIA target.

Glass can make a different claim: instead of only accelerating the five-second
case, enable much longer persistent state:

- minute-scale or episode-scale memory,
- persistent identity/layout/object state,
- multi-camera or multi-agent state,
- stateful interactive loops rather than independent clips.

### 2. Many rollout branches for planning/RL

RL and agentic world models need repeated rollouts. NVIDIA's rack systems have
high bandwidth, but the memory budget is still consumed by model weights,
KV/history, optimizer/checkpoint state, and duplicated parallel groups.

Extra memory panels let the system keep many branches resident:

- `N` candidate futures for planning,
- online replanning with cached histories,
- reward-model and world-model states in the same local memory domain,
- larger rollout batch before every update.

This is a better workload than dense training GEMMs because the win comes from
state residency and avoided communication, not raw FLOPs.

### 3. Distillation with teacher/student/state co-residency

Video distillation can require multiple models or model variants live at once:
teacher, student, reward/preference model, cached trajectories, and activation
or denoising state. Current GPU-attached HBM forces smaller batches, shorter
horizons, offload, or repeated recomputation.

Glass memory panels can support:

- longer teacher rollouts supervising longer student horizons,
- larger distillation batches,
- fewer forced reloads/offloads,
- fewer SP/CP boundaries inside every rollout step.

### 4. Memory-rich serving rather than just a faster GPU island

NVL72's strength is fast scale-up communication among GPU-attached memories.
Glass should be framed as a memory-rich serving substrate:

- memory slots can outnumber compute slots,
- cold/warm/hot tiers can be explicit,
- memory can be provisioned independently from compute,
- long KV/history can stay resident across sessions,
- model weights do not need the same duplication pattern as GPU-parallel
  execution groups.

This is especially relevant for sparse/MoE and long-context decode-style
workloads. Dense compute-bound kernels remain an NVL72 strength.

## How to use 73 TB/s safely

Use it as an adequacy number, not a superiority number:

- Bad: `73 TB/s all-to-all beats NVL72.`
- Good: `Even at 73 TB/s, glass is not trying to win raw all-to-all; it uses
  memory capacity to avoid many all-to-all exchanges that NVL72 must perform
  when rollout state spills across GPUs.`
- Good: `73 TB/s is enough for coarse state exchange if routes are reused over
  many frames/tokens; the goal is not per-token reconfiguration.`

If a reviewer asks why lower all-to-all bandwidth wins, the answer is:

1. The relevant metric is not peak all-to-all bandwidth.
2. The relevant metric is end-to-end time to maintain long-horizon state.
3. More memory reduces the number, size, and criticality of collectives.
4. The remaining fabric traffic is coarser and more schedulable.

## Suggested slide-safe wording

Title:

`What Glass Enables That NVL72 Does Not`

Bullets:

- `NVL72 already has faster raw all-to-all: 130 TB/s vs our 73 TB/s target.`
- `The glass bet is different: expand the memory domain so world-model state
  does not need to be sharded every step.`
- `Long video/world-model state scales with horizon: video tokens, KV/history,
  denoising state, rollout branches, and teacher/student caches.`
- `Current GPUs respond with checkpointing, offload, SP/CP, compression, or
  shorter rollouts.`
- `Dual/triple memory panels enable longer resident histories, larger rollout
  batches, and more concurrent futures.`
- `The win is not faster all-to-all; it is fewer critical-path all-to-alls.`

Shorter version:

`73 TB/s is enough only if memory removes traffic. NVL72 wins raw all-to-all;
glass wins when long-horizon state stays resident instead of crossing GPUs each
frame.`

## Measurement hooks

The right evaluation should avoid a raw all-to-all microbenchmark as the main
result. Suggested figures:

1. **Horizon feasibility curve**
   - x-axis: generated/rolled-out horizon, e.g. `5s, 30s, 60s, 5min`.
   - y-axis: resident state footprint and whether the run needs SP/CP/offload.
   - Claim to test: glass keeps longer state resident before the first forced
     sharding/offload point.

2. **Rollout branch capacity**
   - x-axis: number of parallel candidate futures per agent step.
   - y-axis: per-step latency or maximum branches under a latency target.
   - Claim to test: memory panels increase search breadth for planning/RL
     without multiplying GPU groups.

3. **Critical-path communication**
   - Compare two executions with the same model and horizon:
     `NVL72 with SP/CP` versus `glass with resident state`.
   - y-axis: time spent in NCCL/all-to-all/remote state exchange per generated
     frame or denoising block.
   - Claim to test: glass wins by reducing the number of critical collectives,
     not by making one collective faster.

4. **Teacher/student co-residency**
   - x-axis: teacher horizon and student horizon.
   - y-axis: batch size or memory headroom.
   - Claim to test: glass supports longer distillation rollouts because teacher,
     student, reward model, and cached trajectory state can coexist.

5. **Bandwidth sufficiency sweep**
   - x-axis: fabric bandwidth, including `73 TB/s` and `130 TB/s`.
   - y-axis: end-to-end world-model latency.
   - Claim to test: once state is resident and routes are reused, capacity
     dominates; raw all-to-all bandwidth has diminishing returns.

## References

- NVIDIA GB200 NVL72 product page:
  https://www.nvidia.com/en-us/data-center/gb200-nvl72/
- NVIDIA GB300 NVL72 product page:
  https://www.nvidia.com/en-us/data-center/gb300-nvl72/
- NVIDIA H200 product page:
  https://www.nvidia.com/en-us/data-center/h200/
- NVIDIA H100 product page:
  https://www.nvidia.com/en-us/data-center/h100/
- NVIDIA DGX B200 product page:
  https://www.nvidia.com/en-us/data-center/dgx-b200/
- StreamDiffusionV2, arXiv 2511.07399:
  https://arxiv.org/abs/2511.07399
- PackForcing, arXiv 2603.25730:
  https://arxiv.org/abs/2603.25730
- Quant VideoGen, arXiv 2602.02958:
  https://arxiv.org/abs/2602.02958
