# Session History — Glass Motivation Slides

Date: 2026-05-19

## Goal

Create a serious proposal-style slide section motivating glass substrate from memory-capacity pressure and rollout workloads, especially for world models / video generation.

The user rejected the first generated slide logic as too solution-first. The accepted logic is lifecycle-first:

1. Pretraining: long-video foundation model training is memory-capacity limited.
2. Post-training: distillation and RL are rollout-heavy.
3. Inference: interactive world models are streaming rollout workloads.
4. Glass substrate / dual-panel glass fits because it expands local memory capacity and reduces memory-forced sharding/communication.

## Files Created

- `task.md`
- `short_report.md`
- `flow.md` — user-written desired logic
- `build_motivation_deck.py` — v1 custom card-style deck, rejected stylistically
- `glass_motivation_draft_v1.pptx`
- `build_motivation_deck_v2.py` — style-matched to `Glass ML Presentation Tajana Samsung SK.pptx`, but logic still not ideal
- `glass_motivation_draft_v2_style_matched.pptx`
- `build_motivation_deck_v3_flow.py` — follows `flow.md`
- `glass_motivation_draft_v3_flow.pptx`

The user later said not to update slides unless explicitly asked; provide revised text only.

## Source Decks / Extracted Context

### Main GlassML Deck

`Glass ML Presentation Tajana Samsung SK.pptx`

Checked whether it already mentions dual-panel / dual-sided glass. It does not. Search terms like `dual`, `double`, `sandwich`, `memory-only`, `2-panel`, `3-panel`, `both sides`, `memory panel`, `HBM bottom`, and `GPU top` only matched unrelated `reconfigured`.

### Filtered Team Deck

`ML_IoT Team 25-26 - Mingjia Junda only.pptx`

Use this as the preferred source for Mingjia/Junda numbers.

Key extracted numbers and facts:

- World-model token growth:
  - `1 latent frame ~ 1.5k tokens ~ 0.16s`
  - `1s ~ 10k tokens`
- Internal distillation memory profile:
  - Setting confirmed from slide: `SP=1`, long-video/world-model distillation memory-capacity profiling.
  - `3 latent training: 55 GB peak`
  - `9 latent training: 135 GB peak`
  - `15 latent training: OOM`
  - Exact model, batch size, precision, and GPU are not stated on that slide. Do not add those unless confirmed.
- Wan1.3B causal SP profile:
  - Setting: `Wan1.3B causal`, block size `3 latent frames`, `FlashAttention 4`, `B200`, `SP=1` or `SP=4`.
  - Measurement: one denoising step at the last block of the video, where KV cache is largest.
  - Values extracted from embedded chart on slide 27:

| Resolution / Duration | FlashAttention wall sum | NCCL all-to-all wall sum | Approx total |
|---|---:|---:|---:|
| 480p, 3.4s, SP=4 | ~0.06s | ~0.17s | ~0.23s |
| 480p, 8.4s, SP=4 | ~0.11s | ~0.11s | ~0.22s |
| 1080p, 3.4s, SP=4 | ~0.38s | ~0.18s | ~0.56s |
| 1080p, 8.4s, SP=4 | ~0.84s | ~0.20s | ~1.04s |

  - Slide-safe wording:
    - `Internal profile, Wan1.3B causal on B200, SP=4: for one denoising step at the last video block, NCCL all-to-all adds ~0.17-0.20s wall time. At 480p/3.4s, communication is larger than FlashAttention itself; at 1080p/8.4s, it is still ~24% of FlashAttention time.`
  - Caveat: values are read from an embedded chart image, not raw profiler logs.
- HYWorldPlay / FastVideo inference note:
  - Long video bidirectional transformer generates chunk by chunk, one chunk = `2.5s`.
  - Given `SP=2`, first chunk is `~2.5 minutes`, second chunk is `~7 minutes`.
- Wan-14B training profiling:
  - `4x 80GB H100`, `SP=4`, batch size `2`, HSDP shard `4`.
  - MFU `~40%` for backward/forward.
  - Self-attention MFU `76%`, taking `28%` of single-layer forward.
- Dual-sided glass / Junda:
  - Goal: GPU on one side, HBM on the other.
  - Glass thermal conductivity `k ~ 1 W/mK`; acts as thermal barrier.
  - At `1000W` GPU: `<0.6W` leaks to memory side, `<1C` rise.
  - Silicon interposer `k ~ 130`, about `7x` worse thermal coupling in Junda's slide.
  - Memory per slot: `192 GB -> 384 GB` with HBM3e both sides.
  - Kimi-K2 4TB example: `21 -> 11` memory slots, `48%` fewer, no weight duplication.

## External / Literature Context

### StreamDiffusionV2

`papers/applications/StreamDiffusionV2.pdf`, arXiv `2511.07399`.

Interpretation:

- The `memory-bound regime` language in that paper refers mainly to streaming inference / serving, not training.
- The paper targets real-time live video generation:
  - low time-to-first-frame,
  - strict per-frame deadlines,
  - low jitter,
  - scalable multi-GPU serving.
- It states sequence parallelism has unpredictable communication overhead, and short-frame chunks push real-time workloads into a memory-bound regime.
- Reported numbers:
  - first frame within `0.5s`,
  - `58.28 FPS` with 14B model on `4x H100`,
  - `64.52 FPS` with 1.3B model on `4x H100`,
  - high-quality mode: `31.62 FPS` (14B), `61.57 FPS` (1.3B).

### Gradient Checkpointing

Accepted point: emphasize total/peak memory reduction first, while noting checkpointing operates by reducing stored activations.

Useful wording:

- `Reduces peak training memory by not storing many intermediate activations.`
- `Classic result reports total memory 48 GB -> 7 GB with ~30% runtime overhead.`
- `Transformer caveat: total-memory savings depend on how much peak memory is activations; NeMo reports ~30% extra compute for full layer recompute.`

Estimated total-memory effect:

If peak memory is:

```text
peak = non_activation_memory + activation_memory
```

and activation memory is `50-80%` of peak, then a `4-8x` activation reduction can translate to roughly `1.6-3.6x` lower total peak memory.

Slide-safe wording:

- `If activations are 60-80% of peak memory, a 5x activation reduction translates to roughly 2-3x lower total peak memory, but adds recompute during backward.`

References discussed:

- Chen et al., `Training Deep Nets with Sublinear Memory Cost`, arXiv `1604.06174`.
- Korthikanti et al., `Reducing Activation Recomputation in Large Transformer Models`, MLSys 2023.
- NVIDIA NeMo / Megatron recomputation docs.

### Parallelism / Bubble / Stragglers

Do not overclaim:

- Do not say SP has the biggest bubble.
- Do not claim a universal power-law bubble distribution unless measured.
- Literature typically calls this `exposed communication`, `synchronization overhead`, `idle time`, `straggler propagation`, or `pipeline bubbles`, not `SP bubble`.

Safe claim:

- Memory-forced TP/SP/CP makes long samples fit, but inserts synchronization points.
- When overlap fails, communication becomes exposed idle time.
- More local memory reduces memory-forced sharding and therefore reduces opportunities for synchronization exposure.

Useful references:

- ByteDance OSDI'25 / arXiv `2505.05713`: straggler analysis in hybrid-parallel LLM training.
  - Supports synchronization stalls and propagation through TP/CP/PP/DP.
  - Does not prove which parallelism type has the largest bubble.
  - Reports sequence-length imbalance and PP stage imbalance as important measured causes.
- Megatron-Core context parallelism docs.
- PyTorch profiler communication-overlap guidance.

### LLM / Video RL Trend

Safe claim:

- LLM post-training is increasingly rollout-heavy.
- Recent systems work reports rollout generation can dominate RL runtime.
- For video/world models, evidence is emerging, not yet a clean field-wide `% of all training compute`.
- Do not claim a precise public trend such as `RL is X% of all training compute`.

Good slide wording:

- `LLM trend: reasoning models increasingly scale with RL train-time compute.`
- `LLM systems trend: rollout generation can dominate RL runtime.`
- `Video/world-model rollouts are heavier than text: many frames, multi-step denoising, rolling KV/cache state.`

References discussed:

- OpenAI `Learning to Reason with LLMs` / o1 blog.
- DeepSeek-R1 / GRPO.
- APRIL active partial rollouts.
- Video/world-model RL and inverse-dynamics reward literature.

## Accepted / Improved Phrasing

### Slide Title / Flow

User's desired flow:

1. `Memory has become a big problem in world model / video gen training`
   - subtitle: `From pretraining, post-training (step-distillation, RL), to inference`
2. `Pretraining - foundation of the video model`
3. `Post-training: important for efficiency from step distillation and alignment from RL`
4. `Inference`
5. `Glass Substrate Fits This Trend`

Better final title option:

- `Glass Substrate Fits This Trend`
- If specifically solution slide: `Dual-Panel Glass Substrate Fits This Trend`

### SLO

`SLO` means `Service-Level Objective`.

For world-model / streaming video inference:

- time-to-first-frame target,
- every frame delivered before deadline,
- low jitter,
- maintain 30/60 FPS.

Avoid unexplained `SLO` on slides unless defined.

Better wording:

- `Inference: world-model rollout is latency-critical, not offline batch video generation.`
- `Low time-to-first-frame, per-frame deadlines, and low jitter make memory access hard to hide.`

### Why Memory Matters

The user wanted the title to explicitly say memory, not generic systems.

Good title:

- `Why Memory Matters`

Good bullets:

- `Rollout scaling is state scaling: longer horizons require more video tokens, cache, and intermediate denoising state.`
- `Post-training multiplies this state: RL / self-distillation generate many trajectories before each update.`
- `If memory is insufficient, training falls back to checkpointing, offload, or SP/CP.`
- `These workarounds trade memory pressure for recompute, communication, and synchronization.`
- `More local memory enables longer rollouts and larger rollout batches with less sharding.`

Good concise statement:

- `Compute determines how fast each rollout step runs; memory determines how long the rollout can be, how many rollouts fit in parallel, and how much sharding/offload is required.`

### Long-Horizon Bottleneck Statement

Rejected as vague:

- `The common bottleneck is keeping long-horizon state local.`

Better:

- `Long-horizon world models are limited by how much rollout state can stay near compute, and how fast non-local state can be exchanged.`
- `Long-horizon world models need large local state capacity and low-latency exchange when state spills across GPUs.`
- To include SP/CP explicitly:
  - `If rollout state fits locally, execution stays local. If not, SP/CP shards it across GPUs, adding communication to every rollout step.`
  - `When rollout state does not fit locally, SP/CP shards it across GPUs, making communication part of the critical path.`

### Movement

`movement` was vague. Use:

- `data access`,
- `data exchange`,
- `cache/state residency`,
- `non-local communication`.

Better sentence:

- `Takeaway: inference rollout needs local cache/state residency and low-latency access to remote memory.`
- `Takeaway: inference rollout needs cache/state locality; otherwise SP/CP communication shows up directly in frame latency.`

### Dual-Panel Sentence

Original:

- `Solution: Dual-panel glass: One layer: GPU + memory; second layer: memory-only`

Better:

- `Dual-panel glass: GPU+HBM layer + memory-only layer.`
- `Proposed substrate: stack a GPU+HBM panel with a second memory-only panel to expand the local memory domain.`
- `Dual-panel glass with one compute+memory layer and one memory-only layer.`

### Glass Slide Title

User proposed:

- `Glass Substrate Fits This Trend`

Accepted. Also possible:

- `Dual-Panel Glass Substrate Fits This Trend`

## Current Recommendation for Manual Slide Text

Because the user asked not to update slides automatically anymore, provide text only unless explicitly asked.

Potential revised slide text:

### Slide: Pretraining

Title:

`Pretraining: Foundation Video Models Are Natively Long-Context Training`

Bullets:

- `1 latent frame ~ 1.5k tokens ~ 0.16s; 1s video ~ 10k tokens.`
- `Internal distillation profile, SP=1: 3 latent frames = 55GB peak; 9 = 135GB; 15 = OOM.`
- `Video-generation training utilization is low: StepVideo reports ~35.9% MFU in 540P pretraining.`
- `Memory workaround #1: checkpointing reduces peak memory, e.g. 48GB -> 7GB, but adds ~30% runtime overhead.`
- `Memory workaround #2: SP/CP/TP makes long samples fit, but inserts synchronization and straggler propagation.`
- `More local memory reduces checkpointing and memory-forced sharding.`

### Slide: Post-Training

Title:

`Post-Training: Distillation and RL Both Need Rollout`

Bullets:

- `Step distillation is the efficiency path: 50-step bidirectional/causal model -> 4-step causal model.`
- `RL/reward post-training is the alignment path: action following, preference, robot/world-model rewards.`
- `Both repeatedly generate trajectories before each update.`
- `Self-forcing can require multiple models in memory; current notes flag memory capacity as a blocker.`
- `Video rollout is heavier than text rollout: many frames, multi-step denoising, rolling KV/cache state.`
- `Internal SP=4 profile: NCCL all-to-all adds ~0.17-0.20s per denoising step at the last video block.`

### Slide: Inference

Title:

`Inference: Interactive World Models Are Streaming Rollout`

Bullets:

- `World-model agents run a loop: state/action -> predicted future -> policy/planner -> next action.`
- `This is latency-critical rollout, not offline batch video generation.`
- `StreamDiffusionV2: first frame within 0.5s; 58.28 FPS with 14B model on 4xH100.`
- `It identifies sequence-parallel communication overhead and memory-bound short chunks under real-time serving.`
- `HYWorldPlay note: with SP=2, 2.5s chunk takes ~2.5 min for first chunk and ~7 min for second.`
- `Takeaway: inference rollout needs cache/state locality; otherwise SP/CP communication shows up directly in frame latency.`

### Slide: Glass

Title:

`Glass Substrate Fits This Trend`

Bullets:

- `Long-horizon world models need large local state capacity and low-latency exchange when state spills across GPUs.`
- `Dual-panel glass: GPU+HBM layer + memory-only layer.`
- `Expands local memory capacity for activations, cache/history, and rollout batches.`
- `Reduces checkpointing pressure in pretraining.`
- `Reduces memory-forced SP/CP, so less rollout state crosses GPUs every step.`
- `Junda dual-sided hook: 192GB/slot -> 384GB/slot; at 1000W GPU, <0.6W leaks to memory side (<1C rise).`

