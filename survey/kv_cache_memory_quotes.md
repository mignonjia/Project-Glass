# 实时 / 自回归视频生成：谁说过"KV cache 太大、显存放不下"（2026-10-30）

> 用途：给 paper intro / motivation 证明"长 KV cache 的实时 causal 视频生成是公认的显存容量问题"。
> 方法：逐篇读 arXiv HTML / PDF 原文（博客读原网页），引文为**原文逐字**。
> 状态：**VERIFIED** = 读到原句；**PARAPHRASE** = 意思在、措辞不同，列的是最接近的原句。
> 其中 APT2、Quant VideoGen、Krea、PackForcing、TempoKV、Forcing-KV、Rolling Forcing、MAGI-1、LingBot-World-Infinity 的引文已二次抽查。

## 1. 推荐放进 intro 的 5 条

1. **Quant VideoGen**（arXiv 2602.02958）："In practice, KV-cache quickly dominates GPU memory and becomes the binding resource well before raw compute saturates"；5 s、480p 的上下文 KV 约 34 GB，参数只有 27 GB；"This truncation is primarily driven by GPU memory concerns, where full-context KV-cache becomes quickly infeasible for long-horizon generation."（说的是 Self-Forcing 的 21 帧窗口和 WorldPlay 的 20 帧记忆）
2. **Krea Realtime 14B**（博客，2025-10）："To generate videos longer than the training context length, we cannot let the KV cache of past frames grow indefinitely due to memory constraints. The only practical approach is a sliding context window, where old frames are evicted from the cache to make room for new ones."；训练时 "this cache can grow to 25GB per GPU"。
3. **Seaweed APT2**（arXiv 2506.09350，附录 F）："Note that 1280×720 is trained and inference with a smaller attention window size N=15 to fit the memory."（736×416 时 N=30 个 latent 帧 = 5 s；分辨率一高，记忆减半）
4. **PackForcing**（arXiv 2603.25730）："For a 2-minute, 832×480 video at 16 FPS, the full attention context grows to ~749K tokens, requiring ~138 GB of KV storage across 30 transformer layers, well beyond the memory budget of a single commodity GPU."；"retaining the full history inevitably causes out-of-memory failures for videos exceeding roughly 80 seconds"（A100-80GB）。
5. **TempoKV**（arXiv 2610.15324）："generating a 45-second 480p video requires 51.76 GB solely for the KV cache when stored in BF16"；完整历史 "reaches only about 815 frames on our hardware before running out of memory"。可配 **Forcing-KV**（arXiv 2605.09681）：30 s、1080p 时 "the KV cache alone consumes more than 60 GB of GPU memory"（H200）。

备选：Rolling Forcing（标准方法自己写 cache 会 OOM）、MAGI-1（为装进 RTX 4090 把 KV 卸到 CPU）、Infinite-World（80 GB 卡上 180 帧后 OOM）、WorldAttention（GPU → CPU → NVMe 三级 KV，最贴近体系结构）、LingBot-World-Infinity（大厂 tech report 承认"memory cost grows with the horizon"）。

## 2. 全部例子

| 论文 | 出处 | 原文 | 数字 | 状态 |
|---|---|---|---|---|
| Seaweed APT2（字节） | 2506.09350，附录 F | "Note that 1280×720 is trained and inference with a smaller attention window size N=15 to fit the memory." | 窗口 30 → 15 个 latent 帧 | VERIFIED |
| Krea Realtime 14B | [博客](https://www.krea.ai/blog/krea-realtime-14b) | "we cannot let the KV cache of past frames grow indefinitely due to memory constraints. The only practical approach is a sliding context window" / "With Wan 2.1 14B, this cache can grow to 25GB per GPU, creating significant memory pressure." | 25 GB/GPU（训练） | VERIFIED |
| Quant VideoGen | 2602.02958，§1、§3.1 | "KV-cache quickly dominates GPU memory and becomes the binding resource well before raw compute saturates" / "KV-cache capacity is the primary bottleneck for long-horizon video generation." | 5 s 480p：KV 34 GB vs 参数 27 GB | VERIFIED |
| PackForcing | 2603.25730，§1、附录 H.3 | "requiring ~138 GB of KV storage across 30 transformer layers, well beyond the memory budget of a single commodity GPU" / "out-of-memory failures for videos exceeding roughly 80 seconds" | 2 min 832×480：138 GB | VERIFIED（论文内 4 GB / 4.2 GB 前后不一致） |
| TempoKV | 2610.15324，§1、§4.3 | "requires 51.76 GB solely for the KV cache when stored in BF16, exceeding the memory of most consumer GPUs" | 45 s 480p：51.76 GB；约 815 帧 OOM | VERIFIED |
| Forcing-KV | 2605.09681，§1 | "the KV cache alone consumes more than 60 GB of GPU memory in this setting" | 30 s 1080p：> 60 GB | VERIFIED |
| Rolling Forcing | 2509.25161，§3.3 | "the cache may become large enough to cause out-of-memory errors." | — | VERIFIED |
| MAGI-1（Sand AI） | 2505.13211，§4.2.2、§2.4.4 | "we identified memory insufficiency as the primary bottleneck to serve our model on it" / "KV-offload is a technique that stores the KV cache in CPU memory … to effectively address memory constraints." | RTX 4090（24 GB）；KV 范围 8 s | VERIFIED |
| Infinite-World | 2602.02393，§4.4.1 | "without compression, memory usage scales linearly and triggers an out-of-memory error at over 180 frames." | 80 GB H800 | VERIFIED |
| OmniMem | 2605.30519，§1、附录 B | "As video length increases, the KV cache grows linearly with the number of generated chunks" / "it can still require a large KV memory footprint … We refer to this memory-side blow-up as Union Explosion." / "Offloaded KV blocks live in pinned host memory and are reloaded through PCIe" | 717 帧省 3.3× 显存 | VERIFIED |
| WorldKV | 2605.22718，§1 | "accumulating hundreds of thousands of tokens over a one-minute rollout. The corresponding KV cache rapidly exceeds GPU VRAM capacity" | Matrix-Game-2.0、LingBot-Fast；1 min 内 8.87 → 3.61 FPS | VERIFIED |
| WorldAttention | 2609.34606，摘要 | "the linear growth of the KV cache inevitably leads to GPU memory saturation." | GPU → CPU → NVMe 分级 | VERIFIED |
| LingBot-World-Infinity（蚂蚁） | 2607.07534，§6 | "memory cost grows with the horizon while the available budget is bounded." | — | VERIFIED |
| CORAL-KV | 2610.12514，§1 | "the attention key–value (KV) cache grows with video duration, spatial resolution, and model depth, and can quickly dominate GPU memory." | HY-WorldPlay、MAGI-1 | VERIFIED |
| LongLive-2.0 | 2605.18739，§3.2 | "KV cache memory grows linearly with history and quickly becomes a bottleneck" | BF16 训练 64 s OOM | VERIFIED |
| LongLive | 2509.22622，§3.3 | "the KV cache needed per layer scales with the window rather than the total video." | 短窗口 + sink 峰值显存降 17% | VERIFIED |
| RELIC | 2512.04040，§4.3 | "both the KV-cache memory footprint and the per-token attention cost scale linearly with sequence length." | 14B | VERIFIED |
| VideoMLA | 2605.30351，§1 | "retaining all past KV states would grow linearly with rollout length." | 21 帧窗口 6.0 GB（1.3B） | VERIFIED |
| Sparse Forcing | 2604.21221，§3.2 | "For a 1.3B model generating a 1-minute video, the FP16 KV cache reaches 44.9 GB, i.e., 17.26× the parameter memory even at batch size 1." | 44.9 GB | VERIFIED（按 Wan-1.3B 推算应约 69 GB，配置未写，引用时注明"as reported"） |
| TetherCache | 2606.13035，§1 | "The amount of historical context grows linearly with the generated video length, whereas the KV-cache budget is fixed by memory and latency constraints." | — | VERIFIED |
| Yume 1.5 | 2512.22096，§2.3 | "these methods do not adequately address the issue of increasing memory consumption or computational load as the generated video length grows, and merely truncate previous frames" | — | VERIFIED |
| WorldMem | 2504.12369，§1 | "Due to computational and memory constraints, video generative models operate within a fixed context window" | 8 帧窗口 | VERIFIED |
| Memory Forcing | 2510.03198，§1 | "latency and memory limits bound the window size." | — | VERIFIED |
| MemFlow | 2512.14699，§1 | "the capacity of stored memory must be highly constrained, a necessity dictated by both the hardware limits of GPU memory and the demands of generation efficiency." | — | VERIFIED |
| SANA-Video | 2509.24695，§3.3 | "since the computational and memory costs grow linearly, these methods often restrict the attention window to a local scope" | — | VERIFIED |
| Long-Context SSM World Models | 2505.20171，§4.3 | "fully causal transformers whose memory requirements grow linearly as they store KV-caches for all previous frames." | — | VERIFIED |
| Infinity-RoPE | 2511.20649，§1 | "the heavy memory footprint of uncompressed KV caches." | — | VERIFIED |
| Multiverse（Enigma） | 博客 2025-05-08 | "that would make the model too slow for real time play, increase its memory usage and make training much slower." | 8 帧上下文（不是 KV cache） | VERIFIED |
| Dreamer 4（GDM） | 2509.24527，§3.4 | "limited both by the FLOPs of the MLPs and the memory bandwidth needed to access the KV cache of a long context" | 9.6 s 上下文 | VERIFIED（讲的是带宽，不是容量） |
| HY-WorldPlay | 2512.14614，§3.3 | "naively using all past frames as context … is computationally intractable and redundant for long sequences." | — | PARAPHRASE（我们旧笔记的 "greatly increase" 原文没有） |
| Context-as-Memory | 2506.03141，§1 | "Including all historical frames in computation would be extremely resource-intensive." | — | VERIFIED（讲计算，较弱） |
| Matrix-Game 2.0 | 2508.13009，§4.2 | "evicting oldest tokens when exceeding capacity" | 6 帧窗口（按质量选） | VERIFIED，弱 |
| FramePack | 2504.12626，§1 | "thousands of frames with 13B models even on laptops (e.g., 6GB or 8GB GPU memory)" | — | 不用 KV cache，较弱 |

查过但没有可用说法：Self Forcing、Self-Forcing++、CausVid、SkyReels-V2、StreamDiT、StreamDiffusionV2、Matrix-Game 3.0、Deep Forcing、MemoryPack、Reward Forcing、Light Forcing、Hybrid Forcing、LingBot-World v1（只谈训练显存）。

## 3. 写的时候注意

- **来源要平衡**：Quant VideoGen、PackForcing、TempoKV 等是做 KV 压缩的论文，有动机强调显存。要和系统 / 模型本身的论文（APT2、Krea、MAGI-1、Rolling Forcing、LongLive、RELIC、LingBot-World-Infinity）搭配引用。
- **可能的反驳**：Context Forcing（2602.06028）把短上下文归因于"memoryless teacher"，不是显存；Matrix-Game 2.0 的 6 帧窗口是按质量选的；Dreamer 4 讲的是带宽。
- **我们的实测可以补上**：LingBot 14B 在 8 张 H200 上，120 s 历史每卡 KV 74 GB（`docs/realtime_profile.md`）。
