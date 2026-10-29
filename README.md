# Project Glass × Causal World Models

> 这份文档是项目的"锚点"。每次回来先读这里，确认手上的工作仍然服务于当前方向。
> 目标或假设变了，就更新这里，并在末尾的「决策记录」里写一行。

## 当前方向（2026-10-08，导师确认）：实时推理

**Causal world model 的实时推理**：逐 chunk 生成，每个 chunk 必须在它对应的视频时长内生成完（720p、24 fps 时约 0.67 s），同时保留长历史的 KV cache。

### 讲法（草案，待导师 / David 确认）

**容量是起因，带宽和延迟是瓶颈。** 应用场景和引文见 `docs/motivation.md`。

1. **容量把 KV 挤出 GPU**：每路 10 分钟的流有 2.7 TB KV，HBM 放不下，只能放到别的卡的 HBM、Grace LPDDR 或 glass DRAM。
2. **实时性把取数变成带截止时间的问题**：每个 chunk 要跑 44 层 × 5 次 forward = 220 轮"打分 → 取 → 算"，平均每轮约 3 ms；取哪些块打分后才知道，不能提前预取。
3. **需求在运行时确定、带偏斜**：每步取约 22% 的历史，由内容决定；多路流叠加有热点。形式上和 MoE AllToAll 相同，可以接 GlassML 的调度器。
4. **可以用容量换带宽**：4 步之间选择只变约 7%，把取来的 KV 缓存在 HBM 里，带宽降到约 1/4，但要占 HBM。
5. **传统系统要么缺容量、要么缺带宽**（见下面的估算），glass 把大容量和高带宽放在同一个域里。

对照路线：**只存 latent、每个 chunk 重算选中帧的 KV**（HY-WorldPlay：按相机视野选约 20 帧，几乎不占存储；每个 chunk 先重算 20 帧、再对 4 帧去噪 4 步（约 20 vs 16 帧次 forward），计算约翻倍，记忆只有 20 帧）。PAN VSA 每层、每个 head 按内容选块，必须用每层的 K 打分，所以只能存 KV。

### 估算结果（`docs/realtime_inference_estimate.md`）

PAN、720p、10 分钟流、4 步 + 1 次 commit、无 CFG、remote top-k 5%，按所有流都在最后一个 chunk 算；脚本 `estimate_realtime.py`（worktree）。

- 每路流的完整 KV **2.66 TB**；实时所需算力约 4.6 个 tile（稀疏 attention 读 HBM 受限时约 5 个）。
- 每次 forward 取 21.8% 的远端历史（577 GB）。读带宽取决于复用：每步重读 4.3 TB/s / 路；同一 chunk 的 4 步共用 1.0 TB/s；再加上相邻 chunk 复用（trace：重叠 63%）0.38 TB/s。复用的代价是每路流约 0.7 TB 的 HBM 缓存。
- 按 head 切分时每个 tile 只读自己的 KV，互连只传激活（每 chunk 5–8 ms），稳态下不是瓶颈。

| 系统 | 每系统流数 | 只看容量 | 只看算力 | 卡在哪里 |
|---|---:|---:|---:|---|
| NVL72，只用 HBM | 5 | 5 | 14 | **容量** |
| NVL72 + Grace LPDDR（17 TB，每卡 0.19 TB/s） | 9 | 11 | 14 | **带宽 + HBM 缓存空间** |
| Glass + 85 TB DRAM（每 tile 1 TB/s） | 11–12 | 36 | 12 | **算力**（64 个 tile） |

- glass 的主要收益是容量；5% 比例选块时 attention 随历史增长，glass 很快卡算力，收益 1.3×（vs Grace）/ 2.4×（vs 只用 HBM）。
- 固定选块预算（每个 query 看 20 个 chunk）时：10 分钟 21 / 10 / 5 路，30 分钟 12 / 3 / 1 路，60 分钟 6 / 1 / 0 路（glass / Grace / 只用 HBM）。
- glass DRAM 每 tile 0.1 TB/s（多路流交替）到 2 TB/s（每组只服务一路流）可以到算力上限。
- ⚠️ 对讲法的影响：做好复用后带宽需求不高，"带宽是瓶颈"只对不复用或 Grace 成立。更准确的是"容量是起因，之后是 HBM / DRAM 调度问题"。待和导师 / David 讨论。

## 背景：为什么是 glass

- **只讲容量发不了 paper**（David / Tianqi 的反馈），必须有带宽或调度问题。
- **铜线把 scale-up 域限制在一个机柜**（224 Gb/s 每 lane 时无源铜线不到 1 m）。NVIDIA 2027 年的 NVL576 仍在一个机柜内、用铜背板。业界的光互连目前主要用在 scale-out（CPO 交换机）。
- **业界已经把玻璃当作封装基板**（TSMC CoPoS、Intel，2028–2030）；GlassML 在同一块玻璃里做可重构光网络，把整块面板变成一个更大、带宽更高的 scale-up 域。

## 硬件假设

| | GB300 NVL72 | GlassML 论文配置 | Tianqi 模型（导师转述） |
|---|---|---|---|
| 规模 | 72 GPU，一个机柜 | 600×600 mm 面板，64 tile（每 tile = B300 + 288 GB HBM3e） | — |
| 互连 | NVLink 每卡单向 0.9 TB/s，合计 130 TB/s | 光互连，每 tile 点对点最高 4 TB/s（K16/2T） | 最高 8 PB/s |
| 内存 | 20 TB HBM + 17 TB Grace LPDDR5X（14 TB/s，经 NVLink-C2C） | 18.4 TB HBM，**没有 DRAM 面板，没有近存计算** | ≥ 85 TB DRAM（玻璃双面，可能更多） |

估算里 glass 每卡按单向约 4 TB/s 计（约 NVLink 的 4.4×）。

⚠️ 待确认（找 Tianqi / David）：
- 8 PB/s 是总带宽、对分带宽还是每 tile 注入带宽？8 TB/s 是否为半双工总量？
- DRAM 面板怎么接到 GPU：正下方 TGV 直连（本地），还是走光互连（共享池）？带宽和延迟多少？
- 85 TB 是哪种 DRAM，每个 tile 分到多少？
- 光路重配置要多久？每轮取数只有毫秒级预算时，这可能是关键约束。

## 模型与代码

- **模型**：PAN（29.2B，44 层，hidden 4096，32 head）+ VSA hybrid 稀疏 attention（最近 3 个 chunk 精确 + 远端 top-k，本文默认 5%）。
  - 实时推理用 SF 4 步 checkpoint：`WM-Production/outputs/sf400-integ-f31474c`（代码 commit `f31474c`）。
  - DF checkpoint：`yiyan_recording10s_w3_remote15_1k_v1`（代码 commit `5a5544a`）。
- **代码**：`/mnt/weka/shrd/wm/mingjia/WM-Game-Training`。`vsa/hybrid.py`：窗口 + 远端 top-k 选择；`vsa/DESIGN.md`：压缩分支 + 选择分支。
- **实验代码**：分支 `exp/topk-trace`，目录 `experiments/topk_trace/`（trace hooks、分析和估算脚本）。只留在服务器上，不 push；origin 只同步 main。

## 方法原则

- **infra paper**：仿照 GlassML，用真实 checkpoint 的 forward 采集 trace（选择、取数需求、偏斜），注入内存 / 互连模拟，与 NVL72 对比。
- **不做系统性的质量对比**：motivation 引用已有论文（MoC、WorldMem、WorldTrace、Wonder 等）。只有硬件设计改变了算法（改 k、block 大小、预测预取）时，才做 iso-quality 验证。
- **只用 VSA v1**，不用 VSA2 的细粒度路由；**不引入近存计算**：打分在 GPU 上做，索引（每层 K 的 block 均值，每路流约 10 GB）放 HBM。

## 范围

**做**：causal world model 的实时推理，包括每系统流数、KV 容量与读取带宽、每 chunk 延迟、跨步缓存等调度取舍。

**不做**：训练（已结束，见下）、MoE、多视角、10 分钟以上的流、相机轨迹检索、自己做稀疏算法的质量研究、玻璃基板的物理 / 热 / 光器件设计（用 Tianqi / Junda 的模型作为输入）。

## 近期计划：11 月 review（实时推理）

- [ ] Tianqi：glass DRAM 每 tile 的带宽、延迟和连接方式（每 tile 需 0.1–2 TB/s，取决于调度；64 KiB 粒度的效率；光路重配置时间）
- [ ] 用 PAN SF + VSA 实测：720p、尽量长的流，每 chunk 每层的延迟、显存、有 / 无跨步缓存的 KV 读取量
- [x] 多路流叠加的取数偏斜（8 个样本当作 8 路流：同步时峰值 / 均值 2.3，错开后约 1.4）
- [x] NVL72 基线加入 Grace LPDDR5X（容量 / 带宽），重算每系统流数
- [x] 写 `docs/realtime_inference_estimate.md`：视频长度、top-k 比例 / 固定预算 vs 每系统流数、每 chunk 延迟
- [ ] 问 PAN / VSA 的人：长视频用比例选块还是固定预算（决定 glass 收益是 1.3× 还是 2–6×）
- [ ] 720p 实测选块并集（448p 外推约 33%）和跨 chunk 复用
- [ ] 验证"每步取约 22% 远端历史"能否从 60 s 外推到 10 分钟
- [ ] 可选对照：HY-WorldPlay 4 步 480p 模型，测每 chunk 延迟和重算占比
- [ ] David：GlassML 模拟器能否接推理 trace
- [ ] 投稿：DAC 2027（约 11/18）是否还投
- [x] Ring attention 两节点实测（真实 IB，2 节点 × 4 卡），`docs/ring_attn_overhead.md` 第 5 节
- [x] Ring attention 两节点 4 通道（`NCCL_NCHANNELS_PER_NET_PEER=4`）复测：26–34 GB/s，跨节点 ring 开销约减半

**review 目标图表：**
1. 每系统能同时跑的实时流数：NVL72（只用 HBM / 加 Grace）vs glass，标出容量 / 算力 / 带宽三个上限
2. 可支持的视频长度（完整 KV 历史）vs 系统
3. 每 chunk 延迟拆解：计算 / 读 KV / 通信

## 训练方向（已结束，作对照）

10/6–10/8 分析了 PAN DF + VSA 训练，详见 `docs/training_reference.md`。结论：dense 29B 训练在 NVL72 内是算力瓶颈（10 分钟 720p 选对切法时通信只占计算的 9%），线性层的计算量远超取 KV 的流量，glass 的带宽优势体现不出来；只有一个样本超出快的域、或在 8 卡节点 + IB 集群上训练时，通信才成为瓶颈。

## 人员

- 导师：项目方向与 review 要求（10/8 确认做实时推理）
- David：GlassML 架构与模拟器（HPCA 2027 投稿，`David_MoE_glass.pdf`）
- Tianqi：光带宽与 on-glass DRAM 的硬件模型
- Mingjia / Junda：前期 world model memory 动机分析、双面 glass 热分析（见 `Mingjia-Research/`）

## 材料索引

| 路径 | 内容 |
|---|---|
| `docs/motivation.md` | paper motivation：哪些实时应用需要长历史、现有系统只留几秒、实时把容量变成带宽问题，含可引用原话和预期质疑 |
| `docs/realtime_inference_estimate.md` | 实时推理估算：每系统流数、容量 / 算力 / 带宽上限、DRAM 带宽扫描、视频长度扫描、trace 里的放置统计 |
| `docs/realtime_profile.md` | 实测：LingBot-World v2 14B（8×H200，SP、FA4）和 Cosmos-H-Dreams 2B 的长 KV 实时推理——完整历史 vs sliding window vs 存全量 KV + 选块、SP 通信、容量层取数带宽、长期记忆存 KV vs 存 latent 重算（速度和画质） |
| `docs/report/outline.md` | 给导师汇报的提纲和配图（`docs/report/figures/`）：WorldKV、OmniMem 两篇 motivation，存 KV vs 重算的画质和速度，profiling 设置、访问模式、取数带宽 |
| `docs/attn_trace.md` | 给硬件建模的逻辑 attention trace：LingBot 14B 完整历史 KV，每 chunk / 去噪步 / 层 / head / query 块 attend 到的 64 KB KV 块（VSA 式、LF 式两种选块，120 s），数据在 `data/attn_trace/`（Git LFS） |
| `docs/pan_measurements.md` | PAN 参数与公式、推理显存实测、推理选择 trace（每步 22.9%、跨步并集 27.7%） |
| `docs/training_reference.md` | 训练方向的结论和关键数字（已结束） |
| `docs/ring_attn_overhead.md` | Ring attention（序列并行，训练和推理都用）的开销：4K 视频 / Wan2.1-14B 形状，单节点 NVLink、两节点真实 IB 实测、开销来源（重叠份额、NCCL 中等消息带宽、IB 每对端通道数）、到 NVL72 / glass 的投影 |
| `meetings/Mingjia-1029-update.pptx` | 周会一页（沿用 `meetings/Mingjia-1022-update.pptx` 的格式）：双向视频 attention 用 Ulysses × ring 时，dense 和 sparse 都不是通信瓶颈（训练开销表：实测黑体、投影蓝色，IB / NVL72 / glass），所以互连问题要靠 causal + 完整历史 KV cache + 稀疏选块的 workload；生成脚本在 `exp/ring-attn` 的 `build_slides.py` |
| `survey/causal_video_long_kv_scenarios.md` | 需要长 KV 的实时 causal 视频场景（游戏、3D 探索、多人、机器人、驾驶、RL、avatar），每个场景的时长 / 帧率 / 并发需求和 PAN 等价 KV，用于 motivation |
| `survey/sparse_attn_video_training.md` | 稀疏 attention × 视频 × 多卡系统文献；最相关：SparSP、MTraining、db-SP |
| `survey/glass_memory_papers.md` | 玻璃基板 / scale-up 系统 / AI 内存分层 / 长视频内存文献 |
| `survey/venues_and_timeline.md` | 投稿会议与 deadline（2026-10 至 2027-12） |
| `David_MoE_glass.pdf` | GlassML HPCA 2027 投稿稿（**保密，勿外传；仓库保持私有**） |
| `Mingjia-Research/` | 早期材料：world model memory 论证、动机 slides、团队 deck |

## 待解决问题

- [ ] 硬件假设（见上）
- [ ] 模型只按 10 个 chunk、15% 训过；长视频训练后的选择模式可能不同
- [ ] 720p 下一个 chunk 有 32 个 query block（448p 是 14 个），每步取的比例可能高于 22%

## 历史方向（已被取代）

- **10/4：推理 + KV 容量**。原始设想是 causal 长视频推理时 KV 放不下，把完整历史放 glass DRAM、按需取 top-k block。因"只讲容量发不了 paper"转向训练；10/8 回到推理，但以带宽和调度为主。
- **10/6：Wan2.1-14B（Krea Realtime）+ FastVideo**，当天改回 PAN（公开 FastVideo 没有 causal VSA）。
- **10/6–10/8：DF + VSA 训练**。结论见 `docs/training_reference.md`。

## 决策记录

| 日期 | 决策 / 变更 |
|---|---|
| 2026-10-04 | 确立项目目标：video gen + causal + memory + glass；采用 Tianqi 模型数字（8 PB/s，≥ 85 TB）。完成文献与会议调研（`survey/`），倾向 DAC 2027（约 11/18），DATE LBR 作保底。 |
| 2026-10-04 | 定位为 infra paper：质量对比以引用为主，核心实验 = 真实模型 forward 采集 trace → 内存 / 互连模拟。只用 VSA v1，不引入近存计算。 |
| 2026-10-06 | 主模型定为 PAN（组内支持最好、模型更大），remote top-k 按 5%；方向一度改为 DF + VSA 训练。 |
| 2026-10-07 | 完成训练显存实测（10–60 s）和 60 s 训练 trace。 |
| 2026-10-08 | 向导师汇报训练结果：dense PAN 训练在 NVL72 内受算力限制。**导师确认方向改为 causal model 的实时推理**，训练分析保留作对照。删除 LLM / VLM RL 估算。 |
| 2026-10-09 | 实时推理估算补充跨步缓存（4 步并集 27.7%）和 NVL72 的 Grace LPDDR5X；讲法定为"容量是起因，带宽和延迟是瓶颈"（草案）。主线不再考虑 MoE。 |
| 2026-10-09 | 精简文档：三份训练文档合并为 `docs/training_reference.md`（原文在 commit `cc4a3f0`）；推理相关的实测与 trace 移到 `docs/pan_measurements.md`。 |
| 2026-10-09 | 完成应用场景调研（`survey/causal_video_long_kv_scenarios.md`），整理成 `docs/motivation.md`。 |
| 2026-10-09 | 完成实时推理估算（`docs/realtime_inference_estimate.md`）：glass 的主要收益是容量；做好跨步 / 跨 chunk 复用后 DRAM 带宽需求为每 tile 0.1–2 TB/s；收益大小取决于比例选块还是固定预算。讲法待讨论。 |
| 2026-10-10 | 实测 LingBot-World v2 14B 和 Cosmos-H-Dreams（`docs/realtime_profile.md`）：完整历史 dense 做不到实时（120 s 时慢 4.5 倍）；选块恢复实时但每卡要存 74 GB KV；SP 通信每 chunk 约 70 ms、不随历史增长；CPU 内存取数每卡约 28 GB/s。PAN 不再实测。 |
| 2026-10-14 | 在 LingBot 14B 上接 Light Forcing 选块 + FA4 块稀疏 kernel（SP8，完整 KV）：60 s 历史每 chunk 0.96 s（dense 2.65 s），KV 仍线性增长；同一模型下 VSA 式选块的 chunk 间重合约 70%（与训练过 VSA 的 PAN 一致），Light Forcing 式约 30%，locality 由选块策略决定。见 `docs/realtime_profile.md` 第 3 节。 |
| 2026-10-15 | 长期记忆存 KV vs 存 latent 重算（`docs/realtime_profile.md` 第 6 节）：14B SP8 重算 8 帧每 chunk 多 283 ms，从 CPU 内存取 KV 46 ms；画质上（WorldKV 设置，LingBot-Fast 14B，往返转头 22.5 s，2 个 seed）回到超出窗口的旧视角时，存 KV 检索 PSNR 16.1、完整 KV 15.5、重算 11.4–12.5、滑动窗口 11.8，即不专门训练时重算的 KV 带不来长期记忆（深层 KV 含生成时上下文）。另给 Tianqi 导出了逻辑 attention trace（`docs/attn_trace.md`）。 |
| 2026-10-29 | Ring attention 开销（`docs/ring_attn_overhead.md`，起因是 4K 视频生成里 Ulysses 不够用、要 ring）：4K、5 s 切 8 卡时 NVLink 上 ring 只多 6%（推理）/ 2%（训练）；开销由每卡 token 数决定，每卡 16K token、全走 IB 时多 86% / 54%；ring 只藏住部分通信，另有约 9% / 3% 与带宽无关的固定开销；稀疏 attention 让 ring 变成带宽瓶颈。两节点（2 × 4 卡，真实 IB）实测：ring 8 跨节点每卡 85K token 多 12% / 10%，和单节点 IB 模拟一致；Ulysses 4 × ring 2 只多 2%。NCCL 默认每个 IB 对端 2 个通道，只有 20 GB/s/卡，设 4 个通道后两节点实测 26–34 GB/s，跨节点 ring 8 开销约减半（每卡 85K token：12% → 3%）。投影（4K 5 s）：IB 上 ring 64 多 145% / 80%（默认 / 4 通道），Ulysses 8 × ring 8 多 23% / 20%；稀疏时 115% / 61%，NVL72 / glass 18% / 16%。 |
