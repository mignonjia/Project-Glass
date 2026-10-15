# Motivation：实时 causal 视频生成需要长历史 KV

> 给 paper intro 和 11 月 review 用的论证。每条论据都能追到出处：场景条目、链接和完整引文在 `../survey/causal_video_long_kv_scenarios.md`；KV 公式和 PAN 实测在 `pan_measurements.md`。
> 标 **[原文核对]** 的引文已在 PDF 全文中逐字确认，其余引文来自论文摘要 / 正文 / 官方博客，由调研 agent 核对。"推算"是我们按 PAN 参数算的。

## 一句话

**实时 world model 的应用需要分钟到小时的历史，今天的系统只能留几秒，并把记忆不够列为主要失败原因；截止时间又让"放不下的历史"变成"每几毫秒就要按内容去取的历史"。**

## 论证链

### 1. 应用需要分钟到小时的历史

| 场景 | 需要多长 | 依据 |
|---|---|---|
| 交互游戏 / 开放世界 | 数分钟到数小时 | Genie 3 支持"a few minutes"；Odyssey 目标"hours, days, and eventually years of simulation" |
| 3D 场景探索 | 10 分钟以上，会回访 | World Labs RTFM 目标"interact … forever and the world will never be forgotten" |
| 多人共享世界 | 一局数分钟到数小时 | MIRA 4 人对战实测 5 分钟质量稳定，"rollouts continuing for hours" **[原文核对]** |
| 机器人策略评估 | 1 分钟以上，多视角 | DeepMind Veo 评估 Gemini Robotics："long-horizon (e.g., 1+ minutes) multi-view consistent generation remains a key technical milestone" |
| 自动驾驶闭环仿真 | 几分钟到几十分钟，5–7 相机 | Waymo："the longer the simulation, the tougher it is to compute and maintain stable quality" |
| 在 world model 里训练 agent | 一局 60 分钟 | Dreamer 4："Episodes last 60 minutes" **[原文核对]** |
| 实时数字人 | 一次通话 5–60 分钟 | Tavus 通话上限 60 分钟；HeyGen API 1 小时 ⚠️ |

### 2. 今天的系统只留几秒，失败归因于记忆

实际放进 attention 的历史：WHAMM 0.9 s、Matrix-Game 2.0 约 1 s、DreamDojo 约 1.2 s、MIRA 约 2 s、GAIA-2 约 2 s、HY-WorldPlay 约 2.7 s、GameNGen 3.2 s、Seaweed APT2 5 s、Dreamer 4 9.6 s。最长的视觉记忆约 60 s（Genie 3、LingBot-World）。

作者自己的归因：

- Dreamer 4："Its temporal consistency is limited to a context of 9.6 seconds"，未来方向是"integrating long-term memory" **[原文核对]**
- MIRA："Its rare failures trace mostly to a context window too short to carry match state, which lets the clock and score slip and off-screen cars be forgotten" **[原文核对]**
- WHAMM："the model can and will forget about objects that go out of view for longer than this"
- InteractiveAvatar："may degrade as early latents are discarded, or some generated objects may gradually disappear" **[原文核对]**
- Hunyuan-GameCraft-2：约 500 帧后语义漂移，"partly attributable to our model's lack of an explicit long-term memory mechanism"

记忆长度对质量的影响有量化数据：

- **GigaWorld-1**（机器人，40 s rollout）：Wan2.1 无记忆时 FVD 从 0–8 s 的 197 涨到 32–40 s 的 321；加分层记忆后为 35 → 98。
- **NVIDIA OmniDreams**（驾驶，20 s）：FVD 从 0–5 s 到 15–20 s，短上下文 109 → 409，长上下文 96 → 268。

### 3. 留得短是因为放不下

- **Seaweed APT2**：736×416 时窗口 N = 30 个 latent 帧（5 s）；"1280×720 is trained and inference with a smaller attention window size N = 15 to fit the memory" **[原文核对]**。分辨率一高，记忆就减半。
- **InteractiveAvatar**："storing all historical frames is computationally expensive and impractical for streaming generation" **[原文核对]**
- **HY-WorldPlay**：加大空间记忆会让 teacher 显存"greatly increase"。
- **PAN 实测**：448p、SF 4 步、单张 H200 保留完整历史，60 s 就到 130 GiB（`pan_measurements.md` §3）。

按 PAN 大小的模型推算（每 token 721 KB，720p 每秒 4.43 GB）：

| 场景 | 设定 | 每路 KV |
|---|---|---:|
| 游戏 / 探索 | 720p 单视角，10 分钟 | 2.7 TB |
| 多人 | 4 人 × 720p，10 分钟 | 10.6 TB |
| 机器人评估 | 448p × 4 视角，5 分钟 | 1.6 TB |
| 自动驾驶 | 720p × 7 相机，2 分钟 / 10 分钟 | 3.7 / 18.6 TB |
| agent RL | 360p，60 分钟（再乘并行 rollout 数） | 约 4 TB |
| 数字人 | 448p，60 分钟 | 4.7 TB |

一张 GB300 有 288 GB HBM，NVL72 合计约 20 TB：**单路流就要占多张卡，多路流时 NVL72 先撞容量上限**（README 估算：只用 HBM 时每系统 6–8 路）。

### 4. 实时把容量问题变成带宽问题

- **Dreamer 4**："Inference speed of block-causal transformer architectures is limited both by the FLOPs of the MLPs and the memory bandwidth needed to access the KV cache of a long context to attend into" **[原文核对]**
- **Matrix-Game 3.0**：按相机视野检索历史时，检索不放在 GPU 上，速度从约 40 fps 降到 6.6 fps，检索在关键路径上。
- **Context-as-Memory**：上下文从 1 帧加到 30 帧，帧率从 1.60 降到 0.79 fps。
- **OmniMem**（arXiv 2605.30519，Wan2.1-1.3B + Self-Forcing，按 head 稀疏检索）：保留完整历史 KV，"The full-resolution KV cache is organized as per-head, chunk-level tensors"；放不下的部分 "live in pinned host memory and are reloaded through PCIe when their chunks are selected"，GPU 上只用 LRU 留 7 个 chunk。它把跨 query、跨 head 选块的并集叫 "union explosion"："Although the method remains sparse in computation, it can still require a large KV memory footprint for the current AR step"。717 帧时比完整历史 attention 省 3.3× 显存、快 2.7×。这正是我们的 workload：完整 KV 存在容量层，按内容稀疏取回，瓶颈是取数而不是计算。
- **我们的 PAN trace**：每个 720p chunk 要在 0.67 s 内跑 220 轮"打分 → 取 → 算"，每轮约 3 ms；每步取约 22.9% 的远端历史，每路流约 4.4 TB/s（跨步缓存后约 1.1 TB/s）。

### 5. 访问由内容决定，不能提前预取

在多个场景里，取哪段历史要看到新内容后才知道：

- **3D 探索**：按新相机位姿的视野检索（HY-WorldPlay、Context-as-Memory、Matrix-Game 3.0、RTFM）。
- **多人**：一个玩家的画面要用到其他玩家建立的历史（MIRA、MultiGen、Agora-1）。
- **机器人 / 驾驶**：被遮挡的物体或车辆重新出现（Mem-World、Ctrl-World 按位姿检索、ZYT-World 跨轨迹回访）。
- **数字人**：按 SigLIP 内容选关键帧（InteractiveAvatar）。

- **按 head 选块不能省**：OmniMem 把选块在 12 个 head 间共享，"significantly degrades all metrics"，结论是 "different attention heads attend to genuinely different historical regions"；top-K 从 4 加到 12 质量继续上升（CLIP 26.53 → 27.44），预算越大越好。块只有 30 token（每 head K+V 约 15 KB），我们实测 PCIe 上 29 KB 的碎块只有约 5 GB/s，15 KB 只会更低。

这与 PAN VSA 的"每层、每个 head 打分后才知道取哪些块"同类，也是取数需求运行时确定、带偏斜的原因（README 讲法第 2–3 点）。

### 6. 并发和多视角把需求再放大

- **并发**：WOSAC 每个驾驶场景 32 路随机 rollout；WMPO 每个初始状态 8 路；Genie Envisioner"thousands of episodes per hour"；DreamGen 用 1500 张 L40 跑 54 小时；Waymo"billions of miles in virtual worlds"；Odyssey 商用成本 1–2 美元 / 用户小时。
- **多视角**：机器人 3–4 路，驾驶 5–7 路。OmniDreams 4 视角在一张 GB300 上只有 12 fps（不实时），要 4 卡（48 fps）。

对应 review 图表 1（每系统实时流数）：系统要能同时放下、并按时读到很多路长历史。

### 7. 对照：离线长视频已经用全历史

MoC（64 s，约 180k token）、LCT（约 3 分钟，causal 版保留全部 KV）已经按内容访问全历史，瓶颈是 FLOPs，不是延迟。FramePack 把内存压到 6 GB，代价是每帧 1.5–2.5 s。LoL 能连续生成 12 小时，但只用 12 个 latent 窗口 + 3 个 sink，自己也写"maintaining long-term memory still remains challenging"：**时长不等于记忆**。

所以**是实时的截止时间**让"大历史"变成"大、低延迟、按内容随机访问的历史"，即从容量问题变成带宽和调度问题。这正是 glass 要解决的。

## 预期的质疑与回应

| 质疑 | 回应 |
|---|---|
| 压缩历史就够了（FramePack、PackForcing、Yume 1.5） | 压缩是有损的：Self-Forcing++ 写长时间被遮挡的区域会"content divergence"；GigaWorld 的 FVD 数据显示记忆越完整，长程质量越好。压缩路线和我们正交：压缩后的 KV 同样随时长线性增长 |
| 用显式 3D（Marble、Voyager、RTFM） | 只适用于静态场景（RTFM、Marble 原文都限定静态）；游戏状态、动态物体、多人状态放不进 3D 重建 |
| 只存 latent、按需重算 KV（HY-WorldPlay） | 每个 chunk 重算约 20 帧，计算约翻倍，记忆也只有约 20 帧；PAN VSA 每层、每个 head 按内容选块，必须用每层的 K 打分，只能存 KV（README 对照路线） |
| GQA / MLA 能把 KV 缩小 | 只是常数倍（4–8×）：10 分钟 720p 仍是每路 0.33–0.67 TB，超过一张 GB300 的 288 GB，且随时长线性增长 |
| 便宜的数字人根本不用长历史（AVTR-1 用 0.64 MB 运动状态） | 对，所以数字人不作为主场景；主场景是游戏 / 探索 / 多人 / 机器人 / 驾驶 / RL，它们的历史无法由输入或参考图提供 |
| SSM 用常数大小的状态（Long-Context SSM world model） | 架构路线不同，长程检索精度仍是开放问题（WorldTrace 等）；我们针对的是今天主流的 attention + KV 的 causal DiT |
| 模型训练长度短，用不了长历史 | MIRA、LingBot-World、Self-Forcing++ 都能远超训练长度生成；WorldTrace 处理位置编码越界。长历史的价值在提高，系统需要先能放下、取到 |

## 建议的 intro 叙事（草案）

1. World model 正在从短片段走向长时间交互：游戏、探索、多人、机器人评估、驾驶仿真、agent 训练（第 1 节表）。
2. 这些应用需要分钟到小时的记忆，但现有系统只保留几秒，作者普遍把失败归因于记忆（第 2 节引文）。
3. 原因是容量：PAN 大小的模型 10 分钟 720p 每路 2.7 TB，多视角、多人、并发再放大（第 3、6 节）。
4. 实时让它变成带宽和调度问题：每个 chunk 0.67 s 内要按内容取 TB 级历史中的约 1/4，且取哪些块要打分后才知道（第 4、5 节）。
5. 离线长视频没有这个问题（第 7 节），所以这是实时 world model 特有的系统挑战，glass 的大容量 + 高带宽 scale-up 域正好对应。

## 待补

- [ ] 用 PAN 实测补一条"记忆越长质量越好"的自有数据（可选；原则上以引用为主，见 README 方法原则）。
- [ ] 第 1 节表里 ⚠️ 的产品数字（HeyGen、Mirage 2 等）正式引用前再核对或删除。
- [ ] 第 3 节表和 README 的每系统流数估算对齐，等 `realtime_inference_estimate.md` 写完后统一数字。
