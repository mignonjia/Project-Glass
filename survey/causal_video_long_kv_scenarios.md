# 需要长 KV cache 的 causal 视频生成：应用场景调研（2026-10-09）

> 用途：paper motivation。回答"谁需要实时生成、同时保留很长的历史，要多长、多快、多少路"。
> 方法类工作（CausVid、Self Forcing、LongLive、WorldMem、SSM world model、WorldTrace 等）见 `glass_memory_papers.md` §4，这里不重复，只在需要时引用。
> 条目都经网络检索核对（arXiv / 官方博客 / 产品页）。标 **[原文核对]** 的引文已在 PDF 全文中逐字确认；标 ⚠️ 的只有二手来源或部分信息缺失。"推算"表示数字是我们根据论文给出的参数算出来的。

## 先读这些

1. **Dreamer 4**（arXiv 2509.24527）：任务一局 60 分钟，模型只记得 9.6 s；作者直接写推理受"读长上下文 KV cache 的内存带宽"限制。**最适合放进 intro 的一句话。**
2. **MIRA**（arXiv 2607.05352）：4 人实时对战，能跑几个小时，但失败主要来自"上下文窗口太短，装不下比赛状态"。
3. **Genie 3 / Project Genie**（DeepMind 博客）：工业界标杆，视觉记忆约 1 分钟，产品每次限 60 s。
4. **Evaluating Gemini Robotics Policies in a Veo World Simulator**（arXiv 2512.10675）：把"1 分钟以上、多视角一致"列为关键里程碑。
5. **Seaweed APT2**（arXiv 2506.09350）：720p 时把注意力窗口从 30 个 latent 帧减到 15 个"to fit the memory"，**容量直接限制了记忆长度**的实例。
6. **NVIDIA OmniDreams**（arXiv 2606.03159）：GB300 上的闭环驾驶仿真，4 视角单卡做不到实时，需要 4 卡。

---

## 0. 怎么估每路流的 KV（PAN 等价）

用 `docs/pan_measurements.md` §2 的公式：

- **每个 token 的 KV** = 2（K、V）× 44 层 × 4096（hidden）× 2 B（bf16）= **721 KB**。
- **720p、24 fps**：一个 chunk = 4096 行（补齐后），时长 0.67 s → 每秒 6144 行 → 6144 × 721 KB ≈ **4.43 GB/s**，即每分钟 266 GB、10 分钟 2.66 TB、1 小时 15.9 TB。
- **448p、16 fps**（实测分辨率）：每秒 1792 行 → **1.29 GB/s**，即每分钟 77.5 GB、10 分钟 0.78 TB、1 小时 4.7 TB。
- **多视角**：每个相机都是一路 token，KV 按视角数翻倍（机器人 3–4 路、自动驾驶 5–7 路）。
- **注意**：这是"PAN 大小的模型"（29B，无 GQA）的数字。小模型的 KV 小得多，例如 Wan-1.3B（30 层 × 1536）每 token 184 KB，约 PAN 的 1/4；Wan-14B（40 层 × 5120）每 token 819 KB，与 PAN 相当（推算）。Krea Realtime 14B 的博客写训练时 KV"每张 GPU 可达 25 GB"，可以作为旁证。

---

## 1. 总表

"现有系统实际保留的上下文"指放进 attention 的历史，不含 sink / 参考帧。KV 估算按第 0 节，取场景的目标时长。

| 场景 | 用户需要的历史 | 现有系统实际保留 | 分辨率 / 帧率 | 实时预算 | 历史访问方式 | 并发 | 每路 KV（PAN 等价，推算） |
|---|---|---|---|---|---|---|---|
| A. 交互游戏 / 开放世界 | 数分钟到数小时（Genie 3 几分钟，Odyssey 目标"数小时、数天"） | 0.7–3 s（最长约 60 s 视觉记忆） | 720p、20–30 fps | 每帧 33–50 ms；输入到画面 40–200 ms | 回到旧地点、找回游戏状态：**由内容决定** | 每用户 1 路；约 1–2 美元 / 用户小时 | 720p 10 min 2.7 TB；1 h 16 TB |
| B. 3D 场景探索（回访） | 10 分钟以上；RTFM 目标"永远不忘" | 约 2.7 s（WorldPlay）；按视野检索 20 帧（Context-as-Memory） | 480p–720p、12–24 fps | 同上 | 按相机视野 / 位姿检索：**由内容决定，取哪些要算完才知道** | 每用户 1 路 | 720p 10 min 2.7 TB |
| C. 多人共享世界 | 一局数分钟到数小时（MIRA 能跑几小时） | 0.67–2 s | 288×512 到 720p、20–30 fps；N 个视角 | 每帧约 35–50 ms | 一个玩家的画面要用到其他玩家建立的历史：**跨视角、由内容决定** | 2–4 人 / 局；MASS 1024 人（非实时） | 4 人 × 720p，10 min 10.6 TB |
| D. 具身智能 / 机器人策略评估 | 1 分钟以上（Veo 作者原话）；家务任务数分钟 | 1.2 s（DreamDojo）到约 7–14 s 稀疏（Ctrl-World） | 192×320 到 640×480；3–4 视角 | 闭环评估要实时或快于实时 | 被遮挡物体、任务进度：**由内容决定** | 每个评估几千集 / 小时（GE-Sim） | 448p × 4 视角，5 min 1.6 TB |
| E. 自动驾驶闭环仿真 | 基准 4–20 s；真实路线 2 分钟到几十分钟 | 1.5–2 s（DrivingWorld、GAIA-2）；24–32 帧（OmniDreams） | 720p × 5–7 相机，10–30 Hz | 实时（OmniDreams 以 30 fps 为线） | 被遮挡车辆、环线回访：**由内容决定** | 每场景 32 路随机 rollout（WOSAC）；"数十亿英里"（Waymo） | 720p × 7 视角，2 min 3.7 TB；10 min 18.6 TB |
| F. 在 world model 里训练 agent（RL） | 一局 20–60 分钟（Dreamer 4 Minecraft） | 9.6 s（Dreamer 4） | 640×360、20 fps | 总吞吐要远快于实时 | 同 A | **大量并行 rollout**（WMPO 每状态 8 路；DreamGen 1500 张 L40） | 360p 约 1.1 GB/s，60 min 约 4 TB / 路 |
| G. 实时数字人 / avatar | 一次通话 5–60 分钟（Tavus、HeyGen API 上限 1 h） | 参考帧 + 1.5–4 s 窗口 | 512²–720p、25 fps | 首帧 0.3–1 s；每轮回应 < 1 s | 身份靠固定参考；回忆通话中出现过的物体：**由内容决定（刚出现）** | 每路 2–8 张 H800（14B 模型） | 448p 60 min 4.7 TB |
| H. 实时视频转视频（直播风格化） | 目标"一小时"（StreamDiffusionV2）、"数小时"（Lucy 2） | 3–16 帧，或一帧大小的特征库 | 480p–1080p、20–30 fps | 每帧 33–50 ms | 输入视频本身提供场景，主要只需保持外观 / 身份：**固定参考为主** | 约 1.2 美元 / 分钟（Lucy 2） | 需求弱，不作为主场景 |
| 对照：离线长视频 | 1–3 分钟一个场景（MoC、LCT） | 已经能全历史 + 按内容路由（约 180k token） | 480p | **没有截止时间** | 由内容决定 | 无 | 只是容量 / 算力问题 |

**结论一句话**：A–G 的共同模式是"一个小的近期窗口 + 一个随会话时长线性增长、按内容稀疏访问的大历史"。今天所有实时系统都只保留几秒，把记忆不足列为主要失败原因；离线场景已经能用全历史，因为它没有截止时间。

---

## A. 交互游戏 / 开放世界 world model

**为什么需要长历史**：转身回头要看到原来的东西（物体恒常性），游戏状态（分数、时间、背包、已破坏的方块）要一直保持，动作的后果要延续。

| 工作 | 来源 | 规格 | 记忆机制与长度 | 关于遗忘 / 记忆的原话 |
|---|---|---|---|---|
| Genie 3 / Project Genie（DeepMind） | [博客 2025-08](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)；[Project Genie 2026-01](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/project-genie/) | 720p、24 fps，实时；硬件未公开 | 机制未公开；视觉记忆约 1 分钟，几分钟内基本一致；产品每次限 60 s | "a few minutes of continuous interaction, rather than extended hours"；"inaccuracies tend to accumulate over time" |
| Genie 2（DeepMind） | [博客 2024-12](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) | 未公开；蒸馏版实时但画质下降 | 最长一致约 1 分钟，多数示例 10–20 s | "remembering parts of the world that are no longer in view" |
| Oasis（Decart + Etched） | [项目页 2024-10](https://oasis-model.github.io/) | 20 fps；360p / H100 ⚠️（媒体报道） | 短上下文逐帧预测 | 项目页自己列出 "Limited memory over long horizons"；转身后地形会变 |
| GameNGen（Google） | ICLR 2025，[arXiv 2408.14837](https://arxiv.org/abs/2408.14837) | DOOM，320×240、20 fps，1 张 TPU | 64 帧，约 3.2 s | "only has access to a little over 3 seconds of history" |
| WHAMM（Microsoft） | [博客 2025-04](https://www.microsoft.com/en-us/research/articles/whamm-real-time-world-modelling-of-interactive-environments/) | Quake II，640×360、10+ fps | 9 帧 = 0.9 s | "the model can and will forget about objects that go out of view for longer than this" |
| Matrix-Game 2.0（Skywork） | [arXiv 2508.13009](https://arxiv.org/abs/2508.13009) | 352×640、25 fps，1 张 H100 | 滚动 KV，6 个 latent 帧（约 1 s，推算） | "maintaining content consistency and history over long video generations remains challenging" |
| Matrix-Game 3.0（Skywork） | [arXiv 2604.08995](https://arxiv.org/abs/2604.08995)（2026-04） | 720p、最高 40 fps（5B），8 卡 DiT + 1 卡 VAE | 按相机视野检索的记忆 + 短窗口 + 首帧 sink；能回访分钟级序列 | **系统观察**：检索不放在 GPU 上时，速度从约 40 fps 降到 6.6 fps，检索在关键路径上 |
| Hunyuan-GameCraft-2（腾讯） | [arXiv 2511.23429](https://arxiv.org/abs/2511.23429) | 832×448、16 fps，用序列并行 | sink + 局部窗口 | 约 500 帧后语义漂移，"partly attributable to our model's lack of an explicit long-term memory mechanism" |
| Yan（腾讯） | [arXiv 2508.08601](https://arxiv.org/abs/2508.08601) | 1080p；Yan-Gen 1 张 H20 12–17 fps，4 张 30 fps | KV cache + 移动窗口，窗口大小未公开 | "ensuring visual consistency across long video durations remains a challenge" |
| LingBot-World（蚂蚁） | [arXiv 2601.20540](https://arxiv.org/abs/2601.20540)（2026-01） | 720p；Fast 版 480p、16 fps，"一个 GPU 节点" | 滚动 KV；地标离开视野 60 s 后仍保持；生成最长 10 分钟 | "memory is an emergent ability derived from the context window rather than an explicit storage module" |
| Odyssey-1 / -2 / -2 Pro | [Odyssey-1 2025-05](https://odyssey.systems/introducing-odyssey-1) | 最高 30 fps、40 ms / 帧，H100 集群；-2 Pro 720p、22 fps | 未公开；连续 5 分钟以上 | 成本 **1–2 美元 / 用户小时**；目标"hours, days, and eventually years of simulation" |
| Runway GWM-1 Worlds | [博客 2025-12](https://runway.com/research/introducing-runway-gwm-1) | 720p、24 fps ⚠️（TechCrunch） | 未公开；最长 2 分钟 | "Turn around, and what was behind you is still there" |
| Mirage 2（Dynamics Lab） | ⚠️ 官方博客打不开，只有媒体报道 | 单卡，延迟约 200 ms | 未公开；连续 10 分钟以上 | — |

## B. 3D 场景探索（回访）

**为什么需要长历史**：用户会走回去看。访问的历史**由相机位姿决定**：取哪些帧要等新位姿出来才知道，正好对应我们"打分后才知道取哪些块、不能提前预取"的论点。

| 工作 | 来源 | 规格 | 记忆机制 | 要点 |
|---|---|---|---|---|
| HY-WorldPlay / HY-World 1.5（腾讯混元） | [arXiv 2512.14614](https://arxiv.org/abs/2512.14614) | 720p、24 fps；发布的权重是 480p；单卡 72 GB，sp=8 时每卡 28 GB | 3 个近期 chunk + 1 个按视野 / 距离检索的空间 chunk，共约 16 latent ≈ 2.7 s（推算）；重排位置编码 | 只生成约 30 s；遮挡时视野检索"may fail"；空间记忆加大 "may significantly increase the teacher model's memory size"（指上下文帧数，不是显存；2026-10-30 核对）。**README 里的对照路线** |
| Context-as-Memory（港大 / 快手） | [arXiv 2506.03141](https://arxiv.org/abs/2506.03141) | 640×352，**0.97 fps**，非实时 | 按视野检索 20 帧 | 速度随上下文变慢：1 帧 1.60 fps，20 帧 0.97 fps，30 帧 0.79 fps；"Including all historical frames … would be extremely resource-intensive" |
| World Labs RTFM | [博客 2025-10](https://www.worldlabs.ai/blog/rtfm) | 1 张 H100，"interactive framerates" | 带位姿的帧作空间记忆，按区域检索附近的帧 | "You can interact with RTFM forever and the world will never be forgotten"；只支持静态世界 |
| Yume 1.5（上海 AI Lab / 复旦） | CVPR 2026，[arXiv 2512.22096](https://arxiv.org/abs/2512.22096) | 540p、12 fps，1 张 A100 | 去掉 KV cache，历史分层压缩 | 只展示 30 s；动机就是"memory consumption … as the generated video length grows" |
| 对照：World Labs Marble、Voyager、Matrix-3D | [Marble](https://www.worldlabs.ai/blog/marble-world-model)、[arXiv 2506.04225](https://arxiv.org/abs/2506.04225)、[arXiv 2508.08086](https://arxiv.org/abs/2508.08086) | 离线 | 显式 3D（高斯 / 点云 / 网格） | "用显式 3D 代替 KV 历史"的路线，只适用于静态场景 |

## C. 多人共享世界

**为什么需要长历史**：每个玩家的画面都要和其他玩家建立的状态一致。玩家互相看不见后再相遇，现有模型普遍丢失共享状态。

| 工作 | 来源 | 规格 | 记忆 | 原话 |
|---|---|---|---|---|
| MIRA（General Intuition / Kyutai / Epic） | [arXiv 2607.05352](https://arxiv.org/abs/2607.05352)（2026-07） | Rocket League 2v2，4 个视角 288×512，**1 张 B200 上 20 fps** | 滚动 KV，约 2 s（推算） | **[原文核对]** "Its rare failures trace mostly to a context window too short to carry match state, which lets the clock and score slip and off-screen cars be forgotten"；"in practice we observe rollouts continuing for hours" |
| Multiverse（Enigma Labs） | [HF](https://huggingface.co/Enigma-AI/multiverse)（2025-05） | 双人 GT4，30 fps | 最近 4 帧 + 更早的稀疏帧，约 0.67 s | 更密的窗口会"make the model too slow for real time play, increase its memory usage" |
| Solaris（NYU） | [arXiv 2602.22208](https://arxiv.org/abs/2602.22208) | 双人 Minecraft，非实时 | 6 latent ≈ 1.2 s | "our model lacks persistent memory" |
| MultiGen（Stanford / Google） | [arXiv 2603.06679](https://arxiv.org/abs/2603.06679) | Doom 2–3 人，约 20 fps，每人 1 张 A100 | 显式小地图 + 位姿记忆 | 只靠上下文时"relevant layout cues may fall out of the buffer" |
| Agora-1（Odyssey） | [博客 2026-05](https://odyssey.systems/introducing-agora-1) | 4 人 GoldenEye | 显式共享状态（血量、位置） | 指出 Multiverse、Solaris "struggle … when players lose sight of one another" |
| MASS / Khora | [arXiv 2608.06257](https://arxiv.org/abs/2608.06257)、[arXiv 2608.08600](https://arxiv.org/abs/2608.08600)（2026-08） | MASS 1024 人、1 万步（非实时）；Khora 实时 ⚠️ 规格未公开 | 共享世界状态 + 每人渲染 | MASS 认为每视角各存一份 latent 会"redundant compute… poor scalability" |

## D. 具身智能 / 机器人

**为什么需要长历史**：腕部相机遮挡多、运动快，物体离开视野后要记住；任务进度（已放好几个、抽屉开没开）要保持；评估一个策略要整集 rollout。

| 工作 | 来源 | 规格 | 记忆 / 时长 | 原话 / 数字 |
|---|---|---|---|---|
| Veo World Simulator 评估 Gemini Robotics（DeepMind） | [arXiv 2512.10675](https://arxiv.org/abs/2512.10675) | 4 视角拼图；与 1600+ 次真机实验对照 | 每集 8 s rollout | "Achieving long-horizon (e.g., 1+ minutes) multi-view consistent generation remains a key technical milestone" |
| 1X World Model | [博客 2025-06](https://www.1x.tech/discover/redwood-ai-world-model) | 256² / 512² | 4 s 片段 | 下肢误差累积，"limits the capability … for long-horizon navigation" |
| Ctrl-World（Stanford / 清华） | ICLR 2026，[arXiv 2510.10125](https://arxiv.org/abs/2510.10125) | 3 视角 192×320，1.5B | 按位姿检索 7 帧历史，间隔 1–2 s（约 7–14 s 稀疏）；20 s 以上连贯 | "Removing memory leads to blurry predictions"；用想象数据后训练，成功率 38.7% → 83.4% |
| GigaWorld-1 / WMBench | [arXiv 2607.02642](https://arxiv.org/abs/2607.02642) | 多视角，40 s rollout | 分层历史（首帧 + 短 / 中 / 长程） | 评估质量"dominated by long-horizon, action-faithful rollout consistency"；Wan2.1 无记忆时 FVD 从 0–8 s 的 197 涨到 32–40 s 的 321，有记忆时 35 → 98 |
| Mem-World | [arXiv 2606.18960](https://arxiv.org/abs/2606.18960)，CoRL 2026 | 多视角 | 4D 腕部 surfel 记忆 | 没有记忆时模型会"forget or hallucinate scene details seen in earlier frames"；与真机成功率的相关性比 Ctrl-World 高 14.5% |
| DreamDojo（NVIDIA 等） | [arXiv 2602.06949](https://arxiv.org/abs/2602.06949) | 640×480 单视角；蒸馏后 1 张 H100 10.8 fps | 12 帧滑窗（约 1.2 s）；600 帧约 1 分钟 | 策略评估 Pearson r = 0.995；"generation quality can still degrade over long horizons" |
| Genie Envisioner（智元） | [arXiv 2508.05635](https://arxiv.org/abs/2508.05635) | 3 视角，视频 5 Hz | 4 个稀疏记忆帧 | "thousands of episodes per hour"（集群并行） |
| WPE（Stanford / NYU / GDM） | [arXiv 2506.00613](https://arxiv.org/abs/2506.00613) | 256²，几百步交互 | 9 帧上下文最好 | 长任务偏差大：LIBERO-Long 真值 0.47，估计 0.16 |
| V-JEPA 2-AC（Meta） | [arXiv 2506.09985](https://arxiv.org/abs/2506.09985) | 单相机 256²，4 fps | — | 每个动作 16 s（Cosmos-AC 要 4 分钟，"a full pick & place trajectory takes over one hour"）；长 rollout 精度下降 |
| DreamGen / GR00T-Dreams（NVIDIA） | [arXiv 2505.12705](https://arxiv.org/abs/2505.12705) | 2×2 拼多视角，短片段 | — | 规模参考："240k-sample RoboCasa dataset took 54 hours on 1500 NVIDIA L40 GPUs" |

## E. 自动驾驶闭环仿真

**为什么需要长历史**：被遮挡的车辆和行人要记住；环线回到同一路口要一致；5–7 个相机让 token 数成倍增加。

| 工作 | 来源 | 规格 | 记忆 / 时长 | 要点 |
|---|---|---|---|---|
| NVIDIA OmniDreams | [arXiv 2606.03159](https://arxiv.org/abs/2606.03159)（2026-06） | 704×1280，单视角或 4 视角；GB300 | 局部窗口 24 帧（单视角）/ 32 帧（多视角）；每集 20 s，长期一致性测 60 s | 单视角单卡 68 fps；**4 视角单卡 12 fps（不实时）**，4 卡 48 fps，16 卡 105 fps。FVD 从 0–5 s 到 15–20 s：短上下文 109 → 409，长上下文 96 → 268 |
| GAIA-2（Wayve） | [arXiv 2503.20523](https://arxiv.org/abs/2503.20523) | 最多 5 相机 448×960，20–30 Hz | 窗口 48 帧（约 2 s） | "temporal or semantic inconsistencies, particularly in long-horizon … scenarios" |
| GAIA-3（Wayve） | [博客 2025-12](https://wayve.ai/thinking/gaia-3/) | 15B，5 相机 | 声称长轨迹、遮挡下保持一致 | 离线评估；实时是"a focus" |
| Waymo World Model（基于 Genie 3） | [博客 2026-02](https://blog.waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation) | 相机 + 激光雷达；有实时变体 | 未公开 | "the longer the simulation, the tougher it is to compute and maintain stable quality"；"billions of miles in virtual worlds" |
| X-World（小鹏） | [arXiv 2603.19979](https://arxiv.org/abs/2603.19979) | 7 环视相机，12 fps | 固定大小滚动 KV、FIFO 淘汰；最长展示 24 s | 用于闭环评估和端到端驾驶的在线 RL |
| ZYT-World | [arXiv 2609.21712](https://arxiv.org/abs/2609.21712) | 4 鱼眼 + 3 针孔 | 30 s rollout + 跨轨迹回访的隐式记忆 | ⚠️ 绝对延迟未在摘要中给出 |
| Epona / DrivingWorld / Vista | [2506.24113](https://arxiv.org/abs/2506.24113)、[2412.19505](https://arxiv.org/abs/2412.19505)、Vista（NeurIPS 2024） | 前视单相机 | 上下文 10 帧 / 1.5 s / 3 帧；生成 120 s / 40 s / 15 s | Epona：不加特殊训练时 10–20 s 后"deteriorates rapidly" |
| Tesla 神经世界模拟器 | ⚠️ 只有演讲和二手报道 | 8 相机、24 fps、实时 ⚠️ | — | "每天 500 年驾驶" ⚠️ |

参照：标准闭环基准的时长：NAVSIM 4 s，WOSAC 8 s（每场景 32 路随机 rollout），nuPlan 15 s，CARLA Leaderboard 2.0 路线 7–10 km（约几十分钟 ⚠️ 推算）。

## F. 在 world model 里训练 agent（RL）

**为什么需要长历史**：agent 要学的任务本身很长；world model 忘了，agent 学到的就是错的物理。**这个场景同时需要长历史和大量并发**，是 KV 总量最大的。

| 工作 | 来源 | 规格 | 记忆 / 时长 | 原话 / 数字 |
|---|---|---|---|---|
| Dreamer 4（Hafner 等；GDM） | [arXiv 2509.24527](https://arxiv.org/abs/2509.24527) | 2B，640×360、20 fps，1 张 H100 21 fps | 192 帧 = 9.6 s（之前的模型 0.8–1.6 s） | **[原文核对]** "Episodes last 60 minutes"；"Its temporal consistency is limited to a context of 9.6 seconds"；"Inference speed of block-causal transformer architectures is limited both by the FLOPs of the MLPs and **the memory bandwidth needed to access the KV cache of a long context** to attend into"；未来方向："integrating long-term memory into the world model and agent" |
| Genie 3 + SIMA 2（DeepMind） | [SIMA 2 博客 2025-11](https://deepmind.google/discover/blog/) ⚠️ 具体页面未逐一核对 | 720p、24 fps | Genie 3 视觉记忆约 1 分钟 | SIMA 2 在 Genie 3 世界里自我改进 |
| WMPO（港科大 / 字节） | ICLR 2026，[arXiv 2511.09515](https://arxiv.org/abs/2511.09515) | VLA 在像素 world model 里做 GRPO | 4 帧上下文；每个初始状态 8 路 rollout，几百帧 | "errors can accumulate, leading to severe degradation over long horizons" |
| UniSim | ICLR 2024，[arXiv 2310.06114](https://arxiv.org/abs/2310.06114) | — | — | 首次在学到的视频模拟器里训练高层规划和低层 RL 策略，零样本迁移到真机 |

**规模推算**：Dreamer 4 单视角每秒约 5.1k token，60 分钟一局约 1800 万 token。按 PAN 大小的模型算（360p 约 720p 的 1/4 token），每路约 1.1 GB/s，60 分钟约 4 TB；RL 需要几十到几千路并行。

## G. 实时数字人 / avatar

**为什么需要长历史**：长通话中身份、背景要一致；对话里出现过的物体、话题之后要能"回来"。现在都靠固定参考帧保身份，物体和场景的长期一致性开始被当作问题提出。

| 工作 | 来源 | 规格 | 记忆 | 原话 / 数字 |
|---|---|---|---|---|
| InteractiveAvatar（西交 / 中国电信 TeleAI） | [arXiv 2606.22905](https://arxiv.org/abs/2606.22905)（2026-06） | Wan2.2-5B，1024×576、26.7 fps，H100 | 5 s 压缩短期窗口 + 按内容（SigLIP2 关键帧）选的固定容量长期缓存 | **[原文核对]** "storing all historical frames is computationally expensive and impractical for streaming generation"；局限："may degrade as early latents are discarded, or some generated objects may gradually disappear" |
| Live Avatar（阿里夸克） | [arXiv 2512.04677](https://arxiv.org/abs/2512.04677) | 14B，720×400、20.9 fps，**5 张 H800**（单卡约 5 fps） | 1 个 sink 帧 + 4 个 block 的 KV 窗口 | 10,000 s rollout 指标平稳；"dependence on RSFM may compromise long-term temporal consistency" |
| SoulX-FlashTalk（Soul） | [arXiv 2512.23379](https://arxiv.org/abs/2512.23379) | 14B，720×416、32 fps，**每路一整台 8×H800** | 5 个运动帧 + 参考帧 | 展示 1000 s |
| StreamAvatar | CVPR 2026，[arXiv 2512.22065](https://arxiv.org/abs/2512.22065) | 928×704、25 fps，2×H800 | 4 帧参考 sink + 6 帧滚动 KV | 局限："inconsistent content in regions that remain occluded for an extended period" |
| LiveTalk（上交 / GAIR） | [arXiv 2512.23576](https://arxiv.org/abs/2512.23576) | 512²、24.8 fps | 3 个 sink block + 2 个滚动 block | "accumulated errors can cause color drift and geometric distortions" |
| TalkingMachines（Character.AI） | [arXiv 2506.03099](https://arxiv.org/abs/2506.03099) | 18B，512²，DiT 和 VAE 分卡 | 只看上一个 chunk + 参考 chunk | 声称"infinite"，无时长与漂移数据 |
| 产品：Tavus CVI | [定价页](https://www.tavus.io/pricing) | 1080p，平均回应 < 500 ms | 未公开 | 单次通话上限 5 / 15 / 60 分钟（按档位），并发 1–15 路；超额 0.26–0.35 美元 / 分钟 |
| 对照：AVTR-1（Avaturn） | [arXiv 2609.22913](https://arxiv.org/abs/2609.22913) | 153M 运动模型，RTX 3070 实时 | 每会话状态约 0.64 MB（3 s） | 便宜的生产级 avatar 用小运动 latent，**完全避开长历史** |

## H. 实时视频转视频（直播风格化）

输入视频每帧都提供场景内容，模型主要只需保持外观和身份，**对按内容访问长历史的需求弱**，不作为主场景，可在讨论中一笔带过。

| 工作 | 来源 | 规格 | 记忆 | 要点 |
|---|---|---|---|---|
| StreamDiffusionV2 | MLSys 2026，[arXiv 2511.07399](https://arxiv.org/abs/2511.07399) | 480p；4×H100 58 fps（14B）；首帧 < 0.5 s | 滚动 KV + 自适应 sink + RoPE 重置 | 目标"hour-long session"；之前的模型"maintain coherence only within that range"（约 5–10 s） |
| MirageLSD / Lucy 2（Decart） | ⚠️ 技术报告链接已失效；[fal 页面](https://fal.ai/lucy-2) | Mirage 768×432、20 fps；Lucy 2 1080p、30 fps | 未公开 | 声称"Streams can persist for hours"；0.02 美元 / 秒 |
| StreamV2V | ICLR 2025，[arXiv 2405.15757](https://arxiv.org/abs/2405.15757) | 512²、20 fps，1 张 A100 | 压缩到一帧大小的特征库 | — |
| Krea Realtime 14B | [博客 2025-10](https://www.krea.ai/blog/krea-realtime-14b) | 1 张 B200 11 fps | 只保留 3 个 latent 帧 | 训练时 KV"can grow to 25GB per GPU" |

---

## 对照：离线长视频（没有截止时间）

| 工作 | 来源 | 时长 / 上下文 | 要点 |
|---|---|---|---|
| Mixture of Contexts（MoC） | ICLR 2026，[arXiv 2508.21058](https://arxiv.org/abs/2508.21058) | 8 镜头 × 8 s = 64 s，约 180k token | 已经能按内容路由全部历史；"must retain and retrieve salient events across long range" |
| Long Context Tuning（LCT） | ICCV 2025，[arXiv 2503.10589](https://arxiv.org/abs/2503.10589) | 约 20 个镜头、约 3 分钟 | causal 版本保留全部历史 KV；"a shot may refer to elements from distant earlier shots" |
| FramePack | [arXiv 2504.12626](https://arxiv.org/abs/2504.12626) | 60 s、30 fps | 压缩到恒定长度，6 GB 显存即可，但每帧 1.5–2.5 s（4090）：**内存靠压缩解决，代价是时间** |
| LoL（"Longer than Longer"） | [arXiv 2601.16914](https://arxiv.org/abs/2601.16914) | 12 小时连续流，1 张 H100 20 fps | 只用 12 个 latent 局部窗口 + 3 个 sink；"maintaining long-term memory still remains challenging, especially for multi-hour videos"：**时长不等于记忆** |
| Self-Forcing++ / Rolling Sink | [2510.02283](https://arxiv.org/abs/2510.02283)、[2602.07775](https://arxiv.org/abs/2602.07775) | 4 分 15 秒 / 5–30 分钟 | 滚动窗口；"lack of long-term memory, which can cause content divergence in regions occluded for extended periods" |
| Seaweed APT2（字节 Seed） | NeurIPS 2025，[arXiv 2506.09350](https://arxiv.org/abs/2506.09350) | 实时：736×416 1 张 H100 24.8 fps；720p 8 张 H100 | **[原文核对]** 窗口 N = 30 个 latent 帧（5 s）；"1280×720 is trained and inference with a smaller attention window size N = 15 to fit the memory" |
| 商用：Sora 2、Veo 3.1 | 官方公告 | 15–25 s；Veo 的"场景延伸"只看上一段最后 1 s | ⚠️ 2026 年最新上限未核对 |

**离线和实时的区别**：离线已经能用全历史（MoC、LCT），瓶颈是 FLOPs 和位置编码长度，不是延迟；FramePack 说明只要接受慢，内存可以压缩掉。**实时的截止时间才把"大历史"变成"大、低延迟、按内容随机访问的历史"**，即从容量问题变成带宽问题。

---

## 对 Glass 的含义

1. **容量**：A–G 都要几分钟到几小时，PAN 大小的模型每路 0.3–19 TB（第 1 节总表）。现有系统全部截到几秒，APT2 明确写"to fit the memory"把窗口减半，Krea Realtime 14B 写"cannot let the KV cache … grow indefinitely due to memory constraints"（逐字引文汇总见 `kv_cache_memory_quotes.md`）。
2. **带宽 + 截止时间**：Dreamer 4 直接写推理受"读长上下文 KV 的内存带宽"限制；Matrix-Game 3.0 的检索不放在 GPU 上时从 40 fps 掉到 6.6 fps；Context-as-Memory 的帧率随上下文增大而下降。
3. **按内容、不可预取**：B（按视野检索）、C（跨玩家）、D / E（遮挡后重现）、G（InteractiveAvatar 按 SigLIP 选关键帧）都是"看到新内容才知道取哪段历史"，和 PAN VSA 的"打分后才知道取哪些块"同类。
4. **多路并发**：E（每场景 32 路）、F（RL 并行 rollout、DreamGen 1500 张卡）、商用服务（1–2 美元 / 用户小时）都要求一个系统同时跑很多路，对应 review 图表 1（每系统流数）。
5. **多视角放大**：机器人 3–4 路、驾驶 5–7 路。OmniDreams 4 视角在一张 GB300 上做不到实时，需要 4 卡，对应"每路流跨多张卡、取数走互连"的情形。

**建议的 intro 叙事**：游戏 / 探索（A、B）讲"需要回访"，驾驶 / 机器人（D、E）讲"多视角 + 评估需要分钟级"，RL（F）讲"长且并发"。离线长视频作对照，说明截止时间才是让问题变成带宽问题的原因。

## 未核实项汇总

- Oasis 的 360p / H100、Mirage 2 的规格、Runway GWM-1 的 24 fps、Odyssey-2 Max：只有媒体或二手来源。
- Mirage / MirageLSD 的上下文窗口：技术报告链接失效。
- Tesla 世界模拟器、GAIA-4：只有演讲和二手资料。
- HeyGen 实时 API 的会话上限和并发：只有搜索摘要。
- Sora 2 / Veo 3.1 在 2026 年的上限。
- SIMA 2 的具体页面链接。
- ZYT-World 的绝对延迟、Khora 的硬件规格。
- 所有"推算"的秒数假设 VAE 时间压缩 4×，正式引用前再核对。
