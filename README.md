# Project Glass × World-Model Training

> 这份文档是项目的"锚点"。每次回来先读这里，确认手上的工作仍然服务于下面的核心问题。
> 目标或假设变了，就更新这里，并在末尾的「决策记录」里写一行。早期方向见「历史方向」。

## 当前方向（2026-10-08，导师确认）

**Causal world model 的实时推理（real-time inference）**：逐 chunk 生成，每个 chunk 必须在它对应的视频时长内生成完（720p、24 fps 时约 0.67 s），同时保留长历史的 KV cache。

初步估算（PAN、720p、10 分钟流、4 步 + 1 次 commit、无 CFG、remote top-k 5%；详见 `docs/realtime_inference_estimate.md` 待写）：
- 每路流的完整 KV ≈ **2.7 TB**，单靠 HBM 要约 9 张卡才放得下；实时所需算力约 4–5 张 GB300。
- **NVL72 受容量限制**：约 6–8 路流（算力本可支持约 17 路）。
- **Glass 受算力限制**：容量可放约 39 路，算力约 15 路，**每套系统的流数约为 NVL72 的 2 倍**。
- 前提：每步要读约 585 GB 的 KV（trace：每步取约 22% 的远端历史），每路流约 4.4 TB/s，glass DRAM 的带宽必须跟得上。**这是要问 Tianqi 的关键数字。**

下面「核心问题」一节是训练方向的分析，已完成，结论保留作为对照。

## 核心问题（训练方向，已完成）

> **Why does next-generation world-model training need a larger, higher-bandwidth scale-up domain?**
> 为什么下一代 world model 的训练需要更大、带宽更高的 scale-up 域？

论文和 funding 都从这个问题出发。推理链条：

1. **只讲容量发不了 paper**（David / Tianqi 的反馈），必须有带宽或调度问题。
2. **铜线把 scale-up 域限制在一个机柜**（224 Gb/s 每 lane 时无源铜线不到 1 m）。NVL72 是 72 卡；NVIDIA 2027 年的 NVL576 仍在一个机柜内、用铜背板，靠堆密度（600 kW）。业界的光互连目前主要用在 scale-out（CPO 交换机）。
3. **下一代 world model（视频 + MoE + 长记忆）让一个训练任务内紧耦合的通信组不断变大、更吃带宽**：MoE 专家并行要 AllToAll；长视频必须切序列（10 分钟 720p 就要 16–32 卡起）。
4. **glass 的定位**：业界已经把玻璃当作封装基板（TSMC CoPoS、Intel，2028–2030）；GlassML 在同一块玻璃里做可重构光网络，把整块面板变成一个更大、带宽更高的 scale-up 域。

**我们要提供的是第 3 点的实测和 trace 证据。**

## 训练方向的已有结果（已完成，作对照）

**结果**（PAN DF + VSA 训练，8×H200 实测 + 60 s 训练 trace，详见 `docs/pan_compute_comm_estimate.md`、`docs/david_training_load.md`）：

1. **单卡容量**：448p 下单张 H200 训练只能放约 1 分钟视频，更长必须切序列。
2. **稀疏省计算、不省通信**：5% top-k 的 attention 计算只有 dense 的 13.5%，切到 8 卡时通信仍有 dense 的 49%。
3. **负载不均衡**：causal 下越靠后的 chunk 越重；连续切最慢卡约 1.3×，条纹切约 1.1× 但通信翻倍。CP 的 GPU 对偏斜（最高 3.6×）和 GlassML 测的 MoE AllToAll 偏斜同一量级。
4. **NVL72 内是算力瓶颈**：10 分钟 720p，最优切分（Ulysses 8 × 条纹 CP 9）通信只占计算的 9%；同样切分在 8 卡节点 + IB 上是 1.16×。
5. **容量换带宽**：forward、重算、backward 各拉一次远端 K/V；缓存拉来的 K/V 要每卡约 310 GB（超过 GB300 的 288 GB）。更大的近计算内存能把 3 轮拉取降为 1 轮。


## 硬件假设

| | GB300 NVL72 | GlassML 论文配置 | Tianqi 模型（导师转述） |
|---|---|---|---|
| 规模 | 72 GPU，一个机柜 | 600×600 mm 面板，64 tile（每 tile = B300 + 288 GB HBM3e） | — |
| 互连 | NVLink 每卡单向 0.9 TB/s，合计 130 TB/s | 光互连，每 tile 点对点最高 4 TB/s（K16/2T） | 最高 8 PB/s |
| 内存 | 20 TB HBM | 18.4 TB HBM，**没有 DRAM 面板，没有近存计算** | ≥ 85 TB DRAM（玻璃双面，可能更多） |

估算里 glass 每卡按单向约 4 TB/s 计（约 NVLink 的 4.4×）。

⚠️ 待确认（找 Tianqi / David）：
- 8 PB/s 是总带宽、对分带宽还是每 tile 注入带宽？8 TB/s 是否为半双工总量？
- DRAM 面板怎么接到 GPU：正下方 TGV 直连（本地），还是走光互连（共享池）？带宽和延迟多少？
- 85 TB 是哪种 DRAM，每个 tile 分到多少？

## 模型与代码

- **主模型**：PAN（29.2B，44 层，hidden 4096，32 head），DF（diffusion forcing）causal 训练 + VSA hybrid 稀疏 attention（最近 3 个 chunk 精确 + 远端 top-k，本文默认 5%）。
- **代码**：`/mnt/weka/shrd/wm/mingjia/WM-Game-Training`，commit `5a5544a`。
  - `vsa/hybrid.py`：窗口 + 远端 top-k 选择；训练配方 `config/game/df_vsa.yaml`。
  - `vsa/DESIGN.md`：压缩分支（block 均值，看全部历史）+ 选择分支（top-k block 做 full attention），`out = out_sparse + tanh_gate * out_compress`。
  - ⚠️ 当前 packed（DF + VSA）路径不支持序列并行，所有 SP / CP 结果都是估算。
- **实验代码**：分支 `exp/topk-trace`，目录 `experiments/topk_trace/`（trace hooks、切分模拟、估算脚本）。⚠️ 目前在服务器上，还没 push 到 origin。
- **候选 MoE 模型**：`robbyant/lingbot-video-moe-30b-a3b`（48 层，hidden 2048，128 专家 top-8 + 1 共享，结构和 GlassML 评测的 Qwen3-30B-A3B 相同）。

## 方法原则

- **infra paper**：仿照 GlassML，用真实 checkpoint 的 forward 采集 trace（选择、取数需求矩阵、偏斜），注入内存 / 互连模拟，与 NVL72 和 8 卡节点 + IB 对比。
- **不做系统性的质量对比**：motivation 引用已有论文（MoC、WorldMem、WorldTrace 等）。只有硬件设计改变了算法（改 k、block 大小、预测预取）时，才做 iso-quality 验证。
- **只用 VSA v1**，不用 VSA2 的细粒度路由；**不引入近存计算**。

## 范围

**做（训练方向，已完成）：**
- 长视频 world model 的 DF + 稀疏 attention **训练**：切分方式（Ulysses / CP、连续 / 条纹）、取数流量、负载与偏斜
- 实测 + trace + 模拟投影到 glass
- 可能扩展到 MoE 视频 world model（AllToAll + CP 取数叠加）

**做（主线，10/8 起）：** causal world model 的实时推理：每系统流数、KV 容量与读取带宽、每 chunk 延迟。

**不做：**
- 相机轨迹检索
- 自己做稀疏算法的质量研究
- 玻璃基板的物理 / 热 / 光器件设计（用 Tianqi / Junda 的模型作为输入）

## 近期计划：11 月 review（实时推理）

- [ ] Tianqi：glass DRAM 每 tile 的带宽、延迟和连接方式（实时推理每 tile 约需 1 TB/s 读 KV，是关键数字）
- [ ] 定实时目标：分辨率（448p / 720p）、fps、去噪步数（PAN SF 4 步已有 60 s trace）
- [ ] 写 `docs/realtime_inference_estimate.md`：视频长度、步数、top-k 比例 vs 每系统流数、首帧延迟
- [ ] 验证"每步取约 22% 远端历史"能否从 60 s 外推到 10 分钟
- [ ] 把 `exp/topk-trace` push 到 origin
- [ ] David：GlassML 模拟器能否接推理 trace
- [ ] 投稿：DAC 2027（约 11/18）是否还投

**review 目标图表（实时推理）：**
1. 每系统能同时跑的实时流数：NVL72 vs glass（容量 / 算力 / 带宽三个上限）
2. 可支持的视频长度（完整 KV 历史）vs 系统
3. 每 chunk 延迟拆解：计算 / 读 KV / 通信

## 人员

- 导师：项目方向与 review 要求（10/8 确认做实时推理）
- David：GlassML 架构与模拟器（HPCA 2027 投稿，`David_MoE_glass.pdf`）
- Tianqi：光带宽与 on-glass DRAM 的硬件模型
- Mingjia / Junda：前期 world model memory 动机分析、双面 glass 热分析（见 `Mingjia-Research/`）

## 材料索引

| 路径 | 内容 |
|---|---|
| `docs/pan_compute_comm_estimate.md` | PAN + VSA 显存实测、选择 trace、切分模拟、10 分钟 / NVL72 估算（完整记录） |
| `docs/david_training_load.md` | 逐层训练负载拆解、长度与切分扫描、CP 偏斜 |
| `docs/fixed_budget_reuse_estimate.md` | 固定预算稀疏、复用度 R、16 卡 10 分钟 NVL72 vs glass、k = 320 |
| `survey/sparse_attn_video_training.md` | 稀疏 attention × 视频 × 多卡系统（偏训练）文献；最相关：MTraining、SparSP、Striped Attention |
| `survey/glass_memory_papers.md` | 玻璃基板 / scale-up 系统 / AI 内存分层 / 长视频内存文献；必读顺序、术语表 |
| `survey/venues_and_timeline.md` | 投稿会议与 deadline（2026-10 至 2027-12） |
| `David_MoE_glass.pdf` | GlassML HPCA 2027 投稿稿（**保密，勿外传；仓库保持私有**） |
| `Mingjia-Research/` | 早期材料：world model memory 论证（基于旧的 73 TB/s）、动机 slides、团队 deck |

## 待解决问题

- [ ] 硬件假设（见上）
- [ ] 固定预算稀疏下的复用度 R
- [ ] 显存的超线性增长来源（次要，只影响 30 分钟以上的外推）
- [ ] 模型只按 10 个 chunk、15% 训过；长视频训练后的选择模式可能不同

## 历史方向（已被取代，保留作参考）

- **10/4：推理 + KV 容量**。原始设想是 causal 长视频推理时 KV 放不下，用 sparse attention 把完整历史放 glass DRAM、按需取 top-k block。后因"只讲容量发不了 paper"转向训练和带宽。
  - 当时的 KV 粗估（Wan，bf16，480p）：Wan 1.3B 1 小时约 4–7 TB；Wan 14B 1 小时约 18–30 TB。
  - 当时的 RQ：容量、质量（KV 预算 vs 长程一致性）、内存分层设计、与 NVL72 对比。
- **10/6：Wan2.1-14B（Krea Realtime）+ FastVideo**，当天改回 PAN。
  - 公开 FastVideo 有 causal Wan + Self-Forcing 和 VSA，但没有 causal VSA（`causal` 参数被忽略）。
- **Self-Forcing / 开源 causal checkpoint**：太小众，不在范围内。

## 决策记录

| 日期 | 决策 / 变更 |
|---|---|
| 2026-10-04 | 确立项目目标：video gen + causal + memory + glass。采用 Tianqi 模型数字（8 PB/s，≥ 85 TB），取代旧 memo 的 73 TB/s 假设。 |
| 2026-10-04 | 完成文献与会议调研（`survey/`）。初步论文目标：DATE LBR（11/29）拿早期反馈 → MICRO 2027（约 4 月）或 HPCA 2028（约 7 月底）。 |
| 2026-10-04 | 讨论后倾向主投 DAC 2027（约 11/11 摘要、11/18 全文，待导师确认），DATE LBR 作保底；CVPR 不合适。 |
| 2026-10-04 | 方法方向：causal sparse attention finetune，基于现成的 causal checkpoint；不做 Self-Forcing 训练，不做相机轨迹检索。 |
| 2026-10-04 | 不引入近存计算：路由在 GPU 上做，索引放 HBM。 |
| 2026-10-04 | 只用 VSA v1，不用 VSA2 的细粒度路由。 |
| 2026-10-04 | 写好发给 Tianqi 的项目介绍（`docs/intro_for_tianqi.md`，10/8 已删除）。 |
| 2026-10-04 | 定位为 infra paper：质量对比以引用为主；核心实验 = 真实模型 forward 采集 trace → 内存 / 互连模拟（仿照 GlassML）。 |
| 2026-10-06 | 主模型一度定为 Wan2.1-T2V-14B（Krea Realtime 14B）+ FastVideo。 |
| 2026-10-06 | 改回 PAN 作为主模型（组内支持最好、模型更大）；实验假设 remote top-k 5%（配置为 15%）。 |
| 2026-10-06 | 范围收窄为 DF + VSA，导师更希望做**训练**；Self-Forcing 不在范围内。 |
| 2026-10-07 | 完成 PAN DF + VSA 训练的显存实测（10–60 s，8×H200）和 60 s 训练 trace：稀疏省计算不省通信，causal 负载不均衡明显；10 分钟训练在 NVL72 上选对切法即接近理想效率。 |
| 2026-10-07 | 确立核心问题：*why next-generation world-model training needs a larger, higher-bandwidth scale-up domain*。候选方向：MoE 视频 world model + 长视频序列并行。 |
| 2026-10-08 | 完成逐层训练负载拆解（`docs/david_training_load.md`）。 |
| 2026-10-08 | README 按核心问题重排：当前状态在前，推理 / KV 容量等早期方向移到「历史方向」。 |
| 2026-10-08 | 向导师汇报训练方向的结果：dense PAN 训练在 NVL72 内受算力限制，glass 带宽收益 ≤ 3%。补充 `docs/fixed_budget_reuse_estimate.md`。 |
| 2026-10-08 | **导师确认方向改为 causal model 的实时推理。** 训练分析保留作对照。 |
| 2026-10-08 | 清理：删除 LLM / VLM RL 估算和重算调研，只保留视频相关内容；`david_training_load_and_rl.md` 改名为 `david_training_load.md`。 |
