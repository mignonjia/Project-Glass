# Project Glass × Causal Long Video Generation

> 这份文档是项目的"锚点"。每次回来先读这里，确认手上的工作仍然服务于下面的目标。
> 目标或假设变了，就更新这里，并在末尾的「决策记录」里写一行。

## 一句话目标

**研究 causal（自回归）long video generation 中的 memory 瓶颈，并说明 GlassML 玻璃基板系统（超大 on-glass DRAM + 高带宽光互连）能让这类 video gen / world model 做到当前 GPU 系统（H100 / B200 / NVL72）做不到的事。**

## 核心问题（2026-10-07）

> **Why does next-generation world-model training need a larger, higher-bandwidth scale-up domain?**
> 为什么下一代 world model 的训练需要更大、带宽更高的 scale-up 域？

论文和 funding 都从这个问题出发。推理链条：

1. **只讲容量发不了 paper**（David / Tianqi 的反馈），必须有带宽或调度问题。
2. **铜线把 scale-up 域限制在一个机柜**（224 Gb/s 每 lane 时无源铜线不到 1 m）。NVL72 是 72 卡；NVIDIA 2027 年的 NVL576 仍在一个机柜内、用铜背板，靠堆密度（600 kW）。业界的光互连目前主要用在 scale-out（CPO 交换机）。
3. **下一代 world model（视频 + MoE + 长记忆）让一个训练任务内紧耦合的通信组不断变大、更吃带宽**：MoE 专家并行要 AllToAll；长视频必须切序列（10 分钟 720p 就要 16–32 卡起），30 分钟以上一个样本可能超过 72 卡。
4. **glass 的定位**：业界已经把玻璃当作封装基板（TSMC CoPoS、Intel，2028–2030）；GlassML 在同一块玻璃里做可重构光网络，把整块面板变成一个更大、带宽更高（每卡 8 TB/s，NVLink 的 4.4×）的 scale-up 域。

我们要提供的是第 3 点的实测和 trace 证据（见 `docs/pan_compute_comm_estimate.md` 和 `docs/david_training_load_and_rl.md`）。

四个关键词缺一不可：

| 关键词 | 在本项目中的含义 |
|---|---|
| **Video gen** | 视频生成 / world model，以组内 PAN（DF + VSA，`WM-Game-Training`）为载体，**以训练为主**；候选：MoE 视频 world model（LingBot-Video MoE 30B-A3B） |
| **Causal** | 自回归、逐 chunk 生成（Self-Forcing、CausVid 一类），KV cache / 历史状态随视频长度增长 |
| **Memory** | 核心瓶颈：KV cache、训练 rollout 激活、teacher/student 状态等随 horizon 增长的状态 |
| **Glass** | GlassML 玻璃基板：大容量 DRAM（双面 / 多 plane）+ 光互连，作为解决方案的硬件平台 |

## 核心论点（待验证）

1. Causal long video 的状态随长度线性增长。例如 Wan 14B 在 480p 下，1 小时视频的完整 KV cache 约 30 TB（粗估，见下）。
2. 当前 GPU 系统放不下，只能用 sliding window、attention sink、压缩、KV 量化、SP/CP 切分等手段，**代价是长程一致性下降**（遗忘场景、身份漂移）或通信开销。
3. Glass 的大内存池让完整（或分层保留的）历史常驻在计算附近，光互连让远端历史可以被高效取回。
4. 因此 glass 在 **更长 horizon、更好长程一致性、更少 memory-forced 并行** 上占优。

## 硬件假设（来自导师转述的 Tianqi 模型，含义待确认）

| | GB200 NVL72 | Glass（Tianqi 模型） |
|---|---|---|
| 互连带宽 | 130 TB/s NVLink | 最高 **8 PB/s** 光带宽 |
| 内存容量 | 13.4 TB HBM（GB300：20 TB HBM） | **≥ 85 TB DRAM**（玻璃双面；利用 plane 之间空间可能更多） |

注：GlassML 论文本身（`David_MoE_glass.pdf` §III-A / §V-A）的配置是 600×600 mm 面板、64 个 tile，每个 tile = B300 + 288 GB HBM3e，合计 18.4 TB HBM，单 tile 光互连点对点最高 4 TB/s。**论文里没有 DRAM 面板，也没有近存计算。** 85 TB DRAM 和 8 PB/s 是 Tianqi 模型的扩展，不在论文里。

⚠️ 尚未确认：DRAM 面板怎么接到 GPU？是正下方通过 TGV 垂直直连（本地），还是走光互连（远端）？这决定了取 KV block 走哪条路。另外：8 PB/s 是总带宽 / 二分带宽 / 每 tile 注入带宽？85 TB 是哪种 DRAM，计算单元访问它的带宽和延迟是多少？计算 tile 数量和规格？
（注：`Mingjia-Research/.../argument_memo.md` 基于旧的 73 TB/s 假设，其中"带宽输给 NVL72"的论述已过时，可引用其结构，不要照搬结论。）

## 粗估：KV cache 随视频长度的增长

假设：bf16，832×480，只算 self-attn KV。KV/token = 2 × layers × hidden × 2 bytes。每个 latent 帧约 1560 tokens（VAE 空间 8×、patch 2×2）。

每秒 latent 帧数有两种口径，因此给出一个范围：
- 低值：Wan 原生 16 fps、时间压缩 4× → 4 latent 帧/s（约 6.2k tokens/s）
- 高值：团队 deck 写的"1 latent frame ≈ 0.16s" → 约 6.25 latent 帧/s（约 10k tokens/s）

| 视频长度 | Wan 1.3B（30 层，hidden 1536，≈184 KB/token） | Wan 14B（40 层，hidden 5120，≈0.82 MB/token） |
|---|---:|---:|
| 1 分钟 | ≈ 70–110 GB | ≈ 300–490 GB |
| 10 分钟 | ≈ 0.7–1.1 TB | ≈ 3.1–4.9 TB |
| 1 小时 | ≈ 4.1–6.6 TB | ≈ 18–30 TB |

720p 每帧 token 约为 480p 的 2.3 倍。⚠️ 两种口径的差异要在实测中确认（实际跑 Self-Forcing / CausVid 时数一下每秒的 latent 帧数）。

## 研究问题

- **RQ1（容量）**：不同模型 / 分辨率 / horizon 下，causal 生成（推理 + 训练）的内存需求是多少？各系统在哪个 horizon 被迫切分、offload 或丢弃历史？
- **RQ2（质量，次要）**：丢弃 / 压缩历史对长程一致性的损害。**本项目是 infra paper，不做系统性的算法对比**：
  - motivation 主要引用已有论文（MoC、WorldMem、WorldTrace、PackForcing、Quant VideoGen）；
  - 至多放一张小表（可用组内 Dense W3 与 W3+remote 的已有结果）；
  - 只有当硬件设计改变了算法（如改 k、block 大小、用预测预取）时，才需要做 iso-quality 验证。
  - 参照 GlassML 论文：评测章节没有任何模型质量实验，只用真实 checkpoint 的 forward 采集 routing trace，再注入模拟器。
- **RQ3（系统）**：在 glass 上如何组织内存？例如热窗口放 HBM、完整历史放 glass DRAM、通过光互连按需取回相关 chunk（sparse / retrieval attention）。容量够了之后，带宽和 attention 计算是否成为新瓶颈？
- **RQ4（对比）**：同一 workload 在 NVL72 与 glass 上的端到端延迟、可支持 horizon、并发 rollout 数对比。

## 方法方向（2026-10-04 确定）

**Causal sparse attention finetune**：在已有的 causal 视频模型上，让每个 chunk 只 attend「sink + 最近窗口 + 从完整历史里按内容选出的 top-k chunk」，并做轻量 finetune。

- **当前（2026-10-07）**：模型和代码都用 PAN（`WM-Game-Training` commit `5a5544a`，DF + VSA W3 + remote），**重点是 DF + VSA 训练**；Self-Forcing 不在范围内。实验代码在 `exp/topk-trace` 分支（worktree：`.worktrees/topk-trace/experiments/topk_trace/`），分析文档在 main 的 `docs/`。
- 候选方向：MoE 视频 world model（`robbyant/lingbot-video-moe-30b-a3b`：48 层、hidden 2048、128 专家 top-8 + 1 共享，配置和 GlassML 评测的 Qwen3-30B-A3B 相同），把 GlassML 的 MoE AllToAll 和长视频序列并行结合起来。
- 以下是 2026-10-06 一度考虑、已被取代的 Wan2.1-14B / Krea / FastVideo 方案，保留作参考：
  - 不用 Wan2.2-A14B 的原因：两个 expert 各有一份 KV cache，trace 和内存模型更复杂。
  - Wan-AI 官方截至 2026-10 没有比 Wan2.2 更新的开源通用模型（2.5 / 2.6 只有 API）。
- **FastVideo**（本地 `~/fastvideo/FastVideo`，分支 `worldmodel_exp`）：
  - 有 causal Wan + Self-Forcing：`fastvideo/models/wan/causal_transformer.py`。训练用 flex_attention；推理用 FA + 固定大小的 KV cache（sink + local window）。
  - 有 VSA：`fastvideo/attention/backends/video_sparse_attn.py`，但只接在双向 Wan 上；它的 `causal` 参数被忽略。
  - Krea checkpoint 还没注册进 FastVideo，需要加 registry / 配置。
  - 应对：先让 KV cache 保留完整历史，再用纯 PyTorch 实现 top-k block 选择，用来采集 trace；需要 finetune 或提速时，再移植 VSA kernel。
- **参考实现**：`/mnt/weka/shrd/wm/mingjia/WM-Game-Training`（PAN 的 DF causal 训练 + VSA；`vsa/` 在 `origin/main` 上，不在 dev 分支）。
  - `vsa/hybrid.py`：最近窗口精确 attention + 远端 top-k 检索；训练配方 `config/game/df_vsa.yaml`。
  - `vsa/DESIGN.md`：压缩分支（block 均值）+ 选择分支（top-k block full attention），`out = out_sparse + tanh_gate * out_compress`。
  - 推理用的 cached top-k 选择（`fused_ar_rect_score_topk`、`hyvideo/models/transformers/streaming/causal_vsa.py`）可以移植过来。
- **不自己实现 Self-Forcing 训练**：直接用现成的 causal checkpoint。
- **不做基于相机轨迹的检索**（WorldMem / Context-as-Memory 那类）：按内容选 chunk（pooled key 相似度），更通用。
- **与 glass 的连接**：完整 KV 历史放在 glass DRAM（靠容量），每步只取 top-k chunk（带宽需求小），通过光互连按需取回。这就是 RQ3 的软硬协同点。
- **最近的已有工作，必须讲清差异**：Mixture of Contexts（ICLR 2026，可训练的 top-k chunk 路由）、PackForcing（中段压缩 + top-k）、FAST-AR（ANN 稀疏 attention）。差异点应放在**完整历史 + 内存层级感知**（chunk 粒度、预取、取数量与 glass 带宽匹配），而不只是稀疏算法本身。

## 范围

**做：**
- Causal / 自回归视频生成的推理（KV cache）和训练（Self-Forcing 类 rollout 的激活内存）
- 解析内存模型 + 真实 GPU 小规模实测 + 模拟器投影到 glass
- 面向 world model 的长 horizon、交互式、多 rollout 场景

**暂不做（防止偏移）：**
- 纯 LLM / MoE 训练的 all-to-all 优化（那是 David 的 HPCA 论文主线，可复用但不是本项目重点）
- 双向（非 causal）短视频生成的单纯加速
- 玻璃基板的物理 / 热 / 光器件设计本身（用 Tianqi / Junda 的模型作为输入）
- 训练新的大视频模型

## 近期里程碑：11 月 review

导师要求在 2026 年 11 月 review 前拿出"nice results"。计划约 6 周：

1. **第 1 周**
   - [ ] 与 David、Tianqi 对齐：拿到模拟器 / 模型，确认上面的硬件假设
   - [ ] 将内存解析模型写成脚本，画出 **horizon 可行性曲线**（H100 / B200 / NVL72 / Glass）
   - [ ] 获取原仓库 `research-glass-substrate` 访问权限（含未拷贝的 v4–v6 sweep、`glass_3d_dram` 等配置）
2. **第 2–4 周**
   - [ ] 在 GPU 上跑 causal Wan2.1-14B（Krea Realtime 14B，FastVideo）
   - [ ] 实测内存、延迟随 horizon 的变化
   - [ ] 实测 KV 窗口大小 → 长程一致性指标的关系（RQ2）
3. **第 4–6 周**
   - [ ] 将实测 workload 参数输入 Tianqi / David 的模拟器，投影 glass 性能并与 NVL72 对比
   - [ ] 整理 review 用图表与 slides

**review 目标图表：**
1. Horizon 可行性曲线
2. KV budget vs. 长程一致性
3. NVL72 vs. Glass 端到端对比（每帧延迟 / 可支持 horizon / 并发 rollout 数）
4. 关键通信 / 内存访问时间分解

## 人员

- 导师：项目方向与 review 要求
- David：GlassML 架构与模拟器（HPCA 2027 投稿，`David_MoE_glass.pdf`）
- Tianqi：光带宽与 on-glass DRAM 的硬件模型（8 PB/s，≥ 85 TB）
- Mingjia / Junda：前期 world model memory 动机分析、内存 profiling、双面 glass 热分析（见 `Mingjia-Research/`）

## 已有材料

| 路径 | 内容 |
|---|---|
| `survey/glass_memory_papers.md` | **文献调研**：玻璃基板 / scale-up 系统 / AI 内存分层 / 长视频与 world model 内存；必读顺序、术语表、研究空白 |
| `docs/pan_compute_comm_estimate.md` | PAN + VSA 的显存实测、选择 trace、10 分钟 / NVL72 估算（完整记录） |
| `docs/david_training_load_and_rl.md` | 给 David：逐层训练负载拆解、长度与切分扫描、CP 偏斜、LLM / VLM RL 估算、两个方向的对比 |
| `survey/sparse_attn_video_training.md` | 稀疏 attention × 视频 × GPU 系统（偏训练）文献调研 |
| `survey/venues_and_timeline.md` | **投稿会议与时间线**：组里常投的 venue、Tianqi / David 的论文、2026-10 至 2027-12 的 deadline |
| `David_MoE_glass.pdf` | GlassML HPCA 2027 投稿稿（保密，勿外传）：玻璃面板 + 可重构光互连，面向 MoE 训练 |
| `Mingjia-Research/journal/20260610_world_model_memory/argument_memo.md` | world model memory 论证、测量方案建议（基于旧的 73 TB/s） |
| `Mingjia-Research/journal/20260519_glass_motivation/` | 动机 slides 草稿、flow、`session_history.md`（含团队实测数字） |
| `Mingjia-Research/ML_IoT Team 25-26 - Mingjia Junda only.pptx` | 团队实测数据来源（distillation 内存 profile、Wan SP profile 等） |
| `Mingjia-Research/Glass ML Presentation Tajana Samsung SK.pptx` | GlassML 主 presentation |
| `Mingjia-Research/papers/applications/StreamDiffusionV2.pdf` | 实时交互 world model 推理受内存限制的参考 |

已知团队实测（来自 deck，部分为从图读出的近似值）：
- Distillation（SP=1）：3 latent 帧峰值 55 GB，9 帧 135 GB，15 帧 OOM
- Wan1.3B causal，B200，SP=4：每个去噪步 NCCL all-to-all 约 0.17–0.20 s；480p / 3.4s 时通信超过 FlashAttention 本身

## 待解决问题

- [ ] 8 PB/s 和 85 TB 的精确定义（见「硬件假设」）
- [ ] 模拟器是否支持推理 / 视频 workload，还是需要扩展？
- [ ] review 更看重训练还是推理结果？
- [x] 选哪个开源 causal 视频模型作为主载体？→ PAN（DF + VSA，训练为主）；候选 MoE 视频 world model（见「方法方向」）
- [ ] 长程一致性用什么指标衡量？**评测必须包含「回到之前场景」的情况**（例如 LoopBench 或自建 revisit prompt），否则远处的历史用不上，看不出 full / sparse / window 的差别
- [ ] 只用 teacher forcing 做 finetune 时，长 rollout 是否会漂移？需要先用 training-free 的 top-k 测一个 baseline

## 决策记录

| 日期 | 决策 / 变更 |
|---|---|
| 2026-10-04 | 确立项目目标：video gen + causal + memory + glass。采用 Tianqi 模型数字（8 PB/s，≥ 85 TB），取代旧 memo 的 73 TB/s 假设。 |
| 2026-10-04 | 完成文献与会议调研（`survey/`）。KV 粗估改为区间（fps 口径不同）。初步论文目标：DATE LBR（11/29）拿早期反馈 → MICRO 2027（约 4 月）或 HPCA 2028（约 7 月底）。 |
| 2026-10-04 | 讨论后倾向主投 DAC 2027（约 11/11 摘要、11/18 全文，待导师确认），DATE LBR 作保底；CVPR 不合适。 |
| 2026-10-04 | 方法方向：causal sparse attention finetune，基于现成的 causal checkpoint；不做 Self-Forcing 训练，不做相机轨迹检索。 |
| 2026-10-04 | 不引入近存计算，保持设计简单：路由（池化 key 打分）在 GPU 上做，索引放 HBM（即使按 VSA2 的 R=16，每个 tile 也只要约 9–15 GB，放得下）；完整 KV 放 glass DRAM，只搬运选中的 block。 |
| 2026-10-04 | 只用 VSA v1（128-token block 均值作索引），不用 VSA2 的细粒度路由。 |
| 2026-10-04 | 写好发给 Tianqi 的项目介绍和问题清单：`docs/intro_for_tianqi.md`。 |
| 2026-10-04 | 定位为 infra paper：质量对比以引用为主，不自己系统测试；核心实验 = 真实模型 forward 采集 top-k 选择 trace → glass 内存 / 互连模拟（仿照 GlassML 的 routing trace 方法）。算法实现复用 WM-Game-Training 的 VSA hybrid。 |
| 2026-10-06 | 主模型定为 Wan2.1-T2V-14B（dense，避开 Wan2.2-A14B 的 MoE 双 KV cache），causal checkpoint 用 Krea Realtime 14B；代码载体改为 FastVideo + VSA。WM-Game-Training（PAN）只作为 VSA 参考实现，路径改为 `/mnt/weka/shrd/wm/mingjia/WM-Game-Training`。 |
| 2026-10-06 | 改回 PAN 作为主模型（组内支持最好、模型更大）；实验假设 remote top-k 5%（配置为 15%），视频长度任意讨论。 |
| 2026-10-06 | 范围收窄为 DF + VSA，导师更希望做**训练**；Self-Forcing 太小众，不在范围内。 |
| 2026-10-07 | 完成 PAN DF + VSA 训练的显存实测（10–60 s，8×H200）和 60 s 训练 trace；结论：稀疏省计算不省通信，causal 负载不均衡明显；10 分钟训练在 NVL72 上选对切法即接近理想效率，通信不是瓶颈。详见 `docs/pan_compute_comm_estimate.md`。 |
| 2026-10-07 | 确立核心问题：*why next-generation world-model training needs a larger, higher-bandwidth scale-up domain*（见文首）。候选方向：MoE 视频 world model + 长视频序列并行，对接 GlassML 的 MoE AllToAll。 |
