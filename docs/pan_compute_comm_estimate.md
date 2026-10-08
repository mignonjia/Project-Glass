# PAN + VSA（causal DF）：compute / memory / communication 估算与实测

> 2026-10-07。模型 PAN（29.2B），代码 WM-Game-Training，稀疏 attention 为 W3 + remote top-k（VSA hybrid）。
> 每个数字都标了来源：**[实测]** 是在 8×H200 上跑出来的；**[trace]** 是从真实运行中记录的 top-k 选择统计出来的；**[估]** 是按公式推算的。
> 复现代码和原始数据：`.worktrees/topk-trace/experiments/topk_trace/`（分支 `exp/topk-trace`）（见第 9 节）。
> 项目范围（2026-10-06）：只看 diffusion forcing（DF）+ VSA，**以训练为主**；Self-Forcing 不在范围内，相关数据只作参考。

## 0. 结论摘要

1. **容量**：448p 下，单张 H200 训练只能放约 **1 分钟**视频（[实测] 40 s 峰值 97 GiB，60 s 峰值 120 GiB）；推理保留完整历史时，带 CFG 约 30 s、不带 CFG 约 1 分钟（[实测] 1 分钟峰值 130 GiB）。更长的视频必须切序列或把历史放到别处。
2. **稀疏省计算，不省通信**：DF 训练中 5% top-k 只算 dense causal attention 的 **13.5%**，但序列切到 8 张卡时，通信只降到 dense 的 **49%**（[trace]）。和 SparSP（推理，"计算 −71%，通信 −19%"）结论一致，这里首次在 causal 视频训练上测到。
3. **负载不均衡**：causal 下越靠后的 chunk 越重，按顺序切 8 份时最慢的卡是平均的 **1.29×**（[trace]）。条纹式切分把不均衡降到 1.10×，但**总通信翻倍**（134 → 275 GB/forward）。这是一个直接的调度权衡。
4. **单卡内存和通信是耦合的**：每张卡放的 chunk 越多，需要切的份数越少，总通信越少（2 卡 28 GB，8 卡 134 GB，60 卡 592 GB / forward）。大容量内存（glass）的价值是"少切几份"，不只是"放得下"。
5. **推理侧**（参考）：步数越少，越受容量和带宽限制。DF 推理每个 chunk 少于约 30–60 次 forward 时，NVL72 由算力瓶颈转为容量瓶颈（[估]）。

---

## 1. 估算方法（roofline 思路）

以「一个 chunk 的一次 forward」为单位，算三个量：

1. **FLOPs**：线性层 + attention
2. **本地 HBM 读取字节**：权重、本卡上的 KV
3. **远端取回字节**：被 top-k 选中、但不在本卡上的历史 K/V（NVL72 上来自其他 GPU 的 HBM，glass 上来自 DRAM 或其他 tile）

时间 ≈ max(FLOPs / 算力, 远端字节 / 链路带宽)。哪项大，瓶颈就在哪。

## 2. 模型与 workload 参数

| 项 | 值 | 来源 |
|---|---|---|
| 参数量 | 29.2B（bf16 权重约 58 GB） | checkpoint DCP metadata |
| 层数 / hidden / heads | 44 / 4096 / 32（head_dim 128） | `config/game/df.yaml` |
| 视频 token 的线性层 | 每层 qkv 3D² + out D² + FFN 2·D·8D | checkpoint 张量形状 |
| caption token 的 FFN | 宽度 4D | yaml |
| VAE | Wan2.2，空间 16×、时间 4×；patch 1×2×2 | yaml |
| 分辨率 / fps | 768×448，16 fps | yaml |
| 每 latent 帧 token | (448/32)×(768/32) = 14×24 = 336 | 推算 |
| chunk | 4 latent 帧 = 1 秒视频 | `causal_chunk_size: 4` |
| 每 chunk token | 4×336 + 256 caption slot = **1600**（tile 补齐后 1792 行） | `causal_chunk_caption_slot: 256` |
| 稀疏策略 | 最近 3 个 chunk 精确 + 远端 top-k（配置 15%，本文默认 **5%**）；top-k 按比例，k = ⌈比例 × 远端 tile 数⌉ | `df_vsa.yaml`、`vsa/hybrid.py` |
| tile | 128 token（H200）；GB300 上 256 | `vsa/DESIGN.md` |
| 选择粒度 | **每层、每个 head、每个 query block（128 token）各自选**；每个去噪步重新选 | `causal_vsa.py`、`hybrid.py` |
| 推理 | 50 步 + CFG 4；cond / uncond **各有一份 KV cache**；每个 chunk 结束后各做一次 t=0 commit | `pipeline.py` |

720p 对应 1280×704：每 latent 帧 22×40 = 880 token，每 chunk 3776 token（补齐 4096 行）。24 fps 时每个 chunk = 0.67 秒。

## 3. 基本公式

- **KV / token** = 2 (K,V) × 44 层 × 4096 × 2 B = **721 KB**
- **KV / 秒视频（448p，单分支）** = 1792 行 × 721 KB ≈ **1.3 GB**；CFG 两份 ≈ **2.6 GB**
- **线性层 FLOPs / forward（448p）** ≈ 1344 × 29.5 GFLOP + 256 × 17.7 GFLOP ≈ **44 TFLOP**（720p 约 108 TFLOP）
- **attention FLOPs / 每个历史 token** = 4 × N_q × D × L（448p：1.15 GFLOP）
- **取回 KV 的算术强度上限** = N_q FLOP/byte（每个取回的 token 被 chunk 里全部 query 用到时）。实际要除以 query block 之间的"并集放大"（见第 5 节）。

---

## 4. 实测：显存与速度

### 4.1 DF + VSA 训练（8×H200，FSDP，每卡 1 个样本，W3 + remote 5%）[实测]

| 视频长度 | token | 每步时间 | 每卡峰值显存 |
|---|---:|---:|---:|
| 10 s | 1.7 万 | 4.5 s | 70.3 GiB |
| 20 s | 3.3 万 | 8.5 s | 76.5 GiB |
| 40 s | 6.6 万 | 15.6 s | 97.2 GiB |
| 60 s | 9.7 万 | 31.7 s | 120.0 GiB（显存已用到 135 / 140 GiB 的 reserved） |

- 固定部分：每卡模型状态（fp32 权重、梯度、优化器，8 份分片）43.5 GiB + 其他约 20 GiB。
- 增长部分**超线性**：10→20 s 每 token 约 420 KB，20→40 s 约 700 KB，40→60 s 约 765 KB（每 chunk 按 1600 token 计）。推测来自稀疏 attention 每层为 backward 保存的块索引（与块数平方成正比），**未核实**。
- 时间在 40 s 以内基本线性（稀疏压住了 attention 的平方项）；40→60 s 时间翻倍（15.6 → 31.7 s），原因未查明，可能和显存接近上限有关。10 s 的 4.5 s/步和原始 64 卡训练日志（4.75 s）一致。
- 60 s 的第一次运行在第 1 步出现过一次 CUDA illegal memory access，之后两次重跑都正常（见第 8 节）。

### 4.2 推理（参考）[实测]

| 设置 | 峰值显存（单张 H200） |
|---|---:|
| DF，50 步 + CFG，10 s，448p | 110.7 GiB（含借用训练代码带来的 13.6 GiB fp32 分片） |
| SF，4 步无 CFG，60 s，448p，保留完整历史 | 第一次 OOM（约 142 GiB）；文本编码器挪到 CPU 后 **130.4 GiB** |

---

## 5. trace：top-k 选择的统计

### 5.1 DF 训练（60 s，8 个样本：6 段游戏录像 + 2 段 Sekai）[trace]

整段序列一次 forward，记录每层、每个 head、每个 query block 选中的 key block。模拟把 60 个 chunk 切到 P 张卡上：

| GPU 数 | 切法 | 取数量 / dense | 负载不均衡（最慢 / 平均） | 每张卡最多取 | 总取数 |
|---:|---|---:|---:|---:|---:|
| 2 | 连续 | 72% | 1.17 | 28 GB | 28 GB |
| 4 | 连续 | 61% | 1.25 | 35 GB | 71 GB |
| 8 | 连续 | **49%** | **1.29** | 32 GB | 134 GB |
| 8 | 条纹 | 54% | 1.10 | 37 GB | 275 GB |
| 15 | 连续 | 40% | 1.31 | 27 GB | 215 GB |
| 30 | 连续 | 32% | 1.32 | 22 GB | 354 GB |
| 60 | 连续 | 26% | 1.33 | 18 GB | 592 GB |

（每次 forward，44 层 × 32 head 合计，每个 head-tile 64 KB。backward 和重算大约再各来一轮。）

- 整体 attention 计算量是 dense causal 的 **13.5%**。
- "取数量 / dense"随卡数下降：每张卡的 query 越少，它们选择的并集覆盖的历史越少；dense 不管多少 query 都需要全部历史。但**总通信随卡数上升**，因为 dense 的基数涨得更快。
- 即使 1 个 chunk 一张卡，仍要取 26%：前 2 个 chunk 的窗口必须整块取，且一个 chunk 内 14 个 query block 的并集已约 22%。
- 1 分钟时，最忙的卡每次 forward 取约 32 GB，在 H200 NVLink 上约占计算时间的 6%；这个长度下不均衡比通信更关键，两者都随长度增长。

### 5.2 推理（参考）[trace]

| | DF 50 步 + CFG，10 s（15% / 5%） | SF 4 步，60 s，5%（20–60 s 平台值） |
|---|---|---|
| 一个 chunk 内 query block 选择的并集 / 单个 | 4.5× / 4.6× | 4.3× |
| 每步要取的远端历史比例 | — | 22% |
| 相邻去噪步之间新增的 tile | 0.8% / 1.1% | 约 7% |
| 全部步的并集 / 单步 | 1.16× / 1.31× | 1.19× |
| 一个 head 选中时其他 head 也选中的比例 | 70% / 31% | 26% |

- 选择在去噪步之间非常稳定：50 步时每个 chunk 取一次就能复用约 38 次；4 步只能复用约 3.4 次。
- 这些比值在 20 s 以后基本不再变化，外推到更长视频有一定依据。

---

## 6. 估算：10 分钟、720p、24 fps

### 6.1 训练（GB300 NVL72，72 卡）[估]

- 序列 340 万 token。按 4.1 节的实测斜率，单张 H200 只能放约 1 分钟 448p（约 25 秒 720p），所以必须切成几十份。
- 切 72 份时每卡约 12.5 个 chunk。按 5.1 节，取数量约为 dense 的 40–49%，负载不均衡约 1.3×（该趋势随长度增长，60 s 内的测量外推到 10 min 偏乐观）。
- 每步计算、通信和容量随长度的变化：`estimate_training.py`；10 分钟的具体切分方案见 6.2 节。

### 6.2 训练的序列并行方案（10 min，720p，24 fps，GB300 NVL72）[估]

一个样本切到 P = U × C 张卡（U 路 Ulysses 按 head 切，C 路 context parallel 按 chunk 切），其余卡做数据并行。训练 trace 显示**每个 head 选的块数完全一样**（固定比例 top-k），所以按 head 切没有负载不均衡。脚本：`.worktrees/topk-trace/experiments/topk_trace/estimate_sp.py`。

| 方案 | 卡数 / 样本 | 每卡显存 | 每步计算 | 不均衡 | SP 通信 / 步 | 每台机柜每小时样本数 |
|---|---:|---:|---:|---:|---:|---:|
| 不切（现在的代码） | 1 | 5 TB | — | — | — | 训练不了 |
| Ulysses 32 | 32 | 180 GB | 36.9 s | 1.00 | 0.49 s | 195（8 卡空闲） |
| CP 72，连续 | 72 | 124 GB | 27.3 s | 1.66 | 5.1 s | 132 |
| CP 72，条纹 | 72 | 130 GB | 17.5 s | 1.07 | 5.9 s | 206 |
| Ulysses 8 × CP 9，条纹 | 72 | 98 GB | 16.5 s | 1.01 | 1.2 s | **218**（理想 219） |

- 10 分钟时，选对切法 NVL72 就能接近理想效率，**瓶颈是算力**；真正的系统问题是负载均衡（连续切会损失 40% 吞吐）。
- 换成 **8 卡节点 + InfiniBand**（跨节点每卡约 50 GB/s）时，把单个样本切到 72 卡，通信会到计算的 1.2×（720p）到 2.3×（448p）；只追求吞吐、用 Ulysses 32 时约 0.2×。这是通信加速能带来较大端到端收益的场景（探索性估算，未建模重叠）。
- 30 分钟：含平方项时所有方案每卡都要 500 GB 以上，单台 NVL72 放不下；只算线性部分时都放得下。

### 6.3 推理（GB300 NVL72）[估]

一个 NVL72 能同时服务几个 10 分钟实时视频流：

| DF 推理设置 | 每 chunk forward 次数 | 按算力 | 按 HBM 容量 | 瓶颈 |
|---|---:|---:|---:|---|
| 50 步 + CFG | 102 | 1.6 | 2.3 | 算力 |
| 20 步 + CFG | 42 | 3.9 | 2.3 | 容量 |
| 50 步，无 CFG | 51 | 3.2 | 4.8 | 算力 |
| 20 步，无 CFG | 21 | 7.7 | 4.8 | 容量 |
| 10 步，无 CFG | 11 | 14.8 | 4.8 | 容量（差 3×） |

每个流的完整 KV：带 CFG 5.3 TB，不带 2.7 TB。按 Tianqi 模型的 85 TB 算，glass 能放约 16 / 32 个流（**85 TB 的含义尚未确认**）。

---

## 7. 对 glass 的含义（待和 David / Tianqi 讨论）

> **核心问题：Why does next-generation world-model training need a larger, higher-bandwidth scale-up domain?**

- 铜线把 scale-up 域限制在一个机柜（NVL72；2027 年的 NVL576 仍是单机柜、铜背板）。业界的光互连目前主要用在 scale-out；玻璃在业界的角色是封装基板（TSMC CoPoS、Intel，2028–2030），GlassML 在同一块玻璃里加入可重构光网络。
- 本文的数据对这个问题的回答（目前）：
  - 10 分钟 720p 的 DF + VSA 训练，一个样本要 16–32 卡起，但仍在一台 NVL72 内；选对切法时通信只占计算的约 7%，**glass 的带宽优势在这里体现不出来**。
  - 需要更大、更快 scale-up 域的情形：一个样本超过 72 卡（30 分钟以上，取决于第 4.1 节未核实的平方项）、想用更多卡算单个样本（强扩展）、在 8 卡节点 + InfiniBand 的普通集群上训练（通信可达计算的 1.2–2.3×），以及 **MoE 视频 world model**（专家并行的 AllToAll 估计约为计算的 70%，和 GlassML 的 MoE 场景同类）。

- GlassML 的互连是**电路交换的光网络**：传输前要配置无冲突光路，重配置有开销（δ）；调度器把"谁发给谁多少"的矩阵拆成可并行的点对点传输，已支持运行时才确定的 MoE AllToAll。
- 稀疏 attention 的取数就是这种**运行时确定、带偏斜**的需求矩阵；且在去噪步之间几乎不变（50 步时每步只变约 1%），光路可以配一次用很多步。
- 训练侧可以讨论的调度问题：切分方式（连续 / 条纹 / 按负载）在"计算均衡"和"通信量"之间的权衡；每卡内存越大、切的份数越少、总通信越少。
- 需要 Tianqi 确认：DRAM 面板与 GPU 的连接方式和带宽、8 PB/s 和 85 TB 的定义。

## 8. 局限与未解决问题

- **60 s 训练在 kernel 缓存为空时第 1 步崩溃**（CUDA illegal memory access，固定在 rank 3，可复现）；缓存里已有 60 s 形状时 3 步都正常，loss 和峰值完全一致。最可能是 GEMM 自动调优试跑的某个候选 kernel 越界，具体 kernel 未确认。训练入口只允许 10 或 20 个 chunk，实验时在 driver 里绕过了这个检查（模型代码未改）。trace 和显存数据都来自正常完成的运行。排查记录见 `.worktrees/topk-trace/experiments/topk_trace/README.md`。
- **当前代码不支持序列并行**：packed 路径（DF + VSA 训练用的）遇到 SP 会直接 `NotImplementedError`；代码库里的 SP 是 Ulysses，只在旧的非 packed 路径。6.2 节的 SP 结果全部是估算。
- **模型没有按长视频训练过**：checkpoint 是 10 个 chunk、15% 训练的，trace 在 60 个 chunk、5% 下记录，长视频训练好的模型选择模式可能不同。
- **数据**：没有 60 s 的 YouTube 片段（最长 50 s），游戏部分全部来自录像；latent 用官方导出器从源视频重新编码（和官方 10 s latent 的余弦相似度 0.9999）。
- **分辨率**：所有测量都在 448p；720p 的数字是按 token 数换算的。
- **硬件数字**（GB300 算力、NVLink 带宽、50% 利用率）是近似值，没有建模通信与计算的重叠。

## 9. 复现

代码在 `.worktrees/topk-trace/experiments/topk_trace/`（分支 `exp/topk-trace`）（输出写到同目录的 `runs/`，已被 `.gitignore` 排除）：

| 文件 | 作用 |
|---|---|
| `build_60s_samples.py` | 选 6 段游戏录像 + 2 段 Sekai，用官方导出器重新编码 60 s latent，生成 10/20/40/60 chunk 的训练 CSV |
| `run_df_train.py` / `run_df_train.sh` | 用真实的 `train.py` 跑几步 DF + VSA 训练，记录每步显存；`--trace` 记录第一步 forward 的选择 |
| `analyze_train_trace.py` | 5.1 节的切分模拟 |
| `trace_hooks.py` | 推理和训练的选择记录（运行时 monkeypatch，不改模型代码） |
| `run_trace.py`、`run_sf_trace.py`、`analyze_trace.py` | 5.2 节的推理 trace |
| `estimate_training.py` | 不同长度下训练的计算 / 通信 / 容量 |
| `estimate_sp.py` | 6.2 节的序列并行方案对比 |
| `estimate_nvl72.py` | 6.3 节的推理估算 |

模型代码：WM-Game-Training `5a5544a`（DF + VSA，与 checkpoint 训练时一致），以 detached worktree 方式检出。
