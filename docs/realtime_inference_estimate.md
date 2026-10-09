# 实时推理估算：glass 在容量和带宽上带来什么

> 问题：PAN（SF 4 步 + 1 次 commit，5% 远端 top-k）实时生成 720p 视频时，一个系统能同时跑几路流？卡在容量、算力还是带宽？glass 改变了哪一个？
> 脚本在 worktree `exp/topk-trace` 的 `experiments/topk_trace/`：`estimate_realtime.py`（估算模型）、`analyze_placement.py`（从 trace 统计放置相关的量）。模型参数和 trace 来源见 `pan_measurements.md`；为什么需要长历史见 `motivation.md`。
> 来源标注：**[trace]** 来自 `runs/sf60_r05`（448p、60 s、8 个样本，取第 20 个 chunk 以后的平台期）；**[估]** 按公式推算；**[假设]** 是还没确认的硬件数字。

## 结论

1. **glass 的主要收益是容量。** 10 分钟 720p，只看内存能放几路流：NVL72 只用 HBM 是 5 路，加 Grace LPDDR 是 11 路，glass 加 85 TB DRAM 是 36 路。
2. **容量能不能变成流数，要看算力。** 按 5% 比例选块时，attention 计算随历史线性增长，10 分钟末尾每路要约 5 个 tile 的算力。glass 只有 64 个 tile，算力上限约 12 路，实际 11–12 路。相比 NVL72 只用 HBM 是 2.4×，相比 NVL72 + Grace 是 1.3×。
3. **固定选块预算时 glass 的优势大得多。** 每个 query 固定看 20 个 chunk 的 key，attention 不再随历史增长：10 分钟时 glass 21 路，NVL72 + Grace 10 路，只用 HBM 5 路；30 分钟时分别是 12 / 3 / 1 路；60 分钟时是 6 / 1 / 0 路。
4. **需要的 DRAM 带宽不高，前提是调度做对。** 关键是复用：同一个 chunk 的 4 步共用取来的 KV，相邻 chunk 之间也复用（trace：63% 重叠）。这样每路流从 DRAM 读约 0.2–0.4 TB/s，每 tile 0.05–0.3 TB/s 就接近算力上限。不复用（每步重读）要每 tile 1–2 TB/s。NVL72 + Grace 卡在两头：LPDDR 每卡约 0.19 TB/s 太慢，HBM 又剩不下多少空间做缓存。
5. **稳态下互连不是瓶颈。** 按 head 切分时每个 tile 只读自己那部分 head 的 KV，互连只传激活的 all-to-all，每个 chunk 5–8 ms。glass 的光互连有用的地方是：多路流共享 DRAM 池（错开后峰值 / 均值从 2.3 降到约 1.4）、流长短不一时跨 tile 借容量、整路流的 KV 换入换出。
6. **glass 不占优的情况：** 算力受限时（15% top-k、多视角联合 attention），glass 的 tile 比 NVL72 少（64 vs 72），流数反而略少；视频短到 HBM 放得下（约 1–2 分钟以内）时，也不需要 DRAM。

**对讲法的影响**：容量是起因这一点被数字支持。"带宽是瓶颈"只在不做复用、或 Grace 这种慢的容量层时成立。更准确的说法是：**有了足够容量以后，实时推理变成调度问题**：哪些 KV 放 HBM、哪些放 DRAM，怎么在步之间、chunk 之间复用，多路流怎么错开。DRAM 带宽需求在 0.1–2 TB/s 每 tile 之间，取决于调度做得多好。讲法待和导师 / David 讨论后再改 README。

---

## 1. 方法：一步步算

### 1.1 一路流需要什么（10 分钟 720p，最后一个 chunk）[估]

| 量 | 算法 | 值 |
|---|---|---:|
| 每个 chunk 的时长（截止时间） | 16 帧 / 24 fps | 0.667 s |
| chunk 数 | 600 s / 0.667 s | 900 |
| 每个 token 的 KV | 2 × 44 层 × 4096 × 2 B | 721 KB |
| 每个 chunk 的 KV | 4096 行 × 721 KB | 2.95 GB |
| 一路流的 KV | 900 × 2.95 GB | **2.66 TB** |
| 每次 forward 的计算 | 线性 108 + 窗口 attention 36 + 远端 attention 542 TFLOP | 687 TFLOP |
| 每个 chunk 的 forward 数 | 4 步 + 1 次 commit | 5 |
| 实时所需算力 | 5 × 687 TFLOP / 0.667 s / (2.5 PFLOP/s × 45%) | 4.6 个 tile |

远端 attention 占了每次 forward 的 79%，而且随历史线性增长（5% 比例选块时，历史越长，每个 query 看的 key 越多）。

### 1.2 每次 forward 要取多少 KV [trace]

- 每个 query block 每个 head 选 5% 的远端 block，但一个 chunk 里的 13 个 query block 选得不一样，**并集是 21.8%**（4.36×）。取数按并集算：**每次 forward 取 21.8% × 2.65 TB = 577 GB**。
- 一个 chunk 的 4 步并集是 26.0%（单步的 1.19×）：同一个 chunk 内各步选的块大部分相同。
- **相邻 chunk 的并集重叠 63%**，最近 3 个 chunk 合起来覆盖 77%（新发现，`analyze_placement.py`）。

三种取数策略（假设全部历史都不在 HBM 里，每路流）：

| 策略 | 做法 | 每个 chunk 从容量层读 | 读带宽 | HBM 里的缓存 |
|---|---|---:|---:|---:|
| reread | 每次 forward 读自己的选择 | 5 × 577 GB = 2.9 TB | 4.33 TB/s | 0 |
| cache | 一个 chunk 读一次并集，留在 HBM 给 5 次 forward 用 | 26% × 2.65 TB = 687 GB | 1.03 TB/s | 687 GB |
| reuse | cache，并且上一个 chunk 取来的也留着，只读新增的 37% | 0.37 × 687 GB = 254 GB | 0.38 TB/s | 约 687 GB |

**这就是"用容量换带宽"**：HBM 里留的缓存越多，读的越少。代价是缓存占了本来可以放历史的 HBM。

### 1.3 怎么把一路流放到多个 tile 上：按 head 切

一路流要约 5 个 tile 的算力，所以要切到多个 tile 上。我们假设 **Ulysses 式按 head 切**：每个 tile 负责一部分 head（32 个 head 分到 G 个 tile），**这些 head 的 KV 也存在这个 tile 自己的内存里**。VSA 是每个 head 自己选块，所以每个 tile 只读自己内存里的 KV，**"取远端 KV"变成了读本地内存**（HBM 或本地 DRAM），不走互连。

互连只传每层 attention 前后的激活（Q、K、V 和输出）：每次 forward 44 层 × 4 × 4096 行 × 4096 × 2 B = 5.9 GB（整个组），每个 chunk 每个 tile 约 4 GB。走 NVLink（0.9 TB/s）约 5 ms，走 glass 光互连（4 TB/s）约 1 ms，都远小于 667 ms。

代价是 **head 之间取数不均**：每个 head 选到的块数不同，最慢的 tile 决定这一轮的时间。[trace] 按 head 分成 G 组，每轮最大组 / 平均组：

| G（每路流的 tile 数） | 2 | 4 | 5 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| 最慢 / 平均 | 1.05 | 1.16 | 1.22 | 1.32 | 1.58 | 2.05 |

估算里取数时间乘上这个系数。

### 1.4 算力的 roofline：稀疏 attention 卡在 HBM 带宽上

一个（query tile, key tile, head）对：计算 4 × B × B × 128 FLOP，读 KV 4 × B × 128 字节，所以**每字节 B 次计算**（B = tile 大小，trace 里是 128）。GB300 的平衡点是 2.5 PFLOP/s ÷ 8 TB/s ≈ 312 FLOP/字节。128 < 312，所以**稀疏 attention 读 HBM 比算还慢**：542 TFLOP 的远端 attention 要读 4.2 TB HBM，约 0.53 tile·s，比按算力算的 0.48 tile·s 还长。估算里 attention 取两者的较大值。tile 换成 256 时这一项减半（流数 12 → 12–13）。

### 1.5 系统级：用"tile·秒"记账

- 一个 chunk 内，一路流要用的 tile 时间 = 计算 + 从容量层取数 + 通信（单位 tile·秒）。
- N 路流放在 T 个 tile 上：**N × 每路 tile·秒 ≤ T × 0.667 s**。
- 两种算法：
  - **serial**（保守）：计算、取数、通信串行，一个 tile 只服务一路流时就是这样（取数要等打分结果，不能提前）。
  - **overlap**（乐观）：一个 tile 交替服务多路流，A 流取数时算 B 流，时间取 max(计算, 取数)。
- **容量**：HBM 先放权重（58 GB）、工作区（20 GB）、最近 3 个 chunk 的窗口、缓存，剩下的放历史；放不下的历史进容量层（Grace LPDDR 或 glass DRAM）。
- 所有流都按最后一个 chunk（历史最长、最重）算。

### 1.6 系统参数

| 系统 | tile 数 | HBM / tile | 容量层 / tile | 容量层带宽 / tile | 互连 / tile |
|---|---:|---|---|---|---|
| NVL72，只用 HBM | 72 | 288 GB，8 TB/s | — | — | NVLink 0.9 TB/s |
| NVL72 + Grace LPDDR5X | 72 | 同上 | 240 GB（17.3 TB / 72） | 0.19 TB/s（14 TB/s / 72） | 0.9 TB/s |
| glass，只用 HBM | 64 | 同上 | — | — | 光互连约 4 TB/s [假设] |
| glass + DRAM（本地，TGV 直连） | 64 | 同上 | 1.33 TB（85 TB / 64）[假设] | **扫描 0.1–8 TB/s** [假设] | 约 4 TB/s |
| glass + DRAM（共享池，经光互连） | 64 | 同上 | 85 TB 共享 | min(DRAM 带宽, 4 TB/s) | 约 4 TB/s |

算力：每个 tile 2.5 PFLOP/s BF16 × 45% MFU。glass 的 tile 按 GlassML 论文配置（B300 + 288 GB HBM3e）。

---

## 2. 结果

### 2.1 10 分钟 720p，每系统能跑几路流 [估]

`python3 estimate_realtime.py`（serial），`--mode overlap`：

| 系统 | N（serial） | N（overlap） | 卡在哪里 | 只看容量 | 只看算力 |
|---|---:|---:|---|---:|---:|
| NVL72，只用 HBM | 5 | 5 | 容量 | 5 | 14 |
| NVL72 + Grace | 9 | 9 | 带宽 + HBM 缓存空间 | 11 | 14 |
| glass，只用 HBM | 4 | 4 | 容量 | 4 | 12 |
| **glass + DRAM（1 TB/s）** | **11** | **12** | 接近算力 | 36 | 12 |

每个 chunk 的时间拆解（每路流在自己的 tile 组上，serial，`--breakdown`）：

| 系统 | 路数 | 每路 tile 数 | 线性 | attention | 取数 | 通信 | 合计 / 截止 667 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| NVL72，只用 HBM | 5 | 14.4 | 45 | 184 | 0 | 7 | 235 ms（算力闲着，容量满了） |
| NVL72 + Grace（reuse） | 9 | 8.0 | 80 | 331 | 128 | 8 | 548 ms |
| glass + DRAM（reuse） | 11 | 5.8 | 111 | 455 | 44 | 5 | 615 ms |

NVL72 只用 HBM 时每路流要 14 个 tile 才放得下 KV，算力只用了约 1/3。glass 每路流用约 6 个 tile，**算力和容量都用满**。

### 2.2 glass DRAM 带宽要多少 [估]

`--sweeps`：glass + DRAM（本地），自动选最好的策略。

| 每 tile DRAM 带宽（TB/s） | 0.1 | 0.19 | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5% 比例，serial | 8 | 9 | 9 | 11 | 11 | 12 | 12 |
| 5% 比例，overlap | 12 | 12 | 12 | 12 | 12 | 12 | 12 |
| 固定 20 chunk，serial | 14 | 17 | 18 | 20 | 21 | 22 | 22 |
| 固定 20 chunk，overlap | 23 | 23 | 23 | 23 | 23 | 23 | 23 |

- 能交替服务多路流（overlap）时，**0.1 TB/s 每 tile 就够**，因为 reuse 策略读得很少，取数藏在别的流的计算后面。
- 一个 tile 组只服务一路流（serial）时，**每 tile 0.5–1 TB/s 拿到 90% 以上，约 2 TB/s 到算力上限**。
- 带宽越高，越可以不用缓存（直接 reread），省下 HBM：4 TB/s 时最优策略从 reuse 变成 cache 或 reread。

**要问 Tianqi 的数字**：每 tile DRAM 带宽能否到 0.5–2 TB/s；DRAM 是 tile 正下方直连还是走光互连；访问粒度 64 KiB（一个 head、一个 128-token block 的 K+V）时的有效带宽和延迟。

### 2.3 视频长度 vs 流数（review 图表 2）[估]

`--sweeps`，serial，最优策略。格内为路数，括号里是限制因素（容 = 容量，算 = 算力，带 = 带宽）。

**5% 比例选块**（attention 随历史增长）：

| 视频长度 | NVL72 HBM | NVL72 + Grace | glass HBM | glass + DRAM |
|---|---|---|---|---|
| 1 min | 50（容） | 52（算） | 44（容） | 46（算） |
| 2 min | 25（容） | 36（带） | 22（容） | 35（带） |
| 5 min | 10（容） | 19（带） | 8（容） | 20（带） |
| 10 min | 5（容） | 9（带） | 4（容） | 11（带） |
| 20 min | 2（容） | 4（带） | 2（容） | 6（带） |
| 30 min | 1（容） | 3（容） | 1（容） | 4（算） |
| 60 min | 0（容） | 1（容） | 0（容） | 2（算） |

**固定 20 chunk 预算**（`--budget-chunks 20`，attention 不随历史增长）：

| 视频长度 | NVL72 HBM | NVL72 + Grace | glass HBM | glass + DRAM |
|---|---|---|---|---|
| 1 min | 25（算） | 25（算） | 23（算） | 23（算） |
| 2 min | 25（容） | 25（算） | 22（容） | 23（算） |
| 5 min | 10（容） | 20（带） | 8（容） | 22（带） |
| 10 min | 5（容） | 10（带） | 4（容） | 21（带） |
| 20 min | 2（容） | 5（容） | 2（容） | 17（带） |
| 30 min | 1（容） | 3（容） | 1（容） | 12（容） |
| 60 min | 0（容） | 1（容） | 0（容） | 6（容） |

- 1–2 分钟时 HBM 够用，大家都卡在算力上，glass 因为 tile 少略输。
- 5 分钟以上 NVL72 开始卡容量，glass 的优势随长度增大：固定预算时 10 分钟 2.1×（vs Grace）、30 分钟 4×、60 分钟 6×。
- 5% 比例选块时，长视频的 attention 计算太大，glass 很快也卡在算力上，优势只有约 1.3–2×。**选块方式（比例 vs 固定预算）决定了 glass 的收益有多大**，这是要和 PAN / VSA 的人确认的算法问题。

### 2.4 敏感性（10 分钟 720p，serial）[估]

| 设定 | NVL72 HBM | NVL72 + Grace | glass + DRAM | 说明 |
|---|---:|---:|---:|---|
| 默认（5%） | 5 | 9 | 11 | |
| 720p 选块并集更大（`--qscale 1.53`） | 5 | 8 | 11 | 见 3.2；reuse 时影响小 |
| 15% top-k | 5 | 5 | 4 | 每路 12 个 tile 的算力，全部卡算力 |
| tile 256 | 5 | 9 | 12 | attention 读 HBM 减半 |
| MFU 30% | 5 | 8 | 8 | 算力更紧 |
| 固定 20 chunk 预算 | 5 | 10 | 21 | |
| 机器人：448p × 4 视角，5 分钟 | 8 | 15 | 15 | Grace 和 glass 都接近算力上限（16 路，每路 3.7 tile）；glass 容量上限 62 路 |
| 驾驶：720p × 7 视角联合 attention，2 分钟 | 1 | 1 | 1 | 每路 52 个 tile 的算力；多视角联合 attention 是算力问题 |

---

## 3. trace 里和放置有关的发现 [trace]

`python3 analyze_placement.py runs/sf60_r05`（输出 `runs/sf60_r05/analysis/placement_summary.json`）。

### 3.1 访问不集中在最近的历史

把一个 chunk 选中的远端 block 按"在历史中的位置"分成 10 段（0 = 最老）：

| 位置（老 → 新） | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 取数占比 | 13.8% | 11.2% | 10.2% | 10.7% | 11.3% | 12.3% | 7.6% | 6.7% | 7.2% | 8.9% |
| block 数占比 | 11.3% | 9.9% | 10.1% | 9.9% | 9.5% | 10.4% | 10.2% | 9.9% | 10.1% | 8.7% |

取数在整个历史上几乎均匀，最老的一段还略多。所以**不能只把最近的历史放 HBM、老的放 DRAM 来省带宽**：要的是"按内容随机访问整个历史"，这正是 `motivation.md` 第 5 节的论点在 PAN 上的实测。

### 3.2 720p 的并集会更大

一个 chunk 的 query block 越多，各 block 选择的并集越大。[trace] 只保留 n 个 query block 时的并集比例：

| n | 1 | 2 | 4 | 8 | 13（全部） |
|---|---:|---:|---:|---:|---:|
| 并集 / 远端历史 | 5.1% | 7.8% | 11.9% | 16.9% | 21.7% |

近似按 n^0.51 增长。720p 一个 chunk 约 30 个 query block，外推每步约 **33%**（`--qscale 1.53`）。这是上限估计（实际可能饱和），要用 720p 实测确认。

### 3.3 取数量随层变化，多路流错开可以削峰

- 每轮（一层、一步）的取数量在各层之间差 3 倍（层平均的 0.47–1.47×）。448p 时每轮平均 222 MB（p90 359 MB）；720p、10 分钟时约 13 GB 一轮（每路流），切到 6 个 tile 每个约 2.3 GB，1 TB/s 下约 2.3 ms，而每轮预算约 3 ms（667 ms / 220 轮）。
- 8 个样本当作 8 路流共享一个 DRAM 池：
  - 各流同步（同一时刻在同一层）：总需求峰值 / 均值 2.30，和单路流（2.35）几乎一样。各层的取数模式相同，所以同步时不会抵消。
  - **各流错开**（随机起始相位）：峰值 / 均值约 **1.4**。
  - 也就是说，共享池加上错开调度，带宽只需按均值的 1.4 倍准备，而不是 2.3 倍。这是 GlassML 式调度器可以做的事，也是共享池（光互连）相对本地 DRAM 的一个具体好处。

---

## 4. 假设与局限

- **全部 trace 在 448p**；720p 按 token 数换算，选择比例用 448p 的平台期值（并集可能更大，见 3.2）。
- **从 60 个 chunk 外推到 900 个**：比例在 20 个 chunk 后基本不变，但没在更长的历史上验证。checkpoint 只按 10 个 chunk、15% 训过。
- **MFU 45%** 是假设，稀疏 attention kernel 的实际效率可能更低（MFU 30% 时 glass 降到 8 路）。
- **glass 硬件数字都是假设**：85 TB DRAM、每 tile 约 4 TB/s 光互连、DRAM 带宽未知（扫描）、DRAM 的位置（本地 / 共享池）未知。光路重配置时间没有建模：每个 chunk 有 220 轮取数，如果每轮要重配置，重配置时间必须远小于 3 ms。
- **"tile·秒"是流体模型**：没有考虑每路流的 tile 数必须是整数、32 个 head 能否均分、以及各流处在不同长度时的混合情况（按所有流都在最后一个 chunk 算，偏保守）。
- **跨 chunk 复用（63%）只在 448p 测过**，reuse 策略的缓存大小按一个 chunk 的并集估，没模拟淘汰策略。
- 多视角按"把所有视角拼成一个大帧、联合 attention"算，算力会按视角数的平方增长；如果视角之间只做局部 attention，结果会完全不同。
- 不含 VAE 解码、文本编码器、CFG（SF 不用 CFG）。

## 5. 复现

在 worktree `.worktrees/topk-trace/experiments/topk_trace/` 下：

```bash
python3 analyze_placement.py runs/sf60_r05                 # 第 3 节（CPU，约 3 分钟）
python3 estimate_realtime.py --breakdown --sweeps          # 2.1–2.3（5% 比例）
python3 estimate_realtime.py --mode overlap --sweeps       # overlap 版本
python3 estimate_realtime.py --budget-chunks 20 --sweeps   # 固定预算
python3 estimate_realtime.py --qscale 1.53                 # 2.4 各行：--ratio 0.15 / --tile 256 / --mfu 0.30 /
                                                           # --res 448p --views 4 --minutes 5 / --views 7 --minutes 2
```

## 6. 下一步

- [ ] 和导师 / David 讨论：讲法从"带宽是瓶颈"调整为"容量是起因，之后是 HBM / DRAM 之间的调度问题"？
- [ ] 问 PAN / VSA 的人：长视频用比例选块还是固定预算？（决定 glass 收益是 1.3× 还是 2–6×）
- [ ] Tianqi：DRAM 每 tile 带宽（0.5–2 TB/s 是否可行）、位置（本地 / 共享）、64 KiB 粒度的效率、光路重配置时间。
- [ ] 720p 实测：并集比例（3.2）、跨 chunk 复用、每 chunk 延迟。
- [ ] 画 review 图表 1–3（数据都在脚本输出里）。
