# Ring attention 的开销：单节点实测、IB 实测与多节点投影

> 2026-10-29。**[实测]** 8×H200（NVLink NV18，每卡一块 400G IB 网卡）；**[估]** 由实测校准的模型投影。
> 代码：`.worktrees/ring-attn/experiments/ring_attn/`（分支 `exp/ring-attn`，不 push），原始数据在其 `runs/`。

## 0. 结论

1. **为什么要 ring**：4K 视频的序列太长，一个样本要切到很多卡上；Ulysses 按 head 切，最多切到 head 数（Wan2.1-14B 有 40 个 head，16 卡除不尽），所以超过 8 卡时要么纯 ring，要么 Ulysses（节点内）× ring（跨节点）。ring 训练和推理都要用。
2. **开销由"每卡 token 数"决定**：每一步 ring，计算 ∝ (每卡 token)²，通信 ∝ 每卡 token，所以每卡 token 越多越容易把通信藏住。
   - 4K、5 s（68 万 token）切 8 卡（每卡 8.5 万）：NVLink 上 ring 只慢 **6%（推理）/ 2%（训练）**；Ulysses 只慢 1%。
   - 每卡 4K token 时：NVLink 上 ring 慢 **90% / 55%**，所有流量走 IB 时慢 **415% / 370%**。
3. **跨节点是瓶颈**：同样 8 卡、每卡 1.6 万 token，所有 hop 走 IB（约 20 GB/s/卡）时 ring 慢 **86% / 54%**，NVLink 只慢 11% / 7%。
4. **ring 实际只能藏住一部分通信**：通信比计算少时，仍有约 30%（推理）/ 40%（训练）的通信暴露出来；8 卡 ring 另有约 9% / 3% 的固定开销（NCCL 和 attention kernel 抢 SM、每步同步）。所以"带宽够就零开销"的理想模型偏乐观。
5. **NCCL 的 ring 收发在中等消息上很慢**：8 卡 NVLink ring 里，16–64 MB 的消息只有 65–80 GB/s，单独一对卡同样大小有 240 GB/s，≥512 MB 才到 350 GB/s。每卡 token 少时，ring 的 KV 块正好落在这个区间。
6. **投影（4K、5 s，Wan2.1-14B）**：
   - 节点之间走 IB 时，纯 ring 16 卡推理多花约 50% 的时间，ring 64 卡多花约 200%；改成 Ulysses 8 × ring 只多 5–35%。
   - 换成 NVL72 级（0.9 TB/s）或 glass（4 TB/s）的域，所有配置都在约 20% 以内，NVL72 和 glass 差别不大（剩下的是固定开销，不是带宽）。
   - **稀疏 attention 让 ring 更吃带宽**：计算降到 13.5% 时，IB 上的 Ulysses 8 × ring 8（64 卡）推理多花约 170% 的时间，NVL72 约 20%，glass 约 17%。

**和项目的关系**：dense 的 4K 视频 attention 计算量太大，带宽好藏；真正吃带宽的是"每卡 token 少"（卡多、序列相对短）和"attention 变稀疏"两种情况。稀疏 attention + ring 把我们之前的稀疏故事从推理扩展到训练：稀疏省了计算，但 ring 每一步仍要传完整的 KV 块，通信藏不住。

## 1. 设置

| 项 | 值 |
|---|---|
| 形状 | Wan2.1-14B：40 head × 128 维，bf16；block 的线性层（self / cross attention 投影 + FFN 13824）用来换算整层开销 |
| 4K、5 s | 3840×2160、81 帧 → 21 个 latent 帧 × 135 × 240 = **680,400 token** |
| 实现 | ring-flash-attn 0.1.8（ring：非 causal；zigzag：causal）+ flash_attn 2.8.4；Ulysses 用 all-to-all + 单卡 flash_attn；USP = Ulysses × ring |
| 环境 | torch 2.11、NCCL 2.28.9；ring 组同时跑满 8 卡（例如 P=2 时 4 个 ring 并行，和 DP × SP 训练一样） |
| IB 模拟 | `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1`：节点内的流量也走网卡（GPUDirect RDMA，经交换机回环），实测约 **20 GB/s/卡**（线速 50 GB/s 的 40%）。真实 2 节点的数见第 5 节 |

**每个配置测 4 个时间**（单层 attention，每卡，3 次取中位数，取最慢的卡）：

| 时间 | 含义 |
|---|---|
| ideal | 一个 flash_attn 内核做完本卡那份计算（N/P 个 query × N 个 key），没有通信 |
| compute | 同样的 ring 循环，但把通信换成本地 KV：多出来的是"拆成 P 个小内核 + 合并 softmax"的开销 |
| comm | 只做 KV 轮转，不算 attention：得到实际带宽 |
| dist | 真实的分布式 attention |

开销 = dist / ideal − 1；整层开销 = (dist − ideal) / (ideal + 线性层)。68 万 token 时 attention 占整层 98%，所以两者几乎一样。

正确性：ring / zigzag / Ulysses 的输出和梯度与单卡 flash_attn 的相对误差都 < 1%（bf16 精度）。

## 2. 单节点结果（8 卡，P = 8）

![ring attention overhead](figures/ring_attn_overhead.png)

attention 开销 / 整层开销，推理（forward）| 训练（forward + backward）：

| 实现 | 链路 | 每卡 4K token | 每卡 16K | 每卡 85K（4K 视频） |
|---|---|---|---|---|
| ring（dense） | NVLink | 90% / 62% \| 55% / 41% | 11% / 10% \| 7% / 6% | **6% / 6% \| 2% / 2%** |
| zigzag（causal） | NVLink | 175% / 95% \| 89% / 55% | 7% / 5% \| 8% / 7% | ≈0 \| 3% / 3% |
| Ulysses（dense） | NVLink | 10% / 7% \| 7% / 5% | 4% / 3% \| 2% / 2% | 1% / 1% \| 1% / 1% |
| Ulysses 4 × ring 2（dense） | NVLink | 21% / 15% \| 13% / 10% | 7% / 6% \| 6% / 5% | 6% / 6% \| 2% / 2% |
| ring（dense） | IB 模拟 | 415% / 287% \| 370% / 273% | 86% / 76% \| 54% / 49% | 11% / 11% \| 8% / 8% |
| zigzag（causal） | IB 模拟 | 775% / 425% \| 704% / 435% | 174% / 142% \| 141% / 121% | 16% / 15% \| 18% / 18% |
| Ulysses（dense） | IB 模拟 | 54% / 37% \| 26% / 19% | 12% / 11% \| 7% / 6% | 3% / 2% \| 1% / 1% |

- **Ulysses 一直比 ring 便宜**：Ulysses 每卡收发的量和卡数无关（约 4 × 每卡 token），ring 每卡要收齐其他所有卡的 KV（约 2 × (P−1) × 每卡 token），8 卡时是 Ulysses 的约 4 倍。只是 Ulysses 受 head 数限制。
- **causal 比 dense 更难藏**：causal 每步计算减半，通信不变，所以平衡点翻倍。
- **训练比推理好藏**：训练整步的计算是 forward 的 3.5 倍，通信约 4 倍（KV 再传一次 + fp32 的 dK、dV），比例接近；但 backward 每一步的计算更长，实测能藏住的通信更多（3.1 节 η 0.80 vs 0.48）。

## 3. 开销从哪来

### 3.1 通信 / 计算比 x 决定大部分开销

把每个配置的"暴露时间" (dist − compute) / compute 对 x = comm / compute 画出来，NVLink 和 IB 模拟落在同一条线上，拟合为：

暴露 / 计算 = max(β·x, x − η) + φ

| | β（x < 1 时暴露的份额） | η（x > 1 时能藏住的计算时间） | φ（8 卡 ring 的固定开销） |
|---|---:|---:|---:|
| 推理 | 0.30 | 0.48 | 8.7% |
| 训练 | 0.42 | 0.80 | 3.2% |

- 理想重叠是 β = 0、η = 1：通信比计算少就完全藏住。实测只藏住一部分，因为 NCCL 的收发内核和 attention 同时占用 SM，且 ring 每一步都要等最慢的卡。
- φ 只出现在 ≥ 3 卡的 ring（2 卡没有），和带宽无关：每卡 8.5 万 token 时传输时间只占计算的 1%，开销仍有 6–9%。**更快的互连去不掉这一部分。**

### 3.2 NCCL ring 收发的带宽（`bench_p2p.py`）

| 每条消息 | 8 卡 ring（NVLink） | 单独一对卡（NVLink） | 8 卡 ring（IB 模拟） |
|---:|---:|---:|---:|
| 16 MB | 67 GB/s | 146 GB/s | 18 GB/s |
| 64 MB | 79 GB/s | 241 GB/s | 19 GB/s |
| 128 MB | 156 GB/s | 274 GB/s | 19 GB/s |
| 512 MB | 351 GB/s | — | 20 GB/s |

- ring 的一个 KV 块 = 每卡 token × 40 head × 128 × 2 B，每卡 4K token 时是 42 MB，正好在最慢的区间。
- 试过 `NCCL_MIN_P2P_NCHANNELS`、`NCCL_P2P_NVL_CHUNKSIZE`、`NCCL_P2P_SCHEDULE_GROUP_SIZE`，都没有改善；`NCCL_P2P_USE_CUDA_MEMCPY`（copy engine，不占 SM）在所有大小都只有约 70 GB/s。
- 原因未确认。现象是 ≥ 3 卡的 ring（每卡同时发给一个邻居、从另一个邻居收）比 2 卡互发慢 2–3 倍。把 K 和 V 合成一条消息可以越过 128 MB 的台阶。

## 4. 投影：4K、5 s，Wan2.1-14B [估]

模型（`model_ring.py`）：每步通信时间用实测的带宽–消息大小曲线；暴露时间用 3.1 节的拟合；拆内核的开销按每卡 token 数取实测值。在实测数据上的误差：每卡 ≥ 8K token 时中位数 2.5%（NVLink）/ 4.3%（IB 模拟），最大 13% / 19%。NVL72 和 glass 用 NVLink 单对卡曲线的形状、按峰值缩放，计算按 H200。

attention 开销，推理 | 训练（单节点配置三列相同）：

| 配置 | 卡数 | 节点间 IB（模拟，20 GB/s） | NVL72 级（0.9 TB/s） | glass（4 TB/s） |
|---|---:|---|---|---|
| ring 8（1 节点） | 8 | 10% \| 4% | 10% \| 4% | 10% \| 4% |
| Ulysses 8（1 节点） | 8 | 0% \| 0%（实测 1% \| 1%） | 同左 | 同左 |
| ring 16 | 16 | 53% \| 24% | 12% \| 5% | 11% \| 5% |
| Ulysses 8 × ring 2 | 16 | 5% \| 4% | 2% \| 2% | 2% \| 2% |
| Ulysses 8 × ring 4 | 32 | 21% \| 11% | 13% \| 7% | 13% \| 7% |
| Ulysses 8 × ring 8 | 64 | 36% \| 21% | 16% \| 12% | 16% \| 12% |
| ring 64 | 64 | 195% \| 111% | 20% \| 14% | 17% \| 13% |
| **稀疏（计算 13.5%）**：Ulysses 8 × ring 8 | 64 | 166% \| 89% | 20% \| 14% | 17% \| 13% |
| **稀疏**：ring 64 | 64 | 1346% \| 1305% | 48% \| 26% | 23% \| 15% |

- **平衡点**（每个 ring 成员至少要多少 token 才藏得住通信，= 算力 / 带宽，causal 翻倍）：H200 NVLink 约 1.0K，IB 模拟约 19K，NVL72 约 0.4K，glass 约 0.1K。换成 GB300 级算力（约 2.5×）时都乘 2.5，IB 上约 47K，比 4K 视频切 16 卡后每卡的 42K 还多。
- 算力越快，越吃带宽：GB300 级算力下，节点间 IB 的纯 ring 16 推理开销从约 53% 涨到约 66%，稀疏时约 735%。
- NVL72 和 glass 在 dense 下几乎一样：剩下的是 3.1 节的固定开销和拆内核开销，和带宽无关。glass 的优势只在"稀疏 + 卡多"时明显（ring 64 稀疏：48% → 23%）。

## 5. 两节点实测（真实 IB）

**待补**：作业 `ring_2node.sbatch`（2 节点 × 4 卡，lowprio）在排队。测量内容：ring 8 跨两节点、Ulysses 4 × ring 2、跨节点 2 卡 ring（4 对同时）、真实 IB 的带宽–消息大小曲线。

## 6. 局限

- IB 模拟只有约 20 GB/s/卡，低于线速 50 GB/s，第 4 节的 IB 列偏悲观，待第 5 节校准。
- 只测了 attention 层；没有建模 Ulysses 和线性层、FSDP 通信的重叠。
- NVL72 / glass 的投影沿用 H200 上实测的固定开销和重叠份额，并假设 NCCL 在这些链路上没有 3.2 节的中等消息问题。
- ring-flash-attn 是常用的参考实现，不是最优实现：例如 K、V 分两条消息发，backward 用 fp32 传 dK、dV。更好的实现（合并消息、用 copy engine、双缓冲）可能降低固定开销。
- 稀疏 attention 用"计算乘 13.5%、通信不变"近似（PAN 训练 trace 的比例），没有测真实的块稀疏 ring 内核。

## 7. 复现

```bash
cd .worktrees/ring-attn/experiments/ring_attn && source env.sh
torchrun --nproc_per_node=8 bench_ring.py --check --out /dev/null --tag check   # 正确性
./run_ring.sh nvlink          # 单节点 NVLink 扫描
./run_ring.sh ib-emu          # 所有流量走 IB 网卡
torchrun --nproc_per_node=8 bench_p2p.py --out runs/p2p.jsonl --tag nvlink
sbatch --partition=lowprio --qos=lowprio ring_2node.sbatch                       # 2 节点 × 4 卡
python3 analyze_ring.py runs/nvlink.jsonl runs/ib-emu.jsonl                      # 表格
python3 model_ring.py --bench runs/nvlink.jsonl runs/ib-emu.jsonl --p2p runs/p2p.jsonl [--gpu-scale 2.5]
python3 plot_ring.py runs/nvlink.jsonl runs/ib-emu.jsonl --out <main>/docs/figures/ring_attn_overhead.png
```
