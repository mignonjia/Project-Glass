# 训练方向的结论（已结束，作对照）

> 2026-10-06 至 10-08 做的 PAN DF + VSA 训练分析。10/8 起主线改为实时推理，这里只保留结论和关键数字。
> 完整的逐层拆解、长度扫描和固定预算推导在 git 历史里：commit `cc4a3f0` 的 `docs/pan_compute_comm_estimate.md`、`docs/david_training_load.md`、`docs/fixed_budget_reuse_estimate.md`。
> 模型参数和推理数据见 `pan_measurements.md`。硬件：GB300 2.5 PFLOP/s × 45%，NVLink 单向 0.9 TB/s，IB 0.05 TB/s，glass 单向约 4 TB/s。

## 结论

1. **dense 29B 的训练在 NVL72 内是算力瓶颈。** 10 分钟 720p，最优切分（Ulysses 8 × 条纹 CP 9）通信只占计算的 9%，换成 glass 每步 18.0 → 16.8 s（不重叠时约 1.07×，完全重叠时没有收益）；同样切分放到 8 卡节点 + IB 上，通信是计算的 1.16×。
2. **根本原因是线性层**：一个 chunk-层的线性计算 2.46 TFLOP，对应的 KV 只有 67 MB（约 3.7 万 FLOP / 字节），而 NVLink 的平衡点约 1250 FLOP / 字节。
3. **稀疏省计算、不省通信**：5% top-k 的 attention 计算只有 dense 的 13.5%，切到 8 卡时取数仍是 dense 的 49%（和 SparSP 的"计算 −71%、通信 −19%"一致）。
4. **causal 负载不均衡**：越靠后的 chunk 越重。连续切最慢卡约 1.3×（60 s trace），10 分钟估算约 1.66×；条纹切约 1.1×，但通信翻倍。
5. **容量是触发条件**：一个样本需要的卡数超出快的 scale-up 域时才变成带宽瓶颈（720p 单视角：8 卡 H200 节点约 4–5 分钟，NVL72 约 1.5 小时）。
6. **固定预算稀疏也不改变结论**：每个 query tile 看 k 个远端 tile，通信 / 计算由复用度 R（一个取来的 tile 被本卡多少个 query tile 用到）决定。实际放置下 R ≈ 12.5，NVL72 → glass 约 1.035×；只有 R ≈ 1 且每个 query tile 取 7.8K 以上远端 token 时 NVLink 才成瓶颈。

## 实测：显存与速度（8×H200，FSDP，每卡 1 个样本，448p，5%）[实测]

| 视频长度 | token | 每步时间 | 每卡峰值显存 |
|---|---:|---:|---:|
| 10 s | 1.7 万 | 4.5 s | 70.3 GiB |
| 20 s | 3.3 万 | 8.5 s | 76.5 GiB |
| 40 s | 6.6 万 | 15.6 s | 97.2 GiB |
| 60 s | 9.7 万 | 31.7 s | 120.0 GiB |

显存增长超线性（每 token 420 → 700 → 765 KB），来源未核实。

## 训练 trace：把 60 s 切到 P 张卡（每次 forward）[trace]

| GPU 数 | 切法 | 取数 / dense | 不均衡（最慢 / 平均） | 总取数 | GPU 对偏斜 |
|---:|---|---:|---:|---:|---:|
| 8 | 连续 | 49% | 1.29 | 134 GB | 2.67 |
| 8 | 条纹 | 54% | 1.10 | 275 GB | 1.95 |
| 15 | 连续 | 40% | 1.31 | 215 GB | 3.63 |
| 60 | 连续 | 26% | 1.33 | 592 GB | — |

CP 的 GPU 对偏斜（最高 3.6×）和 GlassML 测的 MoE AllToAll 偏斜（3.6–6.3×）同一量级。

## 10 分钟 720p 的切分（NVL72）[估]

| 方案 | 卡数 / 样本 | 每卡显存 | 每步计算 | 通信 / 计算（NVL72 / IB） |
|---|---:|---:|---:|---:|
| Ulysses 32 | 32 | 180 GB | 36.9 s | 约 0.01 / 约 0.2 |
| Ulysses 8 × CP 9，条纹 | 72 | 98 GB | 16.5 s | **0.09 / 1.16** |
| 固定预算 k = 32，Ulysses 16 | 16 | 204 GB | 31.4 s | 0.038（glass 加速上限 1.03×） |

## 复现

`.worktrees/topk-trace/experiments/topk_trace/`：`run_df_train.sh`、`build_60s_samples.py`（显存实测和训练 trace）；`analyze_train_trace.py`（切分模拟）；`step_breakdown.py`、`estimate_training.py`、`estimate_sp.py`（估算）。60 s 训练在 kernel 缓存为空时第 1 步会崩（CUDA illegal memory access），排查记录见该目录的 `README.md`。
