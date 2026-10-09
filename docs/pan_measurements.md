# PAN + VSA：模型参数、实测与选择 trace

> 实时推理估算的基础数据。来源标注：**[实测]** 8×H200 上跑出来的；**[trace]** 真实运行中记录的 top-k 选择统计；**[估]** 按公式推算。
> 训练方向的结果见 `training_reference.md`。复现代码在 `.worktrees/topk-trace/experiments/topk_trace/`（分支 `exp/topk-trace`）。

## 1. 模型与 workload

| 项 | 值 |
|---|---|
| 模型 | PAN 29.2B（bf16 权重约 58 GB）：44 层，hidden 4096，32 head（head_dim 128） |
| 稀疏 attention | VSA hybrid：最近 3 个 chunk 精确 + 远端 top-k（checkpoint 按 15% 训，本文默认 **5%**）；**每层、每个 head、每个 query block（128 token）各自选，每个去噪步重新选** |
| 检查点 | DF：`yiyan_recording10s_w3_remote15_1k_v1`（代码 `5a5544a`）；SF 4 步：`sf400-integ-f31474c`（代码 `f31474c`） |
| 448p（实测分辨率） | 768×448、16 fps；1 chunk = 4 latent 帧 = 1 s = 1344 视频 + 256 caption = 1600 token（补齐 1792 行） |
| 720p（估算分辨率） | 1280×704、24 fps；1 chunk = 0.67 s = 3520 视频 + 256 caption = 3776 token（补齐 4096 行 = 32 个 tile） |

## 2. 基本公式

- **KV / token** = 2（K、V）× 44 层 × 4096 × 2 B = **721 KB**。10 分钟 720p（900 chunk × 4096 行）≈ **2.7 TB / 路流**；有 CFG 时翻倍。
- **latent / token** ≈ 128 B（2×2 patch × 16 通道 × bf16，通道数按 Wan 2.1 VAE 估算），比 KV 小约 5600 倍。
- **线性层 / chunk-层（720p）** = 3776 × 0.6525 GFLOP ≈ 2.46 TFLOP；44 层 ≈ 108 TFLOP / forward。
- **attention / chunk-层** = 4 × 行数 × 看到的 key 数 × 4096：窗口（3 个 chunk）0.82 TFLOP；5% 远端在 10 分钟末尾约 12.3 TFLOP。
- **一个 tile 的 K+V**（32 head）= 2 MiB；一个 720p chunk 的完整 KV = 32 tile ≈ 67 MB / 层。

## 3. 推理实测 [实测]

| 设置（448p，单张 H200） | 峰值显存 |
|---|---:|
| DF，50 步 + CFG，10 s | 110.7 GiB（含借用训练代码带来的 13.6 GiB fp32 分片） |
| SF，4 步无 CFG，60 s，保留完整历史 | 第一次 OOM（约 142 GiB）；文本编码器挪到 CPU 后 **130.4 GiB** |

单卡只能放约 1 分钟（无 CFG）或约 30 s（有 CFG）的完整历史。

## 4. 推理的选择 trace [trace]

SF 4 步、60 s、5%（`runs/sf60_r05`，8 个样本），以及 DF 50 步 + CFG、10 s（15% / 5%）：

| | SF 4 步，60 s，5% | DF 50 步，10 s，15% / 5% |
|---|---:|---:|
| 一个 chunk 内各 query block 选择的并集 / 单个 | 4.3× | 4.5× / 4.6× |
| **每步要取的远端历史比例**（并集） | **22.9%** | — |
| **一个 chunk 全部步的并集** | **27.7%**（= 单步 × 1.21） | 单步 × 1.16 / 1.31 |
| 相邻步之间新增的 tile | 约 7%（p90 18%） | 0.8% / 1.1% |
| 一个 head 选中时其他 head 也选中的比例 | 26% | 70% / 31% |

按历史长度分段（SF）：

| chunk 序号 | 0–9 | 10–19 | 20–29 | 30–39 | 40–49 | 50–59 |
|---|---:|---:|---:|---:|---:|---:|
| 每步取的远端比例 | 29% | 24% | 22% | 22% | 22% | 22% |
| 全部步的并集 | 35% | 30% | 27% | 26% | 26% | 26% |

- 20 个 chunk 以后比例基本不变，外推到 10 分钟有一定依据，但仍待验证。
- **跨步缓存**：一个 chunk 的 4 步 + commit 共用取来的 KV 时，读取量从约 5 × 22.9% 降到约 27.7%，约 1/4。
- 选择在去噪步之间很稳定：50 步时每个 chunk 取一次能复用约 38 次，4 步时约 3.4 次。

## 5. 局限

- 所有测量都在 448p；720p 的数字按 token 数换算。
- checkpoint 只按 10 个 chunk、15% 训过；trace 在 60 个 chunk、5% 下记录，长视频训练后的选择模式可能不同。
- 没有 60 s 的 YouTube 片段，游戏部分全部来自录像；latent 用官方导出器重新编码（和官方 latent 的余弦相似度 0.9999）。

## 6. 复现

| 文件 | 作用 |
|---|---|
| `run_sf_trace.sh` / `run_sf_trace.py` | SF 推理 trace（`./run_sf_trace.sh sf60_r05 0.05 60 8`） |
| `run_trace.sh` / `run_trace.py` | DF 50 步 + CFG 推理 trace |
| `analyze_trace.py` | 第 4 节的统计（`--bin 10` 按历史长度分段） |
| `trace_hooks.py` | 运行时 monkeypatch 记录选择，不改模型代码 |
