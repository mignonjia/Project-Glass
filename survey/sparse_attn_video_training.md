# 稀疏 attention × 视频生成 × GPU 系统（偏训练）文献调研（2026-10-06）

> 范围：只看 DF + VSA **训练**相关的问题（项目方向见 README，2026-10-06 起以训练为主）。
> 条目都经网络检索核实（arXiv / 会议页面）。标 ⚠️ 的有部分信息没核实。
> 先不涉及 glass；这一层要回答的是"在传统 GPU（NVLink / PCIe）上，稀疏 attention + 长序列训练卡在哪里，别人怎么解决"。

## 先读这些（按顺序）

1. **VSA**（NeurIPS 2025）：我们用的稀疏 attention 本身。看 coarse（tile 均值）+ fine（top-k tile）两段式和 kernel 设计。
2. **MoC: Mixture of Contexts**（ICLR 2026）：长视频 + 可训练 top-k chunk 路由。和 W3 + remote 最像，必须讲清差异。
3. **MTraining**（arXiv 2025-10）：动态稀疏 attention + ring/context parallel **训练**，明确提出 worker 级和 step 级不均衡。训练侧最接近的系统工作（LLM）。
4. **Striped Attention**（arXiv 2023）：causal + ring attention 的负载不均衡及条纹式分配。对应我们测到的"越靠后的 chunk 越重"。
5. **SparSP**（arXiv 2026-09）："稀疏 attention 去掉最多 71.3% 的计算，通信只减少 18.9%"。和我们的 trace 结论一致，但它只做推理、只用 PCIe。
6. 选读：**db-SP**、**FVAttn**（稀疏 + 序列并行的负载均衡，推理）；**NSA / MoBA**（LLM 可训练稀疏 attention）。

---

## 1. 视频扩散的稀疏 attention（算法 + kernel）

| Paper | Venue | 做了什么 | 与 DF+VSA 训练的关系 | Link |
|---|---|---|---|---|
| Faster Video Diffusion with Trainable Sparse Attention（**VSA**；Zhang, …, Stoica, Xing, Hao Zhang） | NeurIPS 2025 | coarse 阶段把 token 池化成 tile 找关键 tile，fine 阶段只在选中的 tile 内算 token 级 attention；训练 FLOPs 降 2.53×，loss 不变 | 我们的 attention 本身（双向；我们用的是 causal + W3 + remote 的改造版） | [arXiv 2505.13389](https://arxiv.org/abs/2505.13389) |
| Fast Video Generation with Sliding Tile Attention（**STA**；Zhang 等） | ICML 2025 | 按 tile 滑窗的局部 attention；HunyuanVideo 端到端 945s → 685s（免训练）/ 268s（微调） | 对应我们的"最近窗口"部分；tile 化的硬件友好设计 | [arXiv 2502.04507](https://arxiv.org/abs/2502.04507) |
| Sparse VideoGen2（**SVG2**；Yang, …, Han Cai, Song Han 等） | NeurIPS 2025 spotlight | k-means 按语义重排 token，把关键 token 聚到一起，减少浪费；免训练 | 另一种选块方式（语义聚类 vs tile 均值） | [arXiv 2505.18875](https://arxiv.org/abs/2505.18875) |
| Radial Attention（Li 等；MIT Han Lab） | NeurIPS 2025 | 发现时空能量衰减，静态 mask，窗口随时间距离缩小，O(n log n)；视频可延长 4× | 静态模式 → 通信可预先确定；我们的是动态选择 | [arXiv 2506.19852](https://arxiv.org/abs/2506.19852) |
| Mixture of Contexts for Long Video Generation（**MoC**；Cai 等） | ICLR 2026 | 每个 query 动态选少量 chunk + 必选 anchor（caption、局部窗口）；剪掉 85% 以上 token 对，FLOPs 降 7×，分钟级场景端到端 2.2× | **和 W3 + remote 最接近的算法**；只讲 FLOPs，不讲多卡通信和内存层级 | [arXiv 2508.21058](https://arxiv.org/abs/2508.21058) |

## 2. LLM 的可训练稀疏 attention / 块检索（"索引在 GPU、按块取 KV"的前辈）

| Paper | Venue | 做了什么 | 关系 | Link |
|---|---|---|---|---|
| Native Sparse Attention（**NSA**；Yuan, Gao, Dai 等；DeepSeek） | ACL 2025 | 压缩分支 + 选择分支 + 滑窗，硬件对齐，可从预训练起使用 | 结构和 VSA 的 compress + select 几乎一样；有训练 kernel | [ACL 2025](https://aclanthology.org/2025.acl-long.1126) |
| MoBA: Mixture of Block Attention（Lu 等；Moonshot） | NeurIPS 2025 | 把上下文分块，无参数 top-k 门控选块，可在 full / sparse 间切换；已上线 | 与"tile 均值打分 + top-k"同类 | [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/117997) |
| Quest（Tang, …, Song Han） | ICML 2024 | 每个 KV page 记录 key 的 min/max，用 query 估计重要性，只取 top-k page；自注意力 2.23× | 推理时"按块选 KV"的经典做法 | [arXiv 2406.10774](https://arxiv.org/abs/2406.10774) |
| InfLLM（Xiao 等） | NeurIPS 2024 | 远端 KV 分块存 CPU，每块用代表 token 做检索，常用块留 GPU | "完整历史放远端、只取相关块"的最早形态之一 | [arXiv 2402.04617](https://arxiv.org/abs/2402.04617) |
| ShadowKV（Sun 等；ByteDance Seed / CMU） | ICML 2025 | 低秩 key 留 GPU，value 放 CPU，用 landmark 选块再取；batch 6×，吞吐 3.04× | 内存层级 + 稀疏选取的系统设计参考（推理） | [arXiv 2410.21465](https://arxiv.org/abs/2410.21465) |

## 3. 稀疏 attention × 多卡并行（**最相关的系统工作**）

| Paper | Venue | 训练 / 推理 | 发现的问题 | 做法与结果 | Link |
|---|---|---|---|---|---|
| **MTraining**（Li, Zhang, Jiang, …, Lili Qiu；Microsoft） | arXiv 2025-10（2026-05 修订）⚠️ 正式 venue 未核实 | **训练**（LLM） | 动态稀疏 + context parallel 时有 worker 级和 step 级不均衡，实际加速远低于理论值 | balanced sparse ring attention + hierarchical sparse ring attention；Qwen2.5-3B，32K → 512K，32×A100，训练吞吐最高 6× | [arXiv 2510.18830](https://arxiv.org/abs/2510.18830) |
| **SparSP**（Sun, Zhong, Wang, Liu；Waterloo / Rice） | arXiv 2026-09 | 推理（视频 DiT） | "sparse attention eliminates up to 71.3% of computation, yet reduces communication by only 18.9%"（§3.1）；计算少了，可用来掩盖通信的时间也少了 | 按稀疏模式放置序列块 + 按需路由 KV + 通信与计算解耦；端到端平均 1.17×（最高 1.69×）；只在 PCIe 服务器上测（A100 / L40S / RTX 6000 PRO），用 SpargeAttn，模型 LTX-Video、LTX-2、HunyuanVideo | [arXiv 2609.32197](https://arxiv.org/abs/2609.32197) |
| **db-SP**（Chen, …, Yu Wang；清华 NICS） | arXiv 2025-11 | 推理 | 块稀疏下各 head 稀疏度不同、dense 块分布不规则 → Ulysses / Ring 都不均衡；提出 sparse imbalance ratio | head 级 + 块级双层划分，按步、按层动态调整并行度；端到端 1.25×，attention 1.40× | [arXiv 2511.23113](https://arxiv.org/abs/2511.23113) |
| **FVAttn**（Liu 等） | arXiv 2026-07 | 推理 | 自适应 top-p 路由让各 head 工作量不均 → 多卡上出现掉队的 rank | 运行时把重的 head 点对点迁移；不均衡度 1.34 → 1.08；Wan2.2 I2V 推理 2.02–2.11× | [arXiv 2607.16190](https://arxiv.org/abs/2607.16190) |
| **OSP-Next**（Ge, …, Li Yuan） | arXiv 2026-05 | 推理为主 ⚠️ 是否含稀疏训练未细查 | Ulysses 序列并行在稀疏模式下浪费通信 | Sparse Sequence Parallelism：一次 All-to-All 切换稀疏模式，通信比 Ulysses 少 75%；H200 8 卡 1.52× | [arXiv 2605.28691](https://arxiv.org/abs/2605.28691) |

## 4. 长序列训练的并行与负载均衡（非稀疏，但问题相同）

| Paper | Venue | 关系 | Link |
|---|---|---|---|
| **Striped Attention**（Brandon 等） ⚠️ venue 未核实 | arXiv 2023 | causal + Ring Attention 时按顺序切会负载不均；条纹式分配后，256K 长度端到端 1.45×（A100 / TPUv4） | [arXiv 2311.09431](https://arxiv.org/abs/2311.09431) |
| **WLB-LLM**（Wang 等） | OSDI 2025 | pipeline 和 context parallel 两层的负载不均；按文档细粒度切分；平均 1.23× | [USENIX](https://www.usenix.org/conference/osdi25/presentation/wang-zheng) |
| **ByteScale** ⚠️ venue 未核实 | arXiv 2025 | 长短序列混合训练，动态 mesh + 负载均衡调度；最高 7.89× | [arXiv 2502.21231](https://arxiv.org/abs/2502.21231) |
| **xDiT / PipeFusion**（Fang 等） | PipeFusion：NeurIPS 2025；xDiT：arXiv 2024 | DiT 推理的 SP + patch 级流水 + CFG 并行；利用相邻扩散步输入相似、复用上一步的特征 | [PipeFusion](https://arxiv.org/abs/2405.14430) · [xDiT](https://arxiv.org/abs/2411.01738) |

---

## 我们的 trace 和这些工作怎么对上

| 我们测到的（PAN，W3 + remote，`experiments/topk_trace`） | 文献里的对应 |
|---|---|
| 一张卡负责的 chunk 越多，需要的远端历史越接近全部（1 个 chunk 22% → 12 个 67% → 24 个 84%） | SparSP："计算少 71%，通信只少 19%" |
| causal：越靠后的 chunk 计算越多，按顺序切分的不均衡约 1.6×（10 分钟 720p 估算） | Striped Attention；MTraining 的 worker 级不均衡 |
| 各 head 的选择差别大（5% 时 head 间重叠只有约 26–31%） | db-SP / FVAttn 的 head 级不均衡 |
| 相邻去噪步的选择几乎不变（50 步时约 1% 变化） | PipeFusion 复用上一步特征；MTraining 的 step 级 |

## 研究空白（训练侧，候选贡献点）

1. **上面的系统工作都没有做"自回归 / causal 视频 + 可训练 top-k 检索"的训练。**
   - MTraining 是 LLM；SparSP、db-SP、FVAttn、OSP-Next 都是双向视频 DiT 的推理。
   - DF 训练里，历史随位置线性增长，选择是按内容动态决定的，两者叠加的负载与通信模式没有人刻画过。
2. **"内存逼出来的序列并行"没被单独讨论。** 现有工作默认 SP 是为了提速，而长视频 DF 训练是因为激活放不下才不得不切（单张 H200 约 1.5 分钟 448p / 25 秒 720p）。切得越细，每张卡的并集越小，但卡数和通信总量越大。这个权衡和单卡内存容量直接相关，也就是后面接 glass 的地方。
3. **SparSP 只在 PCIe 上测**，NVLink / NVL72 级别的结论还没有；跨节点、机柜级更没有。
4. **按块取 KV 的通信需求是运行时才确定的、带偏斜的**，形式上和 MoE 的 AllToAll 类似（GlassML 已经为 MoE 解决了运行时 AllToAll 的调度），但还没有人把稀疏 attention 的通信当作这种问题来处理。

## 未核实项

- MTraining、Striped Attention、ByteScale 的正式 venue。
- OSP-Next 是否包含稀疏 attention 训练（只读了摘要）。
- PipeFusion 的会议版本与 xDiT 的关系（xDiT 只核实了 arXiv）。
