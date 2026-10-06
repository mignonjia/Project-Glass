# Glass + Memory + Long Video 文献调研（截至 2026-10-04）

> 条目的标题、作者、venue 和链接都经网络检索核实。标 ⚠️ 的条目有部分信息没能核实，原因写在条目里。
> 公开文献里搜不到 "GlassML"（双盲在审）。最接近的公开前身看起来是 UCSD 的 Hsueh / Bill Lin / Fainman 的 panel-scale glass photonic interconnect 工作（1.1）。**这是推断，需要向 David / Tianqi 确认。**

## 先读这些（非硬件背景，按顺序）

1. **AI and Memory Wall**（Gholami 等，IEEE Micro 2024）：建立"带宽和容量才是瓶颈"的直觉。
2. **Challenges and Research Directions for LLM Inference Hardware**（Ma & Patterson，IEEE Computer 2026）：顶级架构师眼中的推理硬件地图。
3. **Quant VideoGen**（ICML 2026）：从你熟悉的领域切入，看自回归视频的 KV 有多大，以及 ML 侧现在怎么"省"。
4. **Mooncake**（FAST 2025）：学会把 KV cache 当作分布式存储系统来设计。
5. **Panel-Scale Reconfigurable Photonic Interconnects for Scalable AI Computation**（Hsueh 等，2025）：项目的硬件底座。
6. 选读：**Stratum**（David 的 MICRO 2025 一作，大容量 3D DRAM + 近存）、**TPU v4**（ISCA 2023，可重构光互连的系统级用法）。

---

## 0. 已发表的 Glass + ML 论文（精选，2026-10-04 补充）

公开发表的"玻璃 + ML"论文非常少，下面基本就是全部直接相关的。**没有一篇涉及视频生成或长上下文内存。**

| 优先级 | Paper | Venue | 为什么读 | Link |
|---|---|---|---|---|
| 1 | GlassML（David 等） | HPCA 2027 在审，保密 | 本项目的硬件本身 | `../David_MoE_glass.pdf` |
| 2 | Panel-Scale Reconfigurable Photonic Interconnects for Scalable AI Computation（Hsueh, Lin, Chen, Fainman；UCSD） | IEEE OJ-SSCS 2025 | 玻璃面板 + 光互连 + AI，最接近 GlassML 的公开前身 | [arXiv 2508.06079](https://arxiv.org/abs/2508.06079) |
| 3 | Designing High-Performance and Thermally Feasible Multi-Chiplet Architectures enabled by Non-bendable Glass Interposer（Sharma, Doppa, Ogras, Pande） | ACM TECS 2025 | 唯一一篇在玻璃 interposer 上跑 DNN / LLM 负载（≤1.5B）的架构论文 | [arXiv 2507.18040](https://arxiv.org/abs/2507.18040) |
| 4 | Glass Interposer Integration of Logic and Memory Chiplets（Vanna-Iampikul, Lim 等） | **DAC 2023**；TCAD 2025 | 不是 ML 专用，但证明 **DAC 收玻璃 + 内存论文**，可参考其 6 页结构 | [pdf](https://gtcad.gatech.edu/www/papers/dac23-pruek.pdf) |
| 5 | ChipLight: Cross-Layer Optimization of Chiplet Design with Optical Interconnects for LLM Training | arXiv 2026 | 光互连 chiplet 做 LLM 训练（非玻璃），与 GlassML 同类问题 | [arXiv 2604.18909](https://arxiv.org/abs/2604.18909) |
| 6 | Photonic Fabric Platform for AI Accelerators（Celestial AI / Marvell） | arXiv 2025 | 光互连 + 大容量共享 HBM，"大内存池"思路的产业版 | [arXiv 2507.14000](https://arxiv.org/abs/2507.14000) |

背景综述：[A Review of Glass Substrate Technologies（MDPI 2025）](https://www.mdpi.com/2674-0729/4/3/37)；产业进展：[TrendForce 2026-06](https://www.trendforce.com/news/2026/06/05/news-glass-substrates-eye-2027-launch-scale-toward-2030-as-cowos-costs-rise-and-hyperscaler-demand-grows/)（预计 2027 年开始早期商用，2030 年起大规模量产）。

## 1. Glass interposer / glass substrate / 玻璃中的光子学

**为什么重要：** 今天的 GPU 把计算芯片和 HBM 贴在一块硅中介层上。硅中介层的面积受光刻 reticle 限制，HBM 只能沿芯片边缘摆一圈，显存容量就被卡住了。玻璃可以做成 500 mm 级的大面板，电学损耗低，还能直接在玻璃里刻光波导。这就是 GlassML "两面铺 DRAM、加上光互连"的物理基础。

| Paper | Venue/Year | 与项目的关系 | Link |
|---|---|---|---|
| Panel-Scale Reconfigurable Photonic Interconnects for Scalable AI Computation（T.-C. Hsueh, Bill Lin, Z. Chen, Y. Fainman；UCSD） | IEEE OJ-SSCS, vol. 5, 2025 | 在 ≥500×500 mm 玻璃上做可重构光交换网络，面板一边到另一边不用中继器；集成 chiplet + HBM，用 WDM 和片上光频梳。**与 GlassML 最接近，必读** | [arXiv 2508.06079](https://arxiv.org/abs/2508.06079) |
| Glass Interposer Integration of Logic and Memory Chiplets: PPA and Power/Signal Integrity Benefits（Vanna-Iampikul, S. K. Lim 等；Georgia Tech） | DAC 2023；扩展版 TCAD 2025 | 少见的架构 / EDA 视角玻璃论文。存储 die 可以嵌进玻璃里再叠芯片；面积、线长、PI/SI 有优势，约 15% 温升 | [pdf](https://gtcad.gatech.edu/www/papers/dac23-pruek.pdf) |
| Demonstration of Glass-based 3D Package Architectures with Embedded Dies for HPC（Ravichandran, Smet, Swaminathan, Tummala；Georgia Tech PRC） | ECTC 2022 | Glass Panel Embedding 实物：4 个嵌入 die + 8 个 HBM emulator + 2 个 SoC，不需要 TSV | doi 10.1109/ECTC51906.2022.00180 |
| Through-Package-Via Formation and Metallization of Glass Interposers（Sukumaran, Tummala 等） | ECTC 2010 | TGV（玻璃通孔）的经典早期工作 | — |
| Designing High-Performance and Thermally Feasible Multi-Chiplet Architectures enabled by Non-bendable Glass Interposer（Sharma, Doppa, Ogras, Pande） | ACM TECS 2025 | DNN 负载下联合优化架构和封装，同时处理散热和翘曲。⚠️ 两个来源给的加速比不一致 | [arXiv 2507.18040](https://arxiv.org/abs/2507.18040) |
| FoldedHexaTorus（Iff, Besta, Hoefler；ETH） | arXiv 2025 ⚠️ 正式 venue 未核实 | 针对有机 / 玻璃基板的 chiplet 网络拓扑 | [arXiv 2504.19878](https://arxiv.org/abs/2504.19878) |
| Glass Substrate for Co-Packaged Optics（Brusberg 等；Corning） | IMAPS 2022 | 玻璃里做离子交换波导，损耗 <0.1 dB/cm：玻璃内嵌光波导在工艺上可行 | [link](https://imapsource.org/article/74749-glass-substrate-for-co-packaged-optics) |
| Scaling Routers with In-Package Optics and High-Bandwidth Memories（Keslassy, Lin 等）；前作 Petabit Router-in-a-Package（HotNets 2025） | arXiv 2026 | 同一 UCSD 团队把面板级集成 + HBM + 封装内光学用在路由器上 | [arXiv 2602.10505](https://arxiv.org/abs/2602.10505) |
| Co-packaged optics for HPC and AI（SK hynix + UVA/MIT/UIUC 等） | Nature Electronics 2026-08 | 存储厂商的 CPO 路线图，长期愿景是把光链路直接接到存储接口 | [link](https://www.nature.com/articles/s41928-026-01681-6) |
| Intel 玻璃基板发布（产业新闻） | 2023-09 | Intel 计划约 2030 年出货。⚠️ TSMC CoPoS 约 2029 年量产的说法来自二手来源 | [link](https://www.intel.com/content/www/us/en/newsroom/news/intel-unveils-industry-leading-glass-substrates.html) |

## 2. 大规模 scale-up 系统：wafer-scale、机柜级、光互连（baseline 与邻居）

**为什么重要：** 评审一定会问 GlassML 和 Cerebras（整片晶圆）、NVL72（72 张 GPU 一个 NVLink 域）、TPU（用光开关重构拓扑）比谁强。这些系统在内存容量、带宽、可重构性之间各有取舍，是画对比表时的坐标系。

| Paper | Venue/Year | 与项目的关系 | Link |
|---|---|---|---|
| TPU v4: An Optically Reconfigurable Supercomputer for ML（Jouppi 等；Google） | ISCA 2023 | OCS 动态重构 4096 芯片的拓扑，成本 <5%、功耗 <3%，是可重构光互连的标杆 | [arXiv 2304.01433](https://arxiv.org/abs/2304.01433) |
| Mission Apollo: Landing Optical Circuit Switching at Datacenter Scale（Urata, Vahdat 等） | arXiv 2022 | MEMS OCS 的原理与取舍 | [arXiv 2208.10041](https://arxiv.org/abs/2208.10041) |
| Cerebras Architecture Deep Dive（Sean Lie） | IEEE Micro 2023 | 晶圆级 SRAM 带宽极高，但总容量只有几十 GB，正好和 GlassML 的"海量 DRAM"形成对照 | [link](https://cerebras.ai/blog/cerebras-architecture-deep-dive-first-look-inside-the-hw-sw-co-design-for-deep-learning) |
| WaferLLM: LLM Inference at Wafer Scale（He 等） | OSDI 2025 | 为新硬件建性能模型（PLMR）的方法参考 | [arXiv 2502.04563](https://arxiv.org/abs/2502.04563) |
| TEMP: Memory Efficient Tensor Partition-Mapping for LLM Training on Wafer-Scale Chips | HPCA 2026 | wafer-scale 上内存和算力抢面积 | [arXiv 2512.14256](https://arxiv.org/abs/2512.14256) |
| Theseus: Exploring Efficient Wafer-Scale Chip Design for LLMs | arXiv 2024 ⚠️ venue 未核实 | 设计空间探索（DSE）框架参考 | [arXiv 2407.02079](https://arxiv.org/abs/2407.02079) |
| The Microarchitecture of DOJO（Talpes 等；Tesla） | IEEE Micro 2023 | 为了扩展放弃一致性，依赖高带宽 mesh | [link](https://ieeexplore.ieee.org/document/10078146) |
| Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware | ISCA 2025 | 从模型团队角度讲内存与互连的痛点 | [arXiv 2505.09343](https://arxiv.org/abs/2505.09343) |
| NVIDIA GB200 NVL72（产品资料） | 2024–25 | 130 TB/s NVLink、13.4 TB HBM，**是 85 TB / 8 PB/s 最直接的对比数字** | [link](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| Photonic Fabric Platform for AI Accelerators（Ding, Diep；Celestial AI / Marvell） | arXiv 2025 | 光交换 + HBM3E 共享内存，最多 32 TB，405B 推理吞吐最高 3.66×。思路与 GlassML 非常接近 | [arXiv 2507.14000](https://arxiv.org/abs/2507.14000) |
| TeraPHY（Wade 等；Ayar Labs） | IEEE Micro 2020 | 封装内光 I/O chiplet | IEEE Micro 40(2) |
| Lightmatter Passage（Hot Chips 2022 / 2025） | 产业报告 ⚠️ 非同行评审 | 有源硅光中介层，可编程重构光路 | — |

## 3. AI 的内存容量、解耦与 KV cache 分层

**为什么重要：** 解码阶段的瓶颈是"把权重和 KV 从内存搬到计算单元"，而不是算力。KV 随上下文线性增长。业界的应对是分层（HBM → DRAM → CXL → SSD）、压缩或重算。GlassML 相当于提供了一个超大的近端内存层，这一类论文就是它的软件栈和对比对象。

| Paper | Venue/Year | 与项目的关系 | Link |
|---|---|---|---|
| AI and Memory Wall（Gholami, Keutzer 等） | IEEE Micro 2024 | 算力每 2 年涨 3.0×，DRAM 带宽涨 1.6×，互连涨 1.4×。**动机图** | [arXiv 2403.14123](https://arxiv.org/abs/2403.14123) |
| Challenges and Research Directions for LLM Inference Hardware（Ma, Patterson） | IEEE Computer 2026 | 推理的核心是内存与互连 | [arXiv 2601.05047](https://arxiv.org/abs/2601.05047) |
| PagedAttention / vLLM（Kwon 等） | SOSP 2023 | KV 管理的起点 | [arXiv 2309.06180](https://arxiv.org/abs/2309.06180) |
| vAttention（Prabhu 等） | ASPLOS 2025 | 借助 OS 虚拟内存管理 KV | [arXiv 2405.04437](https://arxiv.org/abs/2405.04437) |
| InfiniGen（Lee, Sim 等） | OSDI 2024 | KV 卸载后只预取重要 token，对应"容量够、带宽不够"的情形 | [arXiv 2406.19707](https://arxiv.org/abs/2406.19707) |
| Mooncake（Qin 等；Moonshot / 清华） | FAST 2025 Best Paper | 全局 KV 池，"用存储换计算"，对应 GlassML 的 85 TB 池 | [link](https://www.usenix.org/conference/fast25/presentation/qin) |
| LMCache | arXiv 2025；MLSys 2026 | 开源 KV 缓存层 | [arXiv 2510.09665](https://arxiv.org/abs/2510.09665) |
| Pond: CXL-Based Memory Pooling（Li 等；Microsoft / CMU） | ASPLOS 2023 | CXL 内存池入门 | [arXiv 2203.00241](https://arxiv.org/abs/2203.00241) |
| Beluga: CXL-Based Memory Architecture for LLM KVCache | SIGMOD 2026 ⚠️ 机构未核实 | GPU 通过 CXL 交换机直接访问 KV 池，TTFT 降低 89.6% | [arXiv 2511.20172](https://arxiv.org/abs/2511.20172) |
| A Photonic-CXL Memory Appliance for KV Cache Management（Ding, Diep 等；Marvell） | arXiv 2026-07 | 光 + CXL KV 池，32 TB，TTFT 提升 6.6×。**与 GlassML 最接近的光学内存解耦工作** | [arXiv 2607.27187](https://arxiv.org/abs/2607.27187) |
| Stratum（Yue (David) Pan 等） | MICRO 2025 | 单片 3D DRAM + 近存，MoE serving。**组内工作** | [arXiv 2510.05245](https://arxiv.org/abs/2510.05245) |
| AttAcc!（SNU） | ASPLOS 2024 | 把 attention（KV 读取）放进 PIM | doi 10.1145/3620665.3640422 |

## 4. 长视频 / 因果视频生成与 world model 的内存

**为什么重要：** 自回归视频每生成一个 chunk 都要 attend 历史 KV。每帧有上千个 token，KV 增长比 LLM 快一到两个数量级。现有工作几乎都在"塞进一张 GPU"的前提下做取舍：窗口、sink、压缩、量化、检索。这些取舍正是 glass 可能消除的。

### 4a. 因果 / 流式视频扩散

| Paper | Venue/Year | 内存上做了什么 | Link |
|---|---|---|---|
| CausVid（Yin, …, X. Huang；MIT / Adobe） | CVPR 2025 | 双向 DiT 蒸馏成因果 4 步模型，用 KV cache 流式生成，历史 KV 持续增长 | [arXiv 2412.07772](https://arxiv.org/abs/2412.07772) |
| Self Forcing（Huang, Li, He, Zhou, Shechtman） | NeurIPS 2025 | **训练时就带 KV cache 做 rollout**（训练期内存负担）；推理用 rolling KV（固定窗口） | [arXiv 2506.08009](https://arxiv.org/abs/2506.08009) |
| Self-Forcing++（Cui, …, Hsieh；UCLA / ByteDance） | ICLR 2026 | 扩展到 100 秒到 4 分 15 秒，仍依赖 rolling window | [arXiv 2510.02283](https://arxiv.org/abs/2510.02283) |
| Rolling Forcing | ICLR 2026 ⚠️ 作者全名未核实 | 滚动窗口联合去噪 + attention sink（永久保留首帧 KV） | [arXiv 2509.25161](https://arxiv.org/abs/2509.25161) |
| LongLive（NVIDIA / MIT 等） | ICLR 2026 | 短窗口 + frame sink；切换 prompt 时 KV-recache；单张 H100 生成 240 秒 | [arXiv 2509.22622](https://arxiv.org/abs/2509.22622) |
| FramePack（Lvmin Zhang, Agrawala） | arXiv 2025 ⚠️ venue 未核实 | 按重要性压缩历史帧，上下文长度恒定（有损） | [arXiv 2504.12626](https://arxiv.org/abs/2504.12626) |
| StreamDiffusionV2（Feng, …, Han, Keutzer, Xu） | MLSys 2026 | sink-token 引导的 rolling KV + SLO 调度 + 多 GPU 流水线 | [arXiv 2511.07399](https://arxiv.org/abs/2511.07399) |
| PackForcing（Mao, …, Zhang） | arXiv 2026-03 | 三段式 KV（sink 全分辨率 / 中段压缩 32× 加 top-k / 近期全分辨率），2 分钟 480p 视频的 KV 限制在 4 GB | [arXiv 2603.25730](https://arxiv.org/abs/2603.25730) |
| Quant VideoGen（Xi, Han, Keutzer 等） | ICML 2026 | 指出 KV 常常 ≥30 GB；2-bit 量化压缩最多 7× | [arXiv 2602.02958](https://arxiv.org/abs/2602.02958) |
| FAST-AR: Temporal Cache Compression and Sparse Attention（Samuel, Chechik 等） | ICML 2026 | 跨帧去重 key + ANN 稀疏 attention，峰值显存近乎恒定 | [arXiv 2602.01801](https://arxiv.org/abs/2602.01801) |

### 4b. World model 的记忆

| Paper | Venue/Year | 内存上做了什么 | Link |
|---|---|---|---|
| Genie（Bruce 等；DeepMind） | ICML 2024 | 可交互 world model 起点，上下文很短 | [arXiv 2402.15391](https://arxiv.org/abs/2402.15391) |
| Genie 3（DeepMind 博客） | 2025-08 | 720p、24 fps，视觉记忆约 1 分钟：工业界 world model 记忆目前是分钟级 | [link](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models) |
| WorldMem（Xiao, Pan 等） | NeurIPS 2025 | 显式 memory bank（帧 + 位姿 + 时间戳），用 memory attention 检索 | [arXiv 2504.12369](https://arxiv.org/abs/2504.12369) |
| Context as Memory（Yu 等；HKU / 快手） | arXiv 2025 | 按相机 FOV 重叠检索历史帧："大存储 + 按需检索" | [arXiv 2506.03141](https://arxiv.org/abs/2506.03141) |
| Long-Context State-Space Video World Models（Po, …, Dao, Wetzstein, Huang） | ICCV 2025 | 用 SSM 的常数大小状态替代增长的 KV，是"架构上避开内存"的路线 | [arXiv 2505.20171](https://arxiv.org/abs/2505.20171) |
| Addressable Memory for Video World Models / WorldTrace（Wu, …, Ošep；NVIDIA 等） | arXiv 2026-08 | 超出训练长度后 KV 检索失效（位置编码越界），用虚拟位置解决；提出 LoopBench | [arXiv 2608.07408](https://arxiv.org/abs/2608.07408) |

背景：StreamingLLM / attention sink（ICLR 2024，[arXiv 2309.17453](https://arxiv.org/abs/2309.17453)）；Diffusion Forcing（NeurIPS 2024，[arXiv 2407.01392](https://arxiv.org/abs/2407.01392)）。

---

## 研究空白（机会点）

1. **没有人从硬件视角刻画自回归视频生成 / world model 的内存需求。** KV 系统工作（Mooncake、LMCache、InfiniGen、Beluga、Photonic-CXL）几乎都只做 LLM 文本。视频工作都默认"必须塞进一张 GPU"。这个负载画像可以直接作为第一篇工作。
2. **"容量免费时，完整历史比压缩好多少？"没有人系统测过。** 在内存不受限的条件下，比较 full KV、sink+window、压缩、SSM 的长期一致性（例如用 LoopBench）。这决定了 glass 对视频的价值上限，对应 README 的 RQ2。
3. **检索式记忆（WorldMem、Context-as-Memory）的访存模式与"大池子 + 可重构光网"天然匹配**：稀疏、随机、远端的访问需要高对分带宽。现有玻璃 / 光论文只评估通用 DNN / LLM 负载。
4. **训练期的 rollout state 没人管。** Self-Forcing 类训练要保存 rollout 的 KV + 激活，视频 RL 后训练会更大。现在只能靠梯度截断。
5. **封装尺度的拓扑重构**：可以按去噪步、prefill/decode 阶段或会话来重构光路。Hsueh 2025 只提出了网络本身，缺少与负载协同的调度。
6. **多租户交互式 world model 服务**（"视频版 Mooncake"）：持久世界状态的迁移、换入换出、跨用户共享，目前完全空白。

## 术语表

- **Interposer（中介层）**：芯片和封装基板之间的高密度布线板，GPU 和 HBM 并排贴在上面。传统是硅，GlassML 用玻璃。
- **Substrate / glass core substrate**：中介层下面的封装基板，玻璃芯更平、更大、更稳定。
- **Panel-level packaging**：在方形大面板（如 510×515 mm）上封装，而不是在 300 mm 圆晶圆上做。
- **Reticle limit**：单次光刻曝光最大约 26×33 mm（约 858 mm²），限制了单颗芯片和硅中介层的大小。
- **Chiplet**：把大芯片拆成小芯片，在封装里再互连。
- **HBM**：竖着堆叠的 DRAM，通过 TSV 连接，贴在 GPU 旁边。带宽高（单栈约 1–2 TB/s），容量小（单栈几十 GB），只能沿芯片边缘（beachfront / shoreline）摆放。
- **TSV / TGV**：硅通孔 / 玻璃通孔，垂直导电孔。"两面放 DRAM"依赖 TGV。
- **CXL**：基于 PCIe 的缓存一致性内存协议，可用 load/store 访问远端内存，延迟约为本地 DRAM 的 2–3 倍。
- **Memory disaggregation（内存解耦）**：内存从计算节点拆出来组成共享池。
- **OCS（光路交换）**：直接切换光路，不做光电转换。可重构、低功耗，切换较慢（MEMS 为毫秒级）。
- **CPO / in-package optics**：光收发器放进封装里。
- **WDM（波分复用）**：一根波导里跑多个波长，每个波长是一条独立通道。
- **Bisection bandwidth（对分带宽）**：系统切成两半，两半之间的总带宽，决定 all-to-all 的快慢。
- **Injection bandwidth（注入带宽）**：单个节点能向网络发送的最大带宽。
- **Scale-up vs scale-out**：scale-up 是紧耦合的高带宽域，表现得像一台大机器（NVLink 域、封装）；scale-out 是用以太网 / InfiniBand 连接多个域。GlassML 是在把 scale-up 域做到极大。
- **SRAM vs DRAM**：SRAM 在芯片上，快但容量小（Cerebras）；DRAM 密度高、便宜，但更慢。
- **pJ/bit**：每搬运 1 bit 的能耗。
- **Arithmetic intensity / roofline**：每字节数据对应多少次计算。强度低（如解码阶段读 KV）的负载受带宽限制，强度高的受算力限制。

## 未核实项汇总

- Theseus、FramePack、FoldedHexaTorus 的正式 venue。
- Beluga 的机构。
- Rolling Forcing 的作者全名。
- Lightmatter Passage、NVL72 只是产业资料，不是同行评审论文。
- TSMC CoPoS 的时间来自二手来源。
- Hsueh 2025 与 GlassML 的关系是推断。
