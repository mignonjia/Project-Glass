# SEELab 投稿会议与时间线（调研于 2026-10-04）

> 来源：Semantic Scholar API（约 163 篇 Rosing 2023–2026 论文）、SEELab 主页、rankdblp、arXiv、各会议 CFP。
> 标注：**[CFP]** 官方已确认；**[3rd]** 仅来自第三方追踪站；**[估]** 按往年规律估计。录用率为近年大致区间，未逐年核实。
> 日期会变，投稿前务必回到官方 CFP 再核对。

## 1. 组里投哪些会

| 类别 | Venue（2023–26 约篇数） |
|---|---|
| EDA / 设计自动化（数量最多） | **DATE (~14)**、TCAD (~8)、**ISLPED (~6)**、**DAC (~5)**、**ASP-DAC (~5)**、**ICCAD (~4)**、GLSVLSI (~4)、ICCD (~3)、ISQED |
| 体系结构顶会 | **ISCA (3)**：HPVM-HDC'24、OptiPIM'25、Hybrid SLC-MLC RRAM PIM'25；**MICRO (2)**：UFC'24、Stratum'25；另有 MEMSYS、TACO、IEEE Micro、Hot Chips'25 |
| 电路 | JSSC、VLSI Symp.'25、CICC、IEDM、ESSERC |
| ML / 其他 | AAAI、IJCAI、WACV、ICASSP、SenSys、IPSN、IMWUT、生物信息期刊 |

- 2023–26 年没检索到 HPCA、ASPLOS、MLSys 论文，所以 GlassML 投 HPCA 2027 对组里算新方向。
- 实验室主页列了 NeurIPS / CVPR，但近年没检索到对应论文。

## 2. Tianqi 与 David

依据 [SEELab People 页](https://seelab.ucsd.edu/people.html)，名单上叫 Tianqi 和 David 的各只有一位。**他们与 GlassML 的对应关系是按名单推断的，未经公开证实**（GlassML 在双盲评审中，搜不到是正常的）。

**Yue (David) Pan**：CSE PhD，2023 秋入学，方向为体系结构与系统、领域专用加速
- **Stratum**：Mono3D DRAM + 近存处理，用于 MoE serving。一作，**MICRO 2025**，[arXiv](https://arxiv.org/abs/2510.05245)。与本项目的"大容量 DRAM"最相关，建议先读。
- PRIMATE：存内计算加速动态 token 剪枝 Transformer。一作，ASP-DAC 2024
- OptiPIM：合作者，ISCA 2025
- CHIME：chiplet 异构近存，用于边缘多模态 LLM。合作者，DATE 2026，[arXiv](https://arxiv.org/abs/2601.19908)

**Tianqi Zhang**：CSE PhD，2023 秋入学，方向为体系结构、内存系统、近存计算
- **FaTRQ**：远端内存感知 ANNS，CXL 加速器。一作，**DATE 2026**，[arXiv](https://arxiv.org/abs/2601.09985)
- SpANNS：稀疏向量 ANNS + 近存处理。一作，ASP-DAC 2026
- SpectraFlux：多 FPGA。一作，DAC 2024
- 另有 HyperSpikeASIC（TCAD 2023）、GenoMiX（BioCAS 2023）、SIMCH（ISCAS 2026），以及多篇合作论文

**可能相关，不是本组**：UCSD 的 Hsueh、Bill Lin、Fainman 发过一篇玻璃基板 panel 级可重构光互连的论文，[arXiv 2508.06079](https://arxiv.org/abs/2508.06079)。它与 GlassML 光互连的关系**未验证**。

## 3. 会议一览（2026–27 周期）

| Venue | 定位 | 截稿（摘要 / 全文） | 通知 | 会期 / 地点 | 评审流程 | 录用率≈ |
|---|---|---|---|---|---|---|
| **HPCA 2027** | 体系结构顶会 | 2026-07-24 / 07-31 [CFP] | **2026-11-06** [CFP] | 2027-03-20~24，Salt Lake City | revision/rebuttal 窗口 9/28 ~ 10/9（researchr 写 10/11）；正文 11 页 | 20–25% |
| **ISCA 2027** | 体系结构顶会 | 约 2026-11 中旬 [估]；2027 CFP 尚未发布 | 约 2027-03 底 [估] | 约 2027-06，Atlanta（FCRC）[3rd] | 两轮评审 + rebuttal | 17–22% |
| **MICRO 2027** | 体系结构顶会 | 约 2027-03 底 / 04 初 [估] | 约 2027-07 [估] | 约 2027-10 | rebuttal / revision | 20–23% |
| **ASPLOS 2027** | 体系结构 / OS / PL | 4 月、9 月两个周期均已过 [CFP] | 9 月周期 2026-12-21 | 2027-04-11~15，Crete | 先快速初审再全审；author response；可给 Major Revision | 17–22% |
| **ASPLOS 2028** | 同上 | 约 2027-04、2027-09 [估] | — | — | 同上 | — |
| **DAC 2027** | EDA 顶会 | 摘要约 11/11–12，全文约 11/18–19 [3rd] | 约 2027-02/03 [估] | 2027-07-10~16，San Jose | rebuttal 未核实 | 22–25% |
| **ICCAD 2027** | EDA 顶会 | 约 2027-04 中旬 [估] | 约 2027-07 [估] | 待定 | 有 rebuttal | 22–25% |
| **DATE 2027** | EDA（欧洲） | 正文已过；**Late Breaking Results 截稿 2026-11-29** [CFP] | LBR 2027-01-13 | 2027-03-22~24，Dresden | 双盲；LBR 2+1 页 | 24–26% |
| **ISLPED 2027** | 低功耗 | 约 2027-03 初 [估] | 约 2027-05 中 [估] | 约 2027-08 | — | 20–25% |
| **MLSys 2027** | ML 系统 | **2026-10-30** [CFP] | 2027-02-28 | 2027-06-21~25，Bellevue [3rd] | OpenReview；有作者回复 | 20–25% |
| **SC27** | HPC | 约 2027-04 初 [估] | 约 2027-06/07 [估] | 2027-11-14~19，Denver [3rd] | 有 rebuttal | 20–24% |
| **ICLR 2027** | ML 顶会 | 已过 [CFP] | 2026-12-16 | 2027-04-26~30 [3rd] | OpenReview 公开讨论 | ~30% |
| **CVPR 2027** | CV 顶会（视频生成） | 注册 2026-11-10，全文 11-16 [CFP] | 2027-02-25 | 2027-06 [3rd] | 1 周 rebuttal | 22–24% |
| **ICML 2027** | ML 顶会 | 约 2027-01-16 / 01-22 [估] | 约 2027-05 | 约 2027-07 | OpenReview rebuttal | ~27% |
| **NeurIPS 2027** | ML 顶会 | 约 2027-05-14 / 05-21 [估] | 约 2027-09 | 约 2027-12 | OpenReview rebuttal | ~25% |
| **HPCA 2028** | 体系结构顶会 | 约 2027-07 底 / 08 初 [估] | 约 2027-11 初 | 约 2028-02/03 | 同 HPCA 2027 | 20–25% |

审稿周期大致是：体系结构和 EDA 会议从截稿到出结果约 **3–4 个月**；ML 会议约 **4 个月**。

## 4. 时间线（★ = 对本项目现实的目标；⚠ = 大概率来不及）

**2026 年**

| 日期 | 事项 |
|---|---|
| 现在 | HPCA 2027 rebuttal 窗口进行中（GlassML 论文） |
| 10/30 | MLSys 2027 ⚠ |
| **11/06** | **HPCA 2027 出结果（GlassML）** |
| 11/10、11/16 | CVPR 2027 ⚠ |
| 约 11 月中 | ISCA 2027、DAC 2027 ⚠ |
| 11 月 | **导师 review** |
| **11/29** | ★ DATE 2027 Late Breaking Results（2 页），适合发早期结果 |

**2027 年**

| 日期 | 事项 |
|---|---|
| 约 1/22 | ICML 2027 [估]，算法侧推进快才考虑 |
| 约 3 月初 | ★ ISLPED 2027 [估] |
| 约 3 月底 / 4 月初 | ★★ **MICRO 2027** [估]：软硬协同 + 玻璃 / 光互连的主攻目标 |
| 约 4 月中 | ★ ICCAD 2027、ASPLOS 2028 第一个周期 [估] |
| 约 5 月中 | ★ NeurIPS 2027 [估]：算法和视频生成侧的目标 |
| 约 7 月底 | ★★ **HPCA 2028** [估]：最稳妥的完整系统论文目标 |
| 约 9 月 | ASPLOS 2028 第二个周期、DATE 2028 [估] |

## 5. 建议

1. 2026 年秋季的 MLSys、CVPR、ISCA、DAC 都来不及，跳过。
2. 早期反馈：DATE LBR（11/29），或 2027 春季的 ISCA / MICRO workshop。
3. 第一篇完整论文：**MICRO 2027（约 4 月）**；备选 **HPCA 2028（约 7 月底）** 或 ASPLOS 2028。
4. 如果长视频记忆的算法部分能独立成文，可另投 NeurIPS 2027。
5. 组里传统强项 DATE、DAC、ICCAD、ASP-DAC 可作为稳妥备选。

## 来源

- 实验室：[SEELab People](https://seelab.ucsd.edu/people.html)
- 体系结构：[HPCA 2027 (sigarch)](https://www.sigarch.org/call-contributions/hpca-2027/) · [HPCA 2027 researchr](https://conf.researchr.org/track/hpca-2027/hpca-2027-main-conference) · [ISCA 2026 CFP](https://www.sigarch.org/call-contributions/isca-2026/) · [MICRO 2026](https://microarch.org/micro59/) · [ASPLOS 2027](https://www.asplos-conference.org/?p=8512)
- EDA：[DAC 2027](https://dac.com/2027/authors/call-for-contributions) · [DATE 2027](https://date27.date-conference.com/call-for-papers) · [ICCAD 2026](https://easychair.org/cfp/ICCAD26) · [ISLPED 2026](https://www.islped.org/2026/files/ISLPED2026CfP_V8.pdf)
- ML / 系统：[MLSys 2027](https://mlsys.org/Conferences/2027/CallForResearchPapers) · [ICLR 2027](https://www.iclr.cc/Conferences/2027/CallForPapers) · [CVPR 2027](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)
- 追踪站：[KAIST deadlines](https://seed.kaist.ac.kr/deadlines/) · [opencurious](https://opencurious.com/ai-conference-deadlines)
