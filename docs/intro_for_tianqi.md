# 发给 Tianqi 的消息

Hi Tianqi，Tajana 让我做 GlassML + world model，想找你聊聊你的 glass 模型。

我的想法：causal 视频生成每生成一段都要 attend 全部历史的 KV，KV 随视频长度线性增长，长视频能到几十 TB，GPU 放不下，所以现在的模型只能留最近几秒，回到之前的场景就忘了。我打算用 sparse attention（VSA）：GPU 上放一个很小的 block 索引来选 top-k，完整 KV 放在 glass DRAM 上，每步只取选中的 block。然后用真实模型的 trace 在你的模型上模拟，和 NVL72 对比。

想问你几个问题：DRAM 面板是在 GPU 正下方直连，还是走光互连？8 PB/s 具体指什么？你的模型我能不能直接接 trace 进去？

这周或下周有空聊半小时吗？
