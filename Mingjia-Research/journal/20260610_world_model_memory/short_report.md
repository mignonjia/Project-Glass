# Short Report

The strongest claim is not that 73 TB/s beats NVL72 all-to-all. It does not: current NVIDIA GB200/GB300 NVL72 publicly lists 130 TB/s NVLink bandwidth across the rack-scale 72-GPU domain.

The stronger claim is capacity-driven:

- NVL72 is a very fast GPU cluster, but its memory is still attached to GPUs and consumed by weight shards, activation/KV state, and duplicated execution groups.
- A dual/triple memory-panel glass system can make the useful working set much larger near compute: long video history, KV/history memory, activations, teacher/student rollout state, multiple rollout branches, and cold/warm memory tiers.
- For world models, that changes the feasible operating point: longer horizon, higher resolution, more concurrent rollout branches, and less sequence/context parallelism on the critical path.
- 73 TB/s aggregate is still meaningful if the architecture uses memory capacity to avoid many all-to-all operations instead of trying to win raw all-to-all bandwidth.

Best one-line thesis:

> Glass does not beat NVL72 by making all-to-all faster; it wins when extra memory makes all-to-all unnecessary for long-horizon world-model state.

Created `argument_memo.md` with public NVIDIA survey numbers, workload implications, and slide-safe wording.

