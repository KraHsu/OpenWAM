# 四卡开发回放自动排队

`scripts/robotwin_dev_queue.py` 从预先审阅、冻结的任务清单自动填补空闲 GPU。
它只启动 `search_robotwin_constraint_counterexamples.py --split dev`，不生成新
候选任务、不执行最终留出评估或训练。没有可运行任务时，状态明确报告等待或
`queue_exhausted`，不会制造 GPU 负载。

已有模拟进程可以加入观察清单。排队器记录 PID、Linux boot ID、进程启动 tick、
命令 hash 和原 launch receipt hash；PID 被复用后不会把新进程当作原任务。
已有进程只观察，退出后根据原输出报告完成或阻断，绝不重启、终止或接管清理。

## 集成已有任务

对仍然活着的原生 PID 执行以下只读命令，输出一个可放入 `jobs` 的观察条目：

```bash
python scripts/robotwin_dev_queue.py capture-observation \
  --job-id lift_v2_seed10010 --output /runs/lift_v2/seed10010 \
  --gpu 0 --pid 12345 --receipt /runs/lift_v2/seed10010_launch.json
```

receipt 默认从 `pid` 字段取进程 ID；其他结构可用 `--pid-field native_pid` 或
点分字段路径指定。命令会检查真实进程的参数确实包含该 output 路径。旧 receipt
没有 start tick 也可以读取，因为新观察条目会捕获当前活进程身份。已经死亡的
PID 不能使用这个命令补造观察身份。

旧 standalone/owner 不应与新队列同时拥有待派发任务。先检查其状态为完成、
`pending=[]`；当前仍在运行的 standalone 原生进程可正常加入观察清单。
此前的一次性 owner 清单不能原地追加或复用为新队列。

## 冻结清单

```json
{
  "version": 1,
  "queue_root": "/runs/dev_queue_v1",
  "lease_root": "/runs/shared_gpu_leases",
  "gpus": [0, 1, 2, 3],
  "poll_seconds": 5,
  "jobs": [
    {
      "job_id": "next_lift_seed10010",
      "mode": "queued",
      "output": "/runs/next_lift/seed10010",
      "cwd": "/snapshots/rax",
      "command": [
        "/envs/RoboTwin/bin/python",
        "/snapshots/rax/scripts/search_robotwin_constraint_counterexamples.py",
        "--reference", "/inputs/reference.json",
        "--reference-manifest", "/inputs/experiment_manifest.json",
        "--template", "/inputs/template.json",
        "--split-plan", "/inputs/split_plan.json",
        "--training-root", "/datasets/robotwin_clean50_v1",
        "--output", "/runs/next_lift/seed10010",
        "--split", "dev",
        "--candidate-budget", "24",
        "--baseline-repeats", "2",
        "--include-outside-controls"
      ],
      "environment": {
        "ROBOTWIN_PATH": "/native/RoboTwin",
        "PYTHONPATH": "/snapshots/rax",
        "VK_ICD_FILENAMES": "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
        "VK_DRIVER_FILES": "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
      },
      "frozen_files": {
        "/snapshots/rax/scripts/search_robotwin_constraint_counterexamples.py": "<真实SHA256>",
        "/inputs/reference.json": "<真实SHA256>",
        "/inputs/experiment_manifest.json": "<真实SHA256>",
        "/inputs/template.json": "<真实SHA256>",
        "/inputs/split_plan.json": "<真实SHA256>"
      }
    }
  ]
}
```

这是结构示例；路径、hash 必须换成已存在的真实文件。应把 RAX 相关源文件一并
加入 `frozen_files`。使用模型输入或开发反馈时，其文件也必须在命令与 hash 表中
出现。每个任务可选 `eligible_gpus` 限制指定卡，默认使用整个选定池。
`queue_root`、`lease_root` 和各任务 output 必须互不包含、不得重叠。

```bash
python scripts/robotwin_dev_queue.py run --plan /inputs/queue.json --check-only
python scripts/robotwin_dev_queue.py run --plan /inputs/queue.json
```

`--check-only` 只校验，不建立队列、不启动 GPU。正式运行封存完整清单及排队器、
guardian 源码 hash。运行中的清单不能追加或改写。新任务应创建新清单及新
`queue_root`，共用同一个 `lease_root`；现有活任务仍只作为观察条目读取。
部署时可把此脚本及 `benchmarks/utils/{owned_processes,resource_process}.py`
复制到单独的 OpenWAM 排队器快照，保留目录结构，不修改正在运行的原生快照。

## 执行、恢复与清理

- 每次循环检查选定 GPU 的实际占用，并把观察中的活任务占用卡计入保留集合。
  一张卡释放后，立即派发下一条已预审任务；其他卡继续独立执行。
- 同意此协议的多个排队器通过共享 GPU lease 防止重复占卡。获得 lease 后再次
  检查 `nvidia-smi`。外来 GPU 进程从不被终止；不遵守 lease 的外部调度器仍可能
  在检查后抢占，因此不能承诺系统级 GPU 独占。
- output 目录用原子创建取得所有权，写入 `.dev_queue_claim.json`，并在启动前写
  `launch_intent.json`。已有 output、失败结果、部分执行或只有启动 intent 的任务
  均不重试、不重新抽样。
- 仅新启动的任务进入 guardian registry。队列停止或所属父进程消失时，guardian
  清理自己的命令进程组；观察到的 PID 和外来进程不在清理范围内。GPU lease 的
  PID 身份记录在 owner 崩溃后仍能保护尚未退出的 guardian。
- `nvidia-smi` 暂时失败只阻断新派发，不终止正在运行的模拟。源文件/输入 hash
  改变会阻断对应任务，其他已预审任务仍可运行。
- 重开同一队列不会重启已有进程：已记录 guardian 只观察；失败、被中断或输出
  不完整的任务保持终态。要重新实验须单独预审新的任务与 output。

进度入口为 `queue_root/state.json`；每个任务的 intent、launch receipt、状态及
日志位于 `queue_root/jobs/<job_id>/`。这里的“完成”仅用于排队；科学验收仍需
使用原生 campaign 的独立 `verify_campaign` 与约束发布门禁。
