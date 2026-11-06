---
translation_status: translated
---

# 使用数据并行提供模型服务 { #serving-models-with-data-parallelism }

本教程说明如何将 vLLM recipe 或 SGLang cookbook 中的数据并行配置用于 GPUStack，包括 GPU 选择、后端参数配置、工作负载检查和故障恢复。

## 前提条件 { #prerequisites }

- GPUStack 已安装并运行，server、参与部署的 Workers 和 UI 均已更新到支持这些工作负载的版本。版本差异提示不能代替功能兼容性验证。
- 准备 **2 个 Linux Workers，每个有 4 张同型号 NVIDIA GPU，每张 80 GB 显存**。vLLM 普通部署使用全部八张卡；SGLang 普通部署只使用其中一个 Worker；PD 示例将两个 Workers 分别用于 Prefill 和 Decode。
- vLLM 选择 `Backend Version` `0.32.0`，SGLang 选择 `0.5.22`，使用支持对应设备的 GPUStack Runner 镜像。普通 SGLang 配套 SMG 还要求镜像依赖元数据包含 `sglang-router=0.3.2`；仅在自定义镜像中安装同名包不足以满足该检查。
- Workers 可以下载 Hugging Face 模型。使用本地模型时，文件在所有参与 Workers 上应位于相同路径。
- Workers 之间可以通过注册 IP 通信，网关可以访问其推理 API 端口；跨 Worker 通信建议使用高速互联。PD 还需满足 [Networking Requirements](../user-guide/pd-disaggregation.md#networking-requirements)。

vLLM 和 SGLang DPA/EP 示例使用 `Qwen/Qwen3-30B-A3B`，SGLang 普通 DP 示例使用 `Qwen/Qwen3-8B`。这些示例说明资源布局；实际可用上下文和并发量取决于权重精度、KV cache、通信库和设备能力，需要在目标环境验证。

## 步骤 1：部署模型 { #step-1-deploy-the-model }

以下示例演示各部署方式在 GPUStack 中的配置。比较多个示例时，先停止上一份部署并释放 GPU。

### 使用 vLLM External-LB 部署 { #deploy-with-vllm-external-lb }

1. 进入 `Deployments`，点击 `Deploy Model`，选择 `Hugging Face`，搜索并选择 `Qwen/Qwen3-30B-A3B`。
2. 将 `Backend` 设为 `vLLM`，`Backend Version` 设为 `0.32.0`。
3. 启用 `Allow Distributed Inference Across Workers`。在 `Scheduling` 中将 `Scheduling Mode` 设为 `Specify GPU`，用 `GPU Selector` 选择两个 Workers 上各四张 GPU。
4. 在 `Backend Parameters` 中逐项添加以下参数：

    ```bash
    --distributed-executor-backend=mp
    --data-parallel-backend=mp
    --data-parallel-size=4
    --tensor-parallel-size=2
    --enable-expert-parallel
    --data-parallel-external-lb
    --max-model-len=8192
    ```

5. 点击 `Save`。

此处 `DP × TP = 4 × 2 = 8`，每个 Worker 承载两个完整 ranks。GPUStack 分配设备和端口、生成各 rank 的启动参数，并在完整实例就绪后发布四个 rank API。

### 使用 vLLM Multi-port External-LB 部署 { #deploy-with-vllm-multi-port-external-lb }

1. 沿用 External-LB 示例的模型、版本、GPU 选择和其他参数。
2. 删除 `--data-parallel-external-lb`，替换为：

    ```bash
    --data-parallel-multi-port-external-lb
    ```

3. 点击 `Save`。

最终仍有四个独立 rank API，每个 Worker 的原生 DPSupervisor 管理两个本地 ranks。不要将其聚合健康端口当作共享推理 API。

此模式要求每个 rank 完整落在一个 Worker 上，各 Worker 的 local DP 相同且至少为 `2`。显式设置 `--api-server-count` 或 `--prefill-context-parallel-size` 时只能为 `1`；不支持 `--headless`、`--grpc` 或 `--uds`。

### 使用 vLLM Hybrid-LB 部署 { #deploy-with-vllm-hybrid-lb }

1. 沿用 External-LB 示例的模型、版本和 GPU 选择。
2. 将 `Backend Parameters` 整体替换为以下参数：

    ```bash
    --distributed-executor-backend=mp
    --data-parallel-backend=mp
    --data-parallel-size=4
    --tensor-parallel-size=2
    --enable-expert-parallel
    --data-parallel-hybrid-lb
    --max-model-len=8192
    ```

3. 点击 `Save`。

本例有两个共享 API，每个管理本地两个 ranks。网关选择共享 API，vLLM 选择其内部 rank。共享 API 的请求计数不等于某个 rank 的独立计数。

普通 External-LB 和 Hybrid-LB 也可安排跨 Worker 的 TP/PP 分片；同一 rank 跨节点时要求均分设备。此时部分 Worker 仅承担计算，不发布 API，应以实际工作负载为准。

### 使用 SGLang DP 和 SMG 部署 { #deploy-with-sglang-dp-and-smg }

1. 进入 `Deployments`，点击 `Deploy Model`，选择 `Hugging Face`，搜索并选择 `Qwen/Qwen3-8B`。
2. 将 `Backend` 设为 `SGLang`，`Backend Version` 设为 `0.5.22`。
3. 在 `Scheduling` 中选择 `Scheduling Mode` `Specify GPU`，使用 `GPU Selector` 选择同一个 Worker 上的四张 GPU。
4. 在 `Backend Parameters` 中添加：

    ```bash
    --tp-size=1
    --dp-size=4
    --context-length=8192
    --policy=cache_aware
    ```

5. 点击 `Save`。

普通 DP 的四份完整模型副本共占 `DP × TP = 4 × 1 = 4` 张 GPU。GPUStack 创建 Engine 和一个使用 CPU 的 SMG 组件，网关连接 SMG。无需额外创建 SMG deployment 或填写 `--worker-urls`。本例不启用 DP attention，四张 GPU 必须位于同一 Worker。

`--policy` 只交给配套 SMG，不会传给 Engine。默认策略是 `cache_aware`，还支持 `random`、`round_robin`、`power_of_two` 和 `manual`。相同前缀可能持续命中同一个目标，不能用请求数是否平均判断路由是否正常。

### 使用 SGLang DPA 和 EP 部署 { #deploy-with-sglang-dpa-and-ep }

1. 沿用上节的单 Worker 四卡选择和 SGLang 版本，将模型改为 `Qwen/Qwen3-30B-A3B`。
2. 将 `Backend Parameters` 整体替换为：

    ```bash
    --tp-size=4
    --dp-size=1
    --attn-dp-size=4
    --ep-size=4
    --context-length=8192
    --policy=cache_aware
    ```

3. 点击 `Save`。

本例在四张 GPU 上组合 attention DP 和 EP，不能按普通 DP 的 `DP × TP` 将它算作十六张卡。`--attn-dp-size` 与 `--attention-data-parallel-size` 是别名，已知 Runner 版本需 SGLang `0.5.22` 或更高；旧版使用 `--dp-size=4 --enable-dp-attention` 的语法，不能混用两种配置来重复表达同一并行度。

普通 SGLang 部署的有效路由 rank 数为 DP size 与 attention DP size 的乘积。新 DPA 语法要求 TP size 能被 attention DP size 整除，且不能同时将 DP size 和 attention DP size 都设为大于 `1`；见 [SGLang Activation Logic](https://docs.sglang.io/docs/advanced_features/dp_dpa_smg_guide#activation-logic)。该数量大于 `1` 时才创建 SMG；只设置 EP 且路由 rank 数为 `1` 时直接使用 Engine，不能再配置普通 SMG 的 `--policy`。SGLang PD 使用独立的 PD router，角色内不会额外创建本文的普通部署 SMG。

### 由 GPUStack 管理拓扑 { #let-gpustack-manage-the-topology }

上述托管示例只配置完整实例的并行度、模式和 GPU 选择。rank、节点序号、通信地址和端口由平台生成，无需为每个 Worker 新建一个 deployment，也无需设置 `GPUSTACK_SKIP_GPU_COUNT_CHECK`。

保持 `--data-parallel-rank`、`--data-parallel-start-rank`、`--data-parallel-address`、`--data-parallel-rpc-port`、`--nnodes`、`--node-rank`、`--host` 和 `--port` 由平台管理。`--data-parallel-size-local` 可以显式设置，但必须与实际分配一致。

!!! warning

    vLLM 的三个 LB 开关互斥。托管 rank API 使用 TCP HTTP，不支持为引擎配置 `--ssl-keyfile` 或 `--ssl-certfile`；客户端 TLS 在网关入口配置。`GPUSTACK_MANUAL_DP=1` 仅用于普通内置 vLLM 的手工 DP 流程，不适用于 PD 或 SGLang；切换此变量需要重建实例。手工流程见 [MultiProcessing tutorial](running-distributed-vllm-with-multiprocessing.md#step-2-deploy-the-model-with-the-mp-backend)。

## 步骤 2：检查工作负载和就绪状态 { #step-2-check-workloads-and-readiness }

1. 在 `Deployments` 展开部署，查看实例状态。
2. 打开实例信息中的 `Workloads / ranks`，检查 Worker、设备、rank/component 和 API 就绪状态。
3. 等待实例进入 `Running`。下载完成、容器启动与 API 就绪是不同阶段。
4. 发生异常时，使用对应工作负载的日志入口或实例的 `View Logs`，确认失败节点和原因；需要完整材料时使用 `Download Logs`。

| Example | Expected layout |
| --- | --- |
| vLLM External-LB / Multi-port External-LB | 四个 rank API |
| vLLM Hybrid-LB | 两个共享 API，每个管理两个本地 ranks |
| SGLang DP | Engine 和 SMG 两个组件，Engine 内部有四个逻辑 ranks |
| SGLang DPA/EP | Engine 和 SMG 两个组件，本例有四个逻辑 ranks |

日志和指标应按实际来源查看。SGLang 的逻辑 DP 视图筛选共享 Engine 的原生 `dp_rank` 指标；没有该维度的指标可能显示无数据。Hybrid 的共享 API 指标和 Engine 聚合指标不能按 ranks 数重复相加。仅缺少指标或请求数暂为零，不表示 Engine 已故障。

## 步骤 3：在 Playground 中验证 { #step-3-verify-in-playground }

1. 在已就绪部署的操作中点击 `Open in Playground`。
2. 确认已选择本次部署的模型，输入“用一句话介绍数据并行。”并发送。
3. 确认收到回复，再结合工作负载日志和监控检查请求路径。

## 与 PD 分离组合使用 { #use-with-pd-disaggregation }

PD 将 Prefill 和 Decode 分开配置、分配资源，再由 PD router 协调请求和 KV 传输。普通部署的网关直接 rank 路由不适用于整个 PD 服务，PD 的统一入口始终是 router。

### 检查传输方案和布局 { #check-the-transport-and-layout }

| vLLM mode | Router members | Transport constraints |
| --- | --- | --- |
| External-LB | 每个 DP rank 的主 API | CUDA NIXL、CUDA Mooncake、Ascend Mooncake；CUDA Mooncake 要求每个 rank 完整落在一个 Worker |
| Multi-port External-LB | 原生 DPSupervisor 启动的逐 rank 主 API，按 DP=1 接入 | 仅 CUDA NIXL；各 Worker local DP 相同且 ≥ 2，每个 rank 完整落在一个 Worker |
| Hybrid-LB | 提供 API 的 Worker 共享入口，按其 local DP 接入 | CUDA NIXL、CUDA Mooncake、Ascend Mooncake；CUDA Mooncake 静态 router 要求各共享 API 的 local DP 相同 |

### 部署 vLLM NIXL 组 { #deploy-a-vllm-nixl-group }

1. 新建 `Qwen/Qwen3-30B-A3B` 部署，选择 `vLLM` 和 `Backend Version` `0.32.0`，启用 `PD Disaggregation`，在 `Transport` 中选择 `NIXL`。
2. 在 `Roles` 中将 `Prefill` 和 `Decode` 的 `Parameters and environment` 都切换为 `Custom`，分别在 `Backend Parameters` 中填写以下完整列表；模型级参数列表留空，保留平台管理的 `Router`：

    ```bash
    --distributed-executor-backend=mp
    --data-parallel-backend=mp
    --data-parallel-size=2
    --tensor-parallel-size=2
    --enable-expert-parallel
    --data-parallel-external-lb
    --max-model-len=8192
    ```

3. 将各角色的 `Resources and scheduling` 切换为 `Custom`，使用 `Specify GPU`：Prefill 选择第一个 Worker 的四张 GPU，Decode 选择第二个 Worker 的四张 GPU。
4. 点击 `Save`，等待 Prefill、Decode 和 Router 都就绪，再按 Step 3 在 Playground 中验证。

本例的 Prefill 和 Decode 各有两个 ranks，合计八张 GPU。角色参数列表覆盖整份模型级列表，不会逐项追加；KV connector、角色地址和连接参数由所选方案生成，无需手工填写 peer 地址。

在上述 CUDA NIXL 示例中，可将两个角色的 `--data-parallel-external-lb` 都替换为 `--data-parallel-multi-port-external-lb`，每侧仍有两个独立 rank API；或替换为 `--data-parallel-hybrid-lb`，并各添加 `--data-parallel-size-local=2`，每侧改为一个管理两个 ranks 的共享 API。每次只保留一个 LB 开关。

Router 在两侧具备可用成员后启动，启动期间显示 `Partially ready` 属于正常阶段。单个 rank 或单个角色就绪不能证明完整 PD 链路可用。CUDA Mooncake 使用静态成员，成员变化可能重建 router 并短暂中断请求；角色管理和 KV 传输配置见 [PD Disaggregation](../user-guide/pd-disaggregation.md)。SGLang PD 的路由参数在 Router 的 `Route arguments` 中配置，不使用普通 SMG 示例中的 Engine 参数 `--policy`。

## 步骤 4：更新和恢复部署 { #step-4-update-and-recover-the-deployment }

### 应用配置变更 { #apply-a-configuration-change }

修改并行度、LB 模式或 GPU 布局后，需要重建完整实例才能应用配置。

1. 保存新配置。
2. 在部署的 `Operations` 中选择 `Stop`。
3. 等待旧实例移除。PD 的停止流程会等待在线 Workers 确认旧工作负载停止。
4. 选择 `Start`，等待新实例就绪，再到 Playground 验证。普通部署启用定时扩缩容时，使用 `Resume Schedule` 恢复当前计划；目标副本数可能仍为零。

| Situation | Recovery behavior |
| --- | --- |
| 普通托管实例的 Engine/rank 故障 | 撤出整个实例入口，确认旧任务停止后，按 `Auto-Restart On Error` 恢复完整实例；普通实例可按当前配置沿用原 Worker/GPU 分配恢复 |
| 普通 SGLang 只有 SMG 故障 | 可以只重建 SMG，保留健康 Engine |
| PD 的 Engine 实例故障，配置未变 | 恢复所属完整实例，保留健康对侧和其他实例，并更新 router 成员 |
| PD 配置已变更 | 暂停旧配置成员的自动恢复和新增补建；对整个部署停止再启动，统一应用配置 |
| Worker 失联，旧任务状态未知 | 自动恢复保留旧任务归属并等待停止确认，不承诺自动异地接管 |

编辑配置不会立即重启健康进程。普通部署的故障恢复也不重新选卡；若新配置需要不同的资源布局，应主动重建部署，不能依赖自动恢复重新调度。

### 替换不可用的 Worker { #replace-an-unavailable-worker }

Worker 失联表示 GPUStack 当前无法确认节点及其任务状态，不能据此判断节点是否可以修复。当节点或网络排查已确定需要替换 Worker 时：

1. 在部署级选择 `Stop` 或 `Delete`。平台会清理已判定失联、心跳过期或已移除 Worker 的旧实例记录，并等待在线 Workers 停止旧工作负载。
2. 保留部署时，注册替代 Worker，并在 `GPU Selector` 中更新 GPU 选择。
3. 等待旧实例移除后再 `Start`；删除了部署时重新创建部署。

!!! warning

    移除失联 Worker 的实例记录不代表远端进程已经停止。若原机器仍运行但网络隔离，应先在机器或基础设施层停止、隔离旧工作负载，再启动替代组。原 Worker 恢复连接后会清理不再属于有效实例的旧任务。恢复不保证保留 KV cache 或在途请求。

## 参考资料 { #references }

- [vLLM 0.32.0: Data Parallel Deployment](https://docs.vllm.ai/en/v0.32.0/serving/data_parallel_deployment/)
- [SGLang: DP, DPA and SGLang DP Router](https://docs.sglang.io/docs/advanced_features/dp_dpa_smg_guide)
