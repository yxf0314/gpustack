# Serving Models with Data Parallelism

This tutorial explains how to use data-parallel configurations from vLLM recipes or the SGLang cookbook in GPUStack, including GPU selection, backend parameters, workload inspection, and recovery.

## Prerequisites

- GPUStack is installed and running. Update the server, participating workers, and UI to versions that support these workloads. Version mismatch notices do not replace feature compatibility checks.
- Prepare **2 Linux workers, each with 4 NVIDIA GPUs of the same model and 80 GB VRAM per GPU**. The ordinary vLLM deployment uses all eight GPUs. The ordinary SGLang deployment uses one worker. The PD example assigns one worker to Prefill and the other to Decode.
- Select `Backend Version` `0.32.0` for vLLM or `0.5.22` for SGLang, using a GPUStack Runner image for your devices. The paired SMG for an ordinary SGLang deployment also requires image dependency metadata containing `sglang-router=0.3.2`. Installing the package in a custom image alone does not satisfy this check.
- Workers can download models from Hugging Face. For local models, use the same model path on every participating worker.
- Workers can communicate through their registered IP addresses, and the gateway can reach their inference API ports. A high-speed interconnect is recommended across workers. PD also requires the [Networking Requirements](../user-guide/pd-disaggregation.md#networking-requirements).

The vLLM and SGLang DPA/EP examples use `Qwen/Qwen3-30B-A3B`. The ordinary SGLang DP example uses `Qwen/Qwen3-8B`. These examples illustrate resource layouts. Validate the available context length and concurrency in your environment: both depend on weight precision, KV cache, communication libraries, and device capabilities.

## Step 1: Deploy the Model

The following examples show how to configure each deployment in GPUStack. Stop the previous deployment and release its GPUs before trying another example.

### Deploy with vLLM External-LB

1. Go to `Deployments`, click `Deploy Model`, select `Hugging Face`, and search for `Qwen/Qwen3-30B-A3B`.
2. Set `Backend` to `vLLM` and `Backend Version` to `0.32.0`.
3. Enable `Allow Distributed Inference Across Workers`. Under `Scheduling`, set `Scheduling Mode` to `Specify GPU`. Use `GPU Selector` to select four GPUs on each worker.
4. Add the following entries to `Backend Parameters`:

    ```bash
    --distributed-executor-backend=mp
    --data-parallel-backend=mp
    --data-parallel-size=4
    --tensor-parallel-size=2
    --enable-expert-parallel
    --data-parallel-external-lb
    --max-model-len=8192
    ```

5. Click `Save`.

This layout uses `DP × TP = 4 × 2 = 8` GPUs, with two complete ranks on each worker. GPUStack assigns devices and ports, generates the launch arguments for each rank, and publishes four rank APIs once the complete instance is ready.

### Deploy with vLLM Multi-port External-LB

1. Keep the model, version, GPU selection, and other parameters from the External-LB example.
2. Replace `--data-parallel-external-lb` with:

    ```bash
    --data-parallel-multi-port-external-lb
    ```

3. Click `Save`.

The deployment still exposes four independent rank APIs. The native DPSupervisor on each worker manages two local ranks. Its aggregate health port is not a shared inference API.

Each rank must fit entirely on one worker. All workers must have the same local DP size, at least `2`. If set explicitly, `--api-server-count` and `--prefill-context-parallel-size` must be `1`. This mode does not support `--headless`, `--grpc`, or `--uds`.

### Deploy with vLLM Hybrid-LB

1. Keep the model, version, and GPU selection from the External-LB example.
2. Replace the entire `Backend Parameters` list with:

    ```bash
    --distributed-executor-backend=mp
    --data-parallel-backend=mp
    --data-parallel-size=4
    --tensor-parallel-size=2
    --enable-expert-parallel
    --data-parallel-hybrid-lb
    --max-model-len=8192
    ```

3. Click `Save`.

This example exposes two shared APIs, each managing two local ranks. The gateway selects a shared API, and vLLM selects a rank behind it. Request counts for a shared API are not counts for an individual rank.

Ordinary External-LB and Hybrid-LB also support TP/PP shards across workers. A rank spanning multiple workers requires an equal device allocation on each node. Some workers then provide compute only, without an API. Check the actual workload layout.

### Deploy with SGLang DP and SMG

1. Go to `Deployments`, click `Deploy Model`, select `Hugging Face`, and search for `Qwen/Qwen3-8B`.
2. Set `Backend` to `SGLang` and `Backend Version` to `0.5.22`.
3. Under `Scheduling`, set `Scheduling Mode` to `Specify GPU`. Use `GPU Selector` to select four GPUs on the same worker.
4. Add the following entries to `Backend Parameters`:

    ```bash
    --tp-size=1
    --dp-size=4
    --context-length=8192
    --policy=cache_aware
    ```

5. Click `Save`.

Ordinary DP runs four complete model copies on `DP × TP = 4 × 1 = 4` GPUs. GPUStack creates an Engine and a CPU-only SMG component. The gateway connects to SMG. No separate SMG deployment or manual `--worker-urls` is needed. This example does not enable DP attention, so all four GPUs must be on one worker.

`--policy` applies only to the paired SMG and is not passed to the Engine. The default is `cache_aware`; other supported values are `random`, `round_robin`, `power_of_two`, and `manual`. Requests with the same prefix may keep reaching the same target. An uneven request distribution does not by itself indicate a routing fault.

### Deploy with SGLang DPA and EP

1. Keep the four-GPU, single-worker selection and SGLang version from the previous example. Change the model to `Qwen/Qwen3-30B-A3B`.
2. Replace the entire `Backend Parameters` list with:

    ```bash
    --tp-size=4
    --dp-size=1
    --attn-dp-size=4
    --ep-size=4
    --context-length=8192
    --policy=cache_aware
    ```

3. Click `Save`.

This example combines attention DP and EP on four GPUs. Do not apply the ordinary DP formula `DP × TP` and count sixteen GPUs. `--attn-dp-size` and `--attention-data-parallel-size` are aliases and require SGLang `0.5.22` or later for known Runner versions. Older versions use `--dp-size=4 --enable-dp-attention`. Do not combine the two syntaxes to express the same parallelism twice.

For an ordinary SGLang deployment, the effective routing rank count is DP size multiplied by attention DP size. The new DPA syntax requires TP size to be divisible by attention DP size, and the two DP sizes cannot both exceed `1`; see [SGLang Activation Logic](https://docs.sglang.io/docs/advanced_features/dp_dpa_smg_guide#activation-logic). GPUStack creates SMG only when the routing rank count exceeds `1`. An EP-only configuration with `1` routing rank uses the Engine directly and cannot set the ordinary SMG `--policy`. SGLang PD uses a separate PD router and does not add the ordinary deployment's SMG to each role.

### Let GPUStack Manage the Topology

The managed examples configure parallelism, mode, and GPU selection for the complete instance. GPUStack generates ranks, node indices, communication addresses, and ports. There is no need to create one deployment per worker or set `GPUSTACK_SKIP_GPU_COUNT_CHECK`.

Leave `--data-parallel-rank`, `--data-parallel-start-rank`, `--data-parallel-address`, `--data-parallel-rpc-port`, `--nnodes`, `--node-rank`, `--host`, and `--port` to GPUStack. You can set `--data-parallel-size-local` explicitly, but it must match the actual allocation.

!!! warning

    The three vLLM LB flags are mutually exclusive. Managed rank APIs use TCP HTTP and do not support Engine-side `--ssl-keyfile` or `--ssl-certfile`; configure client TLS at the gateway. `GPUSTACK_MANUAL_DP=1` is only for manual DP with the ordinary built-in vLLM backend, not PD or SGLang. Changing it requires recreating instances. See the [MultiProcessing tutorial](running-distributed-vllm-with-multiprocessing.md#step-2-deploy-the-model-with-the-mp-backend) for the manual workflow.

## Step 2: Check Workloads and Readiness

1. Expand the deployment in `Deployments` and check its instance state.
2. Open `Workloads / ranks` in the instance information. Check the worker, devices, rank/component, and API readiness.
3. Wait for `Running`. Completing a download, starting a container, and API readiness are separate stages.
4. If a failure occurs, use the workload's log entry or the instance's `View Logs` to identify the worker and cause. Use `Download Logs` to export the logs.

| Example | Expected layout |
| --- | --- |
| vLLM External-LB / Multi-port External-LB | Four rank APIs |
| vLLM Hybrid-LB | Two shared APIs, each managing two local ranks |
| SGLang DP | Engine and SMG components, with four logical ranks inside the Engine |
| SGLang DPA/EP | Engine and SMG components, with four logical ranks in this example |

Read logs and metrics according to their actual source. SGLang's logical DP views filter the shared Engine's native `dp_rank` metrics. Metrics without that dimension may show no data. Do not multiply Hybrid shared API metrics or aggregate Engine metrics by the rank count. Missing metrics or a temporary request count of zero alone do not indicate an Engine failure.

## Step 3: Verify in Playground

1. Click `Open in Playground` for the ready deployment.
2. Confirm that the deployment's model is selected. Enter “Explain data parallelism in one sentence.” and send it.
3. Confirm that a reply arrives, then check the workload logs and monitoring to inspect the request path.

## Use with PD Disaggregation

PD configures and allocates resources to Prefill and Decode separately. A PD router coordinates requests and KV transfer. The ordinary deployment's direct gateway-to-rank routing does not apply to the complete PD service: the PD entry point is always its router.

### Check the Transport and Layout

| vLLM mode | Router members | Transport constraints |
| --- | --- | --- |
| External-LB | Each DP rank's main API | CUDA NIXL, CUDA Mooncake, Ascend Mooncake; CUDA Mooncake requires each rank to fit on one worker |
| Multi-port External-LB | Rank main APIs launched by native DPSupervisor, each registered as DP=1 | CUDA NIXL only; equal local DP ≥ 2 on all workers, with each rank entirely on one worker |
| Hybrid-LB | Shared endpoints on workers that provide APIs, each registered with its local DP | CUDA NIXL, CUDA Mooncake, Ascend Mooncake; CUDA Mooncake's static router requires the same local DP for all shared APIs |

### Deploy a vLLM NIXL Group

1. Create a deployment for `Qwen/Qwen3-30B-A3B`. Select `vLLM` and `Backend Version` `0.32.0`, enable `PD Disaggregation`, and select `NIXL` under `Transport`.
2. Under `Roles`, set `Parameters and environment` to `Custom` for both `Prefill` and `Decode`. Enter the following complete list in each role's `Backend Parameters`. Leave the model-level list empty and keep the platform-managed `Router`:

    ```bash
    --distributed-executor-backend=mp
    --data-parallel-backend=mp
    --data-parallel-size=2
    --tensor-parallel-size=2
    --enable-expert-parallel
    --data-parallel-external-lb
    --max-model-len=8192
    ```

3. Set each role's `Resources and scheduling` to `Custom`. With `Specify GPU`, assign four GPUs on the first worker to Prefill and four on the second worker to Decode.
4. Click `Save`. Wait for Prefill, Decode, and Router to become ready, then verify in Playground as in Step 3.

Prefill and Decode each have two ranks, using eight GPUs in total. A role's parameter list replaces the entire model-level list rather than appending to it. The selected recipe generates the KV connector, role addresses, and connection parameters. No manual peer addresses are needed.

In this CUDA NIXL example, replace `--data-parallel-external-lb` on both roles with `--data-parallel-multi-port-external-lb` to keep two independent rank APIs per side. Alternatively, replace it with `--data-parallel-hybrid-lb` and add `--data-parallel-size-local=2` to each role. Each side then exposes one shared API managing two ranks. Keep only one LB flag at a time.

Router starts after both sides have available members. `Partially ready` is normal during startup. One ready rank or role does not establish that the full PD request path works. CUDA Mooncake uses static membership, so member changes may rebuild the router and briefly interrupt requests. See [PD Disaggregation](../user-guide/pd-disaggregation.md) for role management and KV transfer configuration. Configure SGLang PD routing through Router's `Route arguments`, not the ordinary SMG example's Engine `--policy` parameter.

## Step 4: Update and Recover the Deployment

### Apply a Configuration Change

Changes to parallelism, LB mode, or GPU layout require recreating the complete instance.

1. Save the new configuration.
2. Select `Stop` under the deployment's `Operations`.
3. Wait for the old instances to be removed. The PD stop flow waits for online workers to confirm that old workloads have stopped.
4. Select `Start`, wait for the new instances to become ready, and verify in Playground. For an ordinary deployment with scheduled scaling enabled, use `Resume Schedule` to resume its current schedule. Its target count may still be zero.

| Situation | Recovery behavior |
| --- | --- |
| Engine/rank failure in an ordinary managed instance | Withdraw all endpoints of the instance. After confirming that old workloads have stopped, recover the complete instance according to `Auto-Restart On Error`. Ordinary instances can recover with the current configuration on the original worker/GPU allocation. |
| Only SMG fails in an ordinary SGLang deployment | Rebuild SMG alone while keeping the healthy Engine. |
| A PD Engine instance fails with no configuration change | Recover the complete affected instance, retain the healthy opposite role and other instances, and update router membership. |
| PD configuration has changed | Pause automatic recovery and recreation of members on the old configuration. Stop and start the entire deployment to apply the saved configuration. |
| A worker is unreachable and old workload states are unknown | Retain ownership of old workloads and wait for stop confirmation. Automatic recovery does not promise relocation to another worker. |

Saving changes does not immediately restart healthy processes. Ordinary failure recovery does not select new GPUs. If the new configuration needs a different resource layout, recreate the deployment explicitly rather than relying on recovery to reschedule it.

### Replace an Unavailable Worker

An unreachable worker means GPUStack cannot currently confirm the node or workload state. It does not establish whether the node can be repaired. Once node or network investigation has determined that the worker needs replacement:

1. Select `Stop` or `Delete` for the deployment. GPUStack removes old instance records for workers considered unreachable, with expired heartbeats, or already removed, and waits for online workers to stop old workloads.
2. If keeping the deployment, register a replacement worker and update the GPU selection in `GPU Selector`.
3. Wait for old instances to be removed before selecting `Start`. If you deleted the deployment, create it again.

!!! warning

    Removing an unreachable worker's instance records does not mean its remote processes have stopped. If the original machine is running but isolated from the network, stop or isolate its old workloads at the machine or infrastructure level before starting a replacement group. When the original worker reconnects, it cleans up old workloads that no longer belong to valid instances. Recovery does not guarantee preservation of KV cache or in-flight requests.

## References

- [vLLM 0.32.0: Data Parallel Deployment](https://docs.vllm.ai/en/v0.32.0/serving/data_parallel_deployment/)
- [SGLang: DP, DPA and SGLang DP Router](https://docs.sglang.io/docs/advanced_features/dp_dpa_smg_guide)
