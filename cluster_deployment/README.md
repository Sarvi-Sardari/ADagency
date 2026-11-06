# `cluster_deployment` package

Deployment support shared by every pipeline stage: the local/cluster switch, Ray cluster discovery, the managed vLLM server, and hardware telemetry on remote machines.

In **local mode** (default) models run in-process through Hugging Face Transformers, and this package only reports that mode. In **cluster mode** the pipeline serves the configured model with vLLM across all GPUs of a running Ray cluster. One GPU per PC connected over Ethernet is supported, using pipeline parallelism.

## Selecting the mode

The model configuration contains the selector:

```yaml
deployment:
  mode: cluster                          # local | cluster
  cluster_config: "model_cluster_configs.yaml"   # resolved relative to the model config's folder
```

In cluster mode the cluster file is loaded, and its `models.*` entries are merged key by key over the model configuration (endpoint, served name, launch command). In local mode the cluster file is never opened. A fully commented template with placeholder addresses is provided in [`configs/templates/model_cluster_configs.template.yaml`](../configs/templates/model_cluster_configs.template.yaml). The filled-in cluster file contains IP addresses and user names; keep it private (it is git-ignored).

## Operating a cluster

1. **Same environment everywhere.** Install the identical Python environment (`requirements.txt`) on every PC, and make the model directories available at the same paths.
2. **Password-less SSH** from the head PC to every worker, needed for remote telemetry:

   ```bash
   ssh-keygen -t ed25519
   ssh-copy-id <user>@<worker-ip>
   ssh -o BatchMode=yes <user>@<worker-ip> true
   ```

3. **Start Ray manually.** The pipeline never starts or stops Ray. Use the network interface that connects the PCs (`ip -br link`):

   ```bash
   # head PC
   export NCCL_SOCKET_IFNAME=<interface> GLOO_SOCKET_IFNAME=<interface> VLLM_HOST_IP=<head-ip>
   ray start --head --port=6379 --node-ip-address=<head-ip>

   # each worker PC
   export NCCL_SOCKET_IFNAME=<interface> GLOO_SOCKET_IFNAME=<interface> VLLM_HOST_IP=<worker-ip>
   ray start --address='<head-ip>:6379' --node-ip-address=<worker-ip>
   ```

4. **Preflight** (no model is loaded):

   ```bash
   python -m cluster_deployment.preflight --model-config configs/model_configs.yaml
   # additionally start, sample, stop and collect every telemetry recorder:
   python -m cluster_deployment.preflight --model-config configs/model_configs.yaml --telemetry-smoke-seconds 5
   ```

   The preflight checks the active Ray nodes and their GPU count, the requested parallel topology, that Ray, vLLM, PyTorch and CUDA versions match inside every worker, that the model directories exist on each worker, and that every remote machine has SSH access, the telemetry interpreter, and NVIDIA visibility.

5. **Run any stage as usual.** The stage starts `vllm serve` from `server_launch.command`, waits until the server is healthy (`ready_timeout`), runs the workload, and stops the server. Planner generation swaps the servers VLM → LLM → VLM.

### GPU topology

- `gpu_count: auto` uses every GPU advertised by Ray. An integer limits the count.
- `tensor_parallel_size` (default 1) must divide the GPU count.
- `pipeline_parallel_size = gpu_count / tensor_parallel_size` is derived automatically and inserted into the launch command through the `{pipeline_parallel_size}` placeholder. Adding a PC to the Ray cluster therefore needs no configuration change.

### Remote telemetry

When `remote_monitoring.enabled` is true, the same recorder that runs locally (`anomaly_detection/hardware_monitor.py`) is copied to every remote Ray GPU node and started there. With `discover_from_ray`, nodes are found automatically, and `nodes` entries only override the defaults for individual hosts. Each recorder writes to its own machine's disk and keeps running even if the SSH session drops. At the end, the data is copied back beneath the local telemetry folder together with a clock-offset measurement. With `required: true`, a telemetry start failure stops the run before any model is loaded.

## Module reference

| Module | Responsibility |
|---|---|
| `config.py` | parsing of `deployment`, the cluster file, and the merging of model sections |
| `cluster_runtime.py` | Ray node discovery, GPU topology planning, worker runtime inspection, launch-command placeholder rendering |
| `server_manager.py` | lifecycle of one managed vLLM server (start, readiness, swap, teardown, logs) |
| `remote_monitor.py` | SSH-launched remote recorders, copy-back, clock offset |
| `preflight.py` | read-only validation CLI (`python -m cluster_deployment.preflight`) |

Managed vLLM server logs are written to `cluster_deployment/logs/vllm_server_logs/` (git-ignored).
