# TACC Lonestar6 and Vista for astra-distill GPU jobs

Checked 2026-10-14 against the TACC user guides (Vista page "Last update: June 2, 2026"; Lonestar6 page "Last update: September 18, 2026"). Items marked UNVERIFIED were not stated in any primary source I could read. Several fetched pages were truncated at 100,000 characters by the fetch tool (I read the Lonestar6 page in two chunks, offset 0 and 100000, via a summarising fetch tool, so exact table columns should be re-checked with `qlimits`), so "not found" means "not in the text I received", not "absent from TACC docs". Run `qlimits` on each system for live queue limits (both guides say limits can change without notice).

Frontera is not available to this user and its queues close 2026-10-15; its content was removed from this guide.

## Recommendations

| # | Task | System / queue | Why | SU cost (48 h job) |
|---|---|---|---|---|
| 1 | 27B bf16 LoRA (56 GB weights) | First choice Vista `gh` (96 GB GPU, 1 node). Fallback Lonestar6 `gpu-h100` (2x H100 80 GB, split model across both GPUs) | One 96 GB GPU holds weights plus LoRA state if you use gradient checkpointing (tight; UNVERIFIED by measurement). LS6 A100 40 GB cannot hold 56 GB on one GPU; 3x40 GB = 120 GB with model split (device_map / FSDP) works but over PCIe and is slow. `gpu-h100` is limited to 1 node and 1 running job per user | Vista 48 SU; LS6 `gpu-h100` 288 SU; LS6 `gpu-a100` 144 SU |
| 2 | 27B QLoRA (4-bit, about 16 GB weights) | Lonestar6 `gpu-a100`, 1 node, 1 GPU (or 3 runs in parallel, one per GPU); or Vista `gh` | Fits in one 40 GB A100. x86_64 wheels (bitsandbytes, torch) are standard. Vista needs aarch64 builds | LS6 144 SU/node (3 GPUs busy = 48 SU per GPU-run); Vista 48 SU |
| 3 | 4B fine-tunes | Lonestar6 `gpu-a100` with 3 independent runs per node (`CUDA_VISIBLE_DEVICES`), or `gpu-a100-small` (hardware UNVERIFIED) | 4B bf16 is about 8 GB; one A100 40 GB per run. Nodes are not shared, so fill all 3 GPUs | LS6 3 SU/h per node = 1 SU per GPU-hour; Vista 1 SU/h for 1 GPU |
| 4 | vLLM 27B inference (guided JSON) | Vista `gh`, 1 node, no parallelism needed (56 GB weights + KV on 96 GB). On LS6: `gpu-a100` with tensor-parallel 2 (80 GB, tight KV) or `gpu-h100` TP=2 (160 GB) | Single-GPU serving avoids PCIe all-reduce. On A100 40 GB, TP=3 may not divide the model's head counts: check the config, UNVERIFIED | Vista 1 SU/h; LS6 `gpu-a100` 3 SU/h; `gpu-h100` 6 SU/h |
| 5 | Many small 4B jobs | Lonestar6 `gpu-a100` (3 jobs per node concurrently; up to 12 nodes/user, 8 jobs/user) or Vista `gh` job arrays (up to 20 jobs) | Per-GPU cost is 1 SU/GPU-h on LS6 and 1 SU/GPU-h on Vista; LS6 gives 3 GPUs per allocation of a node, Vista 1 | same as above |

Cross-cutting:
- Vista is aarch64 (Grace CPU); Lonestar6 is x86_64 (AMD EPYC). LS6 uses ordinary PyPI wheels (torch, vLLM, bitsandbytes, flash-attn). That is the main reason to use LS6 when the toolchain is the risk; Vista is the better choice when GPU memory is the constraint.
- Both systems reject `--gres`, `--gpus-per-task`, `--mem`. A node request gives you all of its GPUs.
- The `$WORK` filesystem (Stockyard, 1 TB, 3,000,000 files) is shared between the two, so weights downloaded once are visible on both. Shared policies (Good Conduct, MFA, AI tools on compute nodes only, no `ssh-keygen`) are in section 5 and apply to both.
- Both allocations are needed separately (Vista allocation; Lonestar6 allocation via TxRAS for UT System researchers). `/usr/local/etc/taccinfo` shows balances and quotas on LS6. SU rates are not directly comparable across systems; each draws from its own allocation. UNVERIFIED how allocations are obtained for this project.
- Lonestar6 `$SCRATCH` is BeeGFS; Vista `$SCRATCH` is VAST. Both purge after 10 days without access.

## 1. Hardware

Sources: Lonestar6 guide https://docs.tacc.utexas.edu/hpc/lonestar6/ (2026-10-14); Vista guide https://docs.tacc.utexas.edu/hpc/vista/ (2026-10-14).

### Lonestar6

OS Rocky 8.4, Slurm 20.11.8 (as stated on the guide page). HDR InfiniBand 200 Gb/s. 560 CPU nodes, 88 GPU nodes (84 A100 nodes, 4 H100 nodes). Login: `ls6.tacc.utexas.edu`.

| | A100 node (84) | H100 node (4) |
|---|---|---|
| GPUs | 3x NVIDIA A100 PCIe 40 GB (HBM2); gpu0 on socket 0, gpu1 and gpu2 on socket 1 | 2x NVIDIA H100 PCIe 80 GB (HBM2e); gpu0 on socket 0, gpu1 on socket 1 |
| CPU | 2x AMD EPYC 7763 (Milan), 128 cores, x86_64, 2.45 GHz base | 2x AMD EPYC 9454 (Genoa), 96 cores, 2.75 GHz |
| Host RAM | 256 GB DDR4-3200 | 384 GB |
| Local `/tmp` | 288 GB SSD, purged at job end | 288 GB |
| Peak per GPU | 9.7 TF FP64, 312 TF FP16 tensor | 26 TF FP64, 1513 TF FP16 tensor |

GPU-to-GPU link within a node: the guide mentions only PCIe cards and socket placement; no NVLink is stated. NVLink/NVSwitch absence is therefore likely but UNVERIFIED. GPUs 1 and 2 share a socket, so TP=2 on gpu1+gpu2 is the topologically best pair (inference from the stated socket layout).

### Lonestar6 GPU queues (Production Queues table, as summarised by the fetch tool; columns: nodes per job, max duration, max nodes per user, max running jobs per user, max submitted, SU per node-hour)

| Queue | Hardware | Nodes/job | Max time | Max nodes/user | Max running jobs | Max submitted | SU/node-h |
|---|---|---|---|---|---|---|---|
| `gpu-a100` | A100 node (3x 40 GB) | 8 | 48 h | 12 | 8 | 32 | 3 |
| `gpu-a100-dev` | A100 node | 2 | 2 h | 2 | 1 | 3 | 3 |
| `gpu-a100-small` | "virtual nodes with fewer resources (cores)" | 1 | 48 h | 3 | 3 | 12 | 1.5 |
| `gpu-h100` | H100 node (2x 80 GB) | 1 | 48 h | 1 | 1 | 4 | 6 |
| `normal` (CPU) | 128-core node | 64 | 48 h | 64 | 20 | 100 | 1 |
| `development` (CPU) | | 8 | 2 h | 8 | 1 | 3 | 1 |
| `vm-small` (CPU) | 16 cores, 32 GB | 1 | 48 h | 4 | 4 | 16 | 0.143 |

- `gpu-a100-small`: the guide gives only the sentence "The `gpu-a100-small` and `vm-small` queues contain virtual nodes with fewer resources (cores) than the nodes in the other queues." GPU count, GPU memory (whole A100 or slice), cores and RAM per virtual node: UNVERIFIED. Check with `sinfo`/`scontrol show node` or a one-hour test before using. Do not plan a 27B job on it.
- One summary of the table gave the `gpu-h100` row as "1 node per user, 4 jobs per user", another as "1 running, 4 submitted"; I used the latter, which matches the other rows' column pattern. Confirm with `qlimits`.
- Charge: SUs = nodes x max(wall hours, 0.25) x rate; charged for resources used; nodes are not shared (LS6 guide, SU Charging Policy). Example given: 4 nodes x 0.25 h x 3 = 3 SU for `gpu-a100`.

### Vista

| | Vista GH | Vista GG |
|---|---|---|
| Nodes | 600 | 256 |
| CPU | NVIDIA Grace (Arm, aarch64), 72 cores, 3.1 GHz | 2x Grace, 144 cores, 3.4 GHz |
| Host memory | 116 GB table / "120 GiB LPDDR5X" in text (guide is inconsistent) | 237 GB LPDDR5 |
| GPU | 1x "NVIDIA H200" with 96 GB HBM3 (text: "GH200 Grace-Hopper Superchip", "96 GiB") | none |
| Local /tmp | 286 GB | 286 GB |
| Interconnect | NDR 400 Gb/s | NDR200 200 Gb/s |

- The guide calls the Vista GPU "H200" in the tables and "GH200" in the prose; HBM is 96 GB in both. One GPU per node, so multi-GPU training means multi-node over NDR.
- Vista is Arm: pip/uv wheels must be `aarch64`. Login nodes are also Grace.

| Vista queue | Node | Max nodes/job | Max time | Max nodes/user | Max jobs/user (submit) | SU/node-h |
|---|---|---|---|---|---|---|
| `gh` | GH | 64 | 48 h | 192 | 20 (40) | 1 |
| `gh-dev` | GH | 8 | 2 h | 8 | 1 (3) | 1 |
| `gg` | GG | 32 | 48 h | 128 | 20 (40) | 0.33 |

## 2. Slurm specifics

Both systems (guides, 2026-10-14):
- `-N` and `-t` required; combine `-N` with `-n` or `--ntasks-per-node`. `-p` selects queue; `-A` if you have more than one allocation.
- `--gres` is not supported, `--gpus-per-task` is not supported, `--mem` is not available: Slurm rejects scripts containing them (Lonestar6 guide: "`--gres` is not supported; Slurm rejects any script containing it"). There is no GPU-count request: you get the whole node, so all 3 A100s (or 2 H100s) are visible. Select a subset inside the job with `CUDA_VISIBLE_DEVICES` (standard CUDA variable; not a TACC feature).
- Avoid `--export`. First line `#!/bin/bash`. All `#SBATCH` lines before commands. Jobs inherit the submission environment.
- Do not request or exclude specific nodes (jobs are deleted; both guides).
- Dependencies: `-d afterok:JOBID`. Arrays: `-a` / `--array` is listed among LS6 sbatch options.
- Lonestar6 is fair-share scheduled, not first-come-first-served.
- Monitoring: `qlimits`, `squeue`, `showq`, `sacct`, `scancel`, `scontrol show job=`.
- No `sbatch` from a compute node (Vista guide, Dependent Jobs: "you can't use `sbatch` to submit a job from a compute node"). Chain jobs from the login node with `--dependency=afterok:` or `afterany:`. Neither the Vista guide nor the Good Conduct guide states what dev queues are for, beyond their limits (gh-dev: 2 h, 1 running and 3 submitted per user). Good Conduct asks for only the time and nodes a job needs. Checked 2026-10-25.

idev (Lonestar6 guide, ML section): `idev -p gpu-a100-dev -N 1 -n 1 -t 1:00:00` (one node), `idev -N 2 -n 2 -p gpu-a100-dev -t 01:00:00` (two nodes). Vista: `idev -p gh-dev -N 1 -n 1 -t 1:00:00`. You may ssh to a node you hold a job on.

Single-node multi-GPU on Lonestar6 (documented):
- `torchrun --nproc_per_node=3 examples/distributed/ddp-tutorial-series/multigpu_torchrun.py 50 10` (PyTorch DDP, 3 GPUs)
- `accelerate launch --num_processes 3 ./finetune.py` (from the TACC "Machine-Learning-on-LS6" repo, https://github.com/skye-glitch/Machine-Learning-on-LS6.git)
- Multi-node: `ibrun -np 2 ./run.sh HOST 2` where `run.sh` runs `torchrun` with `PMI_RANK` as node rank; or `mpirun -np 2 --map-by ppr:1:node multinode.sh` (the guide uses both `ibrun` and `mpirun` for ML; its MPI text says use `ibrun`).

Model split across GPUs for fine-tuning: DDP replicates the full model per GPU, so it does not help a 56 GB model on 40 GB cards. Use `device_map="auto"` (naive pipeline, one process, no `torchrun`) or FSDP/DeepSpeed ZeRO-3 with `--nproc_per_node=3`. Those are from general Hugging Face/PyTorch knowledge, not TACC docs; UNVERIFIED on LS6. 27B bf16 with FSDP across 3x40 GB: weights 56 GB sharded about 19 GB per GPU plus activations and optimizer state of the LoRA params; plausible, untested.

Tensor-parallel vLLM across A100s: `vllm serve MODEL --tensor-parallel-size 2` (or 3 only if head counts divide) is standard vLLM usage; vLLM uses NCCL over PCIe here. Not documented by TACC. With TP=2 on 2x40 GB, 56 GB weights leave roughly 80 x 0.9 - 56 = 16 GB for KV cache and CUDA graphs (arithmetic, UNVERIFIED by measurement). Using `--pipeline-parallel-size 3` is the alternative when 3 does not divide the heads.

Vista multi-node (TACC's pattern): one rank per node launched with `mpirun`/`ibrun`, each calling `torchrun --nproc_per_node=1 --nnodes=$NODES --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$HOST ...` (Vista guide, "Running PyTorch (Multi-node)").

Many small tasks: `module load pylauncher` (LS6 guide, serial-application section: "TACC's pylauncher utility provides an easy way to launch more than one serial application in a single job"; documented for CPU serial apps, GPU use UNVERIFIED). Simplest documented-safe alternative: background three processes in one batch script, each with its own `CUDA_VISIBLE_DEVICES`, then `wait`. Vista has 1 GPU per node, so use job arrays instead.

## 3. File systems and policies

Lonestar6 (guide, 2026-10-14):

| FS | Type | Quota | Purge | Backed up |
|---|---|---|---|---|
| `$HOME` | NFS | 10 GB, 200,000 files | no | yes |
| `$WORK` | Lustre (Stockyard, shared across TACC systems) | 1 TB, 3,000,000 files | no | no |
| `$SCRATCH` | BeeGFS, about 8 PB | no quota | "Files that have not been accessed in ten days are subject to purge" | no |
| `/tmp` (node) | local SSD | 288 GB | purged at job end | no |

- The guide says TACC will not grant increases for `/home` or `/work` and points to Corral. Each account gets 2 TB on Ranch (tape archive).
- Striping: `$SCRATCH` default 4 targets, 512 KB chunks, suggested max 72; `$WORK` default 1 stripe, 1 MB, suggested max 18. Set striping on a directory before writing large files, e.g. `lfs setstripe -c 18 $PWD` (WORK) or `beegfs-ctl --setpattern --numtargets=20 $PWD` (SCRATCH). Existing files cannot be restriped in place; `cp` them. Relevant to 15-60 GB checkpoints.
- Access time updates on compute nodes, not login nodes; manipulating access times to avoid the purge is prohibited. `ls -ul` shows access times.
- Aliases: `cdw` (WORK), `cds` (SCRATCH), `cdh` (HOME).

Vista (guide, 2026-10-14):

| FS | Type | Quota | Purge | Backed up |
|---|---|---|---|---|
| `$HOME` | VAST | 23 GB, 500,000 files | no | yes |
| `$WORK` | Lustre (Stockyard, shared) | 1 TB, 3,000,000 files | no | no |
| `$SCRATCH` | VAST | no quota (about 10 PB total) | files not accessed for 10 days | no |
| `/tmp` | local | 286 GB | cleared at job end | no |

Rules (Good Conduct, https://docs.tacc.utexas.edu/basics/conduct/ , 2026-10-14), both systems:
- Run jobs out of `$SCRATCH`; use `$WORK` to stage data; `$HOME` is not for parallel jobs and a full `$HOME` can make jobs fail.
- "Tens of thousands" of files in a directory is too many; avoid tight open/close loops; do not submit many simultaneous I/O-heavy jobs.
- Transfer: 2-3 concurrent scp sessions is fine; tar up directories of many small files. rsync supported.

Placement for us (both systems, since `$WORK` is shared and the same 1 TB and 3M-file quota applies to both):
- `$WORK/pondie/models/` and `$WORK/hf-cache` (HF_HOME): model weights (8-60 GB each; 1 TB covers about 15 bf16 27B-class checkpoints).
- Copy weights to node-local `/tmp` (288 GB on LS6, 286 GB on Vista) at job start, because `$WORK` is not for high-intensity I/O.
- `$SCRATCH/pondie/`: venv, JSONL data, run outputs. Separate venvs per architecture: `$SCRATCH` is per system, so build one on each. Copy adapters back to `$WORK`; never leave the only copy on `$SCRATCH`.
- Data transfer (https://docs.tacc.utexas.edu/datatransfer/datatransferguide/): scp/sftp/rsync for small data, Globus above 200 GB. Endpoint names: UNVERIFIED.

## 4. Software

### Lonestar6 (x86_64)

- Modules (Lmod): `module reset`, `module spider`, `module avail`. The guide's ML recipe uses `module load python3/3.9.7` (guide text; this is the version in the docs, may be outdated). Python 3.9 is too old for recent vLLM/transformers; run `module spider python3` for newer versions. If none, uv can fetch a standalone Python (`uv python install 3.12`): UNVERIFIED on LS6 (needs outbound HTTPS from the node used). Intel is the default compiler; load a `gcc` module for current GCC.
- CUDA module: not listed in the text I read; the guide's PyTorch wheels (`cu128`) bundle their runtime, so no cuda module is needed for torch. `module spider cuda` for the module name; NVIDIA driver version: UNVERIFIED (run `nvidia-smi` in an idev session and check against the CUDA wheel you choose).
- TACC's own PyTorch recipe (LS6 guide, ML section): in an `idev -p gpu-a100-dev` session, `python3 -m venv $SCRATCH/python-envs/test`, `source .../activate`, `pip3 install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128`. Install venvs on `$SCRATCH`, not `$WORK` (guide). This is an ordinary x86_64 PyTorch wheel, so any standard wheel install works the same way.
- bitsandbytes: Linux x86_64 wheels, standard `pip install bitsandbytes` (docs https://huggingface.co/docs/bitsandbytes/main/en/installation, read 2026-10-14; A100 is sm80, H100 sm90, both supported). No TACC-specific steps found.
- vLLM: standard `uv pip install vllm --torch-backend=auto` (https://raw.githubusercontent.com/vllm-project/vllm/main/docs/getting_started/installation/gpu.cuda.inc.md, read 2026-10-14). Whether the wheel's CUDA build runs on LS6's driver, and whether your pinned version supports the Qwen3.5-family architecture: UNVERIFIED. No TACC-specific vLLM instructions found. Guided JSON backends (xgrammar) ship wheels for x86_64; UNVERIFIED for the pinned version.
- flash-attn: standard wheels on x86_64 exist for common torch/CUDA combinations, but I read no primary source this session; UNVERIFIED. Sources compile slowly and need `nvcc`. Prefer PyTorch SDPA (`attn_implementation="sdpa"`) unless the speedup is needed. A100 (sm80) and H100 are both supported by flash-attn v2 in general (from general knowledge, UNVERIFIED here).
- No TACC-specific wheel or install instruction for any of the above beyond the PyTorch recipe.
- Internet from compute nodes: UNVERIFIED. The recipe runs `pip3 install` from download.pytorch.org inside an idev compute session, implying outbound HTTPS works. Prefetch weights and set `HF_HUB_OFFLINE=1`.
- Containers: Apptainer via the `tacc-apptainer` module (TACC AI/ML docs https://aiml-environments-at-tacc.readthedocs.io/en/latest/getting_starting_section/Containers%20Tutorial.html , secondary TACC-affiliated; secondary: https://life-sciences-ml-at-tacc.readthedocs.io/en/latest/section4/running_containers_on_hpc.html). Use `--nv` to expose GPUs (`apptainer exec --nv image.sif ...`). Apptainer is available on compute nodes only (start an idev first; Life Sciences docs). No root: build elsewhere or pull a Docker image (`apptainer pull docker://...`); set `APPTAINER_CACHEDIR` under `$SCRATCH`. The official LS6 guide text I read contains no Apptainer section; Lonestar6-specific module version UNVERIFIED (`module spider apptainer`). A vLLM-openai Docker image on LS6 is x86_64, no arm64 issue; test with `--nv`: UNVERIFIED.
- Conda: the LS6 menu links to "Automatic Conda Startup"; the text was not in what I read. UNVERIFIED.

### Vista (aarch64)

- Modules: `module reset; module load gcc/15.1.0 python3/3.11.8 cuda` (Vista PyTorch recipe). Check `module spider cuda`.
- TACC's recipe: `python3 -m venv` on a compute node, then `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129`.
- bitsandbytes: docs list Linux aarch64 wheels for CUDA 11.8-12.6 (sm75, sm80, sm90), 12.8 and 13.0-13.2.
- vLLM: wheel pattern `CPU_ARCH=$(uname -m) # x86_64 or aarch64` with CUDA 12.8/12.9/13.0 builds on the install page. Whether the default PyPI wheel is aarch64 and supports Qwen3.5 on your pinned version: UNVERIFIED. Older vLLM docs (v0.6.3) required PyTorch nightly for aarch64+CUDA; probably obsolete.
- flash-attn on aarch64: UNVERIFIED; use SDPA.
- uv: not mentioned in TACC docs; installable into `$HOME/.local/bin` (UNVERIFIED on Vista).
- Containers: Apptainer via `tacc-apptainer`, version 1.4.1 as of December 2025 (Cornell Virtual Workshop, secondary: https://cvw.cac.cornell.edu/vista/user-environment/apptainer). Images must be `linux/arm64`. NGC arm64 images on Vista: UNVERIFIED.
- MPS: documented in the Vista guide (`CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps`, `ibrun -np $SLURM_NNODES nvidia-cuda-mps-control -d`). Probably unnecessary.

## 5. Policies (apply to both unless noted)

- Login nodes are shared: no jobs, no intensive computation. LS6 login nodes are for "file management, compilation, and data movement". Use `idev` or `srun` for interactive work.
- AI tools and agents: "All AI-assisted workloads must be executed on compute nodes only", you are responsible for their processes and their SUs are charged to you (Good Conduct; Vista notice 05/07/2026; the LS6 guide says SUs consumed by an AI process you launch are charged to your allocation and "strongly" recommends running AI-assisted tasks on a compute node). VS Code should run on compute nodes.
- MFA: required for all TACC systems (`ssh user@ls6.tacc.utexas.edu` / `vista.tacc.utexas.edu`, password then token code). Pairing steps not read; UNVERIFIED.
- SSH keys: "Do not run the `ssh-keygen` command" (Vista and Frontera guides; not re-checked on LS6, UNVERIFIED there).
- Node requests: specific nodes (`-w`, hostfiles) get jobs deleted.
- Cron jobs: tied to one login node, not allowed on compute nodes.
- Running services and ports: no TACC rule found. Bind vLLM to `127.0.0.1` and run the client in the same job.
- Allocations: LS6 is for UT System researchers (plus Texas A&M, Texas Tech, UNT partners); you must be added to an allocation to log in. Vista logins require a Vista allocation.
- Support: TACC ticket naming the resource; consulting 8am-5pm CST Mon-Fri (Vista guide).

## 6. Sample scripts

### Lonestar6 setup (on an idev A100 node, not a login node)

```bash
idev -p gpu-a100-dev -N 1 -n 1 -t 1:00:00
module reset; module spider python3          # pick the newest; guide uses python3/3.9.7
module load python3/<newest>
cd $SCRATCH/pondie && python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # TACC's recipe uses cu128
pip install vllm peft bitsandbytes transformers accelerate trl            # pin versions; vLLM may pin its own torch
nvidia-smi; python -c "import torch;print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
HF_HOME=$WORK/hf-cache huggingface-cli download <model-id>
```
If vLLM pins its own torch, install vLLM first (`uv pip install vllm --torch-backend=auto`) and add peft/bitsandbytes after.

### Lonestar6 single-node LoRA (3 GPUs, FSDP or device_map)

```bash
#!/bin/bash
#SBATCH -J lora27b
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -A YOUR_ALLOCATION
#SBATCH -o logs/%x-%j.out
module reset; module load python3/<newest>
source $SCRATCH/pondie/venv/bin/activate
export HF_HOME=$WORK/hf-cache HF_HUB_OFFLINE=1
mkdir -p /tmp/model && cp -r $WORK/pondie/models/qwen-27b /tmp/model/   # 288 GB local /tmp
cd $SCRATCH/pondie/experiments/astra-distill
# Option A: QLoRA, one GPU:  CUDA_VISIBLE_DEVICES=0 python train_lora.py --load-in-4bit ...
# Option B: 27B bf16 LoRA split across the 3 GPUs (FSDP via accelerate; config file placeholder)
accelerate launch --num_processes 3 --config_file fsdp_3gpu.yaml train_lora.py \
    --model /tmp/model/qwen-27b --data $SCRATCH/pondie/data/train.jsonl \
    --out $SCRATCH/pondie/runs/$SLURM_JOB_ID --gradient-checkpointing
mkdir -p $WORK/pondie/runs && cp -r $SCRATCH/pondie/runs/$SLURM_JOB_ID $WORK/pondie/runs/
```
`train_lora.py` and `fsdp_3gpu.yaml` are placeholders. `accelerate launch --num_processes 3` is the form in the TACC ML docs; the FSDP config is mine, UNVERIFIED on LS6. For H100, change to `-p gpu-h100` and `--num_processes 2` (288 SU for 48 h). For three independent 4B runs, start three background processes each with `CUDA_VISIBLE_DEVICES=0/1/2` and `wait`.

### Lonestar6 vLLM server plus client, all GPUs of the node

```bash
#!/bin/bash
#SBATCH -J vllm-pipe
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A YOUR_ALLOCATION
module reset; module load python3/<newest>
source $SCRATCH/pondie/venv/bin/activate
export HF_HUB_OFFLINE=1
MODEL=/tmp/model/qwen-27b
mkdir -p /tmp/model && cp -r $WORK/pondie/models/qwen-27b /tmp/model/
# TP=2 on gpu1+gpu2 (same socket); TP=3 only if head counts are divisible.
# For 3 x 4B servers instead, launch three vllm serve procs on ports 8000-8002, one per CUDA_VISIBLE_DEVICES.
CUDA_VISIBLE_DEVICES=1,2 vllm serve $MODEL --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 2 --max-model-len 16384 --gpu-memory-utilization 0.90 \
    > $SCRATCH/vllm-$SLURM_JOB_ID.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
for i in $(seq 1 120); do curl -sf localhost:8000/health && break; sleep 10; done
python run_clients.py --base-url http://127.0.0.1:8000/v1
kill $SERVER; wait $SERVER 2>/dev/null
```
vLLM flags are standard (from my knowledge); confirm against the pinned version. To use all three GPUs for one 27B server, drop `CUDA_VISIBLE_DEVICES` and use `--pipeline-parallel-size 3` if TP=3 does not divide the head count. Kill by PID, not `pkill -f`. For the 4B server, one GPU is enough and the other two can run more servers. On Vista, none of this applies: a single `vllm serve` on the one 96 GB GPU (see below).

### Vista scripts (kept from the previous version)

One-time setup (idev GH node):
```bash
idev -p gh-dev -N 1 -n 1 -t 1:00:00
module reset; module load gcc/15.1.0 python3/3.11.8
cd $SCRATCH/pondie && python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu129   # TACC's recipe
pip install vllm peft bitsandbytes transformers accelerate trl           # aarch64 UNVERIFIED for vllm
```

Single-node fine-tune (Vista `gh`, one GH200):
```bash
#!/bin/bash
#SBATCH -J lora27b
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -A YOUR_ALLOCATION
#SBATCH -o logs/%x-%j.out
module reset; module load gcc/15.1.0 python3/3.11.8
source $SCRATCH/pondie/venv/bin/activate
export HF_HOME=$WORK/hf-cache HF_HUB_OFFLINE=1
mkdir -p /tmp/model && cp -r $WORK/pondie/models/qwen-27b /tmp/model/
cd $SCRATCH/pondie/experiments/astra-distill
python train_lora.py --model /tmp/model/qwen-27b --data $SCRATCH/pondie/data/train.jsonl \
    --out $SCRATCH/pondie/runs/$SLURM_JOB_ID --gradient-checkpointing
cp -r $SCRATCH/pondie/runs/$SLURM_JOB_ID $WORK/pondie/runs/
```

vLLM server plus client (Vista `gh`): same as the LS6 script but `-p gh`, the Vista module line, `vllm serve $MODEL --host 127.0.0.1 --port 8000 --max-model-len 16384 --gpu-memory-utilization 0.90`, no tensor parallelism.

Multi-node fine-tune on Vista (only if one 96 GB GPU is not enough), TACC's pattern: `#SBATCH -p gh -N 2 -n 2`, `HOST=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)`, `ibrun ./run.sh $HOST $SLURM_NNODES`, where `run.sh` calls `torchrun --nproc_per_node=1 --nnodes=$2 --node_rank=${OMPI_COMM_WORLD_RANK} --master_addr=$1 train.py ...`. TACC's guide uses `mpirun -np 2 --map-by ppr:1:node run.sh HOST 2` inside idev; the `ibrun` form is my adaptation (UNVERIFIED).

## Open items to settle with TACC or by a smoke test
1. Allocations: does the project have both a Vista and a Lonestar6 allocation, and what balances?
2. `gpu-a100-small` hardware (GPUs per virtual node, GPU memory).
3. Does the current vLLM release install on each system and serve the Qwen3.5-family architecture with guided JSON (xgrammar wheels on aarch64 for Vista)?
4. LS6 driver and CUDA module version versus the wheel you choose; newest Python module on LS6.
5. Do compute nodes on either system have outbound internet and reach HF?
6. Real peak memory of 27B bf16 LoRA at your sequence lengths: 96 GB on Vista, 2x80 GB on LS6 H100, 3x40 GB on LS6 A100 (FSDP); and PCIe-only TP throughput.
7. Exact `qlimits` columns on both systems (the LS6 table was read through a summarising fetch tool).

## Sources
- Lonestar6 User Guide, https://docs.tacc.utexas.edu/hpc/lonestar6/ (page dated Sept 18, 2026; read 2026-10-14)
- Vista User Guide, https://docs.tacc.utexas.edu/hpc/vista/ (read 2026-10-14)
- TACC Good Conduct, https://docs.tacc.utexas.edu/basics/conduct/ (2026-10-14)
- TACC Data Transfer guide, https://docs.tacc.utexas.edu/datatransfer/datatransferguide/ (2026-10-14)
- bitsandbytes installation, https://huggingface.co/docs/bitsandbytes/main/en/installation (2026-10-14)
- vLLM install source, https://raw.githubusercontent.com/vllm-project/vllm/main/docs/getting_started/installation/gpu.cuda.inc.md (2026-10-14)
- TACC AI/ML containers tutorial (secondary), https://aiml-environments-at-tacc.readthedocs.io/en/latest/getting_starting_section/Containers%20Tutorial.html
- TACC Life Sciences ML, running containers (secondary), https://life-sciences-ml-at-tacc.readthedocs.io/en/latest/section4/running_containers_on_hpc.html
- Cornell Virtual Workshop, Vista Apptainer (secondary), https://cvw.cac.cornell.edu/vista/user-environment/apptainer
