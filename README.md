# d2ql: Double Deep Q-Network VM Scheduler

`d2ql` is a reinforcement learning research platform designed to train a Double Deep Q-Network (DDQN) agent to schedule virtual machines inside a CloudSimPlus simulation environment.

The learning objective is a single, bounded signal:

* **Energy** — the agent is rewarded with `r_t = -ΔE_t / E_step_max ∈ [-1, 0]`, where the datacenter power follows a convex, heterogeneous per-host model.

Makespan and operational cost are logged alongside the reward as business metrics, but they no longer enter the reward (the SLA axis was retired: trace deadlines are absolute timestamps, so the violation counter was structurally zero — see `docs/backlog.md`).


---

## Architecture Overview

The system operates across two separate containerized runtime environments:
* **`java-sim`**: A Java 21 discrete-event simulation service exposing a CloudSimPlus Gateway via Py4J.
* **`python-agent`**: A Python 3.12 service containing the PyTorch DDQN agent, training loop, and Gymnasium wrapper.

The two environments communicate via a Docker bridge network on port `25333`.

```
                  +--------------------------+
                  |  docker-compose network  |
                  +------------+-------------+
                               |
       (Port 25333)            v             (PyTorch Loop)
+------------------------+  Py4J Socket  +------------------------+
|       java-sim         |<=============>|      python-agent      |
|  (CloudSimPlus Engine) |               |  (Gymnasium Wrapper)   |
+------------------------+               +------------------------+
```

---

## Directory Layout

```
d2ql/
├── configs/               # YAML experiment configurations (H3, H4, H5)
├── data/                  # Workspace for preprocessed workload traces
├── docs/                  # Backlog of improvements + H5 heuristic notes
├── notebooks/             # EDA + results analysis (parquet snapshots in data/)
├── outputs/               # Persistent output directory
│   ├── checkpoints/       # Saved PyTorch checkpoints
│   ├── results/           # Aggregate h4_results.csv / h5_results.csv
│   └── tensorboard/       # Standalone TensorBoard event logs
├── java-sim/              # Java simulation backend code
│   └── src/               # Gateway server source code
├── python-agent/          # RL package code
│   └── d2ql/              # Core modules (env, agent, reward, precision, results)
├── scripts/               # Kernel benchmarks and dataset download
├── tests/                 # pytest suite
├── docker-compose.yml     # Multi-container orchestration config
├── Dockerfile.java        # Multi-stage Java compile and runtime build
└── Dockerfile.python      # Python runtime build utilizing uv package manager
```

---

## Prerequisites

To build and run this project, you only need to install:
* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose](https://docs.docker.com/compose/install/)
* [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

*Note: You do not need Java, Maven, or Python installed locally. Compilation and package resolution occur automatically inside isolated Docker builds.*

*after nvidia toolkit installation to verify docker can see the CUDA device*
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```
---

## Getting Started

### 1. Build and Start the Entire System
To compile the Java gateway, pull the Python dependencies, and run the integration test suite, execute:

```bash
docker compose up --build
```

### 2. Run a Specific Experiment Config
The python orchestrator accepts command-line arguments to load different research parameters. To run a specific configuration, use:

```bash
docker compose run --rm python-agent uv run python main.py --config configs/your_config.yaml
```

### 3. Shutting Down
To safely stop and clean up containers and networks:

```bash
docker compose down
```

### 4. Rebuilding after code changes

The images **bake the source** (`COPY python-agent/d2ql ./d2ql`), so `docker compose up` keeps running
the code that was in the image when it was built. After editing Python/Java code, `pyproject.toml` or
`uv.lock`, rebuild:

```bash
GIT_SHA=$(git rev-parse --short HEAD) docker compose build
docker compose up -d
```

`configs/`, `data/` and `outputs/` are bind-mounted, so config and data changes do **not** need a
rebuild. Symptoms of a stale image: a `KeyError: 'datacenter'` in `d2ql/env.py` (that key was removed
in favour of the fixed 4-host topology) or old line numbers in tracebacks.

---

## Telemetry and Logging

TensorBoard event files are written to `./outputs/tensorboard/<experiment_id>/<timestamp>/` with one monotone x-axis per namespace:

* `episode/*` — `total_reward`, `makespan`, `energy`, `dynamic_energy`, `cost`, `epsilon`, `steps`, `reward_per_step`, `energy_per_step`, `avg_loss`, `mean_q`
* `train/*` — per update on the global step counter: `loss`, `mean_q`, `grad_norm`, `td_error_abs`, `lr`
* `buffer/size`, `stopper/window_mean`, `stopper/best_window_mean`
* `eval/*` and `perf/*` — held-out evaluation and deployment benchmarks, once per run
* `run/es_reason` — why the run stopped (text)

At the end of every run the event file is also **exported to tidy CSV series** (no downsampling, `nan` preserved, one file per namespace) under `outputs/results/tb_series/<experiment_id>/<timestamp>/`, together with a copy of `result.json`. Re-export older runs with:

```bash
docker compose run --rm python-agent python -m d2ql.tb_export \
  --logdir outputs/tensorboard --outdir outputs/results/tb_series
```

Per-run results are persisted as JSON (`<checkpoint_dir>/result.json`) and upserted into `outputs/results/h4_results.csv` (agent) / `h5_results.csv` (heuristics). Open caveats are tracked in [`docs/backlog.md`](docs/backlog.md).

---

## Research Hypotheses Under Study

This project evaluates the following experimental targets:
* **H3: Cross-Workload Generalization:** Evaluates performance degradation when agents are evaluated on scale distributions outside their training scale.
* **H4: Native Bit-Width Training:** Trains Q-networks from scratch at ternary (1.58-bit), 4, 8, 16, 32 bits and bf16 (not PTQ, not QAT), starting from a shared hidden width and scaling capacity only for lower precisions.
* **H5: Heuristic Baselines:** Compares the learned policy against round-robin, random, least-loaded and least-combined scheduling on the same held-out workload and simulator.

Retired hypotheses:
* **H1 (adaptive reward weighting)** — superseded when the reward became energy-only (`r_t = -ΔE_t / E_step_max`); the weight vector is gone from both code and configs.
* **H2 (post-training quantization)** — superseded by H4 native bit-width training; the standalone PTQ module and its config were removed.


---

## Distributed Execution on Google Colab (via Official `google-colab-cli`)

For heavy sweeps such as **H4 Native Bit-Width Training** (66 combinations across precisions, capacities, depths, and seeds), you can run parallel instances in background using Google's official [Google Colab CLI](https://developers.googleblog.com/introducing-the-google-colab-cli/) (`google-colab-cli`).

This partitions the 66 combinations across 2 remote Colab GPU instances (33 runs each) in a fire-and-forget manner:
- **Worker 1:** `--shard-id 0 --num-shards 2` (runs combinations 0 to 32)
- **Worker 2:** `--shard-id 1 --num-shards 2` (runs combinations 33 to 65)

### 1. Install & Authenticate the Colab CLI
On your local terminal (Linux, macOS, or Windows WSL2):
```bash
# Install with uv (recommended) or pip
uv tool install google-colab-cli
# or: pip install google-colab-cli

# Authenticate with your Google account (Colab Pro)
colab auth
```

### 2. Launch Worker 1 (Combinations 0..32)
In terminal 1:
```bash
colab run --gpu T4 -- "git clone -b run_exp_colab <YOUR_REPO_URL> d2ql && cd d2ql && bash scripts/colab_runner.sh --shard-id 0 --num-shards 2"
```

### 3. Launch Worker 2 (Combinations 33..65)
In terminal 2:
```bash
colab run --gpu T4 -- "git clone -b run_exp_colab <YOUR_REPO_URL> d2ql && cd d2ql && bash scripts/colab_runner.sh --shard-id 1 --num-shards 2"
```

### How `colab run` works behind the scenes:
1. Provisions remote Colab VMs with requested GPUs (e.g. `T4` or `A100`).
2. Runs `scripts/colab_runner.sh`, which automatically:
   - Configures OpenJDK 21 and builds `java-sim` fat JAR.
   - Starts the Py4J Gateway in background on port `25333`.
   - Downloads and unzips the Azure VM trace dataset (`data/workload.csv.gz`).
   - Syncs dependencies with `uv` and executes the partitioned slice.
3. Once the 33 runs finish on each machine, the remote Colab instances automatically tear down so no extra compute units are consumed.

*(Optional)* You can also monitor or execute via Colab web notebook using [`notebooks/colab_run_experiments.ipynb`](notebooks/colab_run_experiments.ipynb).
