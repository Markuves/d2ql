# d2ql: Double Deep Q-Network VM Scheduler

`d2ql` is a reinforcement learning research platform designed to train a Double Deep Q-Network (DDQN) agent to optimize multi-objective virtual machine scheduling inside a CloudSimPlus simulation environment. 

The system simultaneously targets three competing objectives:
1. **SLA Compliance** (Quality of Service)
2. **Energy Consumption** (Physical host power profiles)
3. **Operational Cost** (Accrued processing fees)

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

---

## Telemetry and Logging

TensorBoard event files (written to `./outputs/tensorboard/<experiment_id>/<timestamp>/`) carry the per-episode series:

* `episode/total_reward`, `episode/makespan`, `episode/energy`, `episode/cost`, `episode/epsilon`
* `buffer/size` (replay-buffer occupancy)
* `train/loss`, `train/mean_q` (per training update)

Per-run results are also persisted as JSON (`<checkpoint_dir>/result.json`) and appended to `outputs/results/h4_results.csv` (agent) / `h5_results.csv` (heuristics).

```bash
tensorboard --logdir=outputs/tensorboard
```

Known caveats about the current scalar layout (mixed step axes on `train/*`, `train/mean_q` logged as a constant, misleading `best_mean_reward`/`norm_reward` columns) are tracked in [`docs/backlog.md`](docs/backlog.md).

---

## Research Hypotheses Under Study

This project evaluates the following experimental targets:
* **H3: Cross-Workload Generalization:** Evaluates performance degradation when agents are evaluated on scale distributions outside their training scale.
* **H4: Native Bit-Width Training:** Trains Q-networks from scratch at ternary (1.58-bit), 4, 8, 16, 32 bits and bf16 (not PTQ, not QAT), starting from a shared hidden width and scaling capacity only for lower precisions.
* **H5: Heuristic Baselines:** Compares the learned policy against round-robin, random, least-loaded and least-combined scheduling on the same held-out workload and simulator.

Retired hypotheses:
* **H1 (adaptive reward weighting)** — superseded when the reward became energy-only (`r_t = -ΔE_t / E_step_max`); the weight vector is gone from both code and configs.
* **H2 (post-training quantization)** — superseded by H4 native bit-width training; the standalone PTQ module and its config were removed.
