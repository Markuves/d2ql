import os
import sys
import argparse
import logging
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "experiment": {
        "id": "baseline_random_run",
        "hypothesis": "H1",
        "seed": 42
    },
    "datacenter": {
        "n_cloud_hosts": 4,
        "n_edge_nodes": 0,
    },
    "agent": {
        "precision": "32",
        "hidden_size": 256,
        "n_hidden_layers": 2,
        "learning_rate": 0.0005,
        "gamma": 0.98,
        "target_update_freq": 750,
        "batch_size": 64,
        "replay_buffer_capacity": 100_000,
        "replay_buffer_warmup": 2000,
        "gradient_clip_norm": 10.0,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.995,
        "per_alpha": 0.6,
        "per_beta_start": 0.4,
        "per_beta_end": 1.0,
    },
    "training": {
        "n_episodes": 600,
        "eval_every_n_episodes": 50,
        "checkpoint_dir": "outputs/checkpoints",
        "tensorboard_dir": "outputs/tensorboard",
    },
    "reward": {
        "target_utilization": 0.70,
        "migration_penalty": 0.10,
        "weight_lr": 0.01,
        "w_perf_floor": 0.2,
        "w_energy_floor": 0.1,
        "w_cost_floor": 0.1,
        "w_perf_init": 0.4,
        "w_energy_init": 0.3,
        "w_cost_init": 0.3,
    },
    "workload": {
        "trace_path": "data/workload.csv.gz",
        "episode_length": 50,
        "shuffle": True,
    },
    "queue": {
        "urgency_weight": 0.6,
        "demand_weight": 0.4,
    },
    "py4j": {
        "host": "java-sim",
        "port": 25333,
    },
}


def load_config(config_path: str) -> dict:
    if not config_path:
        logger.warning("No config path provided. Using DEFAULT_CONFIG.")
        return DEFAULT_CONFIG
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file %s not found. Using DEFAULT_CONFIG.", config_path)
        return DEFAULT_CONFIG
    with open(path, "r") as f:
        loaded = yaml.safe_load(f)
    logger.info("Loaded config from %s.", config_path)
    return loaded


def run_training(config: dict) -> None:
    native_cfg = config.get("native_precision") or {}
    bits_list = native_cfg.get("bits")
    if native_cfg.get("enabled") and bits_list:
        from d2ql.precision import h4_capacity_plan, precision_bits

        lr_overrides = native_cfg.get("learning_rate_overrides") or {}
        capacity_cfg = native_cfg.get("capacity") or {}
        hidden_sizes = capacity_cfg.get("hidden_sizes") or [
            config["agent"].get("hidden_size", 256)
        ]
        n_hidden_layers = int(
            capacity_cfg.get("n_hidden_layers", config["agent"].get("n_hidden_layers", 2))
        )
        max_hidden = capacity_cfg.get("max_hidden_size") or {}
        plan = h4_capacity_plan(bits_list, hidden_sizes, max_hidden)

        for name, hidden_size in plan:
            bits = precision_bits(name)
            run_config = deepcopy(config)
            run_config["agent"]["precision"] = name
            run_config["agent"]["hidden_size"] = hidden_size
            run_config["agent"]["n_hidden_layers"] = n_hidden_layers
            if name in lr_overrides:
                run_config["agent"]["learning_rate"] = lr_overrides[name]
            elif bits in lr_overrides:
                run_config["agent"]["learning_rate"] = lr_overrides[bits]
            elif str(bits) in lr_overrides:
                run_config["agent"]["learning_rate"] = lr_overrides[str(bits)]
            tag = f"{name}_h{hidden_size}"
            base_ckpt = Path(config["training"]["checkpoint_dir"])
            run_config["training"]["checkpoint_dir"] = str(base_ckpt / tag)
            experiment_id = config.get("experiment", {}).get("id", "run")
            run_config.setdefault("experiment", {})["id"] = f"{experiment_id}_{tag}"
            logger.info(
                "H4 run: %s (%.2f-bit) | hidden %d x %d",
                name,
                bits,
                n_hidden_layers,
                hidden_size,
            )
            _run_one_training(run_config)
        return
    _run_one_training(config)


def _run_one_training(config: dict) -> None:
    from d2ql.env import CloudSimEnv
    from d2ql.agent import DDQNAgent
    from d2ql.reward import RewardManager
    from d2ql.metrics import MetricsLogger
    from d2ql.workload import AzureTraceLoader

    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    experiment_id = config.get("experiment", {}).get("id", "run")
    log_dir = (
        Path(config["training"].get("tensorboard_dir", "outputs/tensorboard"))
        / experiment_id
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics = MetricsLogger(log_dir=str(log_dir))
    logger.info("TensorBoard log dir: %s", log_dir.resolve())

    # Initialize workload trace loader
    workload_cfg = config.get("workload", {})
    trace_loader = AzureTraceLoader(
        trace_path=workload_cfg.get("trace_path", "data/workload.csv.gz"),
        episode_length=workload_cfg.get("episode_length", 50),
        seed=config["experiment"]["seed"],
    )
    logger.info("Workload summary: %s", trace_loader.summary())

    n_episodes = config["training"]["n_episodes"]
    shuffle = workload_cfg.get("shuffle", True)
    episode_iter = trace_loader.episodes(n_episodes=n_episodes, shuffle=shuffle)

    logger.info("Initializing CloudSimEnv...")
    env = CloudSimEnv(config)

    logger.info("Initializing DDQNAgent...")
    agent = DDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=config
    )

    initial_checkpoint_path = checkpoint_dir / "checkpoint_initial.pt"
    agent.save(str(initial_checkpoint_path))
    logger.info("Initial checkpoint saved to %s.", initial_checkpoint_path)

    resume_path = config.get("_resume_path", "")
    if resume_path:
        logger.info("Resuming from checkpoint: %s", resume_path)
        agent.load(resume_path)

    reward_manager = RewardManager(config)
    eval_every = config["training"]["eval_every_n_episodes"]

    logger.info(
        "Starting training: %d episodes, evaluating every %d.",
        n_episodes, eval_every
    )

    prev_makespan = None
    prev_energy = None
    prev_cost = None

    for episode in range(n_episodes):
        # Sample next episode workload window from the trace
        cloudlets = next(episode_iter)
        obs, reset_info = env.reset(cloudlets=cloudlets)

        terminated = False
        truncated = False
        episode_reward = 0.0
        episode_loss = 0.0
        episode_mean_q = 0.0
        step_count = 0

        while not (terminated or truncated):
            action = agent.select_action(obs)
            next_obs, _, terminated, truncated, info = env.step(action)

            reward = reward_manager.compute_step_reward(
                energy_this_step=info.get("energy", 0.0),
                sla_violations_this_step=info.get("sla_violations", 0.0),
                host_cpu_utilizations=info.get("cpu_utilizations", []),
                did_migrate=info.get("did_migrate", False)
            )

            done_flag = float(terminated or truncated)
            agent.memory.push(obs, action, reward, next_obs, done_flag)

            loss = agent.train_step(episode)
            episode_reward += reward
            episode_loss += loss
            step_count += 1
            obs = next_obs

        agent.decay_epsilon()

        makespan = info.get("makespan", 0.0)
        energy = info.get("energy", 0.0)
        cost = info.get("cost", 0.0)

        delta_perf = (
            0.0 if prev_makespan is None
            else (prev_makespan - makespan) / max(prev_makespan, 1e-6)
        )
        delta_energy = (
            0.0 if prev_energy is None
            else (prev_energy - energy) / max(prev_energy, 1e-6)
        )
        delta_cost = (
            0.0 if prev_cost is None
            else (prev_cost - cost) / max(prev_cost, 1e-6)
        )

        reward_manager.update_weights(
            delta_perf=delta_perf,
            delta_energy=delta_energy,
            delta_cost=delta_cost
        )

        prev_makespan = makespan
        prev_energy = energy
        prev_cost = cost

        avg_loss = episode_loss / max(step_count, 1)
        avg_q = episode_mean_q / max(step_count, 1)
        w = reward_manager.get_current_weights()

        logger.info(
            "Episode %d/%d | Reward: %.4f | Avg Loss: %.6f | "
            "Epsilon: %.4f | Cloudlets: %d | Weights: %s",
            episode + 1, n_episodes,
            episode_reward,
            avg_loss,
            agent.epsilon,
            len(cloudlets),
            w
        )

        metrics.log_episode(
            episode=episode + 1,
            total_reward=episode_reward,
            makespan=makespan,
            energy=energy,
            cost=cost,
            epsilon=agent.epsilon,
        )
        metrics.log_weights(
            episode=episode + 1,
            w_sla=w["w_sla"],
            w_energy=w["w_energy"],
            w_cost=w["w_cost"],
        )
        metrics.log_training(
            step=episode + 1,
            loss=avg_loss,
            mean_q=avg_q,
        )
        metrics.log_buffer(
            episode=episode + 1,
            buffer_size=len(agent.memory),
        )

        if (episode + 1) % eval_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_ep{episode + 1}.pt"
            agent.save(str(checkpoint_path))
            logger.info("Checkpoint saved to %s.", checkpoint_path)

    final_path = checkpoint_dir / "checkpoint_final.pt"
    agent.save(str(final_path))
    logger.info("Training complete. Final checkpoint saved to %s.", final_path)

    metrics.close()
    env.close()


def main():
    parser = argparse.ArgumentParser(description="d2ql Agent Training Orchestrator")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint file to resume training from"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    seed = config["experiment"]["seed"]
    random.seed(seed)

    logger.info(
        "Starting experiment '%s' | Hypothesis: %s | Seed: %d",
        config["experiment"]["id"],
        config["experiment"]["hypothesis"],
        seed
    )

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            logger.error("Checkpoint %s not found. Aborting.", args.resume)
            sys.exit(1)
        config["_resume_path"] = str(resume_path)

    run_training(config)


if __name__ == "__main__":
    main()
