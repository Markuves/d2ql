
import os
import sys
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter

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
        "n_cloud_hosts": 20,
        "n_edge_nodes": 10,
    },
    "agent": {
        "precision": "fp32",
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
    }
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
    """
    Core training loop. Instantiates CloudSimEnv and DDQNAgent,
    then runs the episode loop for the configured number of episodes. 
    """
    from d2ql.env import CloudSimEnv
    from d2ql.agent import DDQNAgent
    from d2ql.reward import RewardManager

    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = config.get("experiment", {}).get("id", "run")
    log_dir = Path(config["training"].get("tensorboard_dir", "outputs/tensorboard")) / experiment_id
    writer = SummaryWriter(log_dir=str(log_dir))

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
    logger.info("Initial agent checkpoint saved to %s.", initial_checkpoint_path)
    writer.add_scalar("run/agent_initialized", 1, 0)
    writer.add_text("run/experiment", experiment_id, 0)
    writer.add_text("run/checkpoint", str(initial_checkpoint_path), 0)
    writer.flush()

    resume_path = config.get("_resume_path", "")
    if resume_path:
        logger.info("Resuming from checkpoint: %s", resume_path)
        agent.load(resume_path)

    reward_manager = RewardManager(config)
    n_episodes = config["training"]["n_episodes"]
    eval_every = config["training"]["eval_every_n_episodes"]

    logger.info(
        "Starting training: %d episodes, evaluating every %d.",
        n_episodes, eval_every
    )

    for episode in range(n_episodes):
        obs, reset_info = env.reset()
        writer.add_scalar(
            "workload/window_cloudlets",
            reset_info.get("window_cloudlets", 0),
            episode + 1
        )
        writer.add_scalar(
            "workload/window_index",
            reset_info.get("window_index", -1),
            episode + 1
        )
        writer.flush()
        terminated = False
        truncated = False
        episode_reward = 0.0
        episode_loss = 0.0
        step_count = 0

        while not (terminated or truncated):
            action = agent.select_action(obs)
            next_obs, _, terminated, truncated, info = env.step(action)

            # Reward is computed by RewardManager using metrics from info dict.
            # info will carry energy, SLA violations, utilizations, and
            # migration flag once the Java gateway populates them. [32]
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

        # Adaptive weight update — deltas are placeholders until
        # the dataset module provides real per-episode metric deltas
        reward_manager.update_weights(
            delta_perf=info.get("delta_perf", 0.0),
            delta_energy=info.get("delta_energy", 0.0),
            delta_cost=info.get("delta_cost", 0.0)
        )

        avg_loss = episode_loss / max(step_count, 1)
        logger.info(
            "Episode %d/%d | Reward: %.4f | Avg Loss: %.6f | Epsilon: %.4f | Weights: %s",
            episode + 1, n_episodes,
            episode_reward,
            avg_loss,
            agent.epsilon,
            reward_manager.get_current_weights()
        )
        writer.add_scalar("training/episode_reward", episode_reward, episode + 1)
        writer.add_scalar("training/average_loss", avg_loss, episode + 1)
        writer.add_scalar("training/epsilon", agent.epsilon, episode + 1)
        writer.add_scalar(
            "workload/window_cloudlets",
            info.get("window_cloudlets", 0),
            episode + 1
        )
        writer.add_scalar(
            "workload/window_index",
            info.get("window_index", -1),
            episode + 1
        )
        writer.add_scalar("agent/selected_vm", info.get("last_action", -1), episode + 1)
        writer.add_scalar("simulation/makespan", info.get("makespan", 0.0), episode + 1)
        writer.add_scalar("simulation/energy", info.get("energy", 0.0), episode + 1)
        writer.add_scalar("simulation/cost", info.get("cost", 0.0), episode + 1)
        writer.add_scalar(
            "simulation/sla_violations",
            info.get("sla_violations", 0),
            episode + 1
        )
        writer.flush()

        # Periodic checkpoint save 
        if (episode + 1) % eval_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_ep{episode + 1}.pt"
            agent.save(str(checkpoint_path))
            logger.info("Checkpoint saved to %s.", checkpoint_path)

    # Final checkpoint at end of training
    final_path = checkpoint_dir / "checkpoint_final.pt"
    agent.save(str(final_path))
    logger.info("Training complete. Final checkpoint saved to %s.", final_path)

    writer.close()
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

    # Seed Python's random — agent.py seeds numpy and torch internally 
    seed = config["experiment"]["seed"]
    random.seed(seed)

    logger.info(
        "Starting experiment '%s' | Hypothesis: %s | Seed: %d",
        config["experiment"]["id"],
        config["experiment"]["hypothesis"],
        seed
    )

    # Resume from checkpoint if provided 
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            logger.error("Checkpoint file %s not found. Aborting.", args.resume)
            sys.exit(1)
        logger.info("Resuming from checkpoint: %s", args.resume)
        # agent.load() is called inside run_training after instantiation —
        # pass resume path via config so run_training can pick it up
        config["_resume_path"] = str(resume_path)

    run_training(config)


if __name__ == "__main__":
    main()
