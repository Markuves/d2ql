import os
import sys
import argparse
import logging
import random
from copy import deepcopy
from datetime import datetime, timezone
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


def _run_heldout_eval(
    env,
    agent,
    reward_manager,
    eval_cloudlets,
) -> dict:
    """Deterministic, epsilon-free evaluation on held-out workload (C1 fix).

    Runs every held-out episode greedily and reports business metrics
    (mean reward, makespan, SLA violations and SLA rate per cloudlet) so the
    precision x capacity comparison uses clean, workload-independent numbers
    instead of the noisy training-time reward sum.
    """
    n_episodes = len(eval_cloudlets)
    if n_episodes == 0:
        return {
            "eval_episodes": 0,
            "eval_mean_reward": float("nan"),
            "eval_makespan": float("nan"),
            "eval_sla_violations": 0.0,
            "eval_sla_rate": float("nan"),
        }

    total_reward = 0.0
    total_cloudlets = 0
    makespans: list[float] = []

    for cloudlets in eval_cloudlets:
        obs, _ = env.reset(cloudlets=cloudlets)
        ep_reward = 0.0
        terminated = False
        truncated = False
        info: dict = {}
        while not (terminated or truncated):
            action = agent.select_action(obs, evaluate=True)  # greedy
            obs, _, terminated, truncated, info = env.step(action)
            ep_reward += reward_manager.compute_step_reward(
                energy_delta=info.get("energy_delta", 0.0),
                sla_violations_this_step=info.get("sla_violations", 0.0),
                host_cpu_utilizations=info.get("cpu_utilizations", []),
            )
        total_reward += ep_reward
        total_cloudlets += len(cloudlets)
        makespans.append(info.get("makespan", 0.0))

    # Distinct SLA violations (A2 fix) across the held-out episodes.
    sla = float(env.sim.getSlaViolationCount()) if hasattr(env.sim, "getSlaViolationCount") else 0.0
    avg_makespan = sum(makespans) / max(len(makespans), 1)
    avg_reward = total_reward / max(n_episodes, 1)
    sla_rate = sla / max(total_cloudlets, 1)

    logger.info(
        "Held-out eval: %d episodes | mean_reward=%.4f | makespan=%.2f | "
        "sla_violations=%.0f | sla_rate=%.4f",
        n_episodes, avg_reward, avg_makespan, sla, sla_rate,
    )
    return {
        "eval_episodes": n_episodes,
        "eval_mean_reward": avg_reward,
        "eval_makespan": avg_makespan,
        "eval_sla_violations": sla,
        "eval_sla_rate": sla_rate,
    }


def run_training(config: dict) -> None:
    native_cfg = config.get("native_precision") or {}
    bits_list = native_cfg.get("bits") or []
    precisions = native_cfg.get("precisions") or []  # [{precision, device?}, ...]
    if native_cfg.get("enabled") and (bits_list or precisions):
        from d2ql.precision import parse_precision, precision_bits, lookup_max_hidden

        lr_overrides = native_cfg.get("learning_rate_overrides") or {}
        capacity_cfg = native_cfg.get("capacity") or {}
        hidden_sizes = capacity_cfg.get("hidden_sizes") or [
            config["agent"].get("hidden_size", 256)
        ]
        hidden_sizes = [int(h) for h in hidden_sizes]
        n_hidden_layers = int(
            capacity_cfg.get("n_hidden_layers", config["agent"].get("n_hidden_layers", 2))
        )
        max_hidden = capacity_cfg.get("max_hidden_size") or {}

        # Plan: list of (precision, device, hidden). Device defaults to "auto"
        # (CUDA if available). FP32 can appear on both GPU and CPU as distinct runs.
        if precisions:
            plan: list[tuple[str, str, int]] = []
            for spec in precisions:
                name = parse_precision(spec["precision"])
                device = str(spec.get("device", "auto")).strip().lower()
                cap = lookup_max_hidden(max_hidden, name)
                for hidden in hidden_sizes:
                    if cap is not None and hidden > cap:
                        continue
                    plan.append((name, device, hidden))
        else:
            plan = []
            for raw in bits_list:
                nm = parse_precision(raw)
                cap = lookup_max_hidden(max_hidden, nm)
                for h in hidden_sizes:
                    if cap is not None and h > cap:
                        continue
                    plan.append((nm, "auto", h))

        for name, device, hidden_size in plan:
            bits = precision_bits(name)
            run_config = deepcopy(config)
            run_config["agent"]["precision"] = name
            run_config["agent"]["hidden_size"] = hidden_size
            run_config["agent"]["n_hidden_layers"] = n_hidden_layers
            run_config["agent"]["device"] = device
            if name in lr_overrides:
                run_config["agent"]["learning_rate"] = lr_overrides[name]
            elif bits in lr_overrides:
                run_config["agent"]["learning_rate"] = lr_overrides[bits]
            elif str(bits) in lr_overrides:
                run_config["agent"]["learning_rate"] = lr_overrides[str(bits)]
            # Tag includes device so fp32-gpu and fp32-cpu are distinct runs.
            tag = f"{name}_{device}_h{hidden_size}"
            base_ckpt = Path(config["training"]["checkpoint_dir"])
            run_config["training"]["checkpoint_dir"] = str(base_ckpt / tag)
            experiment_id = config.get("experiment", {}).get("id", "run")
            run_config.setdefault("experiment", {})["id"] = f"{experiment_id}_{tag}"
            logger.info(
                "H4 run: %s (%.2f-bit, device=%s) | hidden %d x %d",
                name,
                bits,
                device,
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
    from d2ql.early_stopping import EarlyStopper
    from d2ql.results import RunResult, save_run_result
    from d2ql.precision import packed_size_mb, precision_bits, parse_precision

    import time as _time

    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results_cfg = config.get("results") or {}
    results_dir = results_cfg.get("csv_dir", "outputs/results")

    es_cfg = config.get("early_stopping") or {}
    early_stopping_enabled = bool(es_cfg.get("enabled", True))
    if early_stopping_enabled:
        stopper = EarlyStopper(
            patience=int(es_cfg.get("patience", 3)),
            window_size=int(es_cfg.get("window_size", 3)),
            min_evaluations=int(es_cfg.get("min_evaluations", 4)),
            min_delta=float(es_cfg.get("min_delta", 1e-3)),
        )
    else:
        stopper = None

    latency_cfg = config.get("latency") or {}
    latency_enabled = bool(latency_cfg.get("benchmark", True))
    latency_samples = int(latency_cfg.get("n_samples", 200))
    latency_warmup = int(latency_cfg.get("warmup", 20))
    throughput_batch = int(latency_cfg.get("throughput_batch_size", 64))
    throughput_n_batches = int(latency_cfg.get("throughput_n_batches", 200))

    experiment_id = config.get("experiment", {}).get("id", "run")
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_dir = (
        Path(config["training"].get("tensorboard_dir", "outputs/tensorboard"))
        / experiment_id
        / run_stamp
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
        holdout_frac=float(workload_cfg.get("holdout_frac", 0.0)),
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
        "Starting training: %d episodes (early_stopping=%s, eval_every=%d).",
        n_episodes, early_stopping_enabled, eval_every
    )

    wall_start = _time.perf_counter()

    prev_makespan = None
    prev_energy = None
    prev_cost = None

    # Best *deterministic* evaluation (sum of rewards over an eval window).
    best_eval_reward = float("-inf")
    best_eval_episode = 0
    eval_window_reward = 0.0  # accumulated reward since last eval boundary
    episodes_trained = 0
    steps_total = 0  # C2: accumulate steps across episodes for avg_steps
    stopped_early = False
    final_es_reason = ""

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
                energy_delta=info.get("energy_delta", 0.0),
                sla_violations_this_step=info.get("sla_violations", 0.0),
                host_cpu_utilizations=info.get("cpu_utilizations", []),
            )

            done_flag = float(terminated or truncated)
            agent.memory.push(obs, action, reward, next_obs, done_flag)

            loss = agent.train_step(episode)
            episode_reward += reward
            episode_loss += loss
            step_count += 1
            obs = next_obs
            metrics.log_training(
                step=max(agent.total_steps, step_count),
                loss=loss,
                mean_q=0.0,
            )

        agent.decay_epsilon()
        steps_total += step_count  # C2

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

        eval_window_reward += episode_reward
        is_eval_point = (episode + 1) % eval_every == 0

        if is_eval_point:
            # Deterministic best-eval tracking (this window's summed reward).
            if eval_window_reward > best_eval_reward:
                best_eval_reward = eval_window_reward
                best_eval_episode = episode + 1
            if stopper is not None:
                stopper.update(
                    episode=episode + 1,
                    reward=eval_window_reward,
                    extra={
                        "makespan": makespan,
                        "energy": energy,
                        "cost": cost,
                        "epsilon": agent.epsilon,
                        "avg_loss": avg_loss,
                    },
                )
                if stopper.should_stop():
                    stopped_early = True
                    final_es_reason = stopper.result(episode + 1, True).reason
                    logger.info(
                        "EARLY STOP at episode %d/%d: %s",
                        episode + 1, n_episodes, final_es_reason,
                    )
                    break
            eval_window_reward = 0.0

    # If the loop never ran (n_episodes == 0), treat as a 0-length run.
    episodes_trained = episode + 1 if n_episodes > 0 else 0
    es_result = (
        stopper.result(episodes_trained, stopped_early)
        if stopper is not None
        else None
    )
    es_reason = es_result.reason if es_result is not None else "early_stopping disabled"

    # Final checkpoint (whether stopped early or not).
    final_path = checkpoint_dir / "checkpoint_final.pt"
    agent.save(str(final_path))
    if stopped_early:
        logger.info(
            "Run stopped early. Final checkpoint saved to %s.", final_path
        )
    else:
        logger.info(
            "Training complete (full budget). Final checkpoint saved to %s.", final_path
        )

    # C1: deterministic held-out evaluation with business metrics on
    # workload the agent never trained on (unseen episode windows).
    eval_cfg = config.get("evaluation") or {}
    n_eval_episodes = int(eval_cfg.get("n_episodes", 10))
    eval_cloudlets = trace_loader.eval_episodes(n_eval_episodes)
    eval_metrics = _run_heldout_eval(env, agent, reward_manager, eval_cloudlets)

    # Latency benchmark of the trained model (forward pass only).
    latency = {
        "mean_ms": float("nan"),
        "p50_ms": float("nan"),
        "p95_ms": float("nan"),
        "n_samples": 0,
    }
    if latency_enabled:
        latency = agent.benchmark_inference(
            state_dim=env.observation_space.shape[0],
            n_samples=latency_samples,
            warmup=latency_warmup,
        )

    # Throughput-per-batch (the regime where low-bit kernels pay off). Runs the
    # real deploy kernel on a batched forward and reports samples/sec.
    throughput = {"samples_per_sec": float("nan"), "batch_size": throughput_batch, "n_batches": throughput_n_batches}
    if latency_enabled:
        throughput = agent.benchmark_throughput(
            state_dim=env.observation_space.shape[0],
            batch_size=throughput_batch,
            n_batches=throughput_n_batches,
        )

    wall_clock_s = _time.perf_counter() - wall_start

    # B2 / C2: compute-cost and per-step-normalized reward axes for the Pareto.
    avg_steps = steps_total / max(episodes_trained, 1)
    best_mean_reward = (
        es_result.best_window_reward
        if es_result is not None and es_result.best_window_reward is not None
        else best_eval_reward
    )
    if not isinstance(best_mean_reward, (int, float)) or best_mean_reward in (float("-inf"), float("inf")):
        best_mean_reward = 0.0
    norm_reward = best_mean_reward / max(avg_steps, 1e-6)

    # Persist comparable per-combination results.
    precision_name = parse_precision(config["agent"].get("precision", "32"))
    params = agent.param_count()
    result = RunResult(
        experiment_id=experiment_id,
        precision=precision_name,
        bits=precision_bits(precision_name),
        hidden_size=int(config["agent"].get("hidden_size", 256)),
        n_hidden_layers=int(config["agent"].get("n_hidden_layers", 2)),
        episodes_trained=episodes_trained,
        stopped_early=stopped_early,
        best_episode=es_result.best_episode if es_result is not None else best_eval_episode,
        best_mean_reward=best_mean_reward,
        best_eval_reward=(
            best_eval_reward if best_eval_reward != float("-inf") else 0.0
        ),
        best_eval_episode=best_eval_episode,
        latency_mean_ms=latency["mean_ms"],
        latency_p50_ms=latency["p50_ms"],
        latency_p95_ms=latency["p95_ms"],
        latency_n_samples=latency["n_samples"],
        throughput_pps=throughput["samples_per_sec"],
        throughput_batch_size=throughput["batch_size"],
        params=params,
        packed_size_mb=packed_size_mb(params, precision_name),
        flops=agent.flops(),
        effective_capacity_bits=agent.effective_capacity_bits(),
        avg_steps=avg_steps,
        norm_reward=norm_reward,
        eval_episodes=eval_metrics["eval_episodes"],
        eval_mean_reward=eval_metrics["eval_mean_reward"],
        eval_makespan=eval_metrics["eval_makespan"],
        eval_sla_violations=eval_metrics["eval_sla_violations"],
        eval_sla_rate=eval_metrics["eval_sla_rate"],
        wall_clock_s=wall_clock_s,
        device=str(agent.device),
        seed=int(config["experiment"]["seed"]),
        notes=es_reason,
        extra={
            "eval_every": eval_every,
            "n_episodes_budget": n_episodes,
        },
    )
    save_run_result(
        result,
        checkpoint_dir=str(checkpoint_dir),
        results_dir=results_dir,
    )

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
