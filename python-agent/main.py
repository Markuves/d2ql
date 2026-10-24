import argparse
import logging
import random
import sys
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
        "id": "baseline_run",
        "hypothesis": "none",
        "seed": 42
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
    "workload": {
        "trace_path": "data/workload.csv.gz",
        "episode_length": 50,
        "shuffle": True,
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

    Runs every held-out episode greedily and reports business metrics (mean
    reward, makespan, average episode length and reward per step) so the
    precision x capacity comparison uses clean, workload-independent numbers
    instead of the noisy training-time window sum.

    The SLA axis is gone: trace deadlines are absolute timestamps compared
    against a clock of ~170 simulated seconds, so the violation counter was
    structurally zero (docs/backlog.md, SIM-01).
    """
    n_episodes = len(eval_cloudlets)
    if n_episodes == 0:
        return {
            "eval_episodes": 0,
            "eval_mean_reward": float("nan"),
            "eval_makespan": float("nan"),
            "eval_avg_steps": float("nan"),
            "eval_norm_reward": float("nan"),
        }

    total_reward = 0.0
    total_steps = 0
    makespans: list[float] = []

    for cloudlets in eval_cloudlets:
        obs, _ = env.reset(cloudlets=cloudlets)
        ep_reward = 0.0
        step_count = 0
        terminated = False
        truncated = False
        info: dict = {}
        while not (terminated or truncated):
            action = agent.select_action(obs, evaluate=True)  # greedy
            obs, _, terminated, truncated, info = env.step(action)
            ep_reward += reward_manager.compute_step_reward(info.get("energy_delta", 0.0))
            step_count += 1
        total_reward += ep_reward
        total_steps += step_count
        makespans.append(float(info.get("makespan", 0.0)))

    avg_makespan = sum(makespans) / max(len(makespans), 1)
    avg_reward = total_reward / max(n_episodes, 1)
    avg_steps = total_steps / max(n_episodes, 1)
    # Reward per step: same units as a single step reward, i.e. in [-1, 0].
    norm_reward = avg_reward / avg_steps if avg_steps > 0 else float("nan")

    logger.info(
        "Held-out eval: %d episodes | mean_reward=%.4f | norm_reward=%.4f | "
        "steps/ep=%.1f | makespan=%.2f",
        n_episodes, avg_reward, norm_reward, avg_steps, avg_makespan,
    )
    return {
        "eval_episodes": n_episodes,
        "eval_mean_reward": avg_reward,
        "eval_makespan": avg_makespan,
        "eval_avg_steps": avg_steps,
        "eval_norm_reward": norm_reward,
    }


def run_heuristics(config: dict) -> None:
    """H5: evaluate traditional load-balancing heuristics (no learning)."""
    h5 = config.get("heuristics") or {}
    if not h5.get("enabled"):
        return
    policies = h5.get("policies") or [
        "round_robin",
        "random",
        "least_loaded",
        "least_combined",
    ]
    logger.info(
        "H5 heuristics sweep: %d policies -> %s",
        len(policies),
        ", ".join(str(p) for p in policies),
    )
    for policy_name in policies:
        _run_one_heuristic(config, str(policy_name))


def _run_one_heuristic(config: dict, policy_name: str) -> None:
    """Evaluate a single heuristic on the SAME held-out workload as H4.

    Uses the identical CloudSim env, trace loader settings and RewardManager as
    the DDQN runs, so makespan / energy / cost / reward are directly comparable
    with the H4 agent rows.
    """
    import time as _time

    from d2ql.env import N_HOSTS, CloudSimEnv
    from d2ql.heuristics import build_heuristic
    from d2ql.metrics import MetricsLogger
    from d2ql.results import HeuristicResult, save_heuristic_result
    from d2ql.reward import RewardManager
    from d2ql.workload import AzureTraceLoader

    training_cfg = config.get("training") or {}
    base_ckpt = Path(training_cfg.get("checkpoint_dir", "outputs/checkpoints/h5"))
    checkpoint_dir = base_ckpt / policy_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = (config.get("results") or {}).get("csv_dir", "outputs/results")

    seed = int(config["experiment"]["seed"])
    h5 = config.get("heuristics") or {}
    h5_seed = int(h5.get("seed", seed))
    experiment_id = config.get("experiment", {}).get("id", "h5")

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_dir = (
        Path(training_cfg.get("tensorboard_dir", "outputs/tensorboard"))
        / f"{experiment_id}_{policy_name}"
        / run_stamp
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics = MetricsLogger(log_dir=str(log_dir))

    workload_cfg = config.get("workload", {})
    trace_loader = AzureTraceLoader(
        trace_path=workload_cfg.get("trace_path", "data/workload.csv.gz"),
        episode_length=workload_cfg.get("episode_length", 50),
        seed=seed,
        holdout_frac=float(workload_cfg.get("holdout_frac", 0.0)),
        mi_scale=workload_cfg.get("mi_scale"),
        simulation_time_scale=workload_cfg.get("simulation_time_scale"),
    )
    logger.info(
        "H5 heuristic '%s' | workload summary: %s", policy_name, trace_loader.summary()
    )

    n_hosts = N_HOSTS
    policy = build_heuristic(policy_name, n_hosts=n_hosts, seed=h5_seed)

    n_eval = int((config.get("evaluation") or {}).get("n_episodes", 10))
    eval_cloudlets = trace_loader.eval_episodes(n_eval)

    logger.info("Initializing CloudSimEnv for heuristic '%s'...", policy_name)
    env = CloudSimEnv(config)
    reward_manager = RewardManager()

    total_reward = 0.0
    makespans: list[float] = []
    energies: list[float] = []
    costs: list[float] = []
    decision_times: list[float] = []

    wall_start = _time.perf_counter()
    for cloudlets in eval_cloudlets:
        obs, _ = env.reset(cloudlets=cloudlets)
        policy.reset()
        ep_reward = 0.0
        terminated = False
        truncated = False
        info: dict = {}
        while not (terminated or truncated):
            t0 = _time.perf_counter()
            action = policy.select_action(obs)
            decision_times.append((_time.perf_counter() - t0) * 1000.0)
            obs, _, terminated, truncated, info = env.step(action)
            ep_reward += reward_manager.compute_step_reward(info.get("energy_delta", 0.0))

        mk = float(info.get("makespan", 0.0))
        en = float(info.get("energy", 0.0))
        co = float(info.get("cost", 0.0))
        total_reward += ep_reward
        makespans.append(mk)
        energies.append(en)
        costs.append(co)
        metrics.log_episode(
            episode=len(makespans),
            total_reward=ep_reward,
            makespan=mk,
            energy=en,
            cost=co,
            epsilon=0.0,
        )

    wall_clock_s = _time.perf_counter() - wall_start

    n_eval_eps = len(eval_cloudlets)
    avg_reward = total_reward / max(n_eval_eps, 1)
    avg_makespan = sum(makespans) / max(len(makespans), 1)
    mean_energy = sum(energies) / max(len(energies), 1)
    mean_cost = sum(costs) / max(len(costs), 1)
    dt = np.asarray(decision_times, dtype=np.float64)
    lat_mean = float(dt.mean()) if dt.size else float("nan")
    lat_p50 = float(np.percentile(dt, 50)) if dt.size else float("nan")
    lat_p95 = float(np.percentile(dt, 95)) if dt.size else float("nan")

    logger.info(
        "H5 '%s' eval: mean_reward=%.4f | makespan=%.2f | energy=%.2f | "
        "cost=%.4f | decision_lat=%.4f ms",
        policy_name,
        avg_reward,
        avg_makespan,
        mean_energy,
        mean_cost,
        lat_mean,
    )

    result = HeuristicResult(
        experiment_id=f"{experiment_id}_{policy_name}",
        policy=policy_name,
        n_hosts=n_hosts,
        eval_episodes=n_eval_eps,
        eval_mean_reward=avg_reward,
        eval_makespan=avg_makespan,
        mean_energy=mean_energy,
        mean_cost=mean_cost,
        decision_latency_mean_ms=lat_mean,
        decision_latency_p50_ms=lat_p50,
        decision_latency_p95_ms=lat_p95,
        wall_clock_s=wall_clock_s,
        seed=seed,
        extra={
            "mi_scale": trace_loader.sim_time_scale,
            "simulation_time_scale": trace_loader.sim_time_scale,
        },
    )
    save_heuristic_result(
        result, checkpoint_dir=str(checkpoint_dir), results_dir=results_dir
    )

    metrics.close()
    env.close()


def _experiment_seeds(config: dict) -> list[int]:
    """Seeds each run is replicated with (P2).

    Reads ``experiment.seeds`` (a list) when present, else falls back to the
    single ``experiment.seed``. Replicates are what make reward differences
    readable against run-to-run noise: with one seed and 10 eval episodes the
    spread inside a single precision (~sigma 3) is the same size as the effect
    being measured, so no "optimum" can be claimed from a single seed.
    """
    exp = config.get("experiment", {}) or {}
    seeds = exp.get("seeds")
    if not seeds:
        seeds = [exp.get("seed", 42)]
    return [int(s) for s in seeds]


def _resolve_layer_counts(capacity_cfg: dict, agent_cfg: dict) -> list[int]:
    """Depth values swept by H4: ``n_hidden_layers`` may be an int or a vector.

    Returns the sorted, de-duplicated set of valid depths (>= 1), so a config can
    ask for ``[1, 2]`` and every depth is enumerated by the plan.
    """
    raw = capacity_cfg.get("n_hidden_layers", agent_cfg.get("n_hidden_layers", 2))
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    counts = sorted({int(v) for v in values if int(v) >= 1})
    if not counts:
        raise ValueError(
            "native_precision.capacity.n_hidden_layers must contain a value >= 1"
        )
    return counts


def _build_native_plan(
    precisions: list,
    bits_list: list,
    hidden_sizes: list[int],
    layer_counts: list[int],
    max_hidden: dict,
) -> list[tuple[str, str, int, int]]:
    """Expand the native sweep into (precision, device, hidden, n_layers) tuples.

    Device defaults to "auto" (CUDA if available); fp32 may appear on both GPU and
    CPU as distinct runs. Widths above ``max_hidden_size`` for a precision are
    dropped, and every (width, depth) combination is enumerated.
    """
    from d2ql.precision import lookup_max_hidden, parse_precision

    if precisions:
        specs = [
            (parse_precision(s["precision"]), str(s.get("device", "auto")).strip().lower())
            for s in precisions
        ]
    else:
        specs = [(parse_precision(raw), "auto") for raw in bits_list]

    plan: list[tuple[str, str, int, int]] = []
    for name, device in specs:
        cap = lookup_max_hidden(max_hidden, name)
        for hidden in hidden_sizes:
            if cap is not None and hidden > cap:
                continue
            for layers in layer_counts:
                plan.append((name, device, hidden, layers))
    return plan


def run_training(config: dict) -> None:
    # H5 dispatch: traditional load-balancing heuristics (no learning).
    if (config.get("heuristics") or {}).get("enabled"):
        run_heuristics(config)
        return
    seeds = _experiment_seeds(config)
    native_cfg = config.get("native_precision") or {}
    bits_list = native_cfg.get("bits") or []
    precisions = native_cfg.get("precisions") or []  # [{precision, device?}, ...]
    if native_cfg.get("enabled") and (bits_list or precisions):
        from d2ql.precision import precision_bits

        lr_overrides = native_cfg.get("learning_rate_overrides") or {}
        capacity_cfg = native_cfg.get("capacity") or {}
        hidden_sizes = capacity_cfg.get("hidden_sizes") or [
            config["agent"].get("hidden_size", 256)
        ]
        hidden_sizes = [int(h) for h in hidden_sizes]
        # H4 also sweeps depth: `n_hidden_layers` may be an int or a vector.
        layer_counts = _resolve_layer_counts(capacity_cfg, config.get("agent", {}))
        max_hidden = capacity_cfg.get("max_hidden_size") or {}

        plan = _build_native_plan(
            precisions, bits_list, hidden_sizes, layer_counts, max_hidden
        )
        logger.info(
            "H4 plan: %d combinaciones (precisión, device, ancho, profundidad): "
            "%d precisiones × %d anchos × %d profundidades.",
            len(plan),
            len(precisions) or len(bits_list),
            len(hidden_sizes),
            len(layer_counts),
        )

        for seed in seeds:
            for name, device, hidden_size, n_layers in plan:
                bits = precision_bits(name)
                run_config = deepcopy(config)
                run_config["agent"]["precision"] = name
                run_config["agent"]["hidden_size"] = hidden_size
                run_config["agent"]["n_hidden_layers"] = n_layers
                run_config["agent"]["device"] = device
                # P2: one replicate per seed. The seed is pushed into the agent,
                # the trace loader and the Java gateway (env.reset -> setSeed).
                run_config.setdefault("experiment", {})["seed"] = seed
                if name in lr_overrides:
                    run_config["agent"]["learning_rate"] = lr_overrides[name]
                elif bits in lr_overrides:
                    run_config["agent"]["learning_rate"] = lr_overrides[bits]
                elif str(bits) in lr_overrides:
                    run_config["agent"]["learning_rate"] = lr_overrides[str(bits)]
                # Tag includes device, depth and seed so every run is distinct.
                tag = f"{name}_{device}_h{hidden_size}_l{n_layers}_s{seed}"
                base_ckpt = Path(config["training"]["checkpoint_dir"])
                run_config["training"]["checkpoint_dir"] = str(base_ckpt / tag)
                experiment_id = config.get("experiment", {}).get("id", "run")
                run_config["experiment"]["id"] = f"{experiment_id}_{tag}"
                logger.info(
                    "H4 run: %s (%.2f-bit, device=%s) | hidden %d x %d capas | seed %d",
                    name,
                    bits,
                    device,
                    hidden_size,
                    n_layers,
                    seed,
                )
                _run_one_training(run_config)
        return

    # Non-native single-plan path: still replicate the run across seeds (P2).
    for seed in seeds:
        run_config = deepcopy(config)
        run_config.setdefault("experiment", {})["seed"] = seed
        if len(seeds) > 1:
            base_ckpt = Path(config["training"]["checkpoint_dir"])
            run_config["training"]["checkpoint_dir"] = str(base_ckpt / f"seed_{seed}")
            experiment_id = config.get("experiment", {}).get("id", "run")
            run_config["experiment"]["id"] = f"{experiment_id}_s{seed}"
        _run_one_training(run_config)


def _run_one_training(config: dict) -> None:
    import time as _time

    from d2ql.agent import DDQNAgent
    from d2ql.early_stopping import EarlyStopper
    from d2ql.env import CloudSimEnv
    from d2ql.metrics import MetricsLogger
    from d2ql.precision import packed_size_mb, parse_precision, precision_bits
    from d2ql.results import RunResult, save_run_result
    from d2ql.reward import RewardManager
    from d2ql.workload import AzureTraceLoader

    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results_cfg = config.get("results") or {}
    results_dir = results_cfg.get("csv_dir", "outputs/results")

    es_cfg = config.get("early_stopping") or {}
    early_stopping_enabled = bool(es_cfg.get("enabled", True))
    if early_stopping_enabled:
        stopper = EarlyStopper(
            patience=int(es_cfg.get("patience", 20)),
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
    # P1: batch sizes swept for the throughput-vs-batch curve. Falls back to the
    # single primary batch so old configs keep working.
    throughput_batch_sizes = latency_cfg.get("throughput_batch_sizes")
    if not throughput_batch_sizes:
        throughput_batch_sizes = [throughput_batch]
    throughput_batch_sizes = [int(b) for b in throughput_batch_sizes]

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
        mi_scale=workload_cfg.get("mi_scale"),
        simulation_time_scale=workload_cfg.get("simulation_time_scale"),
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

    reward_manager = RewardManager()
    eval_every = config["training"]["eval_every_n_episodes"]

    logger.info(
        "Starting training: %d episodes (early_stopping=%s, eval_every=%d).",
        n_episodes, early_stopping_enabled, eval_every
    )

    wall_start = _time.perf_counter()

    # Best evaluation window: **mean** per-episode reward over the trailing
    # window of `eval_every` episodes (CSV-01). The window sum is divided by its
    # length, so the value is on the same scale as one episode's reward and
    # `norm_reward` is a genuine per-step reward in [-1, 0].
    eval_window_reward = 0.0  # accumulated reward since last eval boundary
    eval_window_episodes = 0
    best_window_mean = float("-inf")
    best_window_episode = 0
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

            reward = reward_manager.compute_step_reward(info.get("energy_delta", 0.0))

            done_flag = float(terminated or truncated)
            agent.memory.push(obs, action, reward, next_obs, done_flag)

            loss, mean_q = agent.train_step(episode)
            episode_reward += reward
            episode_loss += loss
            episode_mean_q += mean_q
            step_count += 1
            obs = next_obs
            # LOG-01: a single series on a single x-axis (the global update
            # counter). During buffer warm-up there is no update, so nothing is
            # logged and the axis never goes backwards.
            if agent.total_steps > 0:
                metrics.log_training(step=agent.total_steps, loss=loss, mean_q=mean_q)

        agent.decay_epsilon()
        steps_total += step_count  # C2

        makespan = info.get("makespan", 0.0)
        energy = info.get("energy", 0.0)
        cost = info.get("cost", 0.0)

        avg_loss = episode_loss / max(step_count, 1)
        avg_q = episode_mean_q / max(step_count, 1)

        logger.info(
            "Episode %d/%d | Reward: %.4f | Avg Loss: %.6f | Mean Q: %.4f | "
            "Epsilon: %.4f | Cloudlets: %d",
            episode + 1, n_episodes,
            episode_reward,
            avg_loss,
            avg_q,
            agent.epsilon,
            len(cloudlets),
        )

        metrics.log_episode(
            episode=episode + 1,
            total_reward=episode_reward,
            makespan=makespan,
            energy=energy,
            cost=cost,
            epsilon=agent.epsilon,
        )
        # Episode-level diagnostics live under episode/*, never in train/*.
        metrics.log_episode_diagnostics(
            episode=episode + 1, avg_loss=avg_loss, mean_q=avg_q
        )
        metrics.log_buffer(
            episode=episode + 1,
            buffer_size=len(agent.memory),
        )

        eval_window_reward += episode_reward
        eval_window_episodes += 1
        is_eval_point = (episode + 1) % eval_every == 0

        if is_eval_point:
            window_mean = eval_window_reward / max(eval_window_episodes, 1)
            if window_mean > best_window_mean:
                best_window_mean = window_mean
                best_window_episode = episode + 1
            if stopper is not None:
                stopper.update(
                    episode=episode + 1,
                    reward=window_mean,
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
            eval_window_episodes = 0

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

    # Throughput-per-batch (the regime where low-bit kernels pay off). P1: sweep
    # several batch sizes so the throughput-vs-batch curve is reported, not just
    # a single point; `throughput_batch_size` stays the "primary" batch.
    throughput = {"samples_per_sec": float("nan"), "batch_size": throughput_batch, "n_batches": throughput_n_batches}
    throughput_by_batch: dict = {}
    if latency_enabled:
        throughput_by_batch = agent.benchmark_throughput_sweep(
            state_dim=env.observation_space.shape[0],
            batch_sizes=throughput_batch_sizes,
            n_batches=throughput_n_batches,
            warmup=latency_warmup,
        )
        primary = throughput_by_batch.get(str(throughput_batch))
        if primary is None and throughput_by_batch:
            # Fall back to the largest measured batch for the scalar column.
            throughput_batch = max(int(b) for b in throughput_by_batch)
            primary = throughput_by_batch[str(throughput_batch)]
        if primary is not None:
            throughput = {
                "samples_per_sec": primary["samples_per_sec"],
                "batch_size": throughput_batch,
                "n_batches": throughput_n_batches,
            }

    wall_clock_s = _time.perf_counter() - wall_start

    # B2 / C2 / CSV-01: compute-cost and per-step-normalized reward axes.
    avg_steps = steps_total / max(episodes_trained, 1)
    # Mean per-episode reward over the best trailing window (NOT a sum).
    best_mean_reward = (
        es_result.best_window_reward
        if es_result is not None and es_result.best_window_reward is not None
        else best_window_mean
    )
    if not np.isfinite(best_mean_reward):
        # No evaluation ever happened (n_episodes < eval_every): unknown, not 0.
        best_mean_reward = float("nan")
    # Reward per step: same units as one step reward, bounded in [-1, 0].
    norm_reward = (
        best_mean_reward / avg_steps
        if avg_steps > 0 and np.isfinite(best_mean_reward)
        else float("nan")
    )

    # LOG-03: the held-out evaluation and the deployment benchmarks were only
    # persisted to JSON/CSV; surface them in TensorBoard too (x = episode count).
    metrics.log_eval(
        episodes_trained,
        {
            "mean_reward": eval_metrics["eval_mean_reward"],
            "norm_reward": eval_metrics["eval_norm_reward"],
            "makespan": eval_metrics["eval_makespan"],
            "avg_steps": eval_metrics["eval_avg_steps"],
        },
    )
    metrics.log_perf(
        episodes_trained,
        {
            "latency_mean_ms": latency["mean_ms"],
            "latency_p95_ms": latency["p95_ms"],
            "throughput_pps": throughput["samples_per_sec"],
        },
    )

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
        best_episode=es_result.best_episode if es_result is not None else best_window_episode,
        best_mean_reward=best_mean_reward,
        latency_mean_ms=latency["mean_ms"],
        latency_p50_ms=latency["p50_ms"],
        latency_p95_ms=latency["p95_ms"],
        latency_n_samples=latency["n_samples"],
        throughput_pps=throughput["samples_per_sec"],
        throughput_batch_size=throughput["batch_size"],
        throughput_by_batch=throughput_by_batch,
        params=params,
        packed_size_mb=packed_size_mb(params, precision_name),
        flops=agent.flops(),
        effective_capacity_bits=agent.effective_capacity_bits(),
        avg_steps=avg_steps,
        norm_reward=norm_reward,
        eval_episodes=eval_metrics["eval_episodes"],
        eval_mean_reward=eval_metrics["eval_mean_reward"],
        eval_avg_steps=eval_metrics["eval_avg_steps"],
        eval_norm_reward=eval_metrics["eval_norm_reward"],
        eval_makespan=eval_metrics["eval_makespan"],
        wall_clock_s=wall_clock_s,
        device=str(agent.device),
        seed=int(config["experiment"]["seed"]),
        extra={
            "eval_every": eval_every,
            "n_episodes_budget": n_episodes,
            "es_reason": es_reason,
            "simulation_time_scale": trace_loader.sim_time_scale,
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
