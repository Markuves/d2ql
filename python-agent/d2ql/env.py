import gymnasium as gym
from gymnasium import spaces
import numpy as np

from d2ql.queue import PriorityCloudletQueue


class CloudSimEnv(gym.Env):
    """Gymnasium wrapper around the CloudSimPlus Java simulation via Py4J."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        n_hosts = config["datacenter"]["n_cloud_hosts"]

        # Action: assign the pending cloudlet to one of n_cloud_hosts
        self.action_space = spaces.Discrete(n_hosts)

        # Observation: cpu_util per host + ram_util per host + queue depth
        obs_dim = n_hosts * 2 + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Priority-aware cloudlet dispatch queue
        queue_cfg = config.get("queue", {})
        self.cloudlet_queue = PriorityCloudletQueue(
            urgency_weight=queue_cfg.get("urgency_weight", 0.6),
            demand_weight=queue_cfg.get("demand_weight", 0.4),
        )

        self.gateway = None
        self._connect_gateway()

    # ------------------------------------------------------------------
    # Py4J connection
    # ------------------------------------------------------------------

    def _connect_gateway(self):
        from py4j.java_gateway import JavaGateway, GatewayParameters
        host = self.config.get("py4j", {}).get("host", "java-sim")
        port = self.config.get("py4j", {}).get("port", 25333)
        self.gateway = JavaGateway(
            gateway_parameters=GatewayParameters(host=host, port=port)
        )
        self.sim = self.gateway.entry_point

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Clear the cloudlet queue at episode start
        self.cloudlet_queue = PriorityCloudletQueue(
            urgency_weight=self.config.get("queue", {}).get("urgency_weight", 0.6),
            demand_weight=self.config.get("queue", {}).get("demand_weight", 0.4),
        )

        raw = self.sim.reset()
        obs = self._parse_obs(raw)
        info = {}
        return obs, info

    def step(self, action: int):
        n_hosts = self.config["datacenter"]["n_cloud_hosts"]
        host_idx = int(action) % n_hosts

        # Refresh cloudlet priorities before dispatch
        sim_time = float(self.sim.getSimulationTime()) if hasattr(self.sim, "getSimulationTime") else 0.0
        self.cloudlet_queue.reprioritize(current_time=sim_time)

        # Execute the action in the Java simulator
        raw = self.sim.step(host_idx)
        obs = self._parse_obs(raw)

        # Gather step metrics from Java
        cpu_utilizations = list(self.sim.getHostCpuUtilizations())
        energy = float(self.sim.getTotalEnergyConsumed())
        makespan = float(self.sim.getMakespan())
        cost = float(self.sim.getOperationalCost())
        sla_violations = float(self.sim.getSlaViolationCount())
        did_migrate = bool(self.sim.didMigrateLastStep()) if hasattr(self.sim, "didMigrateLastStep") else False

        terminated = bool(self.sim.isFinished())
        truncated = False

        info = {
            "cpu_utilizations": cpu_utilizations,
            "energy": energy,
            "makespan": makespan,
            "cost": cost,
            "sla_violations": sla_violations,
            "did_migrate": did_migrate,
            "sim_time": sim_time,
        }

        # Reward is computed externally in main.py via RewardManager
        reward = 0.0

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        if self.gateway is not None:
            self.gateway.shutdown()
            self.gateway = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_obs(self, raw) -> np.ndarray:
        """Convert the Java observation array to a numpy float32 vector."""
        n_hosts = self.config["datacenter"]["n_cloud_hosts"]
        try:
            obs = np.array(list(raw), dtype=np.float32)
        except Exception:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        expected = self.observation_space.shape[0]
        if len(obs) < expected:
            obs = np.pad(obs, (0, expected - len(obs)))
        elif len(obs) > expected:
            obs = obs[:expected]

        return np.clip(obs, 0.0, 1.0)
