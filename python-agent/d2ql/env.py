import os
import time 
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from py4j.java_gateway import JavaGateway, GatewayParameters

class CloudSimEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Connect to the Java Gateway container
        java_host = os.getenv("JAVA_HOST", "localhost")
        java_port = int(os.getenv("JAVA_PORT", 25333))

        # Retry loop: Java container may still be starting
        max_retries = 10
        for attempt in range(1, max_retries + 1):
            try:
                self.gateway = JavaGateway(
                    gateway_parameters=GatewayParameters(address=java_host, port=java_port)
                )
                # Probe the connection before proceeding
                self.gateway.entry_point.getObservation()
                break
            except Exception:
                if attempt == max_retries:
                    raise RuntimeError(f"Could not connect to java gateway at {java_host}:{java_port} after {max_retries} attempts.")    
                time.sleep(3)
        
        
        # Space specs
        # Java currently exposes three aggregate metrics.
        self.state_dim = 3
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )
        # VM allocation actions + special actions
        self.total_vms = self.config["datacenter"]["n_cloud_hosts"] * 4 # example VM count mapping
        self.action_space = spaces.Discrete(self.total_vms + 3)

        self.java_entry=None
        self.metadata={"render_modes":[]}



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Call the structured gateway reset, not createSimulation()
        self.java_entry = self.gateway.entry_point
        self.java_entry.reset()

        # Fetch the real initial observation from the simulator
        obs_java = self.java_entry.getObservation()  # returns double[]
        obs = np.array(list(obs_java), dtype=np.float32)

        info = {
            "window_index": self.java_entry.getWindowIndex(),
            "window_start": self.java_entry.getWindowStart(),
            "window_end": self.java_entry.getWindowEnd(),
            "window_cloudlets": self.java_entry.getLoadedCloudlets(),
            "last_action": self.java_entry.getLastAction(),
        }
        return obs, info



    def step(self, action: int):
        # 1. Send action to simulator and advance the clock
        self.java_entry.step(action)

        # 2. Retrieve updated observation
        obs_java = self.java_entry.getObservation()
        obs = np.array(list(obs_java), dtype=np.float32)

        # 3. Check termination
        terminated = bool(self.java_entry.isDone())
        truncated = False

        # 4. Reward is computed by RewardManager in main.py using
        #    metrics returned from the simulator — placeholder for now
        reward = 0.0
        info = {
            "window_index": self.java_entry.getWindowIndex(),
            "window_cloudlets": self.java_entry.getLoadedCloudlets(),
            "last_action": self.java_entry.getLastAction(),
            "makespan": self.java_entry.getMakespan(),
            "energy": self.java_entry.getEnergy(),
            "cost": self.java_entry.getCost(),
            "sla_violations": self.java_entry.getSlaViolations(),
        }

        return obs, reward, terminated, truncated, info


    def close(self):
        self.gateway.close()