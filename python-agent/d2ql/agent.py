import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

print(torch.cuda.is_available()) 

class QNetwork(nn.Module):
    """Identical neural network template for Online and Target Q-Networks."""
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (PER) with Proportional Prioritization."""
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size: int, beta: float):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]
            
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Compute Importance Sampling weights to correct bias
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max() # Normalize weights
        
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(weights, dtype=np.float32),
            indices
        )
        
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = max(prio, 1e-6) # Ensure non-zero priority

    def __len__(self) -> int:
        return len(self.buffer)


class DDQNAgent:
    def __init__(self, state_dim: int, action_dim: int, config: dict):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.gamma = config["agent"]["gamma"] # default 0.98
        self.batch_size = config["agent"]["batch_size"] # default 64
        self.warmup = config["agent"]["replay_buffer_warmup"] # default 2000
        self.grad_clip = config["agent"]["gradient_clip_norm"] # default 10.0
        self.target_update_freq = config["agent"]["target_update_freq"] # default 750
        
        # Epsilon annealing details
        self.epsilon = config["agent"]["epsilon_start"] # starts at 1.0
        self.epsilon_end = config["agent"]["epsilon_end"] # ends at 0.05
        self.epsilon_decay = config["agent"]["epsilon_decay"] # default 0.995 per episode
        
        # PER hyperparams
        self.per_beta_start = config["agent"]["per_beta_start"] # default 0.4
        self.per_beta_end = config["agent"]["per_beta_end"] # default 1.0
        self.total_episodes = config["training"]["n_episodes"] # default 600

        # Seed numpy and torch for reproducible runs
        seed = config.get("experiment", {}).get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        
        # Networks initialization
        self.online_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.update_target_network() # Initialize weights identically
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config["agent"]["learning_rate"])
        self.memory = PrioritizedReplayBuffer(
            config["agent"]["replay_buffer_capacity"], 
            alpha=config["agent"]["per_alpha"] # default 0.6
        )
        
        self.total_steps = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
            
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def update_target_network(self) -> None:
        """Performs a hard parameter copy from Online Network to Target Network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def train_step(self, episode_idx: int) -> float:
        """Executes a single DDQN backpropagation update step with PER."""
        if len(self.memory) < self.warmup:
            return 0.0 # Wait for buffer warmup transitions
            
        # Anneal beta from beta_start to beta_end over training episodes
        beta = self.per_beta_start + (self.per_beta_end - self.per_beta_start) * (episode_idx / self.total_episodes)
        beta = min(beta, 1.0)
        
        # Sample mini-batch
        states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size, beta)
        
        # Convert to Tensors
        states = torch.tensor(states,dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions,dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards,dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states,dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones,dtype=torch.float32, device=self.device)
        weights = torch.tensor(weights,dtype=torch.float32, device=self.device)
        
        # 1. Compute current Q-values: Q(s_t, a_t; theta)
        q_values = self.online_net(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 2. Double DQN Target Calculation
        with torch.no_grad():
            # Find best action on next state using Online parameters: argmax_a Q(s_{t+1}, a; theta)
            next_state_actions = self.online_net(next_states).argmax(dim=1)
            # Evaluate that action using Target parameters: Q(s_{t+1}, a*; theta^-)
            next_q_values = self.target_net(next_states)
            next_state_values = next_q_values.gather(1, next_state_actions.unsqueeze(1)).squeeze(1)
            # Double Q TD Target
            expected_state_action_values = rewards + (self.gamma * next_state_values * (1.0 - dones))
            
        # 3. Compute loss and prioritize experiences
        td_errors = torch.abs(state_action_values - expected_state_action_values)
        
        # Importance-weighted Mean Squared Error (MSE) Loss
        loss = (weights * (state_action_values - expected_state_action_values) ** 2).mean()
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Apply strict gradient clipping to prevent weight saturation under multi-objective reward shifts
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()
        
        # Update priorities in buffer using absolute TD Errors
        new_priorities = td_errors.detach().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, new_priorities)
        
        self.total_steps += 1
        
        # Hard synchronize networks copy every target_update_freq steps
        if self.total_steps % self.target_update_freq == 0:
            self.update_target_network()
            
        return float(loss.item())

    def decay_epsilon(self) -> None:
        """Anneal the exploration rate at the end of each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


    def save(self, path: str) -> None:
        """Persist online network weights, optimizer state, and agent metadata."""
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str) -> None:
        """Restore a previously saved checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint["total_steps"]
        self.epsilon = checkpoint["epsilon"]


