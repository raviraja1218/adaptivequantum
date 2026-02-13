"""
Deep Q-Network agent for photonic circuit compilation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple, List, Dict, Any
from pathlib import Path

class DQN(nn.Module):
    """Deep Q-Network for circuit compilation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        return self.network(state)

class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Store experience in buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch of experiences."""
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent for RL compilation."""
    
    def __init__(self, state_dim: int, action_dim: int, device: str = None):
        """
        Initialize DQN agent.
        
        Parameters:
        -----------
        state_dim : int
            Dimension of state space
        action_dim : int
            Number of possible actions
        device : str
            'cuda' or 'cpu'
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Create Q-network and target network
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        # Experience replay
        self.memory = ReplayBuffer(capacity=10000)
        
        # Training parameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.target_update = 10  # Update target network every N episodes
        
        # Training statistics
        self.training_losses = []
        self.episode_rewards = []
        self.episode_gate_reductions = []
        
        # Create checkpoint directory
        self.checkpoint_dir = Path("models/checkpoints/rl_compiler")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using ε-greedy policy.
        
        Parameters:
        -----------
        state : np.ndarray
            Current state
        training : bool
            Whether in training mode (uses exploration)
        
        Returns:
        --------
        int : Selected action index
        """
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randrange(self.action_dim)
        
        # Exploit: choose best action according to policy network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def update_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train_step(self):
        """Perform one training step."""
        # Sample from replay buffer
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return None
        
        states, actions, rewards, next_states, dones = batch
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train_episode(self, env, max_steps: int = 50) -> Dict[str, Any]:
        """
        Train for one episode.
        
        Returns:
        --------
        dict : Episode statistics
        """
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Select action
            action = self.select_action(state, training=True)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            self.memory.push(state, action, reward, next_state, done)
            
            # Train
            loss = self.train_step()
            if loss is not None:
                self.training_losses.append(loss)
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
        
        # Update exploration rate
        self.update_epsilon()
        
        # Get final metrics
        metrics = env.get_metrics()
        metrics.update({
            'total_reward': total_reward,
            'steps': steps,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory)
        })
        
        return metrics
    
    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_checkpoint(self, episode: int, metrics: Dict[str, Any]):
        """Save model checkpoint."""
        checkpoint = {
            'episode': episode,
            'policy_net_state': self.policy_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'metrics': metrics,
            'training_losses': self.training_losses[-100:] if self.training_losses else [],
            'episode_rewards': self.episode_rewards[-100:] if self.episode_rewards else []
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{episode:04d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Also save latest
        torch.save(checkpoint, self.checkpoint_dir / "latest.pt")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state'])
        self.target_net.load_state_dict(checkpoint['target_net_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon']
        
        print(f"Loaded checkpoint from episode {checkpoint['episode']}")
        return checkpoint
    
    def save_model(self, path: Path):
        """Save final model."""
        model_data = {
            'policy_net_state': self.policy_net.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'training_stats': {
                'final_epsilon': self.epsilon,
                'avg_loss': np.mean(self.training_losses) if self.training_losses else 0,
                'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0
            }
        }
        
        torch.save(model_data, path)
        print(f"Model saved to: {path}")
    
    def evaluate(self, env, max_steps: int = 50) -> Dict[str, Any]:
        """
        Evaluate agent without exploration.
        
        Returns:
        --------
        dict : Evaluation metrics
        """
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Select action (no exploration)
            action = self.select_action(state, training=False)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Update
            state = next_state
            total_reward += reward
            steps += 1
        
        # Get final metrics
        metrics = env.get_metrics()
        metrics.update({
            'eval_total_reward': total_reward,
            'eval_steps': steps
        })
        
        return metrics
