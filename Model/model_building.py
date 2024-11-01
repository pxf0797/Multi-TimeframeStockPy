import gym
from gym import spaces
import numpy as np
from typing import Dict, Tuple, List
from collections import deque
import random

class TradingEnvironment(gym.Env):
    """Trading environment for reinforcement learning"""
    
    def __init__(self, data: Dict[str, np.ndarray], config: Dict):
        super().__init__()
        
        self.data = data
        self.config = config
        self.current_step = 0
        self.current_position = 0
        self.cash = config['initial_cash']
        self.portfolio_value = self.cash
        
        # Define action spaces
        self.position_action_space = spaces.Discrete(config['position_actions'])
        self.trading_action_space = spaces.Discrete(config['trading_actions'])
        
        # Define observation space
        self.observation_space = spaces.Dict({
            period: spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(config['lookback'], config['input_dim'])
            ) for period in data.keys()
        })
        
        # Initialize transaction cost and other parameters
        self.transaction_cost = config['transaction_cost']
        self.max_position = config['max_position']
        
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset the environment"""
        self.current_step = self.config['lookback']
        self.current_position = 0
        self.cash = self.config['initial_cash']
        self.portfolio_value = self.cash
        
        return self._get_observation()
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """Execute one step in the environment"""
        position_action = actions['position']
        trading_action = actions['trading']
        
        # Execute actions and calculate rewards
        position_reward = self._execute_position_action(position_action)
        trading_reward = self._execute_trading_action(trading_action)
        
        # Calculate total reward
        total_reward = position_reward + trading_reward
        
        # Update state
        self.current_step += 1
        done = self.current_step >= len(self.data['5min']) - 1
        
        return self._get_observation(), total_reward, done, {}
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation"""
        obs = {}
        for period, data in self.data.items():
            start = self.current_step - self.config['lookback']
            end = self.current_step
            obs[period] = data[start:end]
        return obs
    
    def _execute_position_action(self, action: int) -> float:
        """Execute position adjustment action"""
        old_position = self.current_position
        
        # Map action to position change
        if action == 0:  # Hold
            position_change = 0
        elif action == 1:  # Increase
            position_change = 0.1
        elif action == 2:  # Decrease
            position_change = -0.1
        else:  # Clear position
            position_change = -self.current_position
            
        # Apply position change
        new_position = np.clip(
            old_position + position_change,
            0,
            self.max_position
        )
        
        # Calculate transaction cost
        cost = abs(new_position - old_position) * self.transaction_cost
        
        # Update position and calculate reward
        self.current_position = new_position
        return self._calculate_position_reward(old_position, new_position, cost)
    
    def _execute_trading_action(self, action: int) -> float:
        """Execute trading action"""
        if action == 0:  # Hold
            return 0
            
        current_price = self.data['5min'][self.current_step]['close']
        
        # Calculate trading size based on current position
        max_trade_size = self.current_position * 0.2  # Max 20% of position
        
        if action == 1:  # Buy
            trade_size = max_trade_size
            cost = trade_size * current_price * self.transaction_cost
            self.cash -= (trade_size * current_price + cost)
        else:  # Sell
            trade_size = -max_trade_size
            cost = abs(trade_size) * current_price * self.transaction_cost
            self.cash += (abs(trade_size) * current_price - cost)
            
        return self._calculate_trading_reward(trade_size, cost)
    
    def _calculate_position_reward(self, old_pos: float, new_pos: float, 
                                 cost: float) -> float:
        """Calculate reward for position adjustment"""
        # Calculate return from position change
        current_price = self.data['5min'][self.current_step]['close']
        next_price = self.data['5min'][self.current_step + 1]['close']
        price_change = (next_price - current_price) / current_price
        
        position_return = new_pos * price_change
        
        # Penalize frequent position changes
        stability_penalty = abs(new_pos - old_pos) * 0.001
        
        # Calculate final reward
        reward = position_return - cost - stability_penalty
        
        return reward
    
    def _calculate_trading_reward(self, trade_size: float, cost: float) -> float:
        """Calculate reward for trading action"""
        # Calculate immediate return from trade
        current_price = self.data['5min'][self.current_step]['close']
        next_price = self.data['5min'][self.current_step + 1]['close']
        price_change = (next_price - current_price) / current_price
        
        trade_return = trade_size * price_change
        
        # Penalize frequent trading
        trading_penalty = abs(trade_size) * 0.001
        
        # Calculate final reward
        reward = trade_return - cost - trading_penalty
        
        return reward

class ReplayBuffer:
    """Experience replay buffer for reinforcement learning"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: Dict[str, np.ndarray], 
             action: Dict[str, int], 
             reward: float, 
             next_state: Dict[str, np.ndarray], 
             done: bool):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to torch tensors
        return (
            {k: torch.FloatTensor(np.stack([s[k] for s in states])) 
             for k in states[0].keys()},
            {k: torch.LongTensor([a[k] for a in actions]) 
             for k in actions[0].keys()},
            torch.FloatTensor(rewards),
            {k: torch.FloatTensor(np.stack([s[k] for s in next_states])) 
             for k in next_states[0].keys()},
            torch.BoolTensor(dones)
        )
        
    def __len__(self) -> int:
        return len(self.buffer)

class TrainingManager:
    """Manage the training process of the trading model"""
    
    def __init__(self, model: MultiPeriodTradingModel, 
                 env: TradingEnvironment, 
                 config: Dict):
        self.model = model
        self.env = env
        self.config = config
        
        self.replay_buffer = ReplayBuffer(config['buffer_capacity'])
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['learning_rate']
        )
        
        # Initialize target network for DQN
        self.target_model = copy.deepcopy(model)
        
    def train_episode(self) -> float:
        """Train for one episode"""
        state = self.env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Get model predictions
            with torch.no_grad():
                model_output = self.model(
                    {k: torch.FloatTensor(v).unsqueeze(0) 
                     for k, v in state.items()},
                    torch.LongTensor([0])  # Dummy period_ids
                )
            
            # Select actions using epsilon-greedy
            position_action = self._select_action(
                model_output['position_policy'], 
                self.config['position_epsilon']
            )
            trading_action = self._select_action(
                model_output['trading_policy'],
                self.config['trading_epsilon']
            )
            
            actions = {
                'position': position_action,
                'trading': trading_action
            }
            
            # Execute actions
            next_state, reward, done, _ = self.env.step(actions)
            
            # Store experience
            self.replay_buffer.push(state, actions, reward, next_state, done)
            
            # Update model if enough samples
            if len(self.replay_buffer) >= self.config['batch_size']:
                self._update_model()
                
            # Update target network periodically
            if self.env.current_step % self.config['target_update'] == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                
            state = next_state
            total_reward += reward
            
        return total_reward
    
    def _select_action(self, policy: torch.Tensor, epsilon: float) -> int:
        """Select action using epsilon-greedy policy"""
        if random.random() < epsilon:
            return random.randrange(len(policy[0]))
        return policy.argmax(dim=1).item()
    
    def _update_model(self):
        """Update model using sampled batch"""
        # Sample batch
        batch = self.replay_buffer.sample(self.config['batch_size'])
        states, actions, rewards, next_states, dones = batch
        
        # Get current Q values
        current_output = self.model(states, torch.LongTensor([0]))
        current_q_position = current_output['position_policy'].gather(
            1, actions['position'].unsqueeze(1)
        )
        current_q_trading = current_output['trading_policy'].gather(
            1, actions['trading'].unsqueeze(1)
        )
        
        # Get next Q values from target network
        with torch.no_grad():
            next_output = self.target_model(next_states, torch.LongTensor([0]))
            next_q_position = next_output['position_policy'].max(1)[0]
            next_q_trading = next_output['trading_policy'].max(1)[0]
            
        # Calculate target Q values
        target_q_position = rewards + (1 - dones) * self.config['gamma'] * next_q_position
        target_q_trading = rewards + (1 - dones) * self.config['gamma'] * next_q_trading
        
        # Calculate loss
        position_loss = F.smooth_l1_loss(
            current_q_position.squeeze(), 
            target_q_position
        )
        trading_loss = F.smooth_l1_loss(
            current_q_trading.squeeze(),
            target_q_trading
        )
        loss = position_loss + trading_loss
        
        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train_model(config: Dict):
    """Main training function"""
    # Initialize environment and model
    env = TradingEnvironment(config['data'], config)
    model = MultiPeriodTradingModel(config)
    trainer = TrainingManager(model, env, config)
    
    # Training loop
    rewards = []
    for episode in range(config['num_episodes']):
        episode_reward = trainer.train_episode()
        rewards.append(episode_reward)
        
        # Log progress
        if (episode + 1) % config['log_interval'] == 0:
            avg_reward = np.mean(rewards[-config['log_interval']:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
            
    return model, rewards