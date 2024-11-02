# reinforcement/trainer.py

class ReinforcementTrainer:
    def __init__(self,
                 model: nn.Module,
                 reward_calc: RewardCalculator,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99):
        self.model = model
        self.reward_calc = reward_calc
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.gamma = gamma
        
        # 探索策略
        self.epsilon_short = 0.3
        self.epsilon_long = 0.1
        
    def train_step(self, 
                  state_dict: Dict[str, torch.Tensor],
                  next_state_dict: Dict[str, torch.Tensor],
                  price_changes: torch.Tensor) -> Dict[str, float]:
        """
        训练一个步骤
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 获取当前动作
        predictions = self.model(state_dict)
        decisions = self.model.decision(predictions, 
                                      self.model._get_market_state(state_dict))
        
        # 获取下一状态的值
        with torch.no_grad():
            next_predictions = self.model(next_state_dict)
            next_decisions = self.model.decision(next_predictions,
                                               self.model._get_market_state(next_state_dict))
        
        # 计算仓位调整的奖励和损失
        position_rewards = self._calculate_position_rewards(
            decisions['position'],
            next_decisions['position'],
            price_changes,
            self.model._get_market_state(state_dict)[:, 0]  # volatility
        )
        
        position_loss = self._compute_position_loss(
            decisions['position'],
            next_decisions['position'],
            position_rewards
        )
        
        # 计算做T操作的奖励和损失
        trading_rewards = self._calculate_trading_rewards(
            decisions['trading'],
            price_changes,
            torch.ones_like(price_changes)  # 假设holding_time都是1
        )
        
        trading_loss = self._compute_trading_loss(
            decisions['trading'],
            next_decisions['trading'],
            trading_rewards
        )
        
        # 总损失
        total_loss = position_loss + trading_loss
        
        # 反向传播
        total_loss.backward()
        self.optimizer.step()
        
        # 更新探索率
        self._update_epsilon(position_rewards.mean().item())
        
        return {
            'total_loss': total_loss.item(),
            'position_loss': position_loss.item(),
            'trading_loss': trading_loss.item(),
            'position_reward': position_rewards.mean().item(),
            'trading_reward': trading_rewards.mean().item()
        }
    
    def _calculate_position_rewards(self,
                                  positions: torch.Tensor,
                                  next_positions: torch.Tensor,
                                  price_changes: torch.Tensor,
                                  volatility: torch.Tensor) -> torch.Tensor:
        """
        计算仓位调整的奖励
        """
        rewards = []
        for pos, next_pos, price_change, vol in zip(positions, next_positions, 
                                                   price_changes, volatility):
            reward = self.reward_calc.calculate_position_reward(
                pos.item(), next_pos.item(), price_change.item(), vol.item())
            rewards.append(reward)
        return torch.tensor(rewards, device=positions.device)
    
    def _calculate_trading_rewards(self,
                                 actions: torch.Tensor,
                                 price_changes: torch.Tensor,
                                 holding_times: torch.Tensor) -> torch.Tensor:
        """
        计算做T操作的奖励
        """
        rewards = []
        actions_idx = actions.argmax(dim=1)
        for action, price_change, holding_time in zip(actions_idx, price_changes, holding_times):
            reward = self.reward_calc.calculate_trading_reward(
                action.item(), price_change.item(), holding_time.item())
            rewards.append(reward)
        return torch.tensor(rewards, device=actions.device)
    
    def _compute_position_loss(self,
                             positions: torch.Tensor,
                             next_positions: torch.Tensor,
                             rewards: torch.Tensor) -> torch.Tensor:
        """
        计算仓位调整的损失
        """
        expected_q_values = rewards + self.gamma * next_positions
        return F.mse_loss(positions, expected_q_values.detach())
    
    def _compute_trading_loss(self,
                            actions: torch.Tensor,
                            next_actions: torch.Tensor,
                            rewards: torch.Tensor) -> torch.Tensor:
        """
        计算做T操作的损失
        """
        expected_q_values = rewards.unsqueeze(1) + self.gamma * next_actions.max(dim=1)[0].unsqueeze(1)
        action_q_values = (actions * expected_q_values).sum(dim=1)
        return F.mse_loss(action_q_values, expected_q_values.squeeze(1).detach())
    
    def _update_epsilon(self, reward: float):
        """
        更新探索率
        """
        if reward > 0:
            self.epsilon_short *= 0.995
            self.epsilon_long *= 0.998
        else:
            self.epsilon_short = min(0.4, self.epsilon_short * 1.005)
            self.epsilon_long = min(0.2, self.epsilon_long * 1.002)