# model_training.py

import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from datetime import datetime

class TradingEnvironment:
    """交易环境类"""
    def __init__(self,
                initial_balance: float = 1000000.0,
                transaction_fee_rate: float = 0.0003,
                max_position: float = 1.0,
                stop_loss_rate: float = 0.02):
        self.initial_balance = initial_balance
        self.transaction_fee_rate = transaction_fee_rate
        self.max_position = max_position
        self.stop_loss_rate = stop_loss_rate
        
        self.reset()
    
    def reset(self):
        """重置环境状态"""
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_price = 0.0
        self.total_fees = 0.0
        self.total_trades = 0
        self.history = []
        
    def step(self, action: Dict[str, float], current_price: float) -> Tuple[float, bool]:
        """
        执行交易动作
        返回：reward, done
        """
        old_value = self.calculate_total_value(current_price)
        
        # 执行交易
        if action['type'] == 'buy':
            self._execute_buy(action['amount'], current_price)
        elif action['type'] == 'sell':
            self._execute_sell(action['amount'], current_price)
        
        # 计算reward
        new_value = self.calculate_total_value(current_price)
        reward = (new_value - old_value) / old_value
        
        # 检查是否触发止损
        done = self._check_stop_loss(current_price)
        
        # 记录历史
        self._record_history(action, current_price, reward)
        
        return reward, done
    
    def _execute_buy(self, amount: float, price: float):
        """执行买入操作"""
        max_buyable = min(
            self.balance / price,
            (self.max_position - self.position) * self.initial_balance / price
        )
        amount = min(amount, max_buyable)
        
        if amount > 0:
            fee = amount * price * self.transaction_fee_rate
            total_cost = amount * price + fee
            
            if total_cost <= self.balance:
                self.position += amount
                self.balance -= total_cost
                self.position_price = price
                self.total_fees += fee
                self.total_trades += 1
    
    def _execute_sell(self, amount: float, price: float):
        """执行卖出操作"""
        amount = min(amount, self.position)
        
        if amount > 0:
            fee = amount * price * self.transaction_fee_rate
            total_revenue = amount * price - fee
            
            self.position -= amount
            self.balance += total_revenue
            self.total_fees += fee
            self.total_trades += 1
    
    def calculate_total_value(self, current_price: float) -> float:
        """计算当前总资产价值"""
        return self.balance + self.position * current_price
    
    def _check_stop_loss(self, current_price: float) -> bool:
        """检查是否触发止损"""
        if self.position > 0:
            loss_rate = (self.position_price - current_price) / self.position_price
            if loss_rate > self.stop_loss_rate:
                self._execute_sell(self.position, current_price)
                return True
        return False
    
    def _record_history(self, action: Dict[str, float], price: float, reward: float):
        """记录交易历史"""
        self.history.append({
            'timestamp': datetime.now(),
            'action': action['type'],
            'amount': action['amount'],
            'price': price,
            'balance': self.balance,
            'position': self.position,
            'total_value': self.calculate_total_value(price),
            'reward': reward
        })

class RLTrainer:
    """强化学习训练器"""
    def __init__(self,
                model,
                environment: TradingEnvironment,
                epsilon_start: float = 0.9,
                epsilon_end: float = 0.1,
                epsilon_decay: float = 0.995,
                gamma: float = 0.99,
                batch_size: int = 32,
                memory_size: int = 10000):
        self.model = model
        self.env = environment
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.memory = []
        self.memory_size = memory_size
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Dict[str, float]:
        """选择动作（epsilon-greedy策略）"""
        if training and np.random.random() < self.epsilon:
            # 探索：随机动作
            action_type = np.random.choice(['buy', 'sell', 'hold'])
            amount = np.random.random() * self.env.max_position
        else:
            # 利用：使用模型预测
            predictions = self.model.predict(state)
            action_probs = predictions['signal'][0]
            action_idx = np.argmax(action_probs)
            
            # 将索引转换为动作
            if action_idx == 0:
                action_type = 'buy'
                amount = 0.1  # 基础买入量
            elif action_idx == 1:
                action_type = 'sell'
                amount = 0.1  # 基础卖出量
            else:
                action_type = 'hold'
                amount = 0.0
        
        return {'type': action_type, 'amount': amount}
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """执行一步训练"""
        if len(self.memory) < self.batch_size:
            return
        
        # 随机采样batch
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states = []
        targets = []
        
        for idx in batch:
            state, action, reward, next_state, done = self.memory[idx]
            
            target = self.model.predict(state)
            if not done:
                next_value = np.max(self.model.predict(next_state)['signal'])
                reward = reward + self.gamma * next_value
            
            # 更新目标值
            action_idx = ['buy', 'sell', 'hold'].index(action['type'])
            target['signal'][0][action_idx] = reward
            
            states.append(state)
            targets.append(target)
        
        # 批量训练
        states = np.vstack(states)
        self.model.train_on_batch(states, targets)
        
        # 更新探索率
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

class ModelTrainer:
    """模型训练器"""
    def __init__(self,
                model,
                rl_trainer: RLTrainer,
                validation_split: float = 0.2,
                early_stopping_patience: int = 10):
        self.model = model
        self.rl_trainer = rl_trainer
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """设置日志"""
        logger = logging.getLogger('ModelTrainer')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('training.log')
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def train(self,
            train_data: Dict[str, np.ndarray],
            train_labels: Dict[str, np.ndarray],
            num_epochs: int = 100,
            batch_size: int = 32):
        """训练模型"""
        self.logger.info("Starting model training...")
        
        # 创建验证集
        val_size = int(len(train_data['price']) * self.validation_split)
        val_indices = np.random.choice(len(train_data['price']), val_size, replace=False)
        train_indices = np.array([i for i in range(len(train_data['price'])) if i not in val_indices])
        
        val_data = {k: v[val_indices] for k, v in train_data.items()}
        val_labels = {k: v[val_indices] for k, v in train_labels.items()}
        train_data = {k: v[train_indices] for k, v in train_data.items()}
        train_labels = {k: v[train_indices] for k, v in train_labels.items()}
        
        # 训练监控
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # 训练一个epoch
            train_loss = self._train_epoch(train_data, train_labels, batch_size)
            
            # 验证
            val_loss = self._validate(val_data, val_labels)
            
            self.logger.info(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_model(f"best_model_epoch_{epoch+1}.h5")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    self.logger.info("Early stopping triggered")
                    break
            
            # 强化学习更新
            self._rl_update(train_data)
    
    def _train_epoch(self,
                    train_data: Dict[str, np.ndarray],
                    train_labels: Dict[str, np.ndarray],
                    batch_size: int) -> float:
        """训练一个epoch"""
        total_loss = 0
        num_batches = len(train_data['price']) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch_data = {k: v[start_idx:end_idx] for k, v in train_data.items()}
            batch_labels = {k: v[start_idx:end_idx] for k, v in train_labels.items()}
            
            loss = self.model.train_on_batch(batch_data, batch_labels)
            total_loss += loss
        
        return total_loss / num_batches
    
    def _validate(self,
                val_data: Dict[str, np.ndarray],
                val_labels: Dict[str, np.ndarray]) -> float:
        """验证模型"""
        return self.model.evaluate(val_data, val_labels, verbose=0)
    
    def _rl_update(self, train_data: Dict[str, np.ndarray]):
        """强化学习更新"""
        for _ in range(len(train_data['price'])):
            self.rl_trainer.train_step()
    
    def _save_model(self, filename: str):
        """保存模型"""
        self.model.save(filename)
        self.logger.info(f"Model saved as {filename}")

