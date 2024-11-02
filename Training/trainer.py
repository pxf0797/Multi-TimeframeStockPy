# training/trainer.py

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm
import numpy as np
from collections import defaultdict

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    # 优化器参数
    weight_decay: float = 0.0001
    beta1: float = 0.9
    beta2: float = 0.999
    
    # 学习率调度参数
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    
    # 梯度裁剪
    clip_grad_norm: float = 1.0
    
    # 验证参数
    validation_interval: int = 1
    
    # 日志参数
    log_interval: int = 100

class ModelTrainer:
    """
    模型训练器
    """
    def __init__(self,
                 model: nn.Module,
                 config: TrainingConfig,
                 device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        
        # 将模型移动到指定设备
        self.model = self.model.to(device)
        
        # 初始化优化器
        self.optimizer = Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        
        # 初始化学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.lr_factor,
            patience=config.lr_patience,
            min_lr=config.min_lr
        )
        
        # 初始化训练状态
        self.train_state = TrainingState()
        
        # 设置日志记录器
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
    def _setup_logging(self):
        """
        设置日志记录
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def train(self,
             train_loader: torch.utils.data.DataLoader,
             val_loader: torch.utils.data.DataLoader) -> TrainingState:
        """
        训练模型
        """
        self.logger.info("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            # 训练一个epoch
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # 验证
            if epoch % self.config.validation_interval == 0:
                val_metrics = self._validate(val_loader, epoch)
                
                # 更新学习率
                self.scheduler.step(val_metrics['total_loss'])
                
                # 检查早停
                if self.train_state.should_stop_early(val_metrics['total_loss']):
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # 保存检查点
            if self.train_state.is_best_model():
                self._save_checkpoint(epoch, metrics={**train_metrics, **val_metrics})
            
            # 记录训练状态
            self.train_state.update(epoch, train_metrics, val_metrics)
            
        return self.train_state
    
    def _train_epoch(self, 
                    train_loader: torch.utils.data.DataLoader,
                    epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        """
        self.model.train()
        metrics = defaultdict(float)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (data, targets) in enumerate(pbar):
            # 将数据移动到设备
            data = {k: v.to(self.device) for k, v in data.items()}
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(data)
            
            # 计算损失
            losses = self.model.calculate_loss(outputs, targets)
            total_loss = losses['total_loss']
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.clip_grad_norm
            )
            
            # 优化器步进
            self.optimizer.step()
            
            # 更新指标
            for k, v in losses.items():
                metrics[k] += v.item()
            
            # 更新进度条
            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({
                    'loss': total_loss.item(),
                    'lr': self._get_current_lr()
                })
        
        # 计算平均指标
        metrics = {k: v / len(train_loader) for k, v in metrics.items()}
        
        return metrics
    
    def _validate(self,
                 val_loader: torch.utils.data.DataLoader,
                 epoch: int) -> Dict[str, float]:
        """
        验证模型
        """
        self.model.eval()
        metrics = defaultdict(float)
        
        with torch.no_grad():
            for data, targets in val_loader:
                # 将数据移动到设备
                data = {k: v.to(self.device) for k, v in data.items()}
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # 前向传播
                outputs = self.model(data)
                
                # 计算损失
                losses = self.model.calculate_loss(outputs, targets)
                
                # 更新指标
                for k, v in losses.items():
                    metrics[k] += v.item()
        
        # 计算平均指标
        metrics = {k: v / len(val_loader) for k, v in metrics.items()}
        
        self.logger.info(f"Validation metrics at epoch {epoch}: {metrics}")
        
        return metrics
    
    def _save_checkpoint(self,
                        epoch: int,
                        metrics: Dict[str, float]):
        """
        保存检查点
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_state': self.train_state,
            'metrics': metrics
        }
        
        filename = f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, filename)
        self.logger.info(f"Saved checkpoint: {filename}")
    
    def _get_current_lr(self) -> float:
        """
        获取当前学习率
        """
        return self.optimizer.param_groups[0]['lr']

class TrainingState:
    """
    训练状态跟踪器
    """
    def __init__(self):
        self.epochs = []
        self.train_metrics = []
        self.val_metrics = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def update(self,
              epoch: int,
              train_metrics: Dict[str, float],
              val_metrics: Dict[str, float]):
        """
        更新训练状态
        """
        self.epochs.append(epoch)
        self.train_metrics.append(train_metrics)
        self.val_metrics.append(val_metrics)
        
        # 更新最佳验证损失
        current_val_loss = val_metrics['total_loss']
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.patience_counter = 0
            return True
        else:
            self.patience_counter += 1
            return False
            
    def should_stop_early(self, current_val_loss: float) -> bool:
        """
        检查是否应该早停
        """
        if self.patience_counter >= self.config.early_stopping_patience:
            return True
        return False
    
    def is_best_model(self) -> bool:
        """
        检查是否是最佳模型
        """
        if not self.val_metrics:
            return False
        return self.val_metrics[-1]['total_loss'] == self.best_val_loss
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        获取训练总结
        """
        if not self.epochs:
            return {}
            
        return {
            'total_epochs': len(self.epochs),
            'best_val_loss': self.best_val_loss,
            'final_train_metrics': self.train_metrics[-1],
            'final_val_metrics': self.val_metrics[-1],
            'training_history': {
                'epochs': self.epochs,
                'train_metrics': self.train_metrics,
                'val_metrics': self.val_metrics
            }
        }

def build_trainer(model: nn.Module,
                 config: TrainingConfig,
                 device: torch.device) -> ModelTrainer:
    """
    构建训练器
    """
    return ModelTrainer(model, config, device)