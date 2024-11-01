# project_structure.py

"""
项目目录结构：

trading_system/
│
├── data/                       # 数据目录
│   ├── raw/                   # 原始数据
│   ├── processed/             # 处理后的数据
│   └── features/              # 特征数据
│
├── configs/                   # 配置文件目录
│   ├── data_config.yaml      # 数据处理配置
│   ├── model_config.yaml     # 模型配置
│   └── training_config.yaml  # 训练配置
│
├── src/                      # 源代码
│   ├── data_processing/     # 数据处理模块
│   ├── feature_engineering/ # 特征工程模块
│   ├── models/             # 模型定义
│   ├── training/          # 训练相关
│   ├── evaluation/        # 评估相关
│   └── utils/            # 工具函数
│
├── notebooks/              # Jupyter notebooks
├── tests/                 # 测试代码
├── logs/                  # 日志文件
├── checkpoints/          # 模型检查点
└── results/              # 结果输出
"""

# main.py - 系统主入口
import yaml
import logging
from pathlib import Path
from datetime import datetime
import argparse

class TradingSystem:
    """交易系统主类"""
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
        # 初始化各个模块
        self.data_processor = self._init_data_processor()
        self.feature_engineer = self._init_feature_engineer()
        self.model = self._init_model()
        self.trainer = self._init_trainer()
        self.evaluator = self._init_evaluator()
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('TradingSystem')
        logger.setLevel(logging.INFO)
        
        # 创建日志目录
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # 设置日志文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        handler = logging.FileHandler(f'logs/trading_system_{timestamp}.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _init_data_processor(self):
        """初始化数据处理器"""
        from src.data_processing.data_processor import MultiPeriodDataProcessor
        return MultiPeriodDataProcessor(self.config['data'])
    
    def _init_feature_engineer(self):
        """初始化特征工程"""
        from src.feature_engineering.feature_engineer import FeatureEngineer
        return FeatureEngineer(self.config['features'])
    
    def _init_model(self):
        """初始化模型"""
        from src.models.trading_model import MultiPeriodTradingModel
        return MultiPeriodTradingModel(self.config['model'])
    
    def _init_trainer(self):
        """初始化训练器"""
        from src.training.model_trainer import ModelTrainer
        return ModelTrainer(self.config['training'])
    
    def _init_evaluator(self):
        """初始化评估器"""
        from src.evaluation.evaluator import ModelEvaluator
        return ModelEvaluator(self.config['evaluation'])
    
    def train(self, data_path: str, mode: str = 'full'):
        """训练系统"""
        self.logger.info(f"Starting training in {mode} mode")
        
        try:
            # 1. 数据加载和处理
            raw_data = self._load_data(data_path)
            processed_data = self.data_processor.process_all_periods(raw_data)
            
            # 2. 特征工程
            features = self.feature_engineer.process_features(processed_data)
            
            # 3. 训练数据准备
            train_data, val_data = self._prepare_train_data(features)
            
            # 4. 模型训练
            self.trainer.train(
                model=self.model,
                train_data=train_data,
                val_data=val_data,
                mode=mode
            )
            
            self.logger.info("Training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    def test(self, data_path: str, model_path: str = None):
        """测试系统"""
        self.logger.info("Starting testing")
        
        try:
            # 1. 加载测试数据
            test_data = self._load_data(data_path)
            processed_data = self.data_processor.process_all_periods(test_data)
            features = self.feature_engineer.process_features(processed_data)
            
            # 2. 加载模型（如果指定）
            if model_path:
                self.model.load_weights(model_path)
            
            # 3. 运行测试
            results = self.evaluator.evaluate(
                model=self.model,
                test_data=features,
                raw_data=test_data
            )
            
            # 4. 保存结果
            self._save_results(results)
            
            self.logger.info("Testing completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Testing failed: {str(e)}")
            raise
    
    def _load_data(self, data_path: str) -> dict:
        """加载数据"""
        # 实现数据加载逻辑
        pass
    
    def _prepare_train_data(self, features: dict) -> tuple:
        """准备训练数据"""
        # 实现训练数据准备逻辑
        pass
    
    def _save_results(self, results: dict):
        """保存结果"""
        # 实现结果保存逻辑
        pass

# config.yaml 配置文件示例
"""
data:
  periods: ['5min', '15min', '1h', '1d', '1w']
  features: ['open', 'high', 'low', 'close', 'volume']
  
model:
  input_dim: 29
  conv_filters: [64, 128, 256]
  lstm_units: [128, 64]
  dropout_rate: 0.3
  
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 10
  
evaluation:
  metrics: ['accuracy', 'precision', 'recall', 'f1']
"""

# train.py - 训练脚本
def main():
    parser = argparse.ArgumentParser(description='Trading System Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data', type=str, required=True, help='Path to data directory')
    parser.add_argument('--mode', type=str, default='full', help='Training mode: full/incremental')
    
    args = parser.parse_args()
    
    # 初始化系统
    system = TradingSystem(args.config)
    
    # 开始训练
    system.train(args.data, args.mode)

if __name__ == "__main__":
    main()