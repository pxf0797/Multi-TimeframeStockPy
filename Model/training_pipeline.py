# training_pipeline.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import tensorflow as tf
from datetime import datetime

class DataPipeline:
    """数据处理流水线"""
    def __init__(self, config: dict):
        self.config = config
        self.data_processor = None
        self.feature_engineer = None
    
    def prepare_data(self, data_path: str) -> Tuple[Dict, Dict]:
        """准备训练和验证数据"""
        # 1. 加载原始数据
        raw_data = self._load_raw_data(data_path)
        
        # 2. 数据处理
        processed_data = self._process_data(raw_data)
        
        # 3. 特征工程
        features = self._engineer_features(processed_data)
        
        # 4. 数据分割
        train_data, val_data = self._split_data(features)
        
        return train_data, val_data
    
    def _load_raw_data(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """加载原始数据"""
        data_dir = Path(data_path)
        raw_data = {}
        
        for period in self.config['periods']:
            file_path = data_dir / f"{period}_data.csv"
            raw_data[period] = pd.read_csv(file_path, parse_dates=['timestamp'])
            raw_data[period].set_index('timestamp', inplace=True)
        
        return raw_data
    
    def _process_data(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """处理数据"""
        if self.data_processor is None:
            from src.data_processing.data_processor import MultiPeriodDataProcessor
            self.data_processor = MultiPeriodDataProcessor(self.config)
        
        return self.data_processor.process_all_periods(raw_data)
    
    def _engineer_features(self, processed_data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """特征工程"""
        if self.feature_engineer is None:
            from src.feature_engineering.feature_engineer import FeatureEngineer
            self.feature_engineer = FeatureEngineer(self.config)
        
        return self.feature_engineer.process_features(processed_data)
    
    def _split_data(self, features: Dict[str, np.ndarray]) -> Tuple[Dict, Dict]:
        """分割训练和验证数据"""
        train_ratio = self.config.get('train_ratio', 0.8)
        
        train_data = {}
        val_data = {}
        
        for key, data in features.items():
            split_idx = int(len(data) * train_ratio)
            train_data[key] = data[:split_idx]
            val_data[key] = data[split_idx:]
        
        return train_data, val_data

class TrainingPipeline:
    """训练流水线"""
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.trainer = None
        self.callbacks = self._setup_callbacks()
    
    def train(self, train_data: Dict, val_data: Dict):
        """执行训练"""
        # 1. 初始化模型（如果未初始化）
        if self.model is None:
            self._init_model()
        
        # 2. 创建数据生成器
        train_generator = self._create_data_generator(train_data, is_training=True)
        val_generator = self._create_data_generator(val_data, is_training=False)
        
        # 3. 训练模型
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=self.config['epochs'],
            callbacks=self.callbacks,
            verbose=1
        )
        
        # 4. 保存模型
        self._save_model()
        
        return history
    
    def _init_model(self):
        """初始化模型"""
        from src.models.trading_model import MultiPeriodTradingModel
        self.model = MultiPeriodTradingModel(self.config['model'])
        
        # 配置优化器和损失函数
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate']
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss={
                'price': 'mse',
                'trend': 'categorical_crossentropy',
                'signal': 'categorical_crossentropy'
            },
            metrics={
                'price': ['mae'],
                'trend': ['accuracy'],
                'signal': ['accuracy']
            }
        )
    
    def _setup_callbacks(self) -> list:
        """设置回调函数"""
        callbacks = []
        
        # 早停
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['early_stopping_patience'],
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # 模型检查点
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)
        
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / 'model_{epoch:02d}-{val_loss:.2f}.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        callbacks.append(model_checkpoint)
        
        # TensorBoard
        log_dir = Path('logs/tensorboard') / datetime.now().strftime('%Y%m%d-%H%M%S')
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1
        )
        callbacks.append(tensorboard)
        
        return callbacks
    
    def _create_data_generator(self, data: Dict, is_training: bool = True):
        """创建数据生成器"""
        return DataGenerator(
            data,
            batch_size=self.config['batch_size'],
            is_training=is_training
        )
    
    def _save_model(self):
        """保存模型"""
        save_dir = Path('models')
        save_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model.save(str(save_dir / f'model_{timestamp}.h5'))

class DataGenerator(tf.keras.utils.Sequence):
    """数据生成器"""
    def __init__(self, data: Dict, batch_size: int = 32, is_training: bool = True):
        self.data = data
        self.batch_size = batch_size
        self.is_training = is_training
        self.indices = np.arange(len(next(iter(data.values()))))
        
        # 区分短周期和长周期数据
        self.short_period_data = {k: v for k, v in data.items() if k in ['5min', '15min', '1h']}
        self.long_period_data = {k: v for k, v in data.items() if k in ['1d', '1w']}
    
    def __len__(self):
        """每个epoch的批次数"""
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __getitem__(self, idx):
        """获取一个批次的数据"""
        # 获取当前批次的索引
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # 准备输入数据
        short_period_batch = {
            k: v[batch_indices] for k, v in self.short_period_data.items()
        }
        long_period_batch = {
            k: v[batch_indices] for k, v in self.long_period_data.items()
        }
        
        # 合并短周期和长周期数据
        batch_x = {
            'short_period': np.stack([v for v in short_period_batch.values()], axis=1),
            'long_period': np.stack([v for v in long_period_batch.values()], axis=1),
            'period_ids': np.array([i for i in range(len(self.data))])
        }
        
        # 准备标签数据
        batch_y = {
            'price': self.data['price_labels'][batch_indices],
            'trend': self.data['trend_labels'][batch_indices],
            'signal': self.data['signal_labels'][batch_indices]
        }
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        """每个epoch结束时的操作"""
        if self.is_training:
            # 训练时打乱数据顺序
            np.random.shuffle(self.indices)

class TestingPipeline:
    """测试流程"""
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.evaluator = None
    
    def test(self, test_data: Dict, model_path: str = None) -> Dict:
        """执行测试"""
        # 1. 加载模型
        self._load_model(model_path)
        
        # 2. 创建测试数据生成器
        test_generator = DataGenerator(test_data, batch_size=32, is_training=False)
        
        # 3. 进行预测
        predictions = self.model.predict(test_generator)
        
        # 4. 评估结果
        evaluation_results = self._evaluate_predictions(predictions, test_data)
        
        # 5. 保存结果
        self._save_results(evaluation_results)
        
        return evaluation_results
    
    def _load_model(self, model_path: str = None):
        """加载模型"""
        if model_path is None:
            # 如果未指定模型路径，加载最新的模型
            model_dir = Path('models')
            model_files = sorted(model_dir.glob('*.h5'))
            if not model_files:
                raise ValueError("No model files found")
            model_path = str(model_files[-1])
        
        self.model = tf.keras.models.load_model(model_path)
    
    def _evaluate_predictions(self, predictions: Dict, test_data: Dict) -> Dict:
        """评估预测结果"""
        if self.evaluator is None:
            from src.evaluation.evaluator import ModelEvaluator
            self.evaluator = ModelEvaluator(self.config)
        
        # 计算各项指标
        price_metrics = self.evaluator.evaluate_price_predictions(
            predictions['price'],
            test_data['price_labels']
        )
        
        trend_metrics = self.evaluator.evaluate_trend_predictions(
            predictions['trend'],
            test_data['trend_labels']
        )
        
        signal_metrics = self.evaluator.evaluate_signal_predictions(
            predictions['signal'],
            test_data['signal_labels']
        )
        
        # 计算交易表现指标
        trading_metrics = self.evaluator.evaluate_trading_performance(
            predictions, test_data
        )
        
        return {
            'price_metrics': price_metrics,
            'trend_metrics': trend_metrics,
            'signal_metrics': signal_metrics,
            'trading_metrics': trading_metrics
        }
    
    def _save_results(self, results: Dict):
        """保存测试结果"""
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = results_dir / f'test_results_{timestamp}.json'
        
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

# 使用示例
def run_training():
    """运行训练流程"""
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化数据流水线
    data_pipeline = DataPipeline(config['data'])
    
    # 准备数据
    train_data, val_data = data_pipeline.prepare_data('data/raw')
    
    # 初始化训练流水线
    training_pipeline = TrainingPipeline(config['training'])
    
    # 执行训练
    history = training_pipeline.train(train_data, val_data)
    
    return history

def run_testing(model_path: str = None):
    """运行测试流程"""
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化数据流水线
    data_pipeline = DataPipeline(config['data'])
    
    # 准备测试数据
    test_data = data_pipeline.prepare_data('data/raw/test')
    
    # 初始化测试流水线
    testing_pipeline = TestingPipeline(config['testing'])
    
    # 执行测试
    results = testing_pipeline.test(test_data, model_path)
    
    return results

if __name__ == "__main__":
    # 运行训练
    history = run_training()
    
    # 运行测试
    results = run_testing()