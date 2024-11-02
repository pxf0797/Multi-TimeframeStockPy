# models/prediction.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math
import numpy as np

@dataclass
class PredictionConfig:
    """预测层配置"""
    input_dim: int = 256
    hidden_dim: int = 512
    num_mc_samples: int = 10  # Monte Carlo采样数量
    dropout_rate: float = 0.2
    num_price_quantiles: int = 5  # 分位数预测数量
    num_ensemble_members: int = 5  # 集成成员数量
    temperature: float = 1.0  # 预测校准温度
    
    # 价格预测配置
    price_periods: List[str] = None
    
    # 趋势预测配置
    trend_classes: int = 3
    
    # 信号预测配置
    signal_classes: int = 3
    
    def __post_init__(self):
        if self.price_periods is None:
            self.price_periods = ['5min', '15min', '1h', '1d']

class UncertaintyEstimator(nn.Module):
    """
    不确定性估计器
    包含认知不确定性(模型不确定性)和随机不确定性(数据噪声)
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.epistemic_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, output_dim)
        )
        
        self.aleatoric_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, output_dim),
            nn.Softplus()  # 确保方差为正
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        epistemic_uncertainty = self.epistemic_net(x)  # 认知不确定性
        aleatoric_uncertainty = self.aleatoric_net(x)  # 随机不确定性
        return epistemic_uncertainty, aleatoric_uncertainty

class EnsembleMember(nn.Module):
    """
    集成成员网络
    用于生成多样化的预测
    """
    def __init__(self, config: PredictionConfig):
        super().__init__()
        self.config = config
        
        # 特征转换
        self.feature_transform = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # 价格预测头
        self.price_head = nn.ModuleDict({
            period: nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.hidden_dim // 2, config.num_price_quantiles)
            )
            for period in config.price_periods
        })
        
        # 趋势预测头
        self.trend_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.trend_classes)
        )
        
        # 信号预测头
        self.signal_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.signal_classes)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 特征转换
        features = self.feature_transform(x)
        
        # 多周期价格预测
        price_predictions = {
            period: head(features)
            for period, head in self.price_head.items()
        }
        
        # 趋势预测
        trend_logits = self.trend_head(features)
        
        # 信号预测
        signal_logits = self.signal_head(features)
        
        return {
            'price': price_predictions,
            'trend': trend_logits,
            'signal': signal_logits
        }

class BayesianDropout(nn.Module):
    """
    贝叶斯Dropout层
    用于Monte Carlo采样
    """
    def __init__(self, dropout_rate: float = 0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return F.dropout(x, self.dropout_rate, training=True)
        mask = torch.bernoulli(torch.full_like(x, 1 - self.dropout_rate))
        return x * mask / (1 - self.dropout_rate)

class EnhancedPredictionModule(nn.Module):
    """
    增强的预测模块
    包含不确定性估计和置信度评估
    """
    def __init__(self, config: PredictionConfig):
        super().__init__()
        self.config = config
        
        # 集成成员
        self.ensemble_members = nn.ModuleList([
            EnsembleMember(config)
            for _ in range(config.num_ensemble_members)
        ])
        
        # 不确定性估计器
        self.uncertainty_estimator = UncertaintyEstimator(
            config.input_dim,
            len(config.price_periods) + 2  # 价格、趋势、信号
        )
        
        # 置信度评估器
        self.confidence_estimator = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, 3),  # 价格、趋势、信号的置信度
            nn.Sigmoid()
        )
        
        # 贝叶斯Dropout
        self.bayesian_dropout = BayesianDropout(config.dropout_rate)
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = features.size(0)
        
        # Monte Carlo采样
        mc_predictions = []
        for _ in range(self.config.num_mc_samples):
            # 应用贝叶斯Dropout
            dropped_features = self.bayesian_dropout(features)
            
            # 获取集成成员预测
            ensemble_predictions = []
            for member in self.ensemble_members:
                preds = member(dropped_features)
                ensemble_predictions.append(preds)
            
            mc_predictions.append(ensemble_predictions)
        
        # 计算预测均值和方差
        predictions = self._aggregate_predictions(mc_predictions)
        
        # 估计不确定性
        epistemic_uncertainty, aleatoric_uncertainty = self.uncertainty_estimator(features)
        
        # 评估置信度
        confidence_scores = self.confidence_estimator(features)
        
        # 组合结果
        return {
            'predictions': predictions,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'confidence_scores': confidence_scores,
            'ensemble_predictions': ensemble_predictions
        }
    
    def _aggregate_predictions(self, 
                             mc_predictions: List[List[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        聚合Monte Carlo采样和集成预测
        """
        # 初始化聚合结果
        aggregated = {
            'price': {period: [] for period in self.config.price_periods},
            'trend': [],
            'signal': []
        }
        
        # 收集所有预测
        for mc_sample in mc_predictions:
            for ensemble_pred in mc_sample:
                # 价格预测
                for period in self.config.price_periods:
                    aggregated['price'][period].append(ensemble_pred['price'][period])
                
                # 趋势预测
                aggregated['trend'].append(ensemble_pred['trend'])
                
                # 信号预测
                aggregated['signal'].append(ensemble_pred['signal'])
        
        # 计算均值和方差
        results = {}
        
        # 处理价格预测
        for period in self.config.price_periods:
            price_preds = torch.stack(aggregated['price'][period])
            results[f'{period}_mean'] = price_preds.mean(0)
            results[f'{period}_std'] = price_preds.std(0)
            results[f'{period}_quantiles'] = torch.quantile(
                price_preds, 
                torch.linspace(0, 1, self.config.num_price_quantiles),
                dim=0
            )
        
        # 处理趋势预测
        trend_preds = torch.stack(aggregated['trend'])
        results['trend_mean'] = trend_preds.mean(0)
        results['trend_std'] = trend_preds.std(0)
        results['trend_probs'] = F.softmax(results['trend_mean'] / self.config.temperature, dim=-1)
        
        # 处理信号预测
        signal_preds = torch.stack(aggregated['signal'])
        results['signal_mean'] = signal_preds.mean(0)
        results['signal_std'] = signal_preds.std(0)
        results['signal_probs'] = F.softmax(results['signal_mean'] / self.config.temperature, dim=-1)
        
        return results
    
    def calculate_loss(self, 
                      predictions: Dict[str, torch.Tensor],
                      targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算损失，包含不确定性
        """
        losses = {}
        
        # 价格预测损失
        price_loss = 0
        for period in self.config.price_periods:
            mean_key = f'{period}_mean'
            std_key = f'{period}_std'
            if mean_key in predictions and period in targets:
                # 负对数似然损失
                price_loss += self._gaussian_nll_loss(
                    predictions[mean_key],
                    predictions[std_key],
                    targets[period]
                )
        losses['price_loss'] = price_loss
        
        # 趋势预测损失
        if 'trend_probs' in predictions and 'trend' in targets:
            losses['trend_loss'] = F.cross_entropy(
                predictions['trend_mean'],
                targets['trend']
            )
            
            # KL散度正则化
            losses['trend_kl'] = self._kl_divergence(
                predictions['trend_probs'],
                F.one_hot(targets['trend'], self.config.trend_classes).float()
            )
        
        # 信号预测损失
        if 'signal_probs' in predictions and 'signal' in targets:
            losses['signal_loss'] = F.cross_entropy(
                predictions['signal_mean'],
                targets['signal']
            )
            
            # KL散度正则化
            losses['signal_kl'] = self._kl_divergence(
                predictions['signal_probs'],
                F.one_hot(targets['signal'], self.config.signal_classes).float()
            )
        
        # 不确定性损失
        if 'epistemic_uncertainty' in predictions:
            losses['uncertainty_loss'] = self._uncertainty_loss(
                predictions['epistemic_uncertainty'],
                predictions['aleatoric_uncertainty'],
                targets
            )
        
        # 计算总损失
        losses['total_loss'] = sum(losses.values())
        
        return losses
    
    def _gaussian_nll_loss(self,
                          mean: torch.Tensor,
                          std: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
        """
        计算高斯负对数似然损失
        """
        return 0.5 * torch.log(2 * math.pi * std**2) + \
               0.5 * (target - mean)**2 / std**2
    
    def _kl_divergence(self,
                      p: torch.Tensor,
                      q: torch.Tensor) -> torch.Tensor:
        """
        计算KL散度
        """
        return torch.sum(p * torch.log(p / (q + 1e-8) + 1e-8), dim=-1).mean()
    
    def _uncertainty_loss(self,
                         epistemic: torch.Tensor,
                         aleatoric: torch.Tensor,
                         targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算不确定性损失
        """
        # 计算预测误差
        errors = []
        for period in self.config.price_periods:
            if period in targets:
                mean_key = f'{period}_mean'
                error = torch.abs(predictions[mean_key] - targets[period])
                errors.append(error)
        
        if errors:
            error = torch.cat(errors).mean()
            # 不确定性应该与误差相关
            uncertainty_loss = F.mse_loss(epistemic + aleatoric, error)
            return uncertainty_loss
        
        return torch.tensor(0., device=epistemic.device)

    @torch.no_grad()
    def get_prediction_intervals(self,
                               predictions: Dict[str, torch.Tensor],
                               confidence_level: float = 0.95) -> Dict[str, torch.Tensor]:
        """
        计算预测区间
        """
        intervals = {}
        
        # 计算价格预测区间
        for period in self.config.price_periods:
            mean_key = f'{period}_mean'
            std_key = f'{period}_std'
            if mean_key in predictions and std_key in predictions:
                z_score = torch.tensor(norm.ppf((1 + confidence_level) / 2))
                intervals[f'{period}_lower'] = predictions[mean_key] - \
                                             z_score * predictions[std_key]
                intervals[f'{period}_upper'] = predictions[mean_key] + \
                                             z_score * predictions[std_key]
        
        return intervals

def build_prediction_module(config: PredictionConfig) -> nn.Module:
    """构建预测模块"""
    return EnhancedPredictionModule(config)

# models/prediction_analyzer.py

class PredictionAnalyzer:
    """
    预测分析器
    用于评估预测质量和置信度
    """
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.prediction_history = []
        self.calibration_scores = {
            'price': [],
            'trend': [],
            'signal': []
        }
        
    def analyze_prediction(self, 
                         predictions: Dict[str, torch.Tensor],
                         actual_values: Optional[Dict[str, torch.Tensor]] = None
                         ) -> Dict[str, float]:
        """
        分析预测结果
        """
        # 记录预测历史
        self.prediction_history.append({
            'predictions': {k: v.detach().cpu().numpy() 
                          for k, v in predictions.items()},
            'actual_values': {k: v.detach().cpu().numpy() 
                            for k, v in (actual_values or {}).items()}
        })
        
        analysis_results = {}
        
        # 预测质量分析
        quality_metrics = self._analyze_prediction_quality(predictions, actual_values)
        analysis_results.update(quality_metrics)
        
        # 不确定性分析
        uncertainty_metrics = self._analyze_uncertainty(predictions)
        analysis_results.update(uncertainty_metrics)
        
        # 置信度分析
        confidence_metrics = self._analyze_confidence(predictions, actual_values)
        analysis_results.update(confidence_metrics)
        
        # 预测一致性分析
        consistency_metrics = self._analyze_prediction_consistency(predictions)
        analysis_results.update(consistency_metrics)
        
        return analysis_results
    
    def _analyze_prediction_quality(self,
                                  predictions: Dict[str, torch.Tensor],
                                  actual_values: Optional[Dict[str, torch.Tensor]]
                                  ) -> Dict[str, float]:
        """
        分析预测质量
        """
        metrics = {}
        
        if actual_values is not None:
            # 价格预测误差
            for period in self.config.price_periods:
                mean_key = f'{period}_mean'
                if mean_key in predictions and period in actual_values:
                    mse = F.mse_loss(predictions[mean_key], 
                                   actual_values[period]).item()
                    mae = F.l1_loss(predictions[mean_key], 
                                  actual_values[period]).item()
                    metrics[f'{period}_mse'] = mse
                    metrics[f'{period}_mae'] = mae
            
            # 趋势预测准确率
            if 'trend_probs' in predictions and 'trend' in actual_values:
                trend_acc = (predictions['trend_probs'].argmax(dim=-1) == 
                           actual_values['trend']).float().mean().item()
                metrics['trend_accuracy'] = trend_acc
            
            # 信号预测准确率
            if 'signal_probs' in predictions and 'signal' in actual_values:
                signal_acc = (predictions['signal_probs'].argmax(dim=-1) == 
                            actual_values['signal']).float().mean().item()
                metrics['signal_accuracy'] = signal_acc
        
        return metrics
    
    def _analyze_uncertainty(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        分析预测不确定性
        """
        metrics = {}
        
        # 认知不确定性
        if 'epistemic_uncertainty' in predictions:
            epistemic = predictions['epistemic_uncertainty'].mean().item()
            metrics['epistemic_uncertainty'] = epistemic
        
        # 随机不确定性
        if 'aleatoric_uncertainty' in predictions:
            aleatoric = predictions['aleatoric_uncertainty'].mean().item()
            metrics['aleatoric_uncertainty'] = aleatoric
        
        # 预测区间覆盖率
        if len(self.prediction_history) > 1:
            coverage_rates = self._calculate_interval_coverage()
            metrics.update(coverage_rates)
        
        return metrics
    
    def _analyze_confidence(self,
                          predictions: Dict[str, torch.Tensor],
                          actual_values: Optional[Dict[str, torch.Tensor]]
                          ) -> Dict[str, float]:
        """
        分析预测置信度
        """
        metrics = {}
        
        if 'confidence_scores' in predictions:
            confidence_scores = predictions['confidence_scores']
            
            # 平均置信度
            metrics['avg_price_confidence'] = confidence_scores[:, 0].mean().item()
            metrics['avg_trend_confidence'] = confidence_scores[:, 1].mean().item()
            metrics['avg_signal_confidence'] = confidence_scores[:, 2].mean().item()
            
            # 置信度校准
            if actual_values is not None:
                calibration_metrics = self._calculate_calibration_scores(
                    predictions, actual_values, confidence_scores)
                metrics.update(calibration_metrics)
        
        return metrics
    
    def _analyze_prediction_consistency(self,
                                     predictions: Dict[str, torch.Tensor]
                                     ) -> Dict[str, float]:
        """
        分析预测一致性
        """
        metrics = {}
        
        # 集成成员间的一致性
        if 'ensemble_predictions' in predictions:
            ensemble_preds = predictions['ensemble_predictions']
            
            # 价格预测一致性
            price_agreement = self._calculate_ensemble_agreement(
                [member['price'] for member in ensemble_preds])
            metrics['price_ensemble_agreement'] = price_agreement
            
            # 趋势预测一致性
            trend_agreement = self._calculate_ensemble_agreement(
                [member['trend'] for member in ensemble_preds])
            metrics['trend_ensemble_agreement'] = trend_agreement
            
            # 信号预测一致性
            signal_agreement = self._calculate_ensemble_agreement(
                [member['signal'] for member in ensemble_preds])
            metrics['signal_ensemble_agreement'] = signal_agreement
        
        return metrics
    
    def _calculate_interval_coverage(self) -> Dict[str, float]:
        """
        计算预测区间覆盖率
        """
        coverage_rates = {}
        
        for period in self.config.price_periods:
            actuals = []
            intervals = []
            
            for hist in self.prediction_history:
                if period in hist['actual_values']:
                    actuals.append(hist['actual_values'][period])
                    intervals.append((
                        hist['predictions'][f'{period}_lower'],
                        hist['predictions'][f'{period}_upper']
                    ))
            
            if actuals and intervals:
                actuals = np.stack(actuals)
                lower = np.stack([i[0] for i in intervals])
                upper = np.stack([i[1] for i in intervals])
                
                coverage = np.mean((actuals >= lower) & (actuals <= upper))
                coverage_rates[f'{period}_coverage'] = coverage
        
        return coverage_rates
    
    def _calculate_calibration_scores(self,
                                    predictions: Dict[str, torch.Tensor],
                                    actual_values: Dict[str, torch.Tensor],
                                    confidence_scores: torch.Tensor
                                    ) -> Dict[str, float]:
        """
        计算置信度校准分数
        """
        calibration_metrics = {}
        
        # 价格预测校准
        for period in self.config.price_periods:
            mean_key = f'{period}_mean'
            if mean_key in predictions and period in actual_values:
                price_error = torch.abs(predictions[mean_key] - 
                                      actual_values[period])
                calibration = self._brier_score(
                    confidence_scores[:, 0],
                    price_error
                )
                self.calibration_scores['price'].append(calibration)
                calibration_metrics[f'{period}_calibration'] = calibration
        
        # 趋势预测校准
        if 'trend_probs' in predictions and 'trend' in actual_values:
            trend_error = (predictions['trend_probs'].argmax(dim=-1) != 
                         actual_values['trend']).float()
            trend_calibration = self._brier_score(
                confidence_scores[:, 1],
                trend_error
            )
            self.calibration_scores['trend'].append(trend_calibration)
            calibration_metrics['trend_calibration'] = trend_calibration
        
        # 信号预测校准
        if 'signal_probs' in predictions and 'signal' in actual_values:
            signal_error = (predictions['signal_probs'].argmax(dim=-1) != 
                          actual_values['signal']).float()
            signal_calibration = self._brier_score(
                confidence_scores[:, 2],
                signal_error
            )
            self.calibration_scores['signal'].append(signal_calibration)
            calibration_metrics['signal_calibration'] = signal_calibration
        
        return calibration_metrics
    
    def _calculate_ensemble_agreement(self, predictions: List[torch.Tensor]) -> float:
        """
        计算集成成员间的预测一致性
        """
        predictions = torch.stack(predictions)
        if predictions.dim() > 2:
            predictions = predictions.argmax(dim=-1)
        
        # 计算成对一致性
        agreement_matrix = (predictions.unsqueeze(0) == 
                          predictions.unsqueeze(1)).float()
        
        # 平均一致性得分
        return agreement_matrix.mean().item()
    
    def _brier_score(self, confidence: torch.Tensor, error: torch.Tensor) -> float:
        """
        计算Brier分数（置信度校准指标）
        """
        return F.mse_loss(confidence, error).item()
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        获取分析总结
        """
        if not self.prediction_history:
            return {}
        
        summary = {
            'total_predictions': len(self.prediction_history),
            'calibration_trend': {
                k: np.mean(v[-100:]) if v else 0
                for k, v in self.calibration_scores.items()
            },
            'recent_metrics': self._get_recent_metrics(),
            'stability_metrics': self._calculate_stability_metrics()
        }
        
        return summary
    
    def _get_recent_metrics(self, window: int = 100) -> Dict[str, float]:
        """
        获取最近的性能指标
        """
        recent_history = self.prediction_history[-window:]
        
        metrics = defaultdict(list)
        for hist in recent_history:
            for k, v in hist.items():
                if isinstance(v, (int, float)):
                    metrics[k].append(v)
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def _calculate_stability_metrics(self) -> Dict[str, float]:
        """
        计算预测稳定性指标
        """
        stability_metrics = {}
        
        # 预测波动性
        for period in self.config.price_periods:
            mean_key = f'{period}_mean'
            predictions = [h['predictions'][mean_key] 
                         for h in self.prediction_history]
            if predictions:
                predictions = np.stack(predictions)
                stability_metrics[f'{period}_volatility'] = np.std(predictions)
        
        # 置信度稳定性
        confidence_scores = [h['predictions']['confidence_scores'] 
                           for h in self.prediction_history
                           if 'confidence_scores' in h['predictions']]
        if confidence_scores:
            confidence_scores = np.stack(confidence_scores)
            stability_metrics['confidence_stability'] = np.std(confidence_scores)
        
        return stability_metrics

def build_prediction_analyzer(config: PredictionConfig) -> PredictionAnalyzer:
    """构建预测分析器"""
    return PredictionAnalyzer(config)