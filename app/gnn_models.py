from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 如果torch_geometric不可用，使用简化版本
try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not available, using simplified implementations")

class FraudGNN(nn.Module):
    """欺诈检测图神经网络"""
    
    def __init__(self, 
                 input_dim: int = 64,
                 hidden_dim: int = 128, 
                 output_dim: int = 2,
                 num_layers: int = 3,
                 dropout: float = 0.3):
        super(FraudGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 图卷积层
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=True))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True))
            
        self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x, edge_index, batch=None):
        # 图卷积
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
                
        # 如果有batch，进行图级别预测
        if batch is not None:
            x = torch_geometric.nn.global_mean_pool(x, batch)
            
        # 分类
        out = self.classifier(x)
        return F.log_softmax(out, dim=-1)

class MultiHopDetector:
    """多跳检测器"""
    
    def __init__(self, max_hops: int = 3):
        self.max_hops = max_hops
        
    def detect_multi_hop_patterns(self, graph, source_nodes: List[str]) -> Dict:
        """检测多跳模式"""
        results = {}
        
        for source in source_nodes:
            if source not in graph:
                continue
                
            hop_analysis = self._analyze_hops(graph, source)
            
            # 计算风险传播分数
            risk_propagation = self._calculate_risk_propagation(hop_analysis)
            
            results[source] = {
                'hop_analysis': hop_analysis,
                'risk_propagation_score': risk_propagation,
                'suspicious_paths': self._find_suspicious_paths(graph, source)
            }
            
        return results
    
    def _analyze_hops(self, graph, source: str) -> Dict:
        """分析多跳连接"""
        hop_data = {}
        
        for hop in range(1, self.max_hops + 1):
            # 获取第hop跳的所有节点
            nodes_at_hop = self._get_nodes_at_hop(graph, source, hop)
            
            hop_data[f'hop_{hop}'] = {
                'node_count': len(nodes_at_hop),
                'node_types': self._analyze_node_types(nodes_at_hop),
                'risk_distribution': self._analyze_risk_distribution(nodes_at_hop),
                'connection_strength': self._calculate_connection_strength(
                    graph, source, nodes_at_hop
                )
            }
            
        return hop_data
    
    def _calculate_risk_propagation(self, hop_analysis: Dict) -> float:
        """计算风险传播分数"""
        base_score = 0.0
        
        for hop_key, hop_data in hop_analysis.items():
            hop_num = int(hop_key.split('_')[1])
            
            # 距离衰减因子
            decay_factor = 0.7 ** (hop_num - 1)
            
            # 节点风险加权
            hop_risk = hop_data['risk_distribution'].get('high_risk_ratio', 0)
            
            # 连接强度加权
            connection_weight = hop_data['connection_strength']
            
            base_score += hop_risk * connection_weight * decay_factor
            
        return min(base_score, 1.0)
    
    def optimize_detection_performance(self) -> Dict:
        """性能优化建议"""
        return {
            'indexing_strategy': {
                'create_node_type_index': "CREATE INDEX idx_node_type ON nodes(type)",
                'create_edge_weight_index': "CREATE INDEX idx_edge_weight ON edges(weight)",
                'create_timestamp_index': "CREATE INDEX idx_timestamp ON transactions(ts)"
            },
            
            'caching_strategy': {
                'node_features_cache_ttl': 3600,  # 1小时
                'graph_structure_cache_ttl': 1800,  # 30分钟
                'risk_scores_cache_ttl': 600  # 10分钟
            },
            
            'batch_processing': {
                'max_batch_size': 1000,
                'parallel_workers': 4,
                'processing_window': '5m'
            },
            
            'model_optimization': {
                'use_gpu_acceleration': True,
                'enable_mixed_precision': True,
                'model_quantization': '8bit',
                'inference_batch_size': 256
            }
        }

class EnsembleRiskScorer:
    """集成风险评分器"""
    
    def __init__(self):
        self.models = {
            'graph_neural_network': FraudGNN(),
            'multi_hop_detector': MultiHopDetector(),
            'traditional_ml': None  # 可以集成XGBoost等
        }
        
    def compute_ensemble_score(self, 
                             transaction_data: Dict,
                             graph_features: Dict,
                             context_features: Dict) -> Dict:
        """计算集成风险分数"""
        
        scores = {}
        
        # GNN评分
        if 'graph_neural_network' in self.models:
            scores['gnn_score'] = self._compute_gnn_score(
                transaction_data, graph_features
            )
            
        # 多跳检测评分
        if 'multi_hop_detector' in self.models:
            scores['multi_hop_score'] = self._compute_multi_hop_score(
                transaction_data, context_features
            )
            
        # 传统ML评分
        scores['traditional_ml_score'] = self._compute_traditional_score(
            transaction_data
        )
        
        # 加权集成
        ensemble_score = self._weighted_ensemble(scores)
        
        return {
            'final_score': ensemble_score,
            'component_scores': scores,
            'confidence': self._calculate_confidence(scores),
            'explanation': self._generate_explanation(scores)
        }
    
    def _weighted_ensemble(self, scores: Dict) -> float:
        """加权集成多个模型的分数"""
        weights = {
            'gnn_score': 0.4,
            'multi_hop_score': 0.3,
            'traditional_ml_score': 0.3
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for score_name, score_value in scores.items():
            if score_name in weights and score_value is not None:
                weighted_sum += weights[score_name] * score_value
                total_weight += weights[score_name]
                
        return weighted_sum / total_weight if total_weight > 0 else 0.0