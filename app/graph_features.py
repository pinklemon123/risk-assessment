from typing import Dict, List, Any
import networkx as nx
import pandas as pd
from datetime import datetime, timedelta

class GraphFeatureEngineer:
    """图特征工程器"""
    
    def __init__(self):
        self.graph = nx.Graph()
        
    def build_account_device_transaction_graph(self, transactions: List[Dict]) -> nx.Graph:
        """构建账户-设备-交易图"""
        
        for txn in transactions:
            account_id = txn.get('src_entity_id')
            device_id = txn.get('device_id')
            merchant_id = txn.get('merchant_id')
            
            # 添加节点
            self.graph.add_node(f"account_{account_id}", type="account")
            self.graph.add_node(f"device_{device_id}", type="device") 
            self.graph.add_node(f"merchant_{merchant_id}", type="merchant")
            
            # 添加边权重
            self.graph.add_edge(
                f"account_{account_id}", 
                f"device_{device_id}",
                weight=self._calculate_trust_score(txn),
                last_used=txn.get('ts')
            )
            
            self.graph.add_edge(
                f"device_{device_id}",
                f"merchant_{merchant_id}", 
                weight=float(txn.get('amount', 0)),
                transaction_time=txn.get('ts')
            )
            
        return self.graph
    
    def extract_graph_features(self, entity_id: str) -> Dict[str, float]:
        """提取图结构特征"""
        
        if entity_id not in self.graph:
            return self._get_default_features()
            
        features = {
            # 基础图特征
            'node_degree': self.graph.degree(entity_id),
            'clustering_coefficient': nx.clustering(self.graph, entity_id),
            
            # 中心性特征
            'betweenness_centrality': nx.betweenness_centrality(self.graph)[entity_id],
            'closeness_centrality': nx.closeness_centrality(self.graph)[entity_id],
            'eigenvector_centrality': nx.eigenvector_centrality(self.graph)[entity_id],
            
            # 邻居特征
            'neighbor_risk_avg': self._calculate_neighbor_risk(entity_id),
            'shared_neighbors_count': self._count_shared_neighbors(entity_id),
            
            # 路径特征  
            'avg_shortest_path_length': self._avg_shortest_path_length(entity_id),
            'reachable_nodes_count': len(nx.node_connected_component(self.graph, entity_id))
        }
        
        return features
        
    def detect_suspicious_patterns(self) -> List[Dict]:
        """检测可疑图模式"""
        suspicious_patterns = []
        
        # 1. 检测密集子图（可能的欺诈团伙）
        cliques = list(nx.find_cliques(self.graph))
        for clique in cliques:
            if len(clique) >= 4:  # 4个以上节点的完全子图
                suspicious_patterns.append({
                    'pattern_type': 'dense_subgraph',
                    'nodes': list(clique),
                    'risk_score': len(clique) * 0.2
                })
                
        # 2. 检测异常连接模式
        for node in self.graph.nodes():
            degree = self.graph.degree(node)
            if degree > 50:  # 连接过多设备/账户
                suspicious_patterns.append({
                    'pattern_type': 'high_degree_node',
                    'node': node,
                    'degree': degree,
                    'risk_score': min(degree / 100, 1.0)
                })
                
        return suspicious_patterns
    
    def _calculate_trust_score(self, txn: Dict) -> float:
        """计算信任度分数"""
        # 基于历史行为计算信任度
        base_score = 0.5
        
        # 时间因子：最近的交易权重更高
        time_factor = self._get_time_decay_factor(txn.get('ts'))
        
        # 金额因子：异常金额降低信任度
        amount_factor = self._get_amount_normalcy_factor(txn.get('amount'))
        
        return base_score * time_factor * amount_factor
    
    def _calculate_neighbor_risk(self, entity_id: str) -> float:
        """计算邻居风险平均值"""
        neighbors = list(self.graph.neighbors(entity_id))
        if not neighbors:
            return 0.0
            
        # 这里需要从风险评分系统获取邻居的风险分数
        risk_scores = []
        for neighbor in neighbors:
            # 简化版本，实际需要查询风险数据库
            risk_scores.append(0.1)  # 默认低风险
            
        return sum(risk_scores) / len(risk_scores)
    
    def _get_default_features(self) -> Dict[str, float]:
        """返回默认特征值"""
        return {
            'node_degree': 0,
            'clustering_coefficient': 0,
            'betweenness_centrality': 0,
            'closeness_centrality': 0, 
            'eigenvector_centrality': 0,
            'neighbor_risk_avg': 0,
            'shared_neighbors_count': 0,
            'avg_shortest_path_length': 0,
            'reachable_nodes_count': 0
        }