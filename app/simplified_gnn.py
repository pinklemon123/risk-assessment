"""
简化版本的图神经网络模型，不依赖torch_geometric
专为欺诈检测系统优化
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimplifiedGCNLayer(nn.Module):
    """简化的图卷积层"""
    
    def __init__(self, in_features: int, out_features: int):
        super(SimplifiedGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        x: [N, in_features] 节点特征
        adj_matrix: [N, N] 邻接矩阵
        """
        # 度归一化
        degree = adj_matrix.sum(dim=1, keepdim=True).clamp(min=1)
        adj_norm = adj_matrix / degree
        
        # 图卷积: A * X * W
        support = self.linear(x)
        output = torch.mm(adj_norm, support)
        
        return output


class FraudDetectionGNN(nn.Module):
    """欺诈检测图神经网络"""
    
    def __init__(self, 
                 node_features: int = 64,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_classes: int = 2,
                 dropout: float = 0.1):
        super(FraudDetectionGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN层
        self.gcn_layers = nn.ModuleList()
        
        # 第一层
        self.gcn_layers.append(SimplifiedGCNLayer(node_features, hidden_dim))
        
        # 隐藏层
        for _ in range(num_layers - 2):
            self.gcn_layers.append(SimplifiedGCNLayer(hidden_dim, hidden_dim))
        
        # 输出层
        if num_layers > 1:
            self.gcn_layers.append(SimplifiedGCNLayer(hidden_dim, hidden_dim))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 注意力机制用于图级别预测
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor, 
                batch_idx: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        x: [N, node_features] 节点特征
        adj_matrix: [N, N] 邻接矩阵
        batch_idx: [N] 批次索引（可选）
        """
        
        # GCN层前向传播
        h = x
        for i, gcn_layer in enumerate(self.gcn_layers):
            h = gcn_layer(h, adj_matrix)
            
            if i < len(self.gcn_layers) - 1:  # 除了最后一层
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # 节点级别预测
        node_logits = self.classifier(h)
        
        # 图级别预测（使用注意力聚合）
        if batch_idx is not None:
            graph_embedding = self._graph_level_pooling(h, batch_idx)
        else:
            # 单图情况，使用注意力池化
            attention_weights = torch.softmax(self.attention(h), dim=0)
            graph_embedding = torch.sum(attention_weights * h, dim=0, keepdim=True)
        
        graph_logits = self.classifier(graph_embedding)
        
        return {
            'node_logits': node_logits,
            'graph_logits': graph_logits,
            'node_embeddings': h,
            'graph_embedding': graph_embedding
        }
    
    def _graph_level_pooling(self, x: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """图级别池化"""
        batch_size = batch_idx.max().item() + 1
        graph_embeddings = []
        
        for i in range(batch_size):
            mask = (batch_idx == i)
            if mask.sum() > 0:
                # 对每个图使用注意力池化
                graph_nodes = x[mask]
                attention_weights = torch.softmax(self.attention(graph_nodes), dim=0)
                graph_emb = torch.sum(attention_weights * graph_nodes, dim=0)
                graph_embeddings.append(graph_emb)
        
        return torch.stack(graph_embeddings)


class GraphFeatureExtractor(nn.Module):
    """图特征提取器"""
    
    def __init__(self, input_dim: int, output_dim: int = 64):
        super(GraphFeatureExtractor, self).__init__()
        
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 边特征编码器（如果需要）
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, 16),  # 假设边只有权重特征
            nn.ReLU(),
            nn.Linear(16, 8)
        )
    
    def forward(self, node_features: torch.Tensor, edge_weights: torch.Tensor = None):
        """
        node_features: [N, input_dim] 原始节点特征
        edge_weights: [E, 1] 边权重（可选）
        """
        encoded_nodes = self.node_encoder(node_features)
        
        if edge_weights is not None:
            encoded_edges = self.edge_encoder(edge_weights)
            return encoded_nodes, encoded_edges
        
        return encoded_nodes


class FraudDetectionPipeline:
    """欺诈检测完整流水线"""
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        
        # 模型组件
        self.feature_extractor = GraphFeatureExtractor(
            input_dim=50,  # 根据实际特征维度调整
            output_dim=64
        ).to(self.device)
        
        self.gnn_model = FraudDetectionGNN(
            node_features=64,
            hidden_dim=128,
            num_layers=3,
            num_classes=2
        ).to(self.device)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + 
            list(self.gnn_model.parameters()),
            lr=0.001,
            weight_decay=1e-5
        )
        
    def prepare_graph_data(self, transactions: List[Dict], 
                          accounts: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备图数据"""
        
        # 构建节点特征矩阵
        node_features = []
        for account in accounts:
            features = [
                account.get('balance', 0),
                account.get('transaction_count', 0),
                account.get('avg_transaction_amount', 0),
                # 添加更多特征...
            ]
            # 填充到固定长度
            while len(features) < 50:
                features.append(0)
            node_features.append(features[:50])
        
        node_features = torch.tensor(node_features, dtype=torch.float32)
        
        # 构建邻接矩阵
        num_nodes = len(accounts)
        adj_matrix = torch.zeros(num_nodes, num_nodes)
        
        account_to_idx = {acc['account_id']: i for i, acc in enumerate(accounts)}
        
        for txn in transactions:
            from_id = txn.get('from_account_id')
            to_id = txn.get('to_account_id')
            
            if from_id in account_to_idx and to_id in account_to_idx:
                i, j = account_to_idx[from_id], account_to_idx[to_id]
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1  # 无向图
        
        return node_features.to(self.device), adj_matrix.to(self.device)
    
    def predict(self, transactions: List[Dict], accounts: List[Dict]) -> Dict[str, float]:
        """预测欺诈风险"""
        self.feature_extractor.eval()
        self.gnn_model.eval()
        
        with torch.no_grad():
            # 准备数据
            node_features, adj_matrix = self.prepare_graph_data(transactions, accounts)
            
            # 特征提取
            encoded_features = self.feature_extractor(node_features)
            
            # GNN预测
            outputs = self.gnn_model(encoded_features, adj_matrix)
            
            # 获取图级别预测
            graph_logits = outputs['graph_logits']
            graph_probs = torch.softmax(graph_logits, dim=1)
            
            # 获取节点级别预测
            node_logits = outputs['node_logits']
            node_probs = torch.softmax(node_logits, dim=1)
            
            # 计算综合风险分数
            graph_risk_score = graph_probs[0, 1].item()  # 欺诈类别概率
            avg_node_risk = node_probs[:, 1].mean().item()
            
            # 加权综合
            final_risk_score = 0.7 * graph_risk_score + 0.3 * avg_node_risk
            
            return {
                'risk_score': final_risk_score,
                'graph_risk_score': graph_risk_score,
                'avg_node_risk_score': avg_node_risk,
                'risk_level': 'high' if final_risk_score > 0.7 else 
                            'medium' if final_risk_score > 0.4 else 'low',
                'confidence': min(0.95, max(0.5, abs(final_risk_score - 0.5) * 2))
            }
    
    def train_step(self, transactions_batch: List[List[Dict]], 
                   accounts_batch: List[List[Dict]], 
                   labels: torch.Tensor) -> float:
        """训练步骤"""
        self.feature_extractor.train()
        self.gnn_model.train()
        
        total_loss = 0
        
        for transactions, accounts, label in zip(transactions_batch, accounts_batch, labels):
            self.optimizer.zero_grad()
            
            # 准备数据
            node_features, adj_matrix = self.prepare_graph_data(transactions, accounts)
            
            # 前向传播
            encoded_features = self.feature_extractor(node_features)
            outputs = self.gnn_model(encoded_features, adj_matrix)
            
            # 计算损失
            graph_loss = self.criterion(outputs['graph_logits'], label.unsqueeze(0))
            
            # 反向传播
            graph_loss.backward()
            self.optimizer.step()
            
            total_loss += graph_loss.item()
        
        return total_loss / len(transactions_batch)


# 创建全局模型实例
fraud_gnn_pipeline = FraudDetectionPipeline()

# 简化的预测函数，用于API调用
async def predict_fraud_risk_gnn(transactions: List[Dict], 
                                accounts: List[Dict]) -> Dict[str, float]:
    """使用GNN进行欺诈风险预测的异步接口"""
    try:
        result = fraud_gnn_pipeline.predict(transactions, accounts)
        return result
    except Exception as e:
        # 降级到简单规则
        print(f"GNN prediction failed: {e}, using fallback")
        return {
            'risk_score': 0.5,
            'risk_level': 'medium',
            'confidence': 0.3,
            'error': str(e)
        }