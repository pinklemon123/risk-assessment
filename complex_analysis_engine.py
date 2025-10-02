"""
复杂风险分析引擎 - 包含真正需要计算时间的算法
包括机器学习、图网络分析、时间序列异常检测等
"""

import numpy as np
import time
import json
from datetime import datetime
from collections import defaultdict, deque
import math
import random
from typing import Dict, List, Tuple
import argparse

class AnalysisProgressTracker:
    """分析进度跟踪器"""
    
    def __init__(self):
        self.progress_file = "analysis_progress.json"
        self.start_time = time.time()
        self.current_step = ""
        self.steps_completed = []
        self.total_steps = 10  # 预计总步骤数
    
    def update_progress(self, step_name: str, progress: float = None):
        """更新分析进度"""
        self.current_step = step_name
        if progress is None:
            progress = len(self.steps_completed) / self.total_steps * 100
        
        progress_data = {
            "status": "running",
            "current_step": step_name,
            "steps": self.steps_completed + [step_name],
            "progress_percentage": min(progress, 100),
            "elapsed_time": time.time() - self.start_time
        }
        
        # 写入进度文件
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {step_name} (进度: {progress:.1f}%)")
    
    def complete_step(self, step_name: str):
        """完成一个步骤"""
        if step_name not in self.steps_completed:
            self.steps_completed.append(step_name)
        
        progress = len(self.steps_completed) / self.total_steps * 100
        self.update_progress(f"已完成: {step_name}", progress)
    
    def finish(self):
        """完成所有分析"""
        progress_data = {
            "status": "completed",
            "current_step": "分析完成",
            "steps": self.steps_completed,
            "progress_percentage": 100,
            "elapsed_time": time.time() - self.start_time
        }
        
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎉 分析完成! 总耗时: {time.time() - self.start_time:.2f}秒")

# 全局进度跟踪器
progress_tracker = AnalysisProgressTracker()

class NetworkAnalyzer:
    """交易网络分析器 - 需要大量计算的图算法"""
    
    def __init__(self):
        self.transaction_graph = defaultdict(list)
        self.account_graph = defaultdict(set)
        self.merchant_graph = defaultdict(set)
        
    def build_transaction_network(self, transactions: List[Dict]) -> Dict:
        """构建交易网络图 - 计算密集型操作"""
        progress_tracker.update_progress("构建交易网络图结构...")
        start_time = time.time()
        
        # 构建账户-商户关系图
        for i, txn in enumerate(transactions):
            if i % 500 == 0:  # 更频繁的进度更新
                progress = (i / len(transactions)) * 20  # 网络构建占总进度的20%
                progress_tracker.update_progress(f"处理交易数据: {i}/{len(transactions)}", progress)
                
            account = txn["account_id"]
            merchant = txn["merchant_id"]
            amount = txn["amount"]
            timestamp = datetime.fromisoformat(txn["timestamp"])
            
            # 添加边权重（基于交易频率和金额）
            edge_weight = math.log(1 + amount) * (1 / max(1, len(self.transaction_graph[account])))
            
            self.transaction_graph[account].append({
                "merchant": merchant,
                "weight": edge_weight,
                "timestamp": timestamp,
                "amount": amount
            })
            
            self.account_graph[account].add(merchant)
            self.merchant_graph[merchant].add(account)
        
        progress_tracker.complete_step("交易网络图构建完成")
        
        # 计算网络统计
        progress_tracker.update_progress("计算网络中心性指标...", 25)
        centrality_scores = self.calculate_centrality_measures()
        progress_tracker.complete_step("中心性指标计算完成")
        
        # 检测社群
        progress_tracker.update_progress("检测交易社群结构...", 35)
        communities = self.detect_communities()
        progress_tracker.complete_step("社群检测完成")
        
        # 异常连接检测
        progress_tracker.update_progress("检测异常连接模式...", 40)
        anomalous_patterns = self.detect_anomalous_patterns()
        progress_tracker.complete_step("异常模式检测完成")
        
        elapsed = time.time() - start_time
        print(f"  网络分析完成，耗时: {elapsed:.2f}秒")
        
        return {
            "network_stats": {
                "total_accounts": len(self.account_graph),
                "total_merchants": len(self.merchant_graph),
                "total_edges": sum(len(edges) for edges in self.transaction_graph.values()),
                "avg_degree": np.mean([len(neighbors) for neighbors in self.account_graph.values()]),
                "network_density": self.calculate_network_density()
            },
            "centrality_scores": centrality_scores,
            "communities": communities,
            "anomalous_patterns": anomalous_patterns,
            "processing_time": elapsed
        }
        
    def calculate_centrality_measures(self) -> Dict:
        """计算各种中心性指标 - 计算复杂度O(N²)"""
        centrality = {}
        
        # 度中心性
        for account in self.account_graph:
            degree = len(self.account_graph[account])
            centrality[account] = {"degree": degree}
            
        # 简化的接近中心性计算
        print("   计算接近中心性...")
        for i, account in enumerate(self.account_graph):
            if i % 500 == 0:
                print(f"      进度: {i/len(self.account_graph)*100:.1f}%")
                
            # BFS计算最短路径
            distances = self.bfs_shortest_paths(account)
            avg_distance = np.mean(list(distances.values())) if distances else 0
            closeness = 1 / (1 + avg_distance)
            centrality[account]["closeness"] = closeness
            
        # 介数中心性（简化版本）
        print("   计算介数中心性...")
        betweenness = self.calculate_betweenness_centrality()
        for account in centrality:
            centrality[account]["betweenness"] = betweenness.get(account, 0)
            
        return centrality
        
    def bfs_shortest_paths(self, start_account: str, max_distance: int = 3) -> Dict[str, int]:
        """BFS计算最短路径"""
        distances = {start_account: 0}
        queue = deque([(start_account, 0)])
        
        while queue:
            current, dist = queue.popleft()
            if dist >= max_distance:
                continue
                
            for neighbor in self.account_graph.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
                    
        return distances
        
    def calculate_betweenness_centrality(self) -> Dict[str, float]:
        """计算介数中心性（简化版）"""
        betweenness = defaultdict(float)
        accounts = list(self.account_graph.keys())
        
        # 采样计算以减少计算量
        sample_size = min(500, len(accounts))
        sampled_accounts = random.sample(accounts, sample_size)
        
        for i, source in enumerate(sampled_accounts):
            if i % 50 == 0:
                print(f"      介数中心性进度: {i/len(sampled_accounts)*100:.1f}%")
                
            paths = self.find_shortest_paths_from_source(source)
            for path in paths:
                for node in path[1:-1]:  # 中间节点
                    betweenness[node] += 1.0 / len(paths)
                    
        return dict(betweenness)
        
    def find_shortest_paths_from_source(self, source: str, max_paths: int = 100) -> List[List[str]]:
        """从源节点查找最短路径"""
        paths = []
        visited = set()
        
        def dfs(current, path, depth):
            if len(paths) >= max_paths or depth > 4:
                return
                
            if current in visited and len(path) > 1:
                paths.append(path.copy())
                return
                
            visited.add(current)
            for neighbor in self.account_graph.get(current, []):
                if neighbor not in path:  # 避免循环
                    path.append(neighbor)
                    dfs(neighbor, path, depth + 1)
                    path.pop()
            visited.remove(current)
        
        dfs(source, [source], 0)
        return paths
        
    def detect_communities(self) -> List[Dict]:
        """社群检测算法"""
        communities = []
        visited = set()
        
        for account in self.account_graph:
            if account not in visited:
                community = self.expand_community(account, visited)
                if len(community) >= 3:  # 至少3个成员才算社群
                    communities.append({
                        "members": list(community),
                        "size": len(community),
                        "cohesion": self.calculate_community_cohesion(community)
                    })
                    
        return sorted(communities, key=lambda x: x["size"], reverse=True)
        
    def expand_community(self, seed: str, visited: set) -> set:
        """扩展社群"""
        community = {seed}
        visited.add(seed)
        queue = deque([seed])
        
        while queue:
            current = queue.popleft()
            neighbors = self.account_graph.get(current, set())
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    # 计算连接强度
                    connection_strength = self.calculate_connection_strength(current, neighbor)
                    if connection_strength > 0.3:  # 阈值
                        community.add(neighbor)
                        visited.add(neighbor)
                        queue.append(neighbor)
                        
        return community
        
    def calculate_connection_strength(self, account1: str, account2: str) -> float:
        """计算两个账户之间的连接强度"""
        merchants1 = self.account_graph.get(account1, set())
        merchants2 = self.account_graph.get(account2, set())
        
        if not merchants1 or not merchants2:
            return 0.0
            
        intersection = len(merchants1 & merchants2)
        union = len(merchants1 | merchants2)
        
        return intersection / union if union > 0 else 0.0
        
    def calculate_community_cohesion(self, community: set) -> float:
        """计算社群内聚性"""
        if len(community) < 2:
            return 0.0
            
        total_connections = 0
        possible_connections = len(community) * (len(community) - 1)
        
        for member1 in community:
            for member2 in community:
                if member1 != member2:
                    strength = self.calculate_connection_strength(member1, member2)
                    if strength > 0.1:
                        total_connections += strength
                        
        return total_connections / possible_connections if possible_connections > 0 else 0.0
        
    def detect_anomalous_patterns(self) -> List[Dict]:
        """检测异常交易模式"""
        anomalies = []
        
        # 检测异常高频交易
        for account, transactions in self.transaction_graph.items():
            if len(transactions) > 50:  # 高频交易账户
                time_intervals = []
                sorted_txns = sorted(transactions, key=lambda x: x["timestamp"])
                
                for i in range(1, len(sorted_txns)):
                    interval = (sorted_txns[i]["timestamp"] - sorted_txns[i-1]["timestamp"]).total_seconds()
                    time_intervals.append(interval)
                    
                if time_intervals:
                    avg_interval = np.mean(time_intervals)
                    if avg_interval < 300:  # 平均间隔小于5分钟
                        anomalies.append({
                            "type": "high_frequency_trading",
                            "account": account,
                            "transaction_count": len(transactions),
                            "avg_interval_seconds": avg_interval,
                            "risk_score": 0.8
                        })
        
        # 检测圆形交易（洗钱模式）
        circular_patterns = self.detect_circular_transactions()
        anomalies.extend(circular_patterns)
        
        return anomalies
        
    def detect_circular_transactions(self) -> List[Dict]:
        """检测圆形交易模式"""
        circular_patterns = []
        
        for account in list(self.account_graph.keys())[:100]:  # 采样检测
            cycles = self.find_cycles_from_account(account, max_length=6)
            for cycle in cycles:
                if len(cycle) >= 3:
                    circular_patterns.append({
                        "type": "circular_transaction",
                        "cycle": cycle,
                        "length": len(cycle),
                        "risk_score": min(0.9, 0.4 + len(cycle) * 0.1)
                    })
                    
        return circular_patterns
        
    def find_cycles_from_account(self, start: str, max_length: int = 6) -> List[List[str]]:
        """从账户查找交易循环"""
        cycles = []
        
        def dfs_cycles(current, path, visited):
            if len(path) > max_length:
                return
                
            if current == start and len(path) > 2:
                cycles.append(path.copy())
                return
                
            if current in visited:
                return
                
            visited.add(current)
            for neighbor in self.account_graph.get(current, []):
                if neighbor == start or neighbor not in path:
                    path.append(neighbor)
                    dfs_cycles(neighbor, path, visited.copy())
                    path.pop()
        
        dfs_cycles(start, [start], set())
        return cycles[:10]  # 限制结果数量
        
    def calculate_network_density(self) -> float:
        """计算网络密度"""
        total_nodes = len(self.account_graph) + len(self.merchant_graph)
        total_edges = sum(len(edges) for edges in self.transaction_graph.values())
        max_possible_edges = total_nodes * (total_nodes - 1)
        
        return total_edges / max_possible_edges if max_possible_edges > 0 else 0.0

class TimeSeriesAnomalyDetector:
    """时间序列异常检测器 - CPU密集型算法"""
    
    def __init__(self, window_size: int = 24):
        self.window_size = window_size
        self.seasonal_patterns = {}
        
    def detect_temporal_anomalies(self, transactions: List[Dict]) -> Dict:
        """检测时间序列异常 - 需要大量计算"""
        print(" 开始时间序列异常检测...")
        start_time = time.time()
        
        # 按小时聚合交易数据
        print("   聚合时间序列数据...")
        hourly_data = self.aggregate_by_hour(transactions)
        
        # 季节性分析
        print("   分析季节性模式...")
        seasonal_analysis = self.analyze_seasonal_patterns(hourly_data)
        
        # 异常检测
        print("   检测异常时间点...")
        anomalies = self.detect_anomalies_with_isolation_forest(hourly_data)
        
        # 趋势分析
        print("   计算趋势指标...")
        trend_analysis = self.analyze_trends(hourly_data)
        
        elapsed = time.time() - start_time
        print(f"  时间序列分析完成，耗时: {elapsed:.2f}秒")
        
        return {
            "hourly_statistics": hourly_data,
            "seasonal_patterns": seasonal_analysis,
            "anomalies": anomalies,
            "trend_analysis": trend_analysis,
            "processing_time": elapsed
        }
        
    def aggregate_by_hour(self, transactions: List[Dict]) -> Dict:
        """按小时聚合交易数据"""
        hourly_stats = defaultdict(lambda: {
            "transaction_count": 0,
            "total_amount": 0.0,
            "unique_accounts": set(),
            "unique_merchants": set(),
            "fraud_count": 0
        })
        
        for txn in transactions:
            timestamp = datetime.fromisoformat(txn["timestamp"])
            hour_key = timestamp.strftime("%Y-%m-%d %H")
            
            stats = hourly_stats[hour_key]
            stats["transaction_count"] += 1
            stats["total_amount"] += txn["amount"]
            stats["unique_accounts"].add(txn["account_id"])
            stats["unique_merchants"].add(txn["merchant_id"])
            
            if txn.get("is_fraud", False):
                stats["fraud_count"] += 1
                
        # 转换set为count
        processed_data = {}
        for hour, stats in hourly_stats.items():
            processed_data[hour] = {
                "transaction_count": stats["transaction_count"],
                "total_amount": stats["total_amount"],
                "avg_amount": stats["total_amount"] / stats["transaction_count"],
                "unique_accounts": len(stats["unique_accounts"]),
                "unique_merchants": len(stats["unique_merchants"]),
                "fraud_count": stats["fraud_count"],
                "fraud_rate": stats["fraud_count"] / stats["transaction_count"]
            }
            
        return processed_data
        
    def analyze_seasonal_patterns(self, hourly_data: Dict) -> Dict:
        """分析季节性模式"""
        patterns = {
            "hourly": defaultdict(list),
            "daily": defaultdict(list),
            "weekly": defaultdict(list)
        }
        
        for hour_str, data in hourly_data.items():
            dt = datetime.strptime(hour_str, "%Y-%m-%d %H")
            
            # 小时模式
            patterns["hourly"][dt.hour].append(data["transaction_count"])
            
            # 日模式  
            patterns["daily"][dt.day].append(data["transaction_count"])
            
            # 周模式
            patterns["weekly"][dt.weekday()].append(data["transaction_count"])
            
        # 计算统计值
        seasonal_stats = {}
        for pattern_type, pattern_data in patterns.items():
            seasonal_stats[pattern_type] = {}
            for key, values in pattern_data.items():
                seasonal_stats[pattern_type][key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": min(values),
                    "max": max(values),
                    "samples": len(values)
                }
                
        return seasonal_stats
        
    def detect_anomalies_with_isolation_forest(self, hourly_data: Dict) -> List[Dict]:
        """使用隔离森林检测异常（简化版本）"""
        anomalies = []
        
        # 提取特征
        features = []
        timestamps = []
        
        for hour_str, data in hourly_data.items():
            dt = datetime.strptime(hour_str, "%Y-%m-%d %H")
            
            # 构造特征向量
            feature_vector = [
                data["transaction_count"],
                data["total_amount"],
                data["avg_amount"], 
                data["unique_accounts"],
                data["unique_merchants"],
                data["fraud_rate"],
                dt.hour,  # 时间特征
                dt.weekday(),
                math.sin(2 * math.pi * dt.hour / 24),  # 周期性特征
                math.cos(2 * math.pi * dt.hour / 24)
            ]
            
            features.append(feature_vector)
            timestamps.append(hour_str)
            
        if not features:
            return anomalies
            
        # 简化的异常检测算法
        features_array = np.array(features)
        
        # 标准化特征
        means = np.mean(features_array, axis=0)
        stds = np.std(features_array, axis=0)
        stds[stds == 0] = 1  # 避免除零
        normalized_features = (features_array - means) / stds
        
        # 计算异常分数（基于马氏距离的简化版本）
        for i, (feature_vector, timestamp) in enumerate(zip(normalized_features, timestamps)):
            # 计算到中心点的距离
            distance = np.linalg.norm(feature_vector)
            
            # 设定阈值（基于分布的95%分位数）
            if i % 100 == 0:  # 显示进度
                print(f"      异常检测进度: {i/len(features)*100:.1f}%")
                
            threshold = np.percentile([np.linalg.norm(fv) for fv in normalized_features], 95)
            
            if distance > threshold:
                anomaly_score = min(1.0, (distance - threshold) / threshold)
                anomalies.append({
                    "timestamp": timestamp,
                    "anomaly_score": anomaly_score,
                    "type": "statistical_anomaly",
                    "features": hourly_data[timestamp],
                    "distance": distance,
                    "threshold": threshold
                })
                
        # 按异常分数排序
        anomalies.sort(key=lambda x: x["anomaly_score"], reverse=True)
        
        return anomalies[:50]  # 返回前50个异常
        
    def analyze_trends(self, hourly_data: Dict) -> Dict:
        """分析趋势"""
        if len(hourly_data) < 2:
            return {"trend": "insufficient_data"}
            
        # 按时间排序
        sorted_data = sorted(hourly_data.items(), key=lambda x: x[0])
        
        # 计算移动平均
        window = min(12, len(sorted_data) // 4)  # 12小时或数据量的1/4
        moving_averages = []
        
        for i in range(len(sorted_data) - window + 1):
            window_data = [data[1]["transaction_count"] for data in sorted_data[i:i+window]]
            moving_averages.append(np.mean(window_data))
            
        # 计算趋势斜率
        if len(moving_averages) >= 2:
            x = np.arange(len(moving_averages))
            y = np.array(moving_averages)
            
            # 简单线性回归
            slope = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x))**2)
            
            trend_direction = "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
        else:
            slope = 0
            trend_direction = "stable"
            
        return {
            "trend_direction": trend_direction,
            "slope": slope,
            "moving_averages": moving_averages,
            "volatility": np.std(moving_averages) if moving_averages else 0,
            "data_points": len(sorted_data)
        }

class MachineLearningRiskModel:
    """机器学习风险评估模型 - 需要训练和推理时间"""
    
    def __init__(self):
        self.feature_weights = None
        self.training_time = 0
        
    def train_risk_model(self, transactions: List[Dict]) -> Dict:
        """训练风险评估模型 - 计算密集型"""
        print(" 训练机器学习风险模型...")
        start_time = time.time()
        
        # 特征工程
        print("   提取特征...")
        features, labels = self.extract_features_and_labels(transactions)
        
        # 简化的梯度下降训练
        print("   训练模型 (梯度下降)...")
        self.feature_weights = self.gradient_descent_training(features, labels)
        
        # 模型评估
        print("   评估模型性能...")
        evaluation_metrics = self.evaluate_model(features, labels)
        
        self.training_time = time.time() - start_time
        print(f"  模型训练完成，耗时: {self.training_time:.2f}秒")
        
        return {
            "model_performance": evaluation_metrics,
            "feature_importance": self.calculate_feature_importance(),
            "training_samples": len(transactions),
            "training_time": self.training_time
        }
        
    def extract_features_and_labels(self, transactions: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """特征工程 - 提取和构造特征"""
        features = []
        labels = []
        
        print("      特征提取进度:")
        for i, txn in enumerate(transactions):
            if i % 1000 == 0:
                print(f"        {i/len(transactions)*100:.1f}%")
                
            # 基础特征
            amount = txn.get("amount", 0)
            hour = datetime.fromisoformat(txn["timestamp"]).hour
            
            # 风险特征工程
            feature_vector = [
                math.log(1 + amount),  # 对数金额
                1 if hour < 6 or hour > 22 else 0,  # 异常时间
                len(txn.get("risk_factors", {})),  # 风险因子数量
                txn.get("calculated_risk_score", 0),  # 计算的风险分数
                1 if txn.get("location") == "unknown" else 0,  # 未知位置
                1 if "crypto" in txn.get("merchant_id", "") else 0,  # 加密货币
                1 if "casino" in txn.get("merchant_id", "") else 0,  # 赌博
                txn.get("account_age_days", 0) / 365.0,  # 账户年龄（年）
                math.sin(2 * math.pi * hour / 24),  # 时间周期性
                math.cos(2 * math.pi * hour / 24),  # 时间周期性
                
                # 复杂特征（需要计算）
                self.calculate_velocity_risk(txn),
                self.calculate_merchant_risk(txn),
                self.calculate_geographic_risk(txn),
                self.calculate_behavioral_deviation(txn, transactions)
            ]
            
            features.append(feature_vector)
            labels.append(1 if txn.get("is_fraud", False) else 0)
            
        return np.array(features), np.array(labels)
        
    def calculate_velocity_risk(self, txn: Dict) -> float:
        """计算交易速度风险"""
        velocity = txn.get("velocity_features", {})
        
        # 基于1小时和24小时的交易频率
        txn_1h = velocity.get("transactions_1h", 0)
        amount_1h = velocity.get("amount_1h", 0)
        
        # 速度风险评分
        frequency_risk = min(1.0, txn_1h / 10.0)  # 1小时超过10笔为高风险
        amount_risk = min(1.0, amount_1h / 50000.0)  # 1小时超过5万为高风险
        
        return (frequency_risk + amount_risk) / 2.0
        
    def calculate_merchant_risk(self, txn: Dict) -> float:
        """计算商户风险"""
        category = txn.get("merchant_category", "")
        
        # 高风险商户类别
        high_risk_categories = ["crypto", "casino", "jewelry", "electronics"]
        medium_risk_categories = ["online", "atm_cash"]
        
        if category in high_risk_categories:
            return 0.8
        elif category in medium_risk_categories:
            return 0.4
        else:
            return 0.1
            
    def calculate_geographic_risk(self, txn: Dict) -> float:
        """计算地理风险"""
        location = txn.get("location", "")
        
        high_risk_locations = ["unknown", "overseas", "macau"]
        if location in high_risk_locations:
            return 0.7
        else:
            return 0.2
            
    def calculate_behavioral_deviation(self, txn: Dict, all_transactions: List[Dict]) -> float:
        """计算行为偏差（简化版本）"""
        account_id = txn["account_id"]
        current_amount = txn["amount"]
        
        # 查找同一账户的历史交易（采样以减少计算量）
        account_txns = [t for t in all_transactions[:1000] if t["account_id"] == account_id]
        
        if len(account_txns) < 2:
            return 0.3  # 新账户默认风险
            
        # 计算金额偏差
        amounts = [t["amount"] for t in account_txns if t["id"] != txn["id"]]
        if amounts:
            avg_amount = np.mean(amounts)
            std_amount = np.std(amounts) if len(amounts) > 1 else avg_amount
            
            # Z-score
            z_score = abs(current_amount - avg_amount) / max(std_amount, 1.0)
            deviation_score = min(1.0, z_score / 3.0)  # 3个标准差为满分
            
            return deviation_score
        else:
            return 0.3
            
    def gradient_descent_training(self, features: np.ndarray, labels: np.ndarray, 
                                epochs: int = 100, learning_rate: float = 0.01) -> np.ndarray:
        """简化的梯度下降训练"""
        n_features = features.shape[1]
        weights = np.random.normal(0, 0.1, n_features)
        
        # 标准化特征
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        feature_stds[feature_stds == 0] = 1
        
        normalized_features = (features - feature_means) / feature_stds
        
        for epoch in range(epochs):
            if epoch % 20 == 0:
                print(f"      训练进度: {epoch/epochs*100:.1f}%")
                
            # 前向传播
            logits = np.dot(normalized_features, weights)
            predictions = self.sigmoid(logits)
            
            # 计算损失梯度
            error = predictions - labels
            gradients = np.dot(normalized_features.T, error) / len(labels)
            
            # 更新权重
            weights -= learning_rate * gradients
            
            # 学习率衰减
            if epoch > 50:
                learning_rate *= 0.99
                
        return weights
        
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid激活函数"""
        # 防止溢出
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
        
    def evaluate_model(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """评估模型性能"""
        if self.feature_weights is None:
            return {"error": "Model not trained"}
            
        # 标准化特征
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        feature_stds[feature_stds == 0] = 1
        normalized_features = (features - feature_means) / feature_stds
        
        # 预测
        logits = np.dot(normalized_features, self.feature_weights)
        probabilities = self.sigmoid(logits)
        predictions = (probabilities > 0.5).astype(int)
        
        # 计算指标
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(labels)
        
        return {
            "accuracy": accuracy,
            "precision": precision, 
            "recall": recall,
            "f1_score": f1_score,
            "confusion_matrix": {
                "tp": int(tp), "fp": int(fp),
                "tn": int(tn), "fn": int(fn)
            }
        }
        
    def calculate_feature_importance(self) -> Dict:
        """计算特征重要性"""
        if self.feature_weights is None:
            return {}
            
        feature_names = [
            "log_amount", "unusual_time", "risk_factor_count", "calculated_risk",
            "unknown_location", "crypto_merchant", "casino_merchant", "account_age",
            "time_sin", "time_cos", "velocity_risk", "merchant_risk", 
            "geographic_risk", "behavioral_deviation"
        ]
        
        # 归一化权重作为重要性
        abs_weights = np.abs(self.feature_weights)
        normalized_importance = abs_weights / np.sum(abs_weights)
        
        importance_dict = {}
        for name, importance in zip(feature_names, normalized_importance):
            importance_dict[name] = float(importance)
            
        # 按重要性排序
        sorted_importance = dict(sorted(importance_dict.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return sorted_importance

def run_comprehensive_analysis(dataset_file: str = "complex_transaction_dataset.json") -> Dict:
    """运行完整的复杂分析流程"""
    # 初始化进度跟踪
    global progress_tracker
    progress_tracker = AnalysisProgressTracker()
    progress_tracker.total_steps = 8  # 更新总步骤数
    
    progress_tracker.update_progress("开始综合风险分析...", 0)
    print(" 开始comprehensive analysis...")
    print("  这将需要大量计算时间，请耐心等待...")
    print("="*80)
    
    overall_start = time.time()
    
    # 加载数据
    progress_tracker.update_progress("加载交易数据集...", 5)
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    transactions = dataset["transactions"]
    progress_tracker.complete_step(f"数据集加载完成: {len(transactions)} 笔交易")
    
    results = {"analysis_modules": {}}
    
    # 1. 网络分析
    progress_tracker.update_progress("开始交易网络分析...", 10)
    network_analyzer = NetworkAnalyzer()
    network_results = network_analyzer.build_transaction_network(transactions)
    results["analysis_modules"]["network_analysis"] = network_results
    progress_tracker.complete_step("网络分析模块完成")
    
    # 2. 时间序列异常检测
    progress_tracker.update_progress("开始时间序列异常检测...", 50)
    time_analyzer = TimeSeriesAnomalyDetector()
    time_results = time_analyzer.detect_temporal_anomalies(transactions)
    results["analysis_modules"]["temporal_analysis"] = time_results
    progress_tracker.complete_step("时序分析模块完成")
    
    # 3. 机器学习模型训练
    progress_tracker.update_progress("开始机器学习模型训练...", 75)
    ml_model = MachineLearningRiskModel()
    ml_results = ml_model.train_risk_model(transactions)
    results["analysis_modules"]["machine_learning"] = ml_results
    progress_tracker.complete_step("机器学习模块完成")
    
    # 综合结果
    progress_tracker.update_progress("生成综合分析报告...", 90)
    total_time = time.time() - overall_start
    
    results["comprehensive_summary"] = {
        "total_processing_time": total_time,
        "transactions_analyzed": len(transactions),
        "analysis_modules_completed": len(results["analysis_modules"]),
        "performance_metrics": {
            "transactions_per_second": len(transactions) / total_time,
            "network_analysis_time": network_results["processing_time"],
            "temporal_analysis_time": time_results["processing_time"], 
            "ml_training_time": ml_results["training_time"]
        }
    }
    
    progress_tracker.complete_step("综合报告生成完成")
    progress_tracker.finish()  # 完成所有分析
    
    print("综合分析完成!")
    print(f"  总耗时: {total_time:.2f}秒")
    print(f" 处理速度: {len(transactions)/total_time:.1f} 交易/秒")
    print("="*80)
    
    return results

def run_ml_only(dataset_file: str = "complex_transaction_dataset.json") -> Dict:
    """仅运行机器学习模块，生成与综合分析相同结构的结果文件。"""
    overall_start = time.time()
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    transactions = dataset.get("transactions", [])

    # 机器学习训练
    ml_model = MachineLearningRiskModel()
    ml_results = ml_model.train_risk_model(transactions)

    total_time = time.time() - overall_start
    results = {
        "analysis_modules": {
            "machine_learning": ml_results
        },
        "comprehensive_summary": {
            "total_processing_time": total_time,
            "transactions_analyzed": len(transactions),
            "analysis_modules_completed": 1,
            "performance_metrics": {
                "transactions_per_second": len(transactions) / total_time if total_time > 0 else 0.0,
                "ml_training_time": ml_results.get("training_time", 0.0)
            }
        }
    }
    return results

def run_network_only(dataset_file: str = "complex_transaction_dataset.json") -> Dict:
    overall_start = time.time()
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    transactions = dataset.get("transactions", [])

    network_analyzer = NetworkAnalyzer()
    network_results = network_analyzer.build_transaction_network(transactions)

    total_time = time.time() - overall_start
    return {
        "analysis_modules": {"network_analysis": network_results},
        "comprehensive_summary": {
            "total_processing_time": total_time,
            "transactions_analyzed": len(transactions),
            "analysis_modules_completed": 1,
            "performance_metrics": {
                "transactions_per_second": len(transactions) / total_time if total_time > 0 else 0.0,
                "network_analysis_time": network_results.get("processing_time", 0.0)
            }
        }
    }

def run_time_only(dataset_file: str = "complex_transaction_dataset.json") -> Dict:
    overall_start = time.time()
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    transactions = dataset.get("transactions", [])

    time_analyzer = TimeSeriesAnomalyDetector()
    time_results = time_analyzer.detect_temporal_anomalies(transactions)

    total_time = time.time() - overall_start
    return {
        "analysis_modules": {"temporal_analysis": time_results},
        "comprehensive_summary": {
            "total_processing_time": total_time,
            "transactions_analyzed": len(transactions),
            "analysis_modules_completed": 1,
            "performance_metrics": {
                "transactions_per_second": len(transactions) / total_time if total_time > 0 else 0.0,
                "temporal_analysis_time": time_results.get("processing_time", 0.0)
            }
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complex Analysis Engine")
    parser.add_argument("--mode", choices=["comprehensive", "ml_only", "network_only", "time_only"], default="comprehensive")
    parser.add_argument("--data", default="complex_transaction_dataset.json")
    args = parser.parse_args()

    if args.mode == "ml_only":
        analysis_results = run_ml_only(args.data)
    elif args.mode == "network_only":
        analysis_results = run_network_only(args.data)
    elif args.mode == "time_only":
        analysis_results = run_time_only(args.data)
    else:
        analysis_results = run_comprehensive_analysis(args.data)

    # 保存分析结果
    with open("comprehensive_analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2, default=str)
    print(" 分析结果已保存为 'comprehensive_analysis_results.json'")