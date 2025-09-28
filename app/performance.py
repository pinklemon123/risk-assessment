import asyncio
import time
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import redis.asyncio as redis
import asyncpg
from functools import wraps
import json
import hashlib

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self):
        self.redis_client = None
        self.db_pool = None
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.cache_ttl = {
            'feature_cache': 3600,  # 1小时
            'model_cache': 1800,    # 30分钟
            'result_cache': 300     # 5分钟
        }
        
    async def initialize(self):
        """初始化连接池"""
        # Redis连接池
        self.redis_client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True,
            max_connections=20
        )
        
        # PostgreSQL连接池
        self.db_pool = await asyncpg.create_pool(
            host='localhost',
            port=5432,
            user='postgres',
            password='password',
            database='fraud_detection',
            min_size=5,
            max_size=20,
            command_timeout=60
        )
    
    def cache_result(self, cache_key_prefix: str, ttl: int = 300):
        """缓存装饰器"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = self._generate_cache_key(cache_key_prefix, args, kwargs)
                
                # 尝试从缓存获取
                cached_result = await self._get_from_cache(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # 执行函数
                result = await func(*args, **kwargs)
                
                # 缓存结果
                await self._set_cache(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    def _generate_cache_key(self, prefix: str, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        key_data = {
            'args': str(args),
            'kwargs': sorted(kwargs.items())
        }
        key_hash = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """从缓存获取数据"""
        try:
            cached_data = await self.redis_client.get(key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            print(f"Cache get error: {e}")
        return None
    
    async def _set_cache(self, key: str, value: Any, ttl: int):
        """设置缓存"""
        try:
            await self.redis_client.setex(
                key, 
                ttl, 
                json.dumps(value, default=str)
            )
        except Exception as e:
            print(f"Cache set error: {e}")
    
    async def batch_query_optimization(self, queries: list, batch_size: int = 100):
        """批量查询优化"""
        results = []
        
        # 分批处理
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            
            # 并发执行批次内的查询
            batch_tasks = []
            for query in batch:
                if asyncio.iscoroutinefunction(query):
                    batch_tasks.append(query)
                else:
                    # 包装同步函数
                    batch_tasks.append(
                        asyncio.get_event_loop().run_in_executor(
                            self.thread_pool, query
                        )
                    )
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
            
        return results
    
    async def feature_computation_optimization(self, transactions: list) -> Dict[str, Any]:
        """特征计算优化"""
        
        # 1. 预先批量查询历史数据
        account_ids = list(set(t['account_id'] for t in transactions))
        historical_data = await self._batch_get_historical_data(account_ids)
        
        # 2. 并行计算不同类型的特征
        tasks = [
            self._compute_transaction_features(transactions),
            self._compute_account_features(account_ids, historical_data),
            self._compute_network_features(transactions),
            self._compute_temporal_features(transactions)
        ]
        
        feature_results = await asyncio.gather(*tasks)
        
        # 3. 合并特征结果
        combined_features = {}
        for feature_dict in feature_results:
            combined_features.update(feature_dict)
            
        return combined_features
    
    @cache_result("historical_data", ttl=1800)
    async def _batch_get_historical_data(self, account_ids: list) -> Dict[str, Any]:
        """批量获取历史数据"""
        if not self.db_pool:
            return {}
            
        async with self.db_pool.acquire() as conn:
            # 使用IN查询批量获取
            query = """
            SELECT account_id, 
                   COUNT(*) as total_transactions,
                   AVG(amount) as avg_amount,
                   MAX(amount) as max_amount,
                   COUNT(DISTINCT merchant_id) as unique_merchants
            FROM transactions 
            WHERE account_id = ANY($1)
              AND created_at >= NOW() - INTERVAL '30 days'
            GROUP BY account_id
            """
            
            rows = await conn.fetch(query, account_ids)
            
            return {
                row['account_id']: {
                    'total_transactions': row['total_transactions'],
                    'avg_amount': float(row['avg_amount']),
                    'max_amount': float(row['max_amount']),
                    'unique_merchants': row['unique_merchants']
                }
                for row in rows
            }
    
    async def _compute_transaction_features(self, transactions: list) -> Dict[str, Any]:
        """计算交易特征"""
        # 基础统计特征
        amounts = [t['amount'] for t in transactions]
        
        return {
            'transaction_count': len(transactions),
            'total_amount': sum(amounts),
            'avg_amount': sum(amounts) / len(amounts) if amounts else 0,
            'max_amount': max(amounts) if amounts else 0,
            'min_amount': min(amounts) if amounts else 0
        }
    
    async def _compute_account_features(self, account_ids: list, historical_data: Dict) -> Dict[str, Any]:
        """计算账户特征"""
        return {
            'unique_accounts': len(account_ids),
            'accounts_with_history': len([aid for aid in account_ids if aid in historical_data])
        }
    
    async def _compute_network_features(self, transactions: list) -> Dict[str, Any]:
        """计算网络特征"""
        # 简化的网络特征计算
        unique_merchants = set(t.get('merchant_id') for t in transactions if t.get('merchant_id'))
        
        return {
            'unique_merchants': len(unique_merchants),
            'merchant_diversity': len(unique_merchants) / len(transactions) if transactions else 0
        }
    
    async def _compute_temporal_features(self, transactions: list) -> Dict[str, Any]:
        """计算时间特征"""
        timestamps = [t.get('timestamp', time.time()) for t in transactions]
        
        if len(timestamps) > 1:
            time_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            avg_time_diff = sum(time_diffs) / len(time_diffs)
        else:
            avg_time_diff = 0
            
        return {
            'time_span': max(timestamps) - min(timestamps) if timestamps else 0,
            'avg_time_between_transactions': avg_time_diff
        }
    
    async def model_inference_optimization(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """模型推理优化"""
        
        # 1. 特征预处理缓存
        preprocessed_features = await self._preprocess_features(features)
        
        # 2. 模型预测缓存
        predictions = await self._cached_model_prediction(preprocessed_features)
        
        # 3. 后处理
        results = await self._postprocess_predictions(predictions)
        
        return results
    
    @cache_result("preprocessed_features", ttl=300)
    async def _preprocess_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """预处理特征"""
        # 标准化处理
        processed = {}
        
        for key, value in features.items():
            if isinstance(value, (int, float)):
                # 简单的标准化
                processed[f"{key}_normalized"] = min(max(value / 1000.0, 0), 1)
            else:
                processed[key] = value
                
        return processed
    
    @cache_result("model_prediction", ttl=180)
    async def _cached_model_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """缓存的模型预测"""
        # 模拟模型推理
        await asyncio.sleep(0.1)  # 模拟计算时间
        
        # 简化的风险评分
        risk_score = min(
            features.get('transaction_count', 0) * 0.1 +
            features.get('total_amount_normalized', 0) * 0.5 +
            features.get('unique_merchants', 0) * 0.2,
            1.0
        )
        
        return {
            'risk_score': risk_score,
            'risk_level': 'high' if risk_score > 0.7 else 'medium' if risk_score > 0.4 else 'low'
        }
    
    async def _postprocess_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """后处理预测结果"""
        # 添加置信度和其他元数据
        return {
            **predictions,
            'confidence': 0.85 if predictions['risk_level'] != 'medium' else 0.65,
            'timestamp': time.time()
        }
    
    async def cleanup(self):
        """清理资源"""
        if self.redis_client:
            await self.redis_client.close()
        if self.db_pool:
            await self.db_pool.close()
        self.thread_pool.shutdown(wait=True)

# 全局性能优化器实例
performance_optimizer = PerformanceOptimizer()

# 性能监控装饰器
def monitor_performance(operation_name: str):
    """性能监控装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                success = True
            except Exception as e:
                result = None
                success = False
                raise e
            finally:
                duration = time.time() - start_time
                
                # 记录性能指标
                await _log_performance_metric(
                    operation=operation_name,
                    duration=duration,
                    success=success
                )
                
            return result
        return wrapper
    return decorator

async def _log_performance_metric(operation: str, duration: float, success: bool):
    """记录性能指标"""
    # 这里可以集成到监控系统
    print(f"Performance: {operation} took {duration:.3f}s, success: {success}")