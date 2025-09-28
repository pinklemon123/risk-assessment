# 欺诈检测系统部署指南

## 系统架构

这是一个基于图神经网络(GNN)的实时欺诈检测系统，具备以下核心功能：

### 核心组件

1. **数据流与特征工程**
   - NetworkX图模型构建
   - 多维度特征提取（节点中心性、异常模式检测）
   - 实时特征缓存优化

2. **模型与算法优化** 
   - 简化版GNN模型（兼容无torch_geometric环境）
   - 图级别和节点级别预测
   - 注意力机制聚合
   - 模型推理缓存

3. **系统落地与部署**
   - Docker容器化部署
   - 监控告警系统
   - 性能优化器
   - 多数据源连接器

## 快速部署

### 1. 使用Docker (推荐)

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

### 2. 本地开发部署

```bash
# 安装依赖
pip install -r requirements.txt

# 启动增强版应用
python -m app.main_enhanced

# 或使用原版应用（无欺诈检测）
python -m app.main
```

## 系统接口

### 健康检查
```bash
curl http://localhost:8080/health
```

### 映射配置管理
- **Web界面**: http://localhost:8080
- **获取配置**: `GET /api/admin/mapping`  
- **更新配置**: `PUT /api/admin/mapping`

### 数据处理
- **批量处理**: `POST /api/batch`
- **获取上传URL**: `POST /api/upload/url`

### 欺诈检测 (增强版功能)
- **风险评估**: `POST /api/fraud/assess`
- **账户特征**: `GET /api/fraud/features/{account_id}`
- **监控面板**: `GET /api/monitoring/dashboard`
- **手动告警**: `POST /api/monitoring/alert`

## 配置文件

### 1. 映射配置 (config/mapping.yaml)
```yaml
version: "1.0"
source_type: "transaction"
transformations:
  - field: "amount"
    type: "number"
    validation:
      min: 0
      max: 1000000
```

### 2. 图优化配置 (config/graph_optimization.yaml)
- 图构建策略
- 特征提取规则  
- GNN模型参数
- 性能优化设置

### 3. 连接器配置 (config/connectors.yaml)
- HTTP API连接器
- Kafka消息队列
- JDBC数据库连接
- SFTP文件传输

## 监控告警

### 性能指标
- API响应时间
- 错误率监控
- 队列深度
- 系统资源使用

### 业务指标  
- 高风险交易比例
- 模型漂移检测
- 特征异常计数
- 处理延迟

### 告警渠道
- Slack通知
- 邮件告警
- PagerDuty集成

## 欺诈检测使用示例

### 1. 单笔交易风险评估

```python
import requests

# 准备交易数据
transaction_data = {
    "transactions": [
        {
            "account_id": "acc_123",
            "amount": 5000,
            "merchant_id": "merchant_456",
            "timestamp": 1640995200,
            "from_account_id": "acc_123",
            "to_account_id": "acc_789"
        }
    ],
    "accounts": [
        {
            "account_id": "acc_123",
            "balance": 10000,
            "transaction_count": 50,
            "avg_transaction_amount": 200
        },
        {
            "account_id": "acc_789", 
            "balance": 5000,
            "transaction_count": 20,
            "avg_transaction_amount": 100
        }
    ]
}

# 调用风险评估接口
response = requests.post(
    "http://localhost:8080/api/fraud/assess",
    json=transaction_data
)

result = response.json()
print(f"风险评分: {result['risk_score']}")
print(f"风险等级: {result['risk_level']}")
print(f"置信度: {result['confidence']}")
```

### 2. 批量数据处理与风险评估

```python
import requests

# 批量交易数据
batch_data = {
    "data": [
        {
            "account_id": "acc_001",
            "amount": 1000,
            "merchant": "online_store",
            "timestamp": "2024-01-01T10:00:00Z"
        },
        {
            "account_id": "acc_002", 
            "amount": 50000,
            "merchant": "crypto_exchange",
            "timestamp": "2024-01-01T10:05:00Z"
        }
    ]
}

# 处理批量数据（自动进行欺诈检测）
response = requests.post(
    "http://localhost:8080/api/batch",
    json=batch_data,
    headers={"Authorization": "Bearer your_token"}
)

result = response.json()
for item in result["results"]:
    if "fraud_assessment" in item:
        fraud_info = item["fraud_assessment"]
        print(f"交易 {item['row_index']}: {fraud_info['risk_level']}")
```

### 3. 获取账户特征分析

```python
import requests

# 获取特定账户的特征
account_id = "acc_123"
response = requests.get(
    f"http://localhost:8080/api/fraud/features/{account_id}"
)

features = response.json()["features"]
print("账户特征分析:")
print(f"- 中心性分数: {features.get('centrality_scores', {})}")
print(f"- 异常模式: {features.get('suspicious_patterns', [])}")
print(f"- 网络特征: {features.get('network_features', {})}")
```

## 性能优化

### 1. 缓存配置
- Redis缓存：特征、模型结果
- 缓存TTL：特征1小时，结果5分钟
- 连接池：Redis 20连接，PostgreSQL 20连接

### 2. 并发处理
- 异步批量查询
- 特征并行计算  
- 线程池执行器
- 后台任务队列

### 3. 模型优化
- 预计算特征缓存
- 模型推理缓存
- 批量预测优化
- 降级机制

## 故障排查

### 常见问题

1. **torch_geometric导入错误**
   - 系统会自动使用简化版GNN实现
   - 检查日志中的警告信息

2. **Redis/PostgreSQL连接失败**
   - 欺诈检测功能会降级但不影响核心功能
   - 检查连接配置和服务状态

3. **内存使用过高**
   - 调整缓存TTL设置
   - 减少批处理大小
   - 监控图特征计算复杂度

### 日志分析
```bash
# 查看应用日志
docker-compose logs app

# 查看性能指标
curl http://localhost:8080/api/monitoring/dashboard

# 检查系统健康状态
curl http://localhost:8080/health
```

## 扩展开发

### 1. 添加新的图特征
编辑 `app/graph_features.py`，在 `GraphFeatureEngine` 类中添加新的特征提取方法。

### 2. 自定义GNN模型
修改 `app/simplified_gnn.py`，扩展 `FraudDetectionGNN` 类的架构。

### 3. 新增监控指标
在 `app/monitoring.py` 中添加新的监控规则和告警渠道。

## 生产部署建议

1. **环境配置**
   - 使用环境变量管理敏感配置
   - 配置SSL/TLS加密
   - 设置适当的资源限制

2. **高可用性**
   - 多实例部署
   - 负载均衡配置
   - 数据库主从复制

3. **安全性**
   - API认证授权
   - 网络隔离
   - 定期安全更新

4. **监控运维**
   - 集成APM系统
   - 自动化告警
   - 定期性能调优