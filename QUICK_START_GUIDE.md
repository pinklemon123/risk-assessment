# 🎯 风险评估系统 - 完整操作指南

## 🚀 快速开始

### 1. 访问系统
打开浏览器访问：http://localhost:8080

### 2. 立即测试欺诈检测
访问：http://localhost:8080/fraud-detection

**复制以下测试数据到输入框：**

#### 低风险交易测试：
```json
{
  "transactions": [
    {
      "id": "txn_safe_001",
      "amount": 89.50,
      "merchant_id": "grocery_store_chain",
      "timestamp": "2025-09-28T12:00:00"
    }
  ],
  "accounts": [
    {"account_id": "user_12345", "age_days": 365}
  ]
}
```
**预期结果**: 低风险 (~20% 风险评分)

#### 高风险交易测试：
```json
{
  "transactions": [
    {
      "id": "txn_risk_001", 
      "amount": 50000,
      "merchant_id": "crypto_exchange_suspicious",
      "timestamp": "2025-09-28T02:30:00"
    }
  ],
  "accounts": [
    {"account_id": "user_99999", "age_days": 5}
  ]
}
```
**预期结果**: 高风险 (~80% 风险评分)

### 3. 测试批量处理
访问：http://localhost:8080/batch-process

**点击"加载示例数据"或复制以下数据：**

```json
[
  {
    "account_id": "acc_001",
    "amount": 150,
    "merchant": "grocery_store",
    "timestamp": "2025-09-28T09:30:00Z",
    "transaction_type": "purchase"
  },
  {
    "account_id": "acc_002",
    "amount": 50000,
    "merchant": "crypto_exchange",
    "timestamp": "2025-09-28T23:45:00Z", 
    "transaction_type": "transfer"
  },
  {
    "account_id": "acc_003",
    "amount": 25000,
    "merchant": "casino_macau",
    "timestamp": "2025-09-28T02:15:00Z",
    "transaction_type": "gambling"
  }
]
```

**选择处理模式**：
- `validate`: 仅验证数据格式
- `risk_assessment`: 进行风险评估
- `full`: 完整处理（验证+风险评估）

## 📊 风险评估算法详解

### 风险评分计算 (0.0 - 1.0)

1. **基础风险**: 0.1 (所有交易的基础风险)

2. **金额风险**:
   - > 10,000元: +0.4
   - > 5,000元: +0.2  
   - > 1,000元: +0.1

3. **时间风险**:
   - 深夜交易 (22:00-06:00): +0.1

4. **商户风险**:
   - 加密货币/赌博相关: +0.3
   - 在线/海外商户: +0.1

5. **频率风险**:
   - > 10笔交易: +0.2
   - > 5笔交易: +0.1

### 风险等级划分
- **低风险 (Low)**: 0.0 - 0.39
- **中等风险 (Medium)**: 0.4 - 0.69  
- **高风险 (High)**: 0.7 - 1.0

## 🔧 API接口使用

### 单笔风险评估
```bash
POST /api/fraud/assess
Content-Type: application/json

{
  "transactions": [
    {
      "id": "txn_001",
      "amount": 1500,
      "merchant_id": "merchant_name",
      "timestamp": "2025-09-28T14:30:00"
    }
  ],
  "accounts": [
    {"account_id": "user_123", "age_days": 30}
  ]
}
```

### 批量处理
```bash
POST /api/batch/process
Content-Type: application/json

{
  "batch_data": [...], 
  "process_mode": "risk_assessment"
}
```

### 系统监控
```bash
GET /api/monitoring/dashboard
```

## 📈 结果解读

### 欺诈检测结果
```json
{
  "risk_score": 0.650,           // 风险评分 (0-1)
  "risk_level": "medium",        // 风险等级
  "confidence": 0.850,           // 置信度
  "transaction_count": 1,        // 交易数量
  "account_count": 1,           // 账户数量
  "processing_time": 0.05,      // 处理时间(秒)
  "enhanced_features": {
    "total_amount": 1500.0,     // 总金额
    "average_amount": 1500.0,   // 平均金额
    "transaction_time_risk": "low", // 时间风险
    "volume_risk": "low"        // 频率风险
  }
}
```

### 批量处理结果
```json
{
  "batch_id": "batch_xxx",      // 批次ID
  "processing_time": 1.234,     // 总处理时间
  "total_records": 100,         // 总记录数
  "processed_count": 85,        // 成功处理数
  "failed_count": 15,          // 失败数量
  "success_rate": 85.0,        // 成功率(%)
  "summary": {
    "high_risk_count": 12,      // 高风险数量
    "medium_risk_count": 25,    // 中等风险数量  
    "low_risk_count": 48,       // 低风险数量
    "avg_risk_score": 0.342     // 平均风险评分
  },
  "results": [...]             // 详细结果列表
}
```

## 🛠️ 故障排除

### 常见错误及解决方案

1. **"Assessment failed: 'str' object has no attribute 'get'"**
   - ✅ 已修复！确保输入数据是正确的JSON格式

2. **"No transactions provided"**  
   - 检查JSON中是否包含`transactions`数组

3. **"No valid transactions found"**
   - 确保每个交易都是字典格式，包含必要字段

4. **页面无法访问**
   - 检查Docker容器是否运行：`docker ps`
   - 重新启动：`docker-compose up --build -d`

### 数据格式要求

**交易数据必须包含的字段：**
- `id` (字符串): 交易ID
- `amount` (数字): 交易金额
- `merchant_id` (字符串): 商户ID

**可选字段：**
- `timestamp`: 交易时间
- `location`: 交易地点
- `currency`: 货币类型

## 🔄 下一步改进建议

1. **机器学习集成**: 使用历史数据训练更准确的模型
2. **实时数据流**: 集成Kafka等消息队列
3. **规则引擎**: 支持动态配置风险规则
4. **多维度分析**: 加入设备指纹、地理位置等特征
5. **报告系统**: 自动生成风险报告和趋势分析

现在这个系统已经完全可用了！🎉