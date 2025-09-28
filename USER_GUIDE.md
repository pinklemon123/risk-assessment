# 风险评估系统使用指南

## 🏠 主页面功能介绍

### 1. 配置管理页面 (/)
- **功能**: 编辑和管理YAML映射配置文件
- **用途**: 定义数据字段映射规则、验证规则等
- **操作**: 
  - 直接在编辑器中修改YAML配置
  - 点击"保存配置"应用更改
  - 支持语法高亮和错误检查

### 2. 控制面板 (/dashboard)
- **功能**: 实时监控系统运行状态
- **显示内容**:
  - API请求量和响应时间
  - 交易处理统计
  - 系统资源使用情况
  - 错误率监控

### 3. 欺诈检测 (/fraud-detection) 
- **功能**: 对交易数据进行风险评估
- **操作步骤**:
  1. 在"交易数据"文本框中输入JSON格式的交易数据
  2. 点击"开始风险评估"按钮
  3. 查看风险评分、风险等级和详细分析结果
- **风险评估规则**:
  - 高金额交易 (>10,000元) 增加风险
  - 深夜交易 (22:00-06:00) 增加风险
  - 加密货币、赌博相关商户增加风险
  - 频繁交易增加风险

### 4. 监控中心 (/monitoring)
- **功能**: 查看系统健康状况和业务指标
- **监控项目**:
  - 服务器状态
  - 数据库连接
  - 模型准确率
  - 处理队列状态

### 5. 批量处理 (/batch-process)
- **功能**: 上传和处理大批量交易数据
- **操作**:
  1. 选择CSV或JSON文件上传
  2. 选择处理模式（验证/转换/风险评估）
  3. 查看处理结果和错误报告

## 📊 如何测试系统功能

### 测试欺诈检测功能

1. 访问 http://localhost:8080/fraud-detection
2. 复制以下测试数据到"交易数据"框：

```json
{
  "transactions": [
    {
      "id": "txn_001",
      "amount": 15000,
      "merchant_id": "crypto_exchange_001",
      "timestamp": "2025-09-28T23:30:00"
    }
  ],
  "accounts": [
    {"account_id": "user_12345", "age_days": 30}
  ]
}
```

3. 点击"开始风险评估"查看结果

### 测试批量处理功能

1. 使用我们创建的示例文件 `examples/sample_transactions.json`
2. 访问 http://localhost:8080/batch-process  
3. 上传文件并查看处理结果

### 测试配置管理

1. 访问 http://localhost:8080/ (配置管理)
2. 修改YAML配置文件
3. 保存并查看更改是否生效

## 🔧 API接口使用

### 交易摄取接口
```bash
curl -X POST "http://localhost:8080/ingest/transactions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d @examples/sample_transactions.json
```

### 欺诈检测接口  
```bash
curl -X POST "http://localhost:8080/api/fraud/assess" \
  -H "Content-Type: application/json" \
  -d '{"transactions": [{"id": "test", "amount": 1000}]}'
```

### 监控数据接口
```bash
curl "http://localhost:8080/api/monitoring/dashboard"
```

## 📁 数据存储

- **原始数据**: 存储在 `data/raw/` 目录
- **清理后数据**: 存储在 `data/clean/` 目录  
- **配置文件**: `config/mapping.yaml`

## 🚨 常见问题

1. **页面显示模拟数据**: 这是正常的，系统会在没有真实数据时显示模拟数据
2. **欺诈检测不工作**: 确保输入正确的JSON格式交易数据
3. **配置保存失败**: 检查YAML语法是否正确
4. **Docker容器重启**: 修改后端代码后需要重新构建容器

## 📈 扩展功能

系统支持以下扩展：
- 自定义风险评估规则
- 多种数据格式导入
- 实时数据流处理
- 机器学习模型集成
- 告警和通知系统