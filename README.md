# Risk Ingestion Gateway

一个基于 FastAPI 的网关，用于摄取、验证和处理风险数据（交易、实体、标签），现在包含了可视化的映射配置编辑器。

## 🚀 快速启动

### 方法一：本地开发模式（推荐）

**Windows:**
```cmd
start-local.bat
```

这将自动：
- 检查 Python 和 Node.js 环境
- 安装依赖
- 启动后端和前端服务

### 方法二：Docker 模式

```bash
docker-compose up --build -d
```

## 📱 访问地址

启动后，您可以通过以下地址访问：

- **映射编辑器**: http://localhost （通过 Nginx）
- **前端直接访问**: http://localhost:3000
- **后端 API**: http://localhost:8080
- **API 文档**: http://localhost:8080/docs

## 🛠️ 功能特性

### 后端 API
- 交易数据摄取和验证
- 数据质量检查
- 预签名 URL 生成
- **新增**: 映射配置管理 API (`/api/admin/mapping`)

### 前端编辑器
- 可视化 YAML 映射配置编辑
- 实时预览和验证
- 支持复杂的数据转换规则
- 导入/导出 YAML 配置文件

## 📁 项目结构

```
.
├── app/                    # 后端应用
│   ├── main.py            # FastAPI 主应用
│   ├── admin.py           # 管理 API 路由
│   ├── models.py          # 数据模型
│   ├── transformer.py     # 数据转换
│   ├── auth.py            # 认证
│   ├── dq.py              # 数据质量检查
│   └── connectors/        # 数据连接器
├── pages/                 # 前端页面
│   └── index.tsx          # 映射编辑器主页面
├── config/                # 配置文件
│   └── mapping.yaml       # 映射配置
├── docker-compose.yml     # Docker 配置
├── Dockerfile             # 镜像构建
├── requirements.txt       # Python 依赖
├── package.json           # Node.js 依赖
└── start-local.bat        # 本地启动脚本
```

## 🔧 配置说明

### 环境变量
- `API_KEYS`: API 访问密钥
- `ALLOWED_TENANTS`: 允许的租户
- `MAPPING_PATH`: 映射配置文件路径
- `API_URL`: 前端访问后端的 URL

### 映射配置
通过可视化编辑器或直接编辑 `config/mapping.yaml` 文件来配置数据映射规则。

## 📋 常用命令

```bash
# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 重新构建
docker-compose up --build --force-recreate
```

## 🔍 故障排除

1. **端口冲突**: 确保端口 80, 3000, 8080 未被其他应用占用
2. **Docker 未启动**: 确保 Docker Desktop 正在运行
3. **权限问题**: Linux/Mac 用户可能需要 `sudo` 权限
4. **网络问题**: 检查防火墙和代理设置

## 🌟 新功能：映射配置编辑器

这个可视化编辑器允许您：
- 图形化编辑 YAML 映射配置
- 实时预览配置效果
- 验证配置语法
- 管理数据质量规则
- 配置隐私设置

## 目录
```
app/                # FastAPI + 业务逻辑
  connectors/       # Kafka / SFTP 连接器
config/
  mapping.yaml      # 租户映射与数据质量配置（示例）
data/{raw,clean}/   # 原始与清洗后数据（本地演示用）
Dockerfile
docker-compose.yml
requirements.txt
```

## 快速开始（本地）
```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

### 测试 HTTP 批量写入
```bash
curl -X POST http://127.0.0.1:8080/ingest/transactions   -H "Content-Type: application/json"   -H "X-API-Key: test_key" -H "X-Tenant-Id: acme"   -d @scripts/sample_batch.json
```

### Presign（占位实现，返回伪 URL）
```bash
curl -X POST "http://127.0.0.1:8080/ingest/presign?type=transactions&ext=parquet"   -H "X-API-Key: test_key" -H "X-Tenant-Id: acme"
```

> 生产中请替换 `app/presign.py` 内的占位逻辑为 GCS/S3 的真正 Presign。

## Kafka 消费者
- 配置环境变量后运行：
```bash
python app/connectors/kafka_consumer.py
```

## SFTP 拉取
- 配置环境变量后运行：
```bash
python app/connectors/sftp_pull.py
```

## Docker 运行
```bash
docker build -t risk-ingestion:dev .
docker run --rm -p 8080:8080   -e API_KEYS=test_key   -e ALLOWED_TENANTS=acme   -v $PWD/config/mapping.yaml:/app/config/mapping.yaml   -v $PWD/data:/app/data   risk-ingestion:dev
```

## 环境变量
- `API_KEYS`：逗号分隔的 key 列表，例如 `key1,key2`
- `ALLOWED_TENANTS`：逗号分隔的租户 ID
- `BASE_CURRENCY`：基准货币（默认 CNY）

## 注意
- 本项目将数据写入本地 `data/` 目录以便演示；生产中请替换为 GCS/S3，或通过 Writer 改写。
- Presign、Kafka、SFTP 组件为最小实现/骨架，请结合你的云环境替换。
