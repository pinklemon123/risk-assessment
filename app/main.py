from fastapi import FastAPI, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os, time, uuid

from .models import BatchRequest, BatchResponse, PresignResponse
from .auth import require_auth
from .transformer import load_mapping, apply_mapping
from .dq import validate_row
from .writer import write_raw, write_clean
from .presign import create_presigned_url
from .admin import router as admin_router

MAPPING_PATH = os.getenv("MAPPING_PATH", "config/mapping.yaml")
cfg = load_mapping(MAPPING_PATH)

app = FastAPI(title="Risk Ingestion Gateway", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加管理路由
app.include_router(admin_router)

# 静态文件服务
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 添加根路由，重定向到编辑器
@app.get("/")
async def root():
    return FileResponse("app/static/index.html")

@app.get("/healthz")
async def healthz():
    return {"ok": True, "ts": int(time.time())}

@app.post("/ingest/transactions", response_model=BatchResponse)
async def ingest_transactions(payload: BatchRequest, auth=Depends(require_auth)):
    tenant = auth["tenant_id"]
    batch_id = f"b_{tenant}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    raw_rows = payload.rows

    write_raw(batch_id, raw_rows)

    accepted = 0
    failed = 0
    clean_rows = []
    seen_ids = set()
    for r in raw_rows:
        mapped = apply_mapping(r, cfg)
        ok, errs = validate_row(mapped, cfg)
        row_id = mapped.get("id") or mapped.get("txn_id")
        if row_id in seen_ids:
            ok = False
        else:
            seen_ids.add(row_id)
        if ok:
            clean_rows.append(mapped)
            accepted += 1
        else:
            failed += 1

    write_clean(batch_id, clean_rows)

    return BatchResponse(
        accepted=accepted,
        failed=failed,
        batch_id=batch_id,
        feat_ver_hint=time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())
    )

# 静态页面路由
@app.get("/dashboard")
async def dashboard():
    """控制面板"""
    return FileResponse("app/static/dashboard.html")

@app.get("/dashboard.html")
async def dashboard_html():
    """控制面板"""
    return FileResponse("app/static/dashboard.html")

@app.get("/fraud-detection")
async def fraud_detection():
    """欺诈检测页面"""
    return FileResponse("app/static/fraud-detection.html")

@app.get("/fraud-detection.html")
async def fraud_detection_html():
    """欺诈检测页面"""
    return FileResponse("app/static/fraud-detection.html")

@app.get("/monitoring")
async def monitoring():
    """监控中心页面"""
    return FileResponse("app/static/monitoring.html")

@app.get("/monitoring.html")
async def monitoring_html():
    """监控中心页面"""
    return FileResponse("app/static/monitoring.html")

@app.get("/batch-process")
async def batch_process():
    """批量处理页面"""
    return FileResponse("app/static/batch-process.html")

@app.get("/batch-process.html")
async def batch_process_html():
    """批量处理页面"""
    return FileResponse("app/static/batch-process.html")

@app.get("/config")
async def config():
    """配置管理页面"""
    return FileResponse("app/static/index.html")

@app.get("/index.html")
async def index_html():
    """配置管理页面"""
    return FileResponse("app/static/index.html")

# 欺诈检测API接口
@app.post("/api/fraud/assess")
async def assess_fraud_risk_endpoint(
    request_data: dict
):
    """简化的欺诈风险评估接口"""
    try:
        transactions = request_data.get('transactions', [])
        accounts = request_data.get('accounts', [])
        
        if not transactions:
            return {
                "error": "No transactions provided",
                "risk_score": 0.5,
                "risk_level": "unknown",
                "confidence": 0.3
            }
        
        # 简化的风险评分逻辑
        total_amount = sum(float(t.get('amount', 0)) for t in transactions)
        avg_amount = total_amount / len(transactions) if transactions else 0
        
        # 基于金额的简单风险评分
        risk_score = 0.1  # 基础风险
        
        # 高金额交易增加风险
        if avg_amount > 10000:
            risk_score += 0.4
        elif avg_amount > 5000:
            risk_score += 0.2
        elif avg_amount > 1000:
            risk_score += 0.1
            
        # 深夜交易增加风险
        import datetime
        current_hour = datetime.datetime.now().hour
        if current_hour < 6 or current_hour > 22:
            risk_score += 0.1
            
        # 大量交易增加风险
        if len(transactions) > 10:
            risk_score += 0.2
        elif len(transactions) > 5:
            risk_score += 0.1
            
        # 特殊商户类型增加风险
        for txn in transactions:
            merchant = txn.get('merchant_id', '').lower()
            if any(keyword in merchant for keyword in ['crypto', 'casino', 'gambling']):
                risk_score += 0.3
                break
            elif any(keyword in merchant for keyword in ['online', 'foreign']):
                risk_score += 0.1
                break
                
        # 限制在0-1范围内
        risk_score = min(1.0, max(0.0, risk_score))
        
        # 确定风险等级
        if risk_score >= 0.7:
            risk_level = "high"
        elif risk_score >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
            
        # 计算置信度
        confidence = 0.6 + (abs(risk_score - 0.5) * 0.8)
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "confidence": min(0.95, confidence),
            "transaction_count": len(transactions),
            "account_count": len(accounts) if accounts else 0,
            "processing_time": 0.05,  # 模拟处理时间
            "enhanced_features": {
                "total_amount": total_amount,
                "average_amount": avg_amount,
                "transaction_time_risk": "high" if current_hour < 6 or current_hour > 22 else "low",
                "volume_risk": "high" if len(transactions) > 10 else "medium" if len(transactions) > 5 else "low"
            }
        }
        
    except Exception as e:
        return {
            "error": f"Assessment failed: {str(e)}",
            "risk_score": 0.5,
            "risk_level": "unknown", 
            "confidence": 0.3,
            "fallback": True
        }

@app.get("/api/monitoring/dashboard")
async def get_monitoring_dashboard():
    """获取监控面板数据（模拟）"""
    import random
    return {
        "api_metrics": {
            "requests_per_second": random.randint(50, 200),
            "average_response_time": random.randint(50, 300),
            "error_rate": random.random() * 0.05
        },
        "business_metrics": {
            "transactions_processed": random.randint(1000, 5000),
            "high_risk_rate": random.random() * 0.1,
            "model_accuracy": 0.85 + random.random() * 0.1
        },
        "system_metrics": {
            "cpu_usage": random.randint(30, 80),
            "memory_usage": random.randint(40, 85),
            "disk_usage": random.randint(50, 90)
        }
    }

@app.post("/api/batch/process")
async def process_batch_data(request_data: dict):
    """批量处理数据API"""
    try:
        import time
        start_time = time.time()
        
        # 获取批次数据
        batch_data = request_data.get('batch_data', [])
        process_mode = request_data.get('process_mode', 'validate')
        
        if not batch_data:
            return {
                "error": "No batch data provided",
                "batch_id": None,
                "results": []
            }
        
        # 生成批次ID
        batch_id = f"batch_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        results = []
        processed_count = 0
        failed_count = 0
        
        for idx, record in enumerate(batch_data):
            try:
                # 数据转换和验证
                transformed_data = {}
                errors = []
                
                # 基本字段映射和验证
                required_fields = ['account_id', 'amount', 'merchant', 'timestamp']
                for field in required_fields:
                    if field in record:
                        transformed_data[field] = record[field]
                    else:
                        errors.append(f"Missing required field: {field}")
                
                # 金额验证
                if 'amount' in record:
                    try:
                        amount = float(record['amount'])
                        if amount <= 0:
                            errors.append("Amount must be positive")
                        transformed_data['amount'] = amount
                    except (ValueError, TypeError):
                        errors.append("Invalid amount format")
                
                # 欺诈风险评估
                fraud_assessment = None
                if process_mode in ['risk_assessment', 'full'] and not errors:
                    fraud_result = await assess_fraud_risk_endpoint({
                        'transactions': [record],
                        'accounts': []
                    })
                    fraud_assessment = fraud_result
                
                # 确定处理状态
                status = "success" if not errors else "failed"
                if status == "success":
                    processed_count += 1
                else:
                    failed_count += 1
                
                result_item = {
                    "row_index": idx + 1,
                    "status": status,
                    "transformed_data": transformed_data if not errors else None,
                    "fraud_assessment": fraud_assessment,
                    "errors": errors if errors else None,
                    "processing_time": round(time.time() - start_time, 3)
                }
                
                results.append(result_item)
                
            except Exception as e:
                failed_count += 1
                results.append({
                    "row_index": idx + 1,
                    "status": "error",
                    "transformed_data": None,
                    "fraud_assessment": None,
                    "errors": [f"Processing error: {str(e)}"],
                    "processing_time": round(time.time() - start_time, 3)
                })
        
        total_time = round(time.time() - start_time, 3)
        
        return {
            "batch_id": batch_id,
            "processing_time": total_time,
            "total_records": len(batch_data),
            "processed_count": processed_count,
            "failed_count": failed_count,
            "success_rate": round((processed_count / len(batch_data)) * 100, 2) if batch_data else 0,
            "process_mode": process_mode,
            "results": results,
            "summary": {
                "high_risk_count": len([r for r in results if r.get('fraud_assessment', {}).get('risk_level') == 'high']),
                "medium_risk_count": len([r for r in results if r.get('fraud_assessment', {}).get('risk_level') == 'medium']),
                "low_risk_count": len([r for r in results if r.get('fraud_assessment', {}).get('risk_level') == 'low']),
                "avg_risk_score": round(sum([r.get('fraud_assessment', {}).get('risk_score', 0) for r in results if r.get('fraud_assessment')]) / max(1, len([r for r in results if r.get('fraud_assessment')])), 3)
            }
        }
        
    except Exception as e:
        return {
            "error": f"Batch processing failed: {str(e)}",
            "batch_id": None,
            "results": [],
            "processing_time": 0
        }

@app.post("/ingest/presign", response_model=PresignResponse)
async def presign(obj_type: str = Query(..., pattern="^(transactions|entities|labels)$"),
                  ext: str = Query(..., pattern="^(csv|jsonl|parquet)$"),
                  auth=Depends(require_auth)):
    info = create_presigned_url(obj_type, ext)
    return PresignResponse(**info)
