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

# 静态文件服务
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 添加管理路由
app.include_router(admin_router)

# 添加静态文件服务
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

# 其他静态页面路由
@app.get("/dashboard.html")
async def dashboard():
    """控制面板"""
    return FileResponse("app/static/dashboard.html")

@app.get("/fraud-detection.html")
async def fraud_detection():
    """欺诈检测页面"""
    return FileResponse("app/static/fraud-detection.html")

@app.get("/monitoring.html")
async def monitoring():
    """监控中心页面"""
    return FileResponse("app/static/monitoring.html")

@app.get("/batch-process.html")
async def batch_process():
    """批量处理页面"""
    return FileResponse("app/static/batch-process.html")

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

@app.post("/ingest/presign", response_model=PresignResponse)
async def presign(obj_type: str = Query(..., pattern="^(transactions|entities|labels)$"),
                  ext: str = Query(..., pattern="^(csv|jsonl|parquet)$"),
                  auth=Depends(require_auth)):
    info = create_presigned_url(obj_type, ext)
    return PresignResponse(**info)
