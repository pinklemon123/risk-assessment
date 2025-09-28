"""
集成版本的主应用 - 包含完整的欺诈检测系统
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
import time
import uuid
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional
import logging

# 导入原有模块
from app.models import BatchRequest, BatchResponse, PresignResponse
from app.auth import require_auth
from app.transformer import load_mapping, apply_mapping
from app.dq import validate_row
from app.writer import write_raw, write_clean
from app.presign import create_presigned_url
from app.admin import router as admin_router

# 导入新的欺诈检测组件
try:
    from app.graph_features import GraphFeatureEngine
    from app.simplified_gnn import fraud_gnn_pipeline, predict_fraud_risk_gnn
    from app.monitoring import MonitoringSystem
    from app.performance import performance_optimizer
    FRAUD_DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"Fraud detection components not available: {e}")
    FRAUD_DETECTION_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局组件
monitoring_system = None
graph_engine = None

MAPPING_PATH = os.getenv("MAPPING_PATH", "config/mapping.yaml")
cfg = load_mapping(MAPPING_PATH)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global monitoring_system, graph_engine, FRAUD_DETECTION_AVAILABLE
    
    logger.info("Starting fraud detection system...")
    
    if FRAUD_DETECTION_AVAILABLE:
        try:
            # 初始化监控系统
            monitoring_system = MonitoringSystem()
            monitoring_system.setup_monitoring_rules()
            monitoring_system.setup_alert_channels()
            
            # 初始化性能优化器
            await performance_optimizer.initialize()
            
            # 初始化图特征引擎
            graph_engine = GraphFeatureEngine()
            
            logger.info("Fraud detection system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize fraud detection system: {e}")
            FRAUD_DETECTION_AVAILABLE = False
    
    yield
    
    # 清理资源
    if FRAUD_DETECTION_AVAILABLE:
        try:
            await performance_optimizer.cleanup()
            logger.info("Fraud detection system cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# 创建FastAPI应用
app = FastAPI(
    title="Enhanced Risk Ingestion Gateway with Fraud Detection", 
    version="2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 路由
app.include_router(admin_router, prefix="/api/admin")

@app.get("/")
async def root():
    """首页 - 映射配置编辑器"""
    return FileResponse("app/static/index.html")

@app.get("/health")
async def health_check():
    """健康检查"""
    status = {
        "status": "healthy",
        "timestamp": time.time(),
        "fraud_detection_available": FRAUD_DETECTION_AVAILABLE
    }
    
    if FRAUD_DETECTION_AVAILABLE and monitoring_system:
        try:
            health_data = monitoring_system.get_system_health_dashboard()
            status["system_health"] = health_data
        except Exception as e:
            status["health_check_error"] = str(e)
    
    return status

# 原有的批处理接口
@app.post("/api/batch", response_model=BatchResponse)
async def process_batch(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
    auth_context = Depends(require_auth)
):
    """处理批量数据"""
    start_time = time.time()
    
    try:
        batch_id = str(uuid.uuid4())
        results = []
        
        for idx, row in enumerate(request.data):
            try:
                # 数据验证
                validation_result = validate_row(row, cfg)
                if not validation_result['valid']:
                    results.append({
                        "row_index": idx,
                        "status": "failed",
                        "errors": validation_result['errors']
                    })
                    continue
                
                # 数据转换
                transformed = apply_mapping(row, cfg)
                
                # 如果启用了欺诈检测，进行风险评估
                fraud_result = None
                if FRAUD_DETECTION_AVAILABLE:
                    try:
                        fraud_result = await assess_fraud_risk(transformed, background_tasks)
                    except Exception as e:
                        logger.warning(f"Fraud detection failed for row {idx}: {e}")
                
                # 异步写入数据
                background_tasks.add_task(write_clean, transformed, batch_id)
                
                result_item = {
                    "row_index": idx,
                    "status": "success",
                    "transformed_data": transformed
                }
                
                if fraud_result:
                    result_item["fraud_assessment"] = fraud_result
                
                results.append(result_item)
                
            except Exception as e:
                results.append({
                    "row_index": idx,
                    "status": "failed", 
                    "errors": [str(e)]
                })
        
        # 记录性能指标
        processing_time = time.time() - start_time
        if FRAUD_DETECTION_AVAILABLE and monitoring_system:
            monitoring_system.record_metric(
                "batch_processing_time", 
                processing_time,
                {"batch_size": len(request.data)}
            )
        
        return BatchResponse(
            batch_id=batch_id,
            processed_count=len([r for r in results if r["status"] == "success"]),
            failed_count=len([r for r in results if r["status"] == "failed"]),
            results=results,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 新增的欺诈检测接口
@app.post("/api/fraud/assess")
async def assess_fraud_risk_endpoint(
    transactions: List[Dict[str, Any]],
    accounts: Optional[List[Dict[str, Any]]] = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """欺诈风险评估接口"""
    if not FRAUD_DETECTION_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Fraud detection system not available"
        )
    
    start_time = time.time()
    
    try:
        # 如果没有提供账户信息，从交易中提取
        if not accounts:
            account_ids = set()
            for txn in transactions:
                account_ids.update([
                    txn.get('from_account_id'), 
                    txn.get('to_account_id'),
                    txn.get('account_id')
                ])
            account_ids.discard(None)
            
            accounts = [{'account_id': aid} for aid in account_ids]
        
        # 使用图特征引擎增强特征
        if graph_engine:
            enhanced_features = graph_engine.extract_graph_features(
                transactions, accounts
            )
        else:
            enhanced_features = {}
        
        # GNN模型预测
        gnn_result = await predict_fraud_risk_gnn(transactions, accounts)
        
        # 组合结果
        result = {
            **gnn_result,
            "enhanced_features": enhanced_features,
            "transaction_count": len(transactions),
            "account_count": len(accounts),
            "processing_time": time.time() - start_time
        }
        
        # 记录监控指标
        if monitoring_system:
            monitoring_system.record_metric(
                "fraud_assessment_time",
                result["processing_time"],
                {"risk_level": result["risk_level"]}
            )
            
            monitoring_system.record_metric(
                "fraud_risk_score",
                result["risk_score"],
                {"transaction_count": len(transactions)}
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Fraud assessment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/fraud/features/{account_id}")
async def get_account_features(account_id: str):
    """获取账户特征"""
    if not FRAUD_DETECTION_AVAILABLE or not graph_engine:
        raise HTTPException(
            status_code=503,
            detail="Graph feature engine not available"
        )
    
    try:
        features = graph_engine.get_account_features(account_id)
        return {
            "account_id": account_id,
            "features": features,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/monitoring/dashboard")
async def get_monitoring_dashboard():
    """获取监控面板数据"""
    if not FRAUD_DETECTION_AVAILABLE or not monitoring_system:
        raise HTTPException(
            status_code=503,
            detail="Monitoring system not available"
        )
    
    try:
        dashboard_data = monitoring_system.get_system_health_dashboard()
        return dashboard_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/monitoring/alert")
async def trigger_manual_alert(
    alert_data: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """手动触发告警"""
    if not FRAUD_DETECTION_AVAILABLE or not monitoring_system:
        raise HTTPException(
            status_code=503,
            detail="Monitoring system not available"
        )
    
    try:
        # 在后台处理告警
        background_tasks.add_task(
            monitoring_system.record_metric,
            alert_data.get("metric_name", "manual_alert"),
            alert_data.get("value", 1),
            alert_data.get("tags", {})
        )
        
        return {"status": "alert_triggered", "timestamp": time.time()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 辅助函数
async def assess_fraud_risk(transaction_data: Dict[str, Any], 
                           background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """内部欺诈风险评估函数"""
    if not FRAUD_DETECTION_AVAILABLE:
        return {"risk_level": "unknown", "reason": "fraud detection unavailable"}
    
    try:
        # 构造简单的交易和账户数据
        transactions = [transaction_data]
        accounts = [{"account_id": transaction_data.get("account_id", "unknown")}]
        
        # 调用GNN预测
        result = await predict_fraud_risk_gnn(transactions, accounts)
        
        # 记录评估指标
        background_tasks.add_task(
            monitoring_system.record_metric,
            "fraud_assessments_total",
            1,
            {"risk_level": result.get("risk_level", "unknown")}
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Internal fraud assessment failed: {e}")
        return {
            "risk_level": "error", 
            "risk_score": 0.5,
            "error": str(e)
        }

# 原有的其他接口保持不变
@app.post("/api/upload/url")
async def get_upload_url(
    filename: str = Query(..., description="文件名"),
    auth_context = Depends(require_auth)
):
    """获取上传URL"""
    try:
        upload_url = create_presigned_url(filename)
        return PresignResponse(upload_url=upload_url, expires_in=3600)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app.main_enhanced:app",
        host="0.0.0.0", 
        port=8080,
        reload=True,
        log_level="info"
    )