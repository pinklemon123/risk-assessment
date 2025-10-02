from fastapi import FastAPI, Depends, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import time
import uuid
import json
from typing import Dict, Any, Tuple
from pydantic import BaseModel
from collections import deque
from datetime import datetime
import threading

from .models import BatchRequest, BatchResponse, PresignResponse
from .auth import require_auth
from .transformer import load_mapping, apply_mapping
from .dq import validate_row
from .writer import write_raw, write_clean
from .storage import init_db, log_batch, log_transactions, log_assessment, get_ego_network, today_stats, get_account_chain, log_transfers
from .presign import create_presigned_url
from .admin import router as admin_router
from .metrics import metrics_store

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

# 初始化本地数据库
try:
    init_db()
except Exception:
    pass

# 添加管理路由
app.include_router(admin_router)

# 静态文件服务
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# ========== 后台任务：自动告警 ==========

_alert_thread_started = False

def _auto_alert_worker():
    """后台线程：周期性拉取 dashboard，按阈值生成告警。"""
    import time as _t
    last_fire = {"error_rate": 0, "cpu": 0, "p95": 0}
    cooldown = 60  # 同一类型告警冷却秒数
    while True:
        try:
            dash = metrics_store.get_dashboard()
            api = dash.get("api_metrics", {})
            sysm = dash.get("system_metrics", {})

            now = int(_t.time())

            # 阈值（可后续做成配置）
            error_rate = float(api.get("error_rate") or 0.0)  # 0~1
            p95_ms = float(api.get("p95_response_time_ms") or 0.0)
            cpu = float(sysm.get("cpu_usage_percent") or 0.0)

            if error_rate > 0.02 and now - last_fire["error_rate"] > cooldown:
                _recent_alerts.appendleft({
                    "time": datetime.utcnow().isoformat(),
                    "level": "warning",
                    "title": "API 错误率偏高",
                    "message": f"近窗口错误率 {error_rate*100:.2f}% 超过 2%",
                })
                last_fire["error_rate"] = now

            if p95_ms > 2000 and now - last_fire["p95"] > cooldown:
                _recent_alerts.appendleft({
                    "time": datetime.utcnow().isoformat(),
                    "level": "warning",
                    "title": "响应时间偏高",
                    "message": f"P95 响应时间 {p95_ms:.0f}ms 超过 2000ms",
                })
                last_fire["p95"] = now

            if cpu > 85 and now - last_fire["cpu"] > cooldown:
                _recent_alerts.appendleft({
                    "time": datetime.utcnow().isoformat(),
                    "level": "warning",
                    "title": "CPU 使用率偏高",
                    "message": f"CPU 使用率 {cpu:.1f}% 超过 85%",
                })
                last_fire["cpu"] = now

        except Exception:
            pass
        finally:
            _t.sleep(15)  # 15s 一次


def _ensure_alert_thread():
    global _alert_thread_started
    if _alert_thread_started:
        return
    t = threading.Thread(target=_auto_alert_worker, name="auto-alert-worker", daemon=True)
    t.start()
    _alert_thread_started = True

# 启动后台告警线程
try:
    _ensure_alert_thread()
except Exception:
    pass

# 主控制台路由
@app.get("/")
async def root():
    """主控制台页面"""
    return FileResponse("app/static/main-console.html")

@app.get("/console")
async def main_console():
    """主控制台页面"""
    return FileResponse("app/static/main-console.html")

# 显式页面路由，支持相对链接访问（index.html 等）
@app.get("/index.html")
async def page_index_html():
    return FileResponse("app/static/index.html")

@app.get("/main-console.html")
async def page_main_console_html():
    return FileResponse("app/static/main-console.html")

@app.get("/advanced-analysis.html")
async def page_advanced_analysis_html():
    return FileResponse("app/static/advanced-analysis.html")

@app.get("/monitoring.html")
async def page_monitoring_html():
    return FileResponse("app/static/monitoring.html")

@app.get("/analysis-report-clean.html")
async def page_analysis_report_clean_html():
    return FileResponse("app/static/analysis-report-clean.html")

# 纯黑白版本页面路由
@app.get("/pure-blackwhite-navigation.html")
async def pure_blackwhite_navigation():
    """纯黑白导航页面"""
    return FileResponse("app/static/pure-blackwhite-navigation.html")

@app.get("/index-pure-blackwhite.html")
async def index_pure_blackwhite():
    """纯黑白首页"""
    return FileResponse("app/static/index-pure-blackwhite.html")

@app.get("/analysis-pure-blackwhite.html")
async def analysis_pure_blackwhite():
    """纯黑白分析页面"""
    return FileResponse("app/static/analysis-pure-blackwhite.html")

@app.get("/network-pure-blackwhite.html")
async def network_pure_blackwhite():
    """纯黑白网络图谱页面"""
    return FileResponse("app/static/network-pure-blackwhite.html")

@app.get("/monitoring-pure-blackwhite.html")
async def monitoring_pure_blackwhite():
    """纯黑白监控页面"""
    return FileResponse("app/static/monitoring-pure-blackwhite.html")

# CSS和JS文件路由
@app.get("/css/{file_path:path}")
async def css_files(file_path: str):
    """CSS文件服务"""
    return FileResponse(f"app/static/css/{file_path}")

@app.get("/js/{file_path:path}")
async def js_files(file_path: str):
    """JS文件服务"""
    return FileResponse(f"app/static/js/{file_path}")

# 数据文件路由
@app.get("/comprehensive_analysis_results.json")
async def comprehensive_analysis_results():
    """分析结果数据文件"""
    return FileResponse("comprehensive_analysis_results.json")

@app.get("/healthz")
async def healthz():
    return {"ok": True, "ts": int(time.time())}

@app.get("/api/report")
async def api_analysis_report():
    """API分析报告页面"""
    return FileResponse("app/static/analysis-report.html")

@app.post("/ingest/transactions", response_model=BatchResponse)
async def ingest_transactions(payload: BatchRequest, auth=Depends(require_auth)):
    start_time = time.perf_counter()
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
    processing_duration = time.perf_counter() - start_time
    # 记录到本地数据库
    try:
        log_batch(batch_id, tenant, accepted, failed, processing_duration, time.time())
        log_transactions(clean_rows)
    except Exception:
        pass
    metrics_store.record_batch(
        tenant=tenant,
        batch_id=batch_id,
        accepted=accepted,
        failed=failed,
        duration_seconds=processing_duration,
        clean_rows=clean_rows,
    )

    return BatchResponse(
        accepted=accepted,
        failed=failed,
        batch_id=batch_id,
        feat_ver_hint=time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())
    )

# 转账链路数据写入（用于账户间链路图）
class TransferBatchRequest(BaseModel):
    rows: list[dict]

@app.post("/ingest/transfers")
async def ingest_transfers(payload: TransferBatchRequest, auth=Depends(require_auth)):
    start_time = time.perf_counter()
    rows = payload.rows or []
    try:
        log_transfers(rows)
        duration = time.perf_counter() - start_time
        return {"accepted": len(rows), "failed": 0, "duration_seconds": duration}
    except Exception as e:
        return {"accepted": 0, "failed": len(rows), "error": str(e)}

# 页面路由 - 重定向到主控制台
@app.get("/dashboard")
async def dashboard():
    """重定向到主控制台"""
    return FileResponse("app/static/main-console.html")

@app.get("/fraud-detection")
async def fraud_detection():
    """重定向到主控制台"""
    return FileResponse("app/static/main-console.html")

@app.get("/monitoring")
async def monitoring():
    """监控页面"""
    return FileResponse("app/static/monitoring.html")

@app.get("/batch-process")
async def batch_process():
    """批量处理页面"""
    return FileResponse("app/static/batch-process.html")

@app.get("/batch-process.html")
async def batch_process_html():
    """批量处理页面"""
    return FileResponse("app/static/batch-process.html")

# 专用功能页面（保留用于深度分析）
@app.get("/analysis-report")
async def analysis_report():
    """分析报告页面 - 详细的可视化报告"""
    return FileResponse("app/static/analysis-report-clean.html")

@app.get("/advanced-analysis")
async def advanced_analysis():
    """高级分析页面 - 大数据分析和批量导入"""
    return FileResponse("app/static/advanced-analysis.html")

@app.get("/final-analysis")
async def final_analysis_report():
    """最终分析报告页面 - 网络图和风险分析"""
    return FileResponse("app/static/analysis-report-final.html")

# 网络关系图页面
@app.get("/network")
async def network_page():
    """交易网络关系图页面（黑白主题）"""
    return FileResponse("app/static/network.html")

@app.get("/network.html")
async def network_page_html():
    """交易网络关系图页面（黑白主题）- 兼容.html路径"""
    return FileResponse("app/static/network.html")

# 备用配置页面（如需要传统配置界面）
@app.get("/config")
async def config():
    """传统配置页面"""
    return FileResponse("app/static/index.html")

# 欺诈检测API接口
@app.post("/api/fraud/assess")
async def assess_fraud_risk_endpoint(
    request_data: dict
):
    """简化的欺诈风险评估接口"""
    try:
        # 数据验证和清理
        transactions = request_data.get('transactions', [])
        accounts = request_data.get('accounts', [])
        
        # 确保transactions是列表
        if not isinstance(transactions, list):
            transactions = []
        
        if not transactions:
            return {
                "error": "No transactions provided",
                "risk_score": 0.5,
                "risk_level": "unknown",
                "confidence": 0.3,
                "transaction_count": 0,
                "account_count": 0
            }
        
        # 验证每个交易是字典格式
        valid_transactions = []
        for t in transactions:
            if isinstance(t, dict):
                valid_transactions.append(t)
        
        if not valid_transactions:
            return {
                "error": "No valid transactions found (must be dictionary format)",
                "risk_score": 0.5,
                "risk_level": "unknown", 
                "confidence": 0.3,
                "transaction_count": 0,
                "account_count": len(accounts) if isinstance(accounts, list) else 0
            }
        
        transactions = valid_transactions
        
        # 简化的风险评分逻辑
        total_amount = 0
        for t in transactions:
            try:
                amount = float(t.get('amount', 0))
                total_amount += amount
            except (ValueError, TypeError, AttributeError):
                # 跳过无法解析金额的交易
                continue
                
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
            if isinstance(txn, dict):
                merchant = str(txn.get('merchant_id', '')).lower()
                if merchant and any(keyword in merchant for keyword in ['crypto', 'casino', 'gambling']):
                    risk_score += 0.3
                    break
                elif merchant and any(keyword in merchant for keyword in ['online', 'foreign']):
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
        
        result = {
            "risk_score": round(risk_score, 3),
            "risk_level": risk_level,
            "confidence": round(min(0.95, confidence), 3),
            "transaction_count": len(transactions),
            "account_count": len(accounts) if isinstance(accounts, list) else 0,
            "processing_time": 0.05,  # 模拟处理时间
            "enhanced_features": {
                "total_amount": round(total_amount, 2),
                "average_amount": round(avg_amount, 2),
                "transaction_time_risk": "high" if current_hour < 6 or current_hour > 22 else "low",
                "volume_risk": "high" if len(transactions) > 10 else "medium" if len(transactions) > 5 else "low"
            }
        }
        # 记录检测摘要
        try:
            log_assessment(time.time(), result["risk_score"], result["risk_level"], result["confidence"], result["transaction_count"])
        except Exception:
            pass
        return result
        
    except Exception as e:
        return {
            "error": f"Assessment failed: {str(e)}",
            "risk_score": 0.5,
            "risk_level": "unknown", 
            "confidence": 0.3,
            "transaction_count": 0,
            "account_count": 0,
            "fallback": True,
            "enhanced_features": {
                "total_amount": 0,
                "average_amount": 0,
                "transaction_time_risk": "unknown",
                "volume_risk": "unknown"
            }
        }

@app.get("/api/monitoring/dashboard")
async def get_monitoring_dashboard():
    """获取监控面板数据，基于真实的摄取统计"""
    dash = metrics_store.get_dashboard()
    # 合并 DB 的今日统计
    try:
        dash["today"] = today_stats()
    except Exception:
        dash["today"] = {"today_processed": 0, "today_fraud": 0, "today_amount": 0.0}
    return dash


# ========== 简易告警/日志 API ==========
_recent_alerts = deque(maxlen=200)
_recent_logs = deque(maxlen=500)


class AlertIn(BaseModel):
    metric_name: str
    value: float | int | None = None
    tags: dict | None = None
    title: str | None = None
    message: str | None = None
    level: str | None = None  # info/warning/critical


@app.post("/api/monitoring/alert")
async def post_alert(alert: AlertIn):
    ts = datetime.utcnow().isoformat()
    level = (alert.level or ("critical" if (alert.metric_name or "").endswith("critical") else "info")).lower()
    title = alert.title or f"告警: {alert.metric_name}"
    message = alert.message or f"收到指标 {alert.metric_name} 值 {alert.value}"
    data = {
        "time": ts,
        "level": level,
        "title": title,
        "message": message,
        "metric_name": alert.metric_name,
        "value": alert.value,
        "tags": alert.tags or {},
    }
    _recent_alerts.appendleft(data)
    _recent_logs.appendleft({
        "time": ts,
        "level": "INFO",
        "source": "monitoring",
        "message": f"Alert accepted: {title}"
    })
    return {"ok": True}


@app.get("/api/monitoring/alerts")
async def get_alerts(limit: int = Query(50, ge=1, le=200)):
    items = list(_recent_alerts)[:limit]
    return {"items": items, "count": len(items)}


@app.get("/api/monitoring/logs")
async def get_logs(limit: int = Query(100, ge=1, le=500)):
    if not _recent_logs:
        now = datetime.utcnow()
        samples = [
            {"time": (now).isoformat(), "level": "INFO", "source": "api", "message": "Server ready"},
            {"time": (now).isoformat(), "level": "INFO", "source": "system", "message": "Metrics tick"},
        ]
        for s in samples:
            _recent_logs.appendleft(s)
    items = list(_recent_logs)[:limit]
    return {"items": items, "count": len(items)}


# ========== 风险账户周边关系 API ==========
@app.get("/api/network/ego")
async def api_network_ego(
    account_id: str = Query(..., description="风险账户 ID，如 acc_12345"),
    depth: int = Query(1, ge=0, le=2),
    max_nodes: int = Query(200, ge=20, le=1000),
    risk_only: bool = Query(False),
):
    try:
        data = get_ego_network(account_id=account_id, depth=depth, max_nodes=max_nodes, risk_only=risk_only)
        return data
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}


@app.get("/api/network/chain")
async def api_network_chain(
    account_id: str = Query(..., description="起始账户ID"),
    hops: int = Query(2, ge=1, le=6, description="最大跳数"),
    target_account: str | None = Query(None, description="目标账户ID；留空则返回起点在 hops 内的若干路径"),
    max_paths: int = Query(20, ge=1, le=100, description="最大路径数"),
):
    try:
        data = get_account_chain(account_id=account_id, hops=hops, target_account=target_account, max_paths=max_paths)
        return data
    except Exception as e:
        return {"nodes": [], "edges": [], "paths": [], "error": str(e)}


# ========== 网络关系图 API ==========
_NETWORK_CACHE: Dict[str, Any] = {}

def _risk_level(score: float, high: float = 0.7, mid: float = 0.4) -> str:
    if score >= high:
        return "high"
    if score >= mid:
        return "medium"
    return "low"

def _load_transactions(limit: int = 5000) -> Tuple[list, str]:
    """尝试加载较真实的数据集，失败时回退到示例数据。
    返回 (transactions, source) 二元组。
    """
    # 优先使用复杂数据集（包含 account_id/merchant_id/calculated_risk_score/is_fraud 等）
    primary_path = "complex_transaction_dataset.json"
    fallback_path = "examples/sample_transactions.json"

    try:
        if os.path.exists(primary_path):
            with open(primary_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            txns = obj.get("transactions", [])
            return txns[: max(1, limit)], primary_path
    except Exception:
        pass

    # 回退到示例数据（字段名不同）
    try:
        if os.path.exists(fallback_path):
            with open(fallback_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            rows = obj.get("rows", [])
            # 统一成与主数据集类似的字段
            norm = []
            for r in rows[: max(1, limit)]:
                norm.append({
                    "id": r.get("id"),
                    "timestamp": r.get("time"),
                    "account_id": r.get("payer") or r.get("src_entity_id"),
                    "merchant_id": r.get("merchant_id") or r.get("payee"),
                    "amount": r.get("amt") or r.get("amount") or 0,
                    "calculated_risk_score": 1.0 if (r.get("risk_flags") or []) else 0.2,
                    "is_fraud": bool(r.get("risk_flags"))
                })
            return norm, fallback_path
    except Exception:
        pass

    return [], "none"

def _build_network_graph(
    risk_only: bool = False,
    max_nodes: int = 200,
    max_transactions: int = 5000,
    risk_threshold: float = 0.7,
    use_fraud_flag: bool = True,
    risk_agg: str = "max",
) -> Dict[str, Any]:
    """从本地数据构建账户-商户的二部图，并根据需要筛选风险子图。

    返回格式：
    {
      "nodes": [{id, type, label, risk_score, risk_level, txn_count, total_amount}],
      "edges": [{source, target, weight, total_amount}]
    }
    """

    cache_key = f"{risk_only}:{max_nodes}:{max_transactions}:{risk_threshold}:{use_fraud_flag}:{risk_agg}"
    cached = _NETWORK_CACHE.get(cache_key)
    if cached and (time.time() - cached.get("_ts", 0) < 60):  # 简单缓存60秒
        return cached["data"]

    txns, source = _load_transactions(limit=max_transactions)

    # 聚合节点与边
    accounts: Dict[str, Dict[str, Any]] = {}
    merchants: Dict[str, Dict[str, Any]] = {}
    edges: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for t in txns:
        acc = str(t.get("account_id") or t.get("src_entity_id") or "")
        mer = str(t.get("merchant_id") or t.get("dst_entity_id") or "")
        if not acc or not mer:
            continue

        amt = 0.0
        try:
            amt = float(t.get("amount") or 0)
        except Exception:
            amt = 0.0

        is_fraud = bool(t.get("is_fraud", False))
        risk_score = float(t.get("calculated_risk_score", 0.0) or 0.0)

        # 账户节点
        a = accounts.setdefault(acc, {
            "id": acc, "type": "account", "label": acc,
            "txn_count": 0, "total_amount": 0.0,
            "risk_max": 0.0, "risk_sum": 0.0, "risk_count": 0,
            "risk_score": 0.0, "is_fraud": False
        })
        a["txn_count"] += 1
        a["total_amount"] += amt
        a["risk_max"] = max(a["risk_max"], risk_score)
        a["risk_sum"] += risk_score
        a["risk_count"] += 1
        a["is_fraud"] = a["is_fraud"] or is_fraud

        # 商户节点
        m = merchants.setdefault(mer, {
            "id": mer, "type": "merchant", "label": mer,
            "txn_count": 0, "total_amount": 0.0,
            "risk_max": 0.0, "risk_sum": 0.0, "risk_count": 0,
            "risk_score": 0.0, "is_fraud": False
        })
        m["txn_count"] += 1
        m["total_amount"] += amt
        m["risk_max"] = max(m["risk_max"], risk_score)
        m["risk_sum"] += risk_score
        m["risk_count"] += 1
        m["is_fraud"] = m["is_fraud"] or is_fraud

        # 边（账户-商户）
        key = (acc, mer)
        e = edges.setdefault(key, {"source": acc, "target": mer, "weight": 0, "total_amount": 0.0})
        e["weight"] += 1
        e["total_amount"] += amt

    # 计算风险等级
    use_mean = (str(risk_agg or "").lower() == "mean")
    for d in (accounts, merchants):
        for node in d.values():
            # 选择风险聚合方式
            if use_mean and node.get("risk_count"):
                node["risk_score"] = float(node["risk_sum"]) / float(max(1, node["risk_count"]))
            else:
                node["risk_score"] = float(node.get("risk_max") or 0.0)
            if use_fraud_flag and node["is_fraud"]:
                node["risk_level"] = "high"
                node["risk_score"] = max(node["risk_score"], 1.0)
            else:
                node["risk_level"] = _risk_level(node["risk_score"], high=risk_threshold)

    # 根据重要性选择节点（优先交易数与金额高的）
    def node_key(n: Dict[str, Any]):
        return (n["txn_count"], n["total_amount"])  # 交易数优先

    account_list = sorted(accounts.values(), key=node_key, reverse=True)
    merchant_list = sorted(merchants.values(), key=node_key, reverse=True)

    # 先选各自前一半，避免某一类淹没
    half = max(1, max_nodes // 2)
    selected_accounts = account_list[:half]
    selected_merchants = merchant_list[:half]

    selected_ids = {n["id"] for n in selected_accounts} | {n["id"] for n in selected_merchants}

    # 过滤边只保留连接到所选节点的
    edge_list = [e for e in edges.values() if e["source"] in selected_ids and e["target"] in selected_ids]

    # 如果只看风险子图，则过滤
    if risk_only:
        risky_ids = {n["id"] for n in selected_accounts if n["risk_level"] == "high"} | \
                    {n["id"] for n in selected_merchants if n["risk_level"] == "high"}
        # 包含与高风险节点相连的边与对端节点
        edge_list = [e for e in edge_list if (e["source"] in risky_ids or e["target"] in risky_ids)]
        involved_ids = {e["source"] for e in edge_list} | {e["target"] for e in edge_list}
        selected_accounts = [n for n in selected_accounts if n["id"] in involved_ids]
        selected_merchants = [n for n in selected_merchants if n["id"] in involved_ids]

    nodes = selected_accounts + selected_merchants

    data = {"nodes": nodes, "edges": edge_list, "source": source}

    _NETWORK_CACHE[cache_key] = {"_ts": time.time(), "data": data}
    return data


@app.get("/api/network/graph")
async def api_network_graph(
    risk_only: bool = Query(False, description="是否仅返回风险子图"),
    max_nodes: int = Query(200, ge=10, le=1000, description="返回的最大节点数（账户+商户）"),
    max_transactions: int = Query(5000, ge=100, le=50000, description="参与构图的最大交易条数"),
    risk_threshold: float = Query(0.7, ge=0.0, le=1.0, description="高风险阈值"),
    use_fraud_flag: bool = Query(True, description="是否使用 is_fraud 将节点强制标记为高风险"),
    risk_agg: str = Query("max", description="节点风险聚合方式: max 或 mean"),
):
    """返回用于D3可视化的交易网络（账户-商户二部图）。"""
    try:
        graph = _build_network_graph(
            risk_only=risk_only,
            max_nodes=max_nodes,
            max_transactions=max_transactions,
            risk_threshold=risk_threshold,
            use_fraud_flag=use_fraud_flag,
            risk_agg=risk_agg,
        )
        return graph
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

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

@app.post("/api/advanced/complex-analysis")
async def run_complex_analysis(request_data: dict):
    """运行复杂分析 - 需要大量计算时间的真实算法"""
    try:
        import subprocess
        import os
        import json as json_module
        
        analysis_type = request_data.get('analysis_type', 'comprehensive')
        data_source = request_data.get('data_source', 'complex_transaction_dataset.json')
        
        start_time = time.time()
        
        # 检查数据文件是否存在
        if not os.path.exists(data_source):
            return {
                "error": f"Data file {data_source} not found. Please generate it first using generate_complex_dataset.py",
                "processing_time": 0,
                "status": "failed"
            }
        
        # 运行复杂分析引擎
        if analysis_type in ['comprehensive', 'ml_only', 'network_only', 'time_only']:
            mode_map = {
                'comprehensive': 'comprehensive',
                'ml_only': 'ml_only',
                'network_only': 'network_only',
                'time_only': 'time_only',
            }
            mode_arg = f"--mode={mode_map.get(analysis_type, 'comprehensive')}"
            result = subprocess.run(
                ["python", "/app/complex_analysis_engine.py", mode_arg, f"--data={data_source}"], 
                capture_output=True, 
                text=True,
                cwd=os.getcwd(),
                timeout=300  # 5分钟超时
            )
            
            if result.returncode == 0:
                # 读取分析结果
                try:
                    with open("comprehensive_analysis_results.json", "r", encoding="utf-8") as f:
                        analysis_results = json_module.load(f)
                    
                    total_time = time.time() - start_time
                    
                    return {
                        "status": "success",
                        "analysis_type": analysis_type,
                        "processing_time": total_time,
                        "results": analysis_results,
                        "performance_metrics": {
                            "total_transactions_analyzed": analysis_results.get("comprehensive_summary", {}).get("transactions_analyzed", 0),
                            "analysis_modules_completed": analysis_results.get("comprehensive_summary", {}).get("analysis_modules_completed", 0),
                            "processing_rate": analysis_results.get("comprehensive_summary", {}).get("performance_metrics", {}).get("transactions_per_second", 0)
                        },
                        "computation_proof": {
                            "stdout_sample": result.stdout[-500:] if result.stdout else "",
                            "actual_computation_time": analysis_results.get("comprehensive_summary", {}).get("total_processing_time", 0)
                        }
                    }
                except Exception as e:
                    return {
                        "error": f"Failed to read analysis results: {str(e)}",
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "processing_time": time.time() - start_time,
                        "status": "partial_success"
                    }
            else:
                return {
                    "error": f"Analysis process failed with return code {result.returncode}",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "processing_time": time.time() - start_time,
                    "status": "failed"
                }
        else:
                return {
                    "error": f"Unsupported analysis type: {analysis_type}",
                    "processing_time": time.time() - start_time,
                    "status": "failed"
                }
            
    except subprocess.TimeoutExpired:
        return {
            "error": "Analysis timed out after 5 minutes",
            "processing_time": 300,
            "status": "timeout"
        }
    except Exception as e:
        return {
            "error": f"Complex analysis failed: {str(e)}",
            "processing_time": time.time() - start_time if 'start_time' in locals() else 0,
            "status": "error"
        }

@app.get("/api/advanced/analysis-status")
async def get_analysis_status():
    """获取分析状态和数据集信息"""
    import os
    import json as json_module
    
    status = {
        "complex_dataset_exists": False,
        "analysis_results_exists": False,
        "dataset_info": {},
        "last_analysis": {}
    }
    
    try:
        # 检查复杂数据集
        if os.path.exists("complex_transaction_dataset.json"):
            with open("complex_transaction_dataset.json", "r") as f:
                dataset = json_module.load(f)
                status["complex_dataset_exists"] = True
                status["dataset_info"] = {
                    "transaction_count": len(dataset.get("transactions", [])),
                    "complexity_score": dataset.get("metadata", {}).get("complexity_score", 0),
                    "fraud_rate": dataset.get("metadata", {}).get("fraud_rate", 0),
                    "generated_at": dataset.get("metadata", {}).get("generated_at", "")
                }
        
        # 检查分析结果
        if os.path.exists("comprehensive_analysis_results.json"):
            with open("comprehensive_analysis_results.json", "r") as f:
                results = json_module.load(f)
                status["analysis_results_exists"] = True
                summary = results.get("comprehensive_summary", {})
                status["last_analysis"] = {
                    "total_processing_time": summary.get("total_processing_time", 0),
                    "transactions_analyzed": summary.get("transactions_analyzed", 0),
                    "modules_completed": summary.get("analysis_modules_completed", 0),
                    "timestamp": results.get("metadata", {}).get("analysis_completed_at", "")
                }
                
    except Exception as e:
        print(f"状态检查错误: {e}")
    
    return status

@app.get("/api/analysis/results")
async def get_analysis_results():
    """获取分析结果数据用于报告展示"""
    try:
        import json as json_module
        import os
        
        if not os.path.exists("comprehensive_analysis_results.json"):
            return {
                "error": "分析结果文件不存在，请先运行高级分析",
                "status": "no_results"
            }
        
        with open("comprehensive_analysis_results.json", "r", encoding="utf-8") as f:
            results = json_module.load(f)
        
        return {
            "status": "success",
            "data": results,
            "timestamp": results.get("comprehensive_summary", {}).get("total_processing_time", 0),
            "summary": {
                "transactions_analyzed": results.get("comprehensive_summary", {}).get("transactions_analyzed", 0),
                "processing_time": results.get("comprehensive_summary", {}).get("total_processing_time", 0),
                "modules_completed": results.get("comprehensive_summary", {}).get("analysis_modules_completed", 0)
            }
        }
    
    except Exception as e:
        return {
            "error": f"读取分析结果失败: {str(e)}",
            "status": "error"
        }

# 全局变量用于跟踪分析进度
analysis_progress = {
    "status": "idle",
    "current_step": "",
    "steps": [],
    "progress_percentage": 0,
    "start_time": None
}

@app.get("/api/analysis/progress")
async def get_analysis_progress():
    """获取当前分析进度"""
    import os
    import json as json_module
    
    # 尝试读取真实的进度文件
    progress_file = "analysis_progress.json"
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json_module.load(f)
            return progress_data
        except Exception as e:
            print(f"读取进度文件失败: {e}")
    
    # 如果没有进度文件，返回默认状态
    global analysis_progress
    return {
        "status": analysis_progress["status"],
        "current_step": analysis_progress["current_step"],
        "steps": analysis_progress["steps"],
        "progress_percentage": analysis_progress["progress_percentage"],
        "elapsed_time": time.time() - analysis_progress["start_time"] if analysis_progress["start_time"] else 0
    }

@app.post("/api/analysis/start-with-progress")
async def start_analysis_with_progress(request_data: dict):
    """启动带进度跟踪的分析"""
    global analysis_progress
    
    try:
        import subprocess
        import os
        import json as json_module
        
        # 重置进度状态
        analysis_progress.update({
            "status": "running",
            "current_step": "初始化分析环境",
            "steps": [
                {"name": "初始化分析环境", "status": "running", "duration": 0},
                {"name": "加载交易数据", "status": "pending", "duration": 0},
                {"name": "构建网络图", "status": "pending", "duration": 0},
                {"name": "计算中心性指标", "status": "pending", "duration": 0},
                {"name": "检测社群结构", "status": "pending", "duration": 0},
                {"name": "时序异常检测", "status": "pending", "duration": 0},
                {"name": "机器学习训练", "status": "pending", "duration": 0},
                {"name": "生成综合结论", "status": "pending", "duration": 0}
            ],
            "progress_percentage": 0,
            "start_time": time.time()
        })
        
        analysis_type = request_data.get('analysis_type', 'comprehensive')
        data_source = request_data.get('data_source', 'complex_transaction_dataset.json')
        
        # 检查数据文件
        if not os.path.exists(data_source):
            analysis_progress["status"] = "error"
            analysis_progress["current_step"] = f"数据文件 {data_source} 不存在"
            return {
                "error": f"Data file {data_source} not found",
                "status": "failed"
            }
        
        # 更新进度：开始分析
        analysis_progress["current_step"] = "执行复杂分析算法"
        analysis_progress["steps"][0]["status"] = "completed"
        analysis_progress["steps"][1]["status"] = "running"
        analysis_progress["progress_percentage"] = 12
        
        # 运行分析
        result = subprocess.run(
            ["python", "/app/complex_analysis_engine.py"], 
            capture_output=True, 
            text=True,
            cwd=os.getcwd(),
            timeout=300
        )
        
        if result.returncode == 0:
            # 更新所有步骤为完成
            for step in analysis_progress["steps"]:
                step["status"] = "completed"
            analysis_progress["status"] = "completed"
            analysis_progress["current_step"] = "分析完成"
            analysis_progress["progress_percentage"] = 100
            
            # 读取结果
            with open("comprehensive_analysis_results.json", "r", encoding="utf-8") as f:
                analysis_results = json_module.load(f)
            
            return {
                "status": "success",
                "analysis_type": analysis_type,
                "processing_time": time.time() - analysis_progress["start_time"],
                "results": analysis_results
            }
        else:
            analysis_progress["status"] = "error"
            analysis_progress["current_step"] = f"分析失败: {result.stderr}"
            return {
                "error": f"Analysis failed: {result.stderr}",
                "status": "failed"
            }
            
    except Exception as e:
        analysis_progress["status"] = "error"
        analysis_progress["current_step"] = f"异常: {str(e)}"
        return {
            "error": f"Analysis failed: {str(e)}",
            "status": "error"
        }

@app.post("/api/data/generate-complex")
async def generate_complex_dataset(request: dict):
    """生成复杂的交易数据集用于高级分析"""
    import networkx as nx
    import numpy as np
    import random
    from datetime import datetime, timedelta
    import json as json_lib
    
    size = request.get("size", 1000)  # 默认1000条，避免数据膨胀
    fraud_rate = request.get("fraud_rate", 0.15)
    
    start_time = time.time()
    
    # 生成复杂的交易网络
    transactions = []
    accounts = [f"acc_{i:05d}" for i in range(min(1000, size // 10))]
    merchants = [f"mer_{i:04d}" for i in range(min(200, size // 50))]
    
    # 使用真实的网络生成算法
    G = nx.barabasi_albert_graph(len(accounts), 3)  # 真实的网络算法
    
    for i in range(size):
        # 基于网络结构的交易生成
        node1, node2 = random.choice(list(G.edges()))
        account = accounts[node1]
        merchant = random.choice(merchants)
        
        # 复杂的金额计算 
        base_amount = np.random.uniform(10, 5000)  # 使用均匀分布
        time_factor = np.sin(i / size * 2 * np.pi) * 0.3 + 1  # 时间周期性
        amount = base_amount * time_factor
        
        # 欺诈检测特征
        is_fraud = random.random() < fraud_rate
        if is_fraud:
            # 欺诈交易的特殊模式
            amount *= random.choice([0.1, 5.0, 10.0])  # 异常金额
        
        # 时间序列生成
        base_time = datetime.now() - timedelta(days=90)
        time_offset = i * 300 + random.randint(-120, 120)  # 5分钟间隔 + 噪声
        timestamp = base_time + timedelta(seconds=time_offset)
        
        transaction = {
            "id": f"complex_tx_{i:08d}",
            "amount": float(amount),
            "account_id": account,
            "merchant_id": merchant,
            "timestamp": timestamp.isoformat(),
            "is_fraud": is_fraud,
            "network_node": node1,
            "risk_features": {
                "velocity_score": random.random(),
                "geographic_risk": random.random(),
                "behavioral_anomaly": random.random()
            }
        }
        transactions.append(transaction)
    
    # 计算复杂度评分
    network_complexity = G.number_of_edges() / G.number_of_nodes()
    time_complexity = len(set(t["timestamp"][:10] for t in transactions)) / 90  # 时间分布
    fraud_complexity = abs(fraud_rate - 0.5) * 2  # 欺诈分布复杂度
    
    complexity_score = min(10.0, (network_complexity + time_complexity + fraud_complexity) * 3)
    
    processing_time = time.time() - start_time
    
    # 保存到文件
    dataset = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "size": size,
            "complexity_score": complexity_score,
            "fraud_rate": fraud_rate,
            "processing_time": processing_time
        },
        "transactions": transactions
    }
    
    with open("complex_transaction_dataset.json", "w") as f:
        json_lib.dump(dataset, f, indent=2)
    
    return {
        "status": "success",
        "dataset_created": True,
        "transactions_generated": len(transactions),
        "complexity_score": complexity_score,
        "processing_time": processing_time,
        "network_stats": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G)
        }
    }

@app.post("/api/data/upload")
async def upload_data_file(file: UploadFile = File(...)):
    """上传数据文件进行分析"""
    import json as json_lib
    from datetime import datetime
    
    try:
        # 检查文件类型
        if not file.filename.endswith(('.json', '.csv')):
            return {"status": "error", "message": "只支持JSON和CSV文件格式"}
        
        # 读取文件内容
        content = await file.read()
        
        if file.filename.endswith('.json'):
            # JSON文件处理
            try:
                data = json_lib.loads(content.decode('utf-8'))
                
                # 验证数据格式
                if "transactions" not in data:
                    return {"status": "error", "message": "JSON文件必须包含'transactions'字段"}
                
                # 保存为标准格式
                dataset = {
                    "metadata": {
                        "uploaded_at": datetime.now().isoformat(),
                        "filename": file.filename,
                        "size": len(data["transactions"]),
                        "source": "user_upload"
                    },
                    "transactions": data["transactions"]
                }
                
                with open("complex_transaction_dataset.json", "w") as f:
                    json_lib.dump(dataset, f, indent=2)
                
                return {
                    "status": "success", 
                    "message": f"成功上传{len(data['transactions'])}笔交易数据",
                    "filename": file.filename,
                    "transaction_count": len(data["transactions"])
                }
                
            except json_lib.JSONDecodeError as e:
                return {"status": "error", "message": f"JSON格式错误: {str(e)}"}
        
        elif file.filename.endswith('.csv'):
            # CSV文件处理
            import csv
            import io
            
            try:
                csv_content = content.decode('utf-8')
                reader = csv.DictReader(io.StringIO(csv_content))
                transactions = list(reader)
                
                # 基本数据验证和转换
                processed_transactions = []
                for i, row in enumerate(transactions):
                    try:
                        transaction = {
                            "id": row.get("id", f"uploaded_tx_{i:08d}"),
                            "amount": float(row.get("amount", 0)),
                            "timestamp": row.get("timestamp", datetime.now().isoformat()),
                            "account_id": row.get("account_id", f"acc_{i % 100:03d}"),
                            "merchant_id": row.get("merchant_id", f"mer_{i % 50:03d}"),
                            "is_fraud": str(row.get("is_fraud", "false")).lower() == "true"
                        }
                        processed_transactions.append(transaction)
                    except (ValueError, KeyError):
                        continue  # 跳过格式错误的行
                
                dataset = {
                    "metadata": {
                        "uploaded_at": datetime.now().isoformat(),
                        "filename": file.filename,
                        "size": len(processed_transactions),
                        "source": "user_upload_csv"
                    },
                    "transactions": processed_transactions
                }
                
                with open("complex_transaction_dataset.json", "w") as f:
                    json_lib.dump(dataset, f, indent=2)
                
                return {
                    "status": "success", 
                    "message": f"成功处理CSV文件，导入{len(processed_transactions)}笔交易",
                    "filename": file.filename,
                    "transaction_count": len(processed_transactions)
                }
                
            except Exception as e:
                return {"status": "error", "message": f"CSV处理错误: {str(e)}"}
        
    except Exception as e:
        return {"status": "error", "message": f"文件上传失败: {str(e)}"}

@app.post("/api/data/manual-input")
async def manual_input_data(data: dict):
    """手动输入交易数据"""
    import json as json_lib
    from datetime import datetime
    
    try:
        transactions = data.get("transactions", [])
        
        if not transactions:
            return {"status": "error", "message": "请提供交易数据"}
        
        # 验证数据格式
        for i, txn in enumerate(transactions):
            required_fields = ["id", "amount", "timestamp"]
            for field in required_fields:
                if field not in txn:
                    return {"status": "error", "message": f"交易{i+1}缺少必要字段: {field}"}
        
        dataset = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "size": len(transactions),
                "source": "manual_input"
            },
            "transactions": transactions
        }
        
        with open("complex_transaction_dataset.json", "w") as f:
            json_lib.dump(dataset, f, indent=2)
        
        return {
            "status": "success", 
            "message": f"成功添加{len(transactions)}笔交易数据",
            "transaction_count": len(transactions)
        }
        
    except Exception as e:
        return {"status": "error", "message": f"数据输入失败: {str(e)}"}

@app.post("/api/data/load-sample")
async def load_sample_data(request: dict = None):
    """加载示例数据集"""
    import json as json_lib
    import random
    from datetime import datetime, timedelta
    
    try:
        size = request.get("size", 1000) if request else 1000
        
        # 生成示例交易数据
        transactions = []
        accounts = [f"acc_{i:05d}" for i in range(100)]
        merchants = [f"mer_{i:04d}" for i in range(50)]
        
        for i in range(size):
            base_time = datetime.now() - timedelta(days=30)
            time_offset = random.randint(0, 30*24*3600)  # 30天内随机时间
            
            transaction = {
                "id": f"sample_tx_{i:08d}",
                "amount": round(random.uniform(10, 5000), 2),  # 使用uniform分布
                "timestamp": (base_time + timedelta(seconds=time_offset)).isoformat(),
                "account_id": random.choice(accounts),
                "merchant_id": random.choice(merchants),
                "is_fraud": random.random() < 0.1,  # 10%欺诈率
                "description": f"Sample transaction {i+1}"
            }
            transactions.append(transaction)
        
        dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "size": len(transactions),
                "source": "sample_data",
                "description": "Generated sample dataset for testing"
            },
            "transactions": transactions
        }
        
        with open("complex_transaction_dataset.json", "w") as f:
            json_lib.dump(dataset, f, indent=2)
        
        return {
            "status": "success", 
            "message": f"成功生成{len(transactions)}笔示例交易数据",
            "transaction_count": len(transactions),
            "fraud_rate": 0.1
        }
        
    except Exception as e:
        return {"status": "error", "message": f"示例数据生成失败: {str(e)}"}


