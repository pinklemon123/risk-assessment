import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Tuple
from datetime import datetime

DB_PATH = os.getenv("DB_PATH", "data/app.db")


def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


@contextmanager
def _connect():
    _ensure_parent_dir(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS batches (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              batch_id TEXT,
              tenant TEXT,
              accepted INTEGER,
              failed INTEGER,
              duration_seconds REAL,
              ts TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
              id TEXT PRIMARY KEY,
              ts TEXT,
              account_id TEXT,
              merchant_id TEXT,
              amount REAL,
              currency TEXT,
              is_fraud INTEGER DEFAULT 0,
              risk_score REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assessments (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts TEXT,
              risk_score REAL,
              risk_level TEXT,
              confidence REAL,
              txn_count INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transfers (
                id TEXT PRIMARY KEY,
                ts TEXT,
                src_account TEXT,
                dst_account TEXT,
                amount REAL,
                currency TEXT,
                is_fraud INTEGER DEFAULT 0,
                risk_score REAL
            )
            """
        )


def log_batch(batch_id: str, tenant: str, accepted: int, failed: int, duration_seconds: float, ts: float) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO batches(batch_id, tenant, accepted, failed, duration_seconds, ts) VALUES(?,?,?,?,?,?)",
            (batch_id, tenant, accepted, failed, duration_seconds, datetime.utcfromtimestamp(ts).isoformat()),
        )


def log_transactions(rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows) or []
    if not rows:
        return
    to_upsert: List[Tuple] = []
    for r in rows:
        tid = str(r.get("id") or r.get("txn_id") or "")
        ts = str(r.get("ts") or "")
        account = str(r.get("src_entity_id") or r.get("account_id") or "")
        merchant = str(r.get("merchant_id") or r.get("dst_entity_id") or "")
        try:
            amount = float(r.get("amount") or 0.0)
        except Exception:
            amount = 0.0
        currency = str(r.get("currency") or "")
        is_fraud = 1 if bool(r.get("is_fraud") or False) else 0
        risk_score = r.get("calculated_risk_score")
        risk_score = float(risk_score) if risk_score is not None else None
        if not tid:
            continue
        to_upsert.append((tid, ts, account, merchant, amount, currency, is_fraud, risk_score))

    if not to_upsert:
        return
    with _connect() as conn:
        conn.executemany(
            """
            INSERT INTO transactions(id, ts, account_id, merchant_id, amount, currency, is_fraud, risk_score)
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
              ts=excluded.ts,
              account_id=excluded.account_id,
              merchant_id=excluded.merchant_id,
              amount=excluded.amount,
              currency=excluded.currency
            """,
            to_upsert,
        )


def log_assessment(ts: float, risk_score: float, risk_level: str, confidence: float, txn_count: int) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO assessments(ts, risk_score, risk_level, confidence, txn_count) VALUES(?,?,?,?,?)",
            (datetime.utcfromtimestamp(ts).isoformat(), risk_score, risk_level, confidence, txn_count),
        )

def log_transfers(rows: Iterable[Dict[str, Any]]) -> None:
    """批量写入/更新 transfers 表。
    期望字段: id, ts, src_account, dst_account, amount, currency, is_fraud(可选), risk_score(可选)
    """
    rows = list(rows) or []
    if not rows:
        return
    to_upsert: List[Tuple] = []
    for r in rows:
        tid = str(r.get("id") or r.get("transfer_id") or "")
        ts = str(r.get("ts") or r.get("timestamp") or "")
        src = str(r.get("src_account") or r.get("src") or r.get("from_account") or "")
        dst = str(r.get("dst_account") or r.get("dst") or r.get("to_account") or "")
        try:
            amount = float(r.get("amount") or 0.0)
        except Exception:
            amount = 0.0
        currency = str(r.get("currency") or "")
        is_fraud = 1 if bool(r.get("is_fraud") or False) else 0
        risk_score = r.get("risk_score")
        try:
            risk_score = float(risk_score) if risk_score is not None else None
        except Exception:
            risk_score = None
        if not tid or not src or not dst:
            continue
        to_upsert.append((tid, ts, src, dst, amount, currency, is_fraud, risk_score))

    if not to_upsert:
        return
    with _connect() as conn:
        conn.executemany(
            """
            INSERT INTO transfers(id, ts, src_account, dst_account, amount, currency, is_fraud, risk_score)
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
              ts=excluded.ts,
              src_account=excluded.src_account,
              dst_account=excluded.dst_account,
              amount=excluded.amount,
              currency=excluded.currency,
              is_fraud=excluded.is_fraud,
              risk_score=excluded.risk_score
            """,
            to_upsert,
        )


def get_ego_network(account_id: str, depth: int = 1, max_nodes: int = 200, risk_only: bool = False) -> Dict[str, Any]:
    """基于 DB 构建账户的邻域(账户-商户二部图)。"""
    with _connect() as conn:
        # 0度: 该账户直接交易的商户
        merchants = conn.execute(
            "SELECT merchant_id, COUNT(1) cnt, SUM(amount) amt, MAX(COALESCE(risk_score,0)) r FROM transactions WHERE account_id=? GROUP BY merchant_id ORDER BY cnt DESC LIMIT ?",
            (account_id, max(1, max_nodes)),
        ).fetchall()

        merchant_ids = [row["merchant_id"] for row in merchants if row["merchant_id"]]

        # 1度: 这些商户涉及的其它账户
        other_accounts = []
        if depth >= 1 and merchant_ids:
            qmarks = ",".join(["?"] * len(merchant_ids))
            other_accounts = conn.execute(
                f"SELECT account_id, merchant_id, COUNT(1) cnt, SUM(amount) amt, MAX(COALESCE(risk_score,0)) r FROM transactions WHERE merchant_id IN ({qmarks}) AND account_id!=? GROUP BY account_id, merchant_id ORDER BY cnt DESC LIMIT ?",
                (*merchant_ids, account_id, max(1, max_nodes)),
            ).fetchall()

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    # 起始账户节点
    nodes.append({"id": account_id, "type": "account", "label": account_id, "risk_level": "high", "risk_score": 1.0})

    # 商户节点与边
    for row in merchants:
        if not row["merchant_id"]:
            continue
        risk_level = "high" if row["r"] and row["r"] >= 0.7 else ("medium" if row["r"] and row["r"] >= 0.4 else "low")
        if risk_only and risk_level == "low":
            continue
        nodes.append({"id": row["merchant_id"], "type": "merchant", "label": row["merchant_id"], "risk_level": risk_level, "risk_score": float(row["r"] or 0.0)})
        edges.append({"source": account_id, "target": row["merchant_id"], "weight": int(row["cnt"]), "total_amount": float(row["amt"] or 0.0)})

    # 其它账户节点与边（通过共同商户）
    for row in other_accounts:
        acc = row["account_id"]
        mer = row["merchant_id"]
        if not acc or not mer:
            continue
        risk_level = "high" if row["r"] and row["r"] >= 0.7 else ("medium" if row["r"] and row["r"] >= 0.4 else "low")
        if risk_only and risk_level == "low":
            continue
        if not any(n["id"] == acc for n in nodes):
            nodes.append({"id": acc, "type": "account", "label": acc, "risk_level": risk_level, "risk_score": float(row["r"] or 0.0)})
        if not any(n["id"] == mer for n in nodes):
            nodes.append({"id": mer, "type": "merchant", "label": mer, "risk_level": risk_level, "risk_score": float(row["r"] or 0.0)})
        edges.append({"source": acc, "target": mer, "weight": int(row["cnt"]), "total_amount": float(row["amt"] or 0.0)})

    # 截断到 max_nodes
    if len(nodes) > max_nodes:
        nodes = nodes[:max_nodes]
        node_ids = {n["id"] for n in nodes}
        edges = [e for e in edges if e["source"] in node_ids and e["target"] in node_ids]

    return {"nodes": nodes, "edges": edges, "source": "db"}


def today_stats() -> Dict[str, Any]:
    """返回今日统计（UTC 日界）。"""
    start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    with _connect() as conn:
        c1 = conn.execute("SELECT COUNT(1) FROM transactions WHERE ts>=?", (start.isoformat(),)).fetchone()[0]
        c2 = conn.execute("SELECT COUNT(1) FROM transactions WHERE ts>=? AND is_fraud=1", (start.isoformat(),)).fetchone()[0]
        amt = conn.execute("SELECT SUM(amount) FROM transactions WHERE ts>=?", (start.isoformat(),)).fetchone()[0]
    return {"today_processed": int(c1 or 0), "today_fraud": int(c2 or 0), "today_amount": float(amt or 0.0)}


def get_account_chain(account_id: str, hops: int = 2, target_account: str | None = None, max_paths: int = 20) -> Dict[str, Any]:
    """在 transfers 表上进行 BFS 寻找账户间路径。
    如果缺少 transfers 数据，返回空结构。
    """
    from collections import deque

    with _connect() as conn:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='transfers'").fetchone()
        if not exists:
            return {"nodes": [], "edges": [], "paths": []}

        # 预载邻接表
        rows = conn.execute("SELECT src_account, dst_account, amount, COALESCE(risk_score,0) r FROM transfers").fetchall()
        if not rows:
            return {"nodes": [], "edges": [], "paths": []}

    adj: Dict[str, List[Tuple[str, float, float]]] = {}
    for r in rows:
        s = r["src_account"]
        d = r["dst_account"]
        amt = float(r["amount"] or 0.0)
        risk = float(r["r"] or 0.0)
        if not s or not d:
            continue
        adj.setdefault(s, []).append((d, amt, risk))

    # BFS 路径搜索
    paths: List[List[str]] = []
    q = deque([(account_id, [account_id], 0)])
    # visited set not required since paths already prevent cycles via membership check
    target = target_account
    while q and len(paths) < max_paths:
        node, path, depth = q.popleft()
        if target and node == target:
            paths.append(path)
            continue
        if depth >= hops:
            continue
        for nxt, amt, risk in adj.get(node, []):
            if nxt in path:  # 避免环
                continue
            new_path = path + [nxt]
            if not target:
                paths.append(new_path)
                if len(paths) >= max_paths:
                    break
            q.append((nxt, new_path, depth + 1))

    # 构建子图
    node_ids = set()
    edges: List[Dict[str, Any]] = []
    for p in paths:
        for i in range(len(p) - 1):
            s, d = p[i], p[i + 1]
            node_ids.add(s)
            node_ids.add(d)
            edges.append({"source": s, "target": d, "weight": 1, "type": "transfer"})
    nodes = [{"id": nid, "type": "account", "label": nid} for nid in sorted(node_ids)]
    return {"nodes": nodes, "edges": edges, "paths": paths}
