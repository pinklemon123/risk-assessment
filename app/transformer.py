import yaml, re, hashlib
from dateutil import parser, tz
from typing import Any, Dict

def load_mapping(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def to_utc(dt_str: str, timezone: str) -> str:
    dt = parser.isoparse(dt_str)
    if dt.tzinfo is None:
        local = tz.gettz(timezone)
        dt = dt.replace(tzinfo=local)
    return dt.astimezone(tz.UTC).isoformat()

def to_decimal(x: Any) -> float:
    return float(x)

def enum_map(val: Any, mapping: Dict[str, str]) -> str:
    return mapping.get(str(val), str(val))

def mask_card(card: str) -> str:
    h = hashlib.sha256(card.encode('utf-8')).hexdigest()[:16]
    return f"card_{h}"

def apply_mapping(row: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    import json, re
    def pick(expr: str):
        if expr == "$":
            return row
        m = re.match(r"^\$\.([\w\-]+)$", expr)
        if not m:
            return None
        return row.get(m.group(1))

    out: Dict[str, Any] = {}
    mp = cfg["mappings"]["transactions"]
    tzname = cfg.get("timezone", "UTC")
    for field, rule in mp.items():
        if isinstance(rule, str):
            out[field] = pick(rule)
        elif isinstance(rule, dict):
            expr = rule.get("expr", "$")
            val = pick(expr)
            op = rule.get("op")
            if op == "to_utc" and val is not None:
                val = to_utc(str(val), tzname)
            elif op == "to_decimal" and val is not None:
                val = to_decimal(val)
            elif op == "enum_map" and val is not None:
                val = enum_map(val, rule.get("params", {}))
            out[field] = val
        else:
            out[field] = None

    privacy = cfg.get("privacy", {})
    if privacy.get("mask_card_id") and out.get("card_id"):
        out["card_id"] = mask_card(str(out["card_id"]))
    return out
