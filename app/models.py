from pydantic import BaseModel
from typing import Any, List, Dict

class BatchRequest(BaseModel):
    source_schema_ver: str
    rows: List[Dict[str, Any]]

class BatchResponse(BaseModel):
    accepted: int
    failed: int
    batch_id: str
    feat_ver_hint: str

class PresignResponse(BaseModel):
    upload_url: str
    expire_in_seconds: int = 900
    path_hint: str

class Evidence(BaseModel):
    summary: Dict[str, Any]
    path: List[Dict[str, Any]]
    graph_stats: Dict[str, Any]
