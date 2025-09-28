import os
from fastapi import Header, HTTPException

API_KEYS = set([k.strip() for k in os.getenv("API_KEYS", "test_key").split(",") if k.strip()])
TENANTS = set([t.strip() for t in os.getenv("ALLOWED_TENANTS", "acme").split(",") if t.strip()])

async def require_auth(x_api_key: str = Header(...), x_tenant_id: str = Header(...)):
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if TENANTS and x_tenant_id not in TENANTS:
        raise HTTPException(status_code=403, detail="Tenant not allowed")
    return {"tenant_id": x_tenant_id}
