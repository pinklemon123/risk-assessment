from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import Response
import os
import yaml
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])

MAPPING_PATH = os.getenv("MAPPING_PATH", "config/mapping.yaml")

@router.get("/mapping")
async def get_mapping():
    """获取当前的映射配置 YAML"""
    try:
        if not os.path.exists(MAPPING_PATH):
            logger.warning(f"Mapping file not found: {MAPPING_PATH}")
            # 返回默认配置
            default_config = {
                "tenant_id": "",
                "source_schema_ver": "v1", 
                "timezone": "UTC",
                "base_currency": "CNY",
                "mappings": {"transactions": {}},
                "dq_rules": {
                    "required": ["id", "ts", "src_entity_id", "dst_entity_id", "amount", "currency"],
                    "unique_key": ["id"],
                    "ranges": {"amount": {"min": 0.01, "max": 100000000}}
                },
                "privacy": {"mask_card_id": True}
            }
            return Response(
                content=yaml.dump(default_config, default_flow_style=False, allow_unicode=True),
                media_type="application/x-yaml"
            )
        
        with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return Response(content=content, media_type="application/x-yaml")
    
    except Exception as e:
        logger.error(f"Failed to read mapping file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read mapping file: {str(e)}")

@router.put("/mapping") 
async def put_mapping(payload: dict = Body(...)):
    """更新映射配置 YAML"""
    try:
        content = payload.get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise HTTPException(status_code=400, detail="Content is required and must be a non-empty string")
        
        # 验证 YAML 格式
        try:
            yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise HTTPException(status_code=400, detail=f"Invalid YAML format: {str(e)}")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(MAPPING_PATH), exist_ok=True)
        
        # 写入文件
        with open(MAPPING_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
            
        logger.info(f"Mapping configuration updated successfully: {MAPPING_PATH}")
        return {"ok": True, "message": "Mapping configuration updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update mapping file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update mapping file: {str(e)}")