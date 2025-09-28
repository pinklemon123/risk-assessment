import time, uuid

def create_presigned_url(obj_type: str, ext: str):
    token = uuid.uuid4().hex
    path = f"incoming/{obj_type}/{int(time.time())}.{ext}"
    url = f"https://upload.example.com/{path}?token={token}"
    return {"upload_url": url, "path_hint": path, "expire_in_seconds": 900}
