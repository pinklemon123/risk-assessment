import os, json, time
import paramiko
from pathlib import Path
from app.transformer import load_mapping, apply_mapping
from app.dq import validate_row
from app.writer import write_raw, write_clean

HOST = os.getenv("SFTP_HOST", "localhost")
PORT = int(os.getenv("SFTP_PORT", "22"))
USER = os.getenv("SFTP_USER", "user")
PWD  = os.getenv("SFTP_PASS", "pass")
REMOTE_DIR = os.getenv("SFTP_DIR", "/uploads")
LOCAL_TMP = Path(os.getenv("SFTP_TMP", "data/sftp_tmp"))
MAPPING_PATH = os.getenv("MAPPING_PATH", "config/mapping.yaml")

LOCAL_TMP.mkdir(parents=True, exist_ok=True)

def fetch_files():
    t = paramiko.Transport((HOST, PORT))
    t.connect(username=USER, password=PWD)
    sftp = paramiko.SFTPClient.from_transport(t)
    try:
        for fname in sftp.listdir(REMOTE_DIR):
            if not fname.endswith(".jsonl"):
                continue
            remote = f"{REMOTE_DIR}/{fname}"
            local  = LOCAL_TMP / fname
            sftp.get(remote, str(local))
            process_file(local)
            sftp.remove(remote)
    finally:
        sftp.close()
        t.close()

def process_file(path: Path):
    cfg = load_mapping(MAPPING_PATH)
    batch_id = f"sftp_{int(time.time())}_{path.stem}"
    raws = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                raws.append(json.loads(line))
            except:
                continue
    write_raw(batch_id, raws)
    clean = []
    for r in raws:
        mapped = apply_mapping(r, cfg)
        ok, _ = validate_row(mapped, cfg)
        if ok:
            clean.append(mapped)
    write_clean(batch_id, clean)
    print(f"[SFTP] batch {batch_id}: accepted={len(clean)}/{len(raws)}")

if __name__ == "__main__":
    fetch_files()
