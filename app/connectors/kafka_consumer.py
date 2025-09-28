import asyncio, os, json, time
from aiokafka import AIOKafkaConsumer
from app.transformer import load_mapping, apply_mapping
from app.dq import validate_row
from app.writer import write_raw, write_clean

TOPIC = os.getenv("KAFKA_TOPIC", "risk-transactions")
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
GROUP = os.getenv("KAFKA_GROUP", "risk-ingestion")
MAPPING_PATH = os.getenv("MAPPING_PATH", "config/mapping.yaml")

async def main():
    cfg = load_mapping(MAPPING_PATH)
    consumer = AIOKafkaConsumer(
        TOPIC, bootstrap_servers=BOOTSTRAP, group_id=GROUP,
        value_deserializer=lambda m: json.loads(m.decode("utf-8"))
    )
    await consumer.start()
    try:
        while True:
            result = await consumer.getmany(timeout_ms=1000)
            for tp, messages in result.items():
                if not messages:
                    continue
                batch_id = f"kafka_{int(time.time())}"
                raw_rows = [m.value for m in messages]
                write_raw(batch_id, raw_rows)

                clean = []
                for r in raw_rows:
                    mapped = apply_mapping(r, cfg)
                    ok, _ = validate_row(mapped, cfg)
                    if ok:
                        clean.append(mapped)
                write_clean(batch_id, clean)
                print(f"[Kafka] batch {batch_id}: accepted={len(clean)}/{len(raw_rows)}")
    finally:
        await consumer.stop()

if __name__ == "__main__":
    asyncio.run(main())
