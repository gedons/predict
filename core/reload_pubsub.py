# app/core/reload_pubsub.py
import json
import threading
import time
from typing import Optional

from core.redis_client import redis_client
from api.predict import load_model_by_id 

CHANNEL = "model_reload"

# a thread-safe event to signal shutdown
subscriber_stop_event = threading.Event()
subscriber_thread: Optional[threading.Thread] = None

def publish_model_reload(model_id: int):
    """
    Publish a reload message. Payload is JSON: {"model_id": <int>}
    """
    payload = json.dumps({"model_id": int(model_id)})
    redis_client.publish(CHANNEL, payload)

def _subscriber_loop():
    pubsub = redis_client.pubsub()
    pubsub.subscribe(CHANNEL)
    try:
        while not subscriber_stop_event.is_set():
            # use non-blocking get_message with short timeout to allow graceful stop
            message = pubsub.get_message(timeout=1.0)
            if not message:
                continue            
            if message.get("type") != "message":
                continue
            data = message.get("data")
            if isinstance(data, bytes):
                try:
                    data = data.decode("utf-8")
                except Exception:
                    # already string likely
                    data = data
            try:
                payload = json.loads(data) # type: ignore
            except Exception:
                print("reload_pubsub: invalid payload, skipping:", data)
                continue
            model_id = payload.get("model_id")
            if model_id is None:
                print("reload_pubsub: no model_id in payload:", payload)
                continue
            print(f"reload_pubsub: received reload for model_id={model_id}, loading...")
            try:
                load_model_by_id(model_id)
                print("reload_pubsub: reload success")
            except Exception as e:
                print("reload_pubsub: reload failed:", e)
    finally:
        try:
            pubsub.close()
        except Exception:
            pass

def start_subscriber_thread():
    global subscriber_thread
    if subscriber_thread and subscriber_thread.is_alive():
        return
    subscriber_stop_event.clear()
    subscriber_thread = threading.Thread(target=_subscriber_loop, daemon=True, name="model-reload-subscriber")
    subscriber_thread.start()
    print("reload_pubsub: subscriber thread started")

def stop_subscriber_thread():
    subscriber_stop_event.set()
    print("reload_pubsub: stop signal set")
    # give thread a moment to exit
    if subscriber_thread:
        subscriber_thread.join(timeout=3.0)
        print("reload_pubsub: subscriber thread joined")
